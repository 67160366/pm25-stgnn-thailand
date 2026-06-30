# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Significance / confidence intervals for the MTGNN vs persistence RMSE gap.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

The point-estimate gaps (MTGNN beats persistence by only +0.3% @24h and +2.4% @48h on
val 2025) are tiny, so a single number cannot say whether the long-horizon edge is real
or noise. This script attaches a confidence interval via a *paired block bootstrap*:

    * Paired - the same resampled days score both methods, so the bootstrap targets the
      RMSE *difference* directly (per-day errors are highly correlated between methods;
      pairing removes that shared variance).
    * Block  - the resampling unit is a whole forecast-origin DAY, not an individual
      (hour, station) row. Hourly errors within a day, and across the 18 stations, are
      strongly autocorrelated; an i.i.d. row bootstrap treats them as independent and
      badly over-states significance. Resampling calendar-day blocks preserves the
      within-day dependence.

For each horizon we report the point estimate, the bootstrap mean, the 95% percentile CI
of both the absolute RMSE difference (ug/m3) and the percentage improvement, the bootstrap
probability that the model beats persistence, and whether the CI excludes zero.

Denormalisation, masking and the persistence baseline are reused verbatim from
``src/training/evaluation.py`` so the numbers are identical to ``scripts/04_evaluate.py``.

Reproduces ``outputs/significance_val2025.json``.

Usage:
    uv run python scripts/07_significance.py
    uv run python scripts/07_significance.py split=test output=outputs/significance_test2025.json
    uv run python scripts/07_significance.py n_boot=5000 seed=0 device=cpu
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader

from src.data.loader import PM25GraphDataset
from src.training.evaluation import (
    build_ground_truth,
    denorm_pred,
    predict,
    station_scalers,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_MTGNN_CKPT = "checkpoints/mtgnn/best_model.pt"
_LOCAL_KEYS = {"device", "output", "split", "n_boot", "seed", "ckpt"}


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available - falling back to CPU.")
        return "cpu"
    return requested


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys (device/output/split/n_boot/seed) from Hydra overrides."""
    local: dict[str, str] = {}
    overrides: list[str] = []
    for arg in argv:
        if "=" not in arg:
            continue
        key, _, val = arg.partition("=")
        if key in _LOCAL_KEYS:
            local[key] = val
        else:
            overrides.append(arg)
    return local, overrides


def _load_mtgnn(
    ds: PM25GraphDataset, horizons: list[int], overrides: list[str], ckpt: str
) -> torch.nn.Module:
    """Instantiate MTGNN from Hydra config and load the trained checkpoint."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        mcfg = compose(config_name="config", overrides=["model=mtgnn", *overrides])
    model = instantiate(mcfg.model, n_stations=ds.n_stations, horizons=horizons)
    state = torch.load(_PROJECT_ROOT / ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state)
    return model


def _row_day_ids(ds: PM25GraphDataset) -> np.ndarray:
    """Integer forecast-origin-day id for each prediction row (sample-major, station-minor)."""
    anchor_norm = pd.DatetimeIndex(ds._timestamps[ds._anchor_indices]).normalize()
    codes, _ = pd.factorize(anchor_norm)
    return np.repeat(codes, ds.n_stations)


def _bootstrap_horizon(
    se_model: np.ndarray,
    se_pers: np.ndarray,
    row_day: np.ndarray,
    valid: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Paired by-day block bootstrap of the (persistence - model) RMSE gap for one horizon."""
    day_ids = row_day[valid]
    _, comp = np.unique(day_ids, return_inverse=True)
    d = int(comp.max()) + 1 if comp.size else 0
    n_d = np.bincount(comp, minlength=d).astype(np.float64)
    sm_d = np.bincount(comp, weights=se_model[valid], minlength=d)
    sp_d = np.bincount(comp, weights=se_pers[valid], minlength=d)

    rmse_m0 = float(np.sqrt(sm_d.sum() / n_d.sum()))
    rmse_p0 = float(np.sqrt(sp_d.sum() / n_d.sum()))
    diff0 = rmse_p0 - rmse_m0

    # Resample whole days with replacement, B times (fully vectorised over the bootstrap).
    idx = rng.integers(0, d, size=(n_boot, d))
    tot_n = n_d[idx].sum(axis=1)
    rmse_m = np.sqrt(sm_d[idx].sum(axis=1) / tot_n)
    rmse_p = np.sqrt(sp_d[idx].sum(axis=1) / tot_n)
    diff = rmse_p - rmse_m
    pct = 100.0 * diff / rmse_p

    lo_d, hi_d = np.percentile(diff, [2.5, 97.5])
    lo_p, hi_p = np.percentile(pct, [2.5, 97.5])
    return {
        "rmse_model": round(rmse_m0, 3),
        "rmse_persistence": round(rmse_p0, 3),
        "rmse_diff_point": round(diff0, 3),
        "pct_improvement_point": round(100.0 * diff0 / rmse_p0, 2),
        "rmse_diff_boot_mean": round(float(diff.mean()), 3),
        "rmse_diff_ci95": [round(float(lo_d), 3), round(float(hi_d), 3)],
        "pct_improvement_ci95": [round(float(lo_p), 2), round(float(hi_p), 2)],
        "prob_model_better": round(float((diff > 0).mean()), 3),
        "significant_at_95": bool(lo_d > 0),
        "n_days": d,
        "n_valid_rows": int(valid.sum()),
    }


def main() -> None:
    """Bootstrap the MTGNN vs persistence RMSE gap per horizon; print + save results."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    split = local_args.get("split", "val")
    n_boot = int(local_args.get("n_boot", 2000))
    seed = int(local_args.get("seed", 42))
    ckpt = local_args.get("ckpt", _MTGNN_CKPT)
    output_path = Path(local_args.get("output", "outputs/significance_val2025.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")

    ds = PM25GraphDataset(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=Path(cfg.data.hotspots_path),
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        split=split,
        window_in=cfg.data.window_in,
        horizons=horizons,
        exclude_stations=list(cfg.data.exclude_stations),
        graph_config={"wind_mode": wind_mode},
    )
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    logger.info("Split=%s: %d samples, %d stations, B=%d", split, len(ds), ds.n_stations, n_boot)

    centers, scales = station_scalers(ds)
    y_ug, mask, pers_ug = build_ground_truth(ds, horizons)
    raw_pred = predict(_load_mtgnn(ds, horizons, overrides, ckpt), loader, device)
    pred_ug = denorm_pred(raw_pred, centers, scales, ds.n_stations)

    se_model = (pred_ug - y_ug) ** 2
    se_pers = (pers_ug - y_ug) ** 2
    row_day = _row_day_ids(ds)
    rng = np.random.default_rng(seed)

    per_horizon: dict[str, object] = {}
    for h_idx, h in enumerate(horizons):
        per_horizon[f"{h}h"] = _bootstrap_horizon(
            se_model[:, h_idx], se_pers[:, h_idx], row_day, mask[:, h_idx], n_boot, rng
        )

    results: dict[str, object] = {
        "split": split,
        "n_stations": ds.n_stations,
        "n_samples": len(ds),
        "horizons_h": horizons,
        "n_boot": n_boot,
        "seed": seed,
        "method": (
            "Paired block bootstrap by forecast-origin day (resample whole calendar days "
            "with replacement) of the RMSE difference persistence - MTGNN, in ug/m3. "
            "Positive diff/pct means MTGNN beats persistence. RMSE, mask and persistence "
            "from src/training/evaluation.py (identical to scripts/04_evaluate.py)."
        ),
        "per_horizon": per_horizon,
    }

    print("\n" + "=" * 78)
    header = f"{'Horizon':<9}{'MTGNN':>9}{'Persist':>9}{'diff':>8}{'%impr':>8}"
    print(f"{header}    {'95% CI (% impr)':>18}  sig")
    print("-" * 78)
    for h in horizons:
        r = per_horizon[f"{h}h"]
        ci = r["pct_improvement_ci95"]  # type: ignore[index]
        print(
            f"{f'{h}h':<9}{r['rmse_model']:>9.2f}{r['rmse_persistence']:>9.2f}"  # type: ignore[index]
            f"{r['rmse_diff_point']:>8.2f}{r['pct_improvement_point']:>8.2f}"  # type: ignore[index]
            f"  [{ci[0]:>7.2f},{ci[1]:>7.2f}]   {'YES' if r['significant_at_95'] else 'no'}"  # type: ignore[index]
        )
    print("=" * 78 + "  (RMSE ug/m3; sig = 95% CI of RMSE diff excludes 0)\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Saved significance results to %s", output_path)


if __name__ == "__main__":
    main()
