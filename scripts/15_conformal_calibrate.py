# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Calibrate split-conformal prediction intervals on the 2024 validation split.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Loads the same MTGNN pitch checkpoint the dashboard serves
(``checkpoints/mtgnn/best_model.pt``), runs it over the val split with the same
``wind_mode`` the app uses (``from_field``), denormalises via
``src.training.evaluation`` (identical to ``scripts/04_evaluate.py``), and writes
per-horizon conformal half-widths (µg/m³) to
``outputs/conformal/conformal_intervals.json``.

Nonconformity score = absolute residual in µg/m³. Quantile per horizon via the
finite-sample split-conformal rule ``k = ceil((n + 1) * (1 - alpha))`` (Lei et
al. 2018). Per-station quantiles are emitted alongside a pooled fallback; the
chosen serving mode is decided from the minimum per-station calibration count.

Usage:
    uv run python scripts/15_conformal_calibrate.py
    uv run python scripts/15_conformal_calibrate.py alpha=0.1
    uv run python scripts/15_conformal_calibrate.py output=outputs/conformal/repro.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader

from src.data.loader import _SPLIT_BOUNDS, PM25GraphDataset
from src.training.conformal import calibrate_conformal
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

# The dashboard serves this checkpoint (app.lib.data_access.CHECKPOINT_PATH); the
# intervals must be calibrated on exactly the same weights.
_CHECKPOINT = "checkpoints/mtgnn/best_model.pt"
_DEFAULT_OUTPUT = "outputs/conformal/conformal_intervals.json"

# Serve per-station quantiles only if every (station, horizon) cell has at least
# this many calibration residuals; otherwise fall back to pooled quantiles.
_MIN_PER_STATION = 100


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        return "cpu"
    return requested


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys (device, output, alpha) from Hydra overrides."""
    local_keys = {"device", "output", "alpha"}
    local: dict[str, str] = {}
    hydra_overrides: list[str] = []
    for arg in argv:
        if "=" not in arg:
            continue
        key, _, val = arg.partition("=")
        if key in local_keys:
            local[key] = val
        else:
            hydra_overrides.append(arg)
    return local, hydra_overrides


def _jsonify(obj: object) -> object:
    """Recursively convert ``inf`` half-widths to null so the JSON stays valid."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main() -> None:
    """Calibrate conformal intervals and write the quantiles JSON."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=mtgnn", *overrides])

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    alpha = float(local_args.get("alpha", 0.1))
    output_path = Path(local_args.get("output", _DEFAULT_OUTPUT))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "from_field")

    ds = PM25GraphDataset(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=Path(cfg.data.hotspots_path),
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        split="val",
        window_in=cfg.data.window_in,
        horizons=horizons,
        exclude_stations=list(cfg.data.exclude_stations),
        graph_config={"wind_mode": wind_mode},
    )
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    logger.info(
        "Val split: %d samples, %d stations, wind_mode=%s, alpha=%.3f",
        len(ds),
        ds.n_stations,
        wind_mode,
        alpha,
    )

    # Point forecasts in µg/m³ via the shared evaluation helpers (no ad-hoc denorm).
    model = instantiate(cfg.model, n_stations=ds.n_stations, horizons=horizons)
    ckpt = torch.load(_PROJECT_ROOT / _CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt)
    centers, scales = station_scalers(ds)
    pred_ug = denorm_pred(predict(model, loader, device), centers, scales, ds.n_stations)

    # Ground-truth targets + validity mask (raw µg/m³), same as scripts/04_evaluate.py.
    target_ug, mask, _pers = build_ground_truth(ds, horizons)

    # Rows are sample-major, station-minor (loader concatenates graphs in order;
    # station nodes are contiguous and in ds._station_ids order within each graph).
    n_samples = len(ds)
    row_station_ids = np.tile(ds._station_ids, n_samples)
    assert row_station_ids.shape[0] == pred_ug.shape[0], "row/station alignment mismatch"

    cal = calibrate_conformal(
        pred=pred_ug,
        target=target_ug,
        mask=mask,
        horizons=horizons,
        alpha=alpha,
        row_station_ids=row_station_ids,
        min_per_station=_MIN_PER_STATION,
    )

    min_count = int(cal.get("min_station_count", 0))
    mode = "per_station" if min_count >= _MIN_PER_STATION else "pooled"
    logger.info("Min per-station calibration count = %d → serving mode = %s", min_count, mode)

    v_start, v_end = _SPLIT_BOUNDS["val"]
    result: dict[str, object] = {
        "meta": {
            "checkpoint": _CHECKPOINT,
            "split": "val",
            "split_range": f"{v_start.date()} to {v_end.date()}",
            "wind_mode": wind_mode,
            "alpha": alpha,
            "coverage_target": round(1.0 - alpha, 3),
            "method": (
                "split-conformal, symmetric absolute-residual score in ug/m3; "
                "finite-sample quantile k=ceil((n+1)(1-alpha)) (Lei et al. 2018)"
            ),
            "unit": "ug/m3",
            "mode": mode,
            "n_stations": ds.n_stations,
            "n_val_samples": n_samples,
            "horizons_h": horizons,
            "min_station_count": min_count,
            "lower_clip": 0.0,
            "caveat": (
                "Marginal coverage over the 2024 ERA5 validation distribution. "
                "Live NWP-driven forecasts break exchangeability and may mis-cover."
            ),
        },
        "pooled": cal["pooled"],
        "pooled_coverage": cal["pooled_coverage"],
        "pooled_n": cal["pooled_n"],
        "per_station": cal["per_station"],
        "per_station_coverage": cal["per_station_coverage"],
        "per_station_n": cal["per_station_n"],
    }

    # --- Console summary ---
    print("\n" + "=" * 60)
    print(f"Split-conformal half-widths (µg/m³) @ {(1 - alpha) * 100:.0f}% target")
    print("-" * 60)
    print(f"{'horizon':<10}{'pooled q':>12}{'in-sample cov':>16}{'n':>10}")
    for h in horizons:
        key = f"{h}h"
        print(
            f"{key:<10}{cal['pooled'][key]:>12.2f}"
            f"{cal['pooled_coverage'][key]:>16.3f}{cal['pooled_n'][key]:>10}"
        )
    print("=" * 60)
    print(f"serving mode = {mode} (min per-station count = {min_count})\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonify(result), fh, indent=2, ensure_ascii=False)
    logger.info("Saved conformal intervals to %s", output_path)


if __name__ == "__main__":
    main()
