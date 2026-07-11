# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Re-score existing PM2.5 point forecasts as exceedance (binary event) forecasts.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Persistence dominates RMSE exactly when PM2.5 does not change level; it is
weakest at the moments a forecaster actually cares about — level crossings.
This script re-scores the SAME point forecasts already evaluated by
``scripts/11_ablation_eval.py`` (RMSE-based) as a binary classification
problem: "will PM2.5 exceed threshold X in H hours?" No retraining, no new
data — pure post-hoc re-scoring via ``src/training/exceedance.py``.

Checkpoints scored (same held-out 2025 test split used by
``outputs/evaluation_test2025.json``, so ``n_samples``/``n_stations`` match):
    primary   -- checkpoints_split2/mtgnn (3-way split retrain, the held-out
                 report protocol behind outputs/evaluation_test2025.json).
    secondary -- checkpoints/mtgnn/best_model.pt (the demo/app checkpoint;
                 trained under the original 2-way split where 2025 was its
                 *validation* split used for early-stop model selection, not
                 gradient updates -- see docs/SESSION8_NOTES.md section 1).
                 Scored here on the same 2025 window for comparability, but
                 this is NOT a fully held-out evaluation for that checkpoint;
                 the JSON note flags this explicitly.

Thresholds: 37.5 ug/m3 (Thai AQI orange boundary; matches
``app/lib/telegram.DEFAULT_ALERT_THRESHOLD_UG``) and 75.0 ug/m3 (red / roadmap
headline "will PM2.5 exceed 75 ug/m3"). Horizons: 6h/12h/24h/48h.

No-leakage climatology: the constant reference probability is the observed
exceedance base rate over the TRAIN+VAL portion of the dataset (2022-01-01 to
2024-12-31 23:00) -- never the scored 2025 test period. See
``src/training/exceedance.py`` module docstring.

Usage:
    uv run python scripts/18_exceedance_eval.py
    uv run python scripts/18_exceedance_eval.py device=cpu
    uv run python scripts/18_exceedance_eval.py output=outputs/exceedance_repro.json
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
from src.training.evaluation import (
    build_ground_truth,
    denorm_pred,
    predict,
    station_scalers,
)
from src.training.exceedance import (
    binary_event,
    brier_score,
    climatology_base_rate,
    exceedance_scores,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_LOCAL_KEYS = {"device", "output"}

_THRESHOLDS_UGM3: list[float] = [37.5, 75.0]

# (label, checkpoint path relative to project root, human note)
_CHECKPOINTS: list[tuple[str, str, str]] = [
    (
        "primary",
        "checkpoints_split2/mtgnn/best_model.pt",
        "3-way split retrain (train 2022-23 / val 2024 / test 2025); the held-out "
        "report protocol behind outputs/evaluation_test2025.json.",
    ),
    (
        "secondary",
        "checkpoints/mtgnn/best_model.pt",
        "demo/app checkpoint; original split where 2025 was its VALIDATION split "
        "(used for early-stop model selection, not gradient updates) -- see "
        "docs/SESSION8_NOTES.md section 1. Scored on the same 2025 window for "
        "comparability, NOT a fully held-out evaluation for this checkpoint.",
    ),
]


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available - falling back to CPU.")
        return "cpu"
    return requested


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys (device, output) from Hydra config overrides."""
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


def _jsonify(obj: object) -> object:
    """Recursively convert ``inf``/``nan`` floats to null so the JSON stays valid."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _round4(x: object) -> object:
    """Round a float to 4dp, passing ``None``/int through unchanged."""
    return round(x, 4) if isinstance(x, float) else x


def _format_block(scored: dict[str, object], *, with_bss: bool) -> dict[str, object]:
    """Trim an ``exceedance_scores`` dict to the JSON-reported fields.

    Args:
        scored: Output of ``exceedance_scores``.
        with_bss: Whether to include Brier skill score (model/persistence
            forecasters only; climatology is scored via Brier alone).

    Returns:
        Dict with rounded pod/far/csi/accuracy/brier(/bss)/base_rate_test and
        the raw contingency counts.
    """
    block = {
        "pod": _round4(scored["pod"]),
        "far": _round4(scored["far"]),
        "csi": _round4(scored["csi"]),
        "accuracy": _round4(scored["accuracy"]),
        "brier": _round4(scored["brier"]),
        "base_rate_test": _round4(scored["base_rate"]),
        "n": scored["n"],
        "counts": scored["counts"],
    }
    if with_bss:
        block["bss"] = _round4(scored["bss"])
    return block


def _load_mtgnn(
    ckpt_rel: str, ds: PM25GraphDataset, horizons: list[int], overrides: list[str]
) -> torch.nn.Module:
    """Instantiate the (full-gate) MTGNN model and load a checkpoint's weights."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        mcfg = compose(config_name="config", overrides=["model=mtgnn", *overrides])
    model = instantiate(mcfg.model, n_stations=ds.n_stations, horizons=horizons)
    ckpt = torch.load(_PROJECT_ROOT / ckpt_rel, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt)
    return model.eval()


def _trainval_base_rates(ds: PM25GraphDataset) -> dict[float, float]:
    """No-leakage climatology base rate per threshold over train+val (2022-2024).

    Args:
        ds: A constructed test-split dataset (arrays span the full 2022-2025
            index regardless of ``split``; only ``_anchor_indices`` differ).

    Returns:
        Dict mapping threshold (µg/m³) to the observed exceedance frequency
        over ``[train_start, val_end]``, never the scored test period.
    """
    t_start = _SPLIT_BOUNDS["train"][0]
    t_end = _SPLIT_BOUNDS["val"][1]
    full_idx = ds._timestamps
    i0 = int(full_idx.searchsorted(t_start, side="left"))
    i1 = int(full_idx.searchsorted(t_end, side="right"))
    values = ds._pm25_raw[i0:i1, :]
    valid = (~ds._mask_in_loss[i0:i1, :]) & (~ds._exclude[i0:i1, :]) & np.isfinite(values)
    return {thr: climatology_base_rate(values, thr, valid) for thr in _THRESHOLDS_UGM3}


def _score_table(
    pred_ug: np.ndarray,
    y_ug: np.ndarray,
    mask: np.ndarray,
    horizons: list[int],
    base_rate_trainval: dict[float, float],
) -> dict[str, dict[str, object]]:
    """Per-horizon, per-threshold exceedance table for one deterministic forecaster."""
    table: dict[str, dict[str, object]] = {}
    for j, h in enumerate(horizons):
        hkey = f"{h}h"
        table[hkey] = {}
        for thr in _THRESHOLDS_UGM3:
            scored = exceedance_scores(
                pred_ug[:, j], y_ug[:, j], mask[:, j], thr, base_rate_trainval[thr]
            )
            table[hkey][str(thr)] = _format_block(scored, with_bss=True)
    return table


def _climatology_table(
    y_ug: np.ndarray,
    mask: np.ndarray,
    horizons: list[int],
    base_rate_trainval: dict[float, float],
) -> dict[str, dict[str, object]]:
    """Climatology-only table: Brier of the constant train+val reference probability.

    Independent of any forecaster's point prediction (the climatology forecast
    is a constant), so this is computed once and shared across checkpoints.
    """
    table: dict[str, dict[str, object]] = {}
    for j, h in enumerate(horizons):
        hkey = f"{h}h"
        table[hkey] = {}
        for thr in _THRESHOLDS_UGM3:
            obs_event = binary_event(y_ug[:, j], thr)
            clim_prob = np.full_like(y_ug[:, j], base_rate_trainval[thr], dtype=np.float64)
            bs_clim = brier_score(clim_prob, obs_event, mask[:, j])
            base_rate_test = climatology_base_rate(y_ug[:, j], thr, mask[:, j])
            table[hkey][str(thr)] = {
                "brier": _round4(bs_clim),
                "base_rate_trainval": _round4(base_rate_trainval[thr]),
                "base_rate_test": _round4(base_rate_test),
            }
    return table


def main() -> None:
    """Re-score the 2025 held-out test forecasts as exceedance events; save JSON."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    output_path = Path(local_args.get("output", "outputs/exceedance_test2025.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")

    ds = PM25GraphDataset(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=Path(cfg.data.hotspots_path),
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        split="test",
        window_in=cfg.data.window_in,
        horizons=horizons,
        exclude_stations=list(cfg.data.exclude_stations),
        graph_config={"wind_mode": wind_mode},
    )
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    logger.info(
        "Test split (held-out): %d samples, %d stations, wind_mode=%s",
        len(ds),
        ds.n_stations,
        wind_mode,
    )

    centers, scales = station_scalers(ds)
    y_ug, mask, pers_ug = build_ground_truth(ds, horizons)

    # --- No-leakage climatology: base rate over train+val (2022-01-01..2024-12-31), never test ---
    base_rate_trainval = _trainval_base_rates(ds)
    logger.info(
        "Climatology base rate (train+val 2022-2024, no test leakage): %s", base_rate_trainval
    )
    t_start, t_end = _SPLIT_BOUNDS["train"][0], _SPLIT_BOUNDS["val"][1]
    t_test_start, t_test_end = _SPLIT_BOUNDS["test"]

    # --- Persistence + climatology tables (shared across checkpoints: same test data) ---
    persistence_table = _score_table(pers_ug, y_ug, mask, horizons, base_rate_trainval)
    climatology_table = _climatology_table(y_ug, mask, horizons, base_rate_trainval)

    # --- Each checkpoint's model forecast ---
    checkpoints_out: dict[str, dict[str, object]] = {}
    for label, ckpt_rel, ckpt_note in _CHECKPOINTS:
        ckpt_path = _PROJECT_ROOT / ckpt_rel
        if not ckpt_path.exists():
            logger.warning("Checkpoint missing for %s (%s) - skipping.", label, ckpt_path)
            continue
        model = _load_mtgnn(ckpt_rel, ds, horizons, overrides)
        pred_ug = denorm_pred(predict(model, loader, device), centers, scales, ds.n_stations)
        model_table = _score_table(pred_ug, y_ug, mask, horizons, base_rate_trainval)
        checkpoints_out[label] = {
            "checkpoint": ckpt_rel,
            "note": ckpt_note,
            "model": model_table,
        }
        logger.info("Scored checkpoint '%s' (%s)", label, ckpt_rel)

    event_def_note = "event = value > threshold (strict); see src/training/exceedance.py"
    result: dict[str, object] = {
        "meta": {
            "split": "test",
            "split_range": f"{t_test_start.date()} to {t_test_end.date()}",
            "n_stations": ds.n_stations,
            "n_samples": len(ds),
            "horizons_h": horizons,
            "thresholds_ugm3": _THRESHOLDS_UGM3,
            "wind_mode": wind_mode,
            "climatology_period": (
                f"{t_start.date()} to {t_end.date()} (train+val; disjoint from the "
                "scored test period, no leakage)"
            ),
            "event_definition": event_def_note,
        },
        "base_rate_trainval_ugm3": {str(thr): _round4(v) for thr, v in base_rate_trainval.items()},
        "persistence": persistence_table,
        "climatology": climatology_table,
        "checkpoints": checkpoints_out,
        "note": (
            "Deterministic (model/persistence) forecasters are scored for Brier as a "
            "degenerate probabilistic forecast p in {0,1} derived from the point forecast "
            "(Brier 1950; Wilks 2011 sec. 8.4.2). Climatology is a constant probability "
            "equal to the exceedance base rate over train+val (2022-2024) ONLY -- never "
            "the scored 2025 test period -- so its Brier score is not optimistic by "
            "construction (see src/training/exceedance.py). BSS = 1 - BS/BS_clim; positive "
            "means the forecaster beats climatology at that horizon/threshold. POD/FAR/CSI "
            "are undefined (reported null) when their denominator is zero, e.g. FAR is "
            "null if the forecaster never predicted an event, and POD is null if the "
            "threshold was never actually exceeded in this sample -- this happens at 75 "
            "ug/m3 for some horizons; check 'n'/'counts' before trusting a ratio. This is a "
            "pure re-scoring of already-computed point forecasts: no retraining, no new "
            "data, same test split and mask as outputs/evaluation_test2025.json "
            "(n_samples/n_stations should match)."
        ),
    }

    # --- Console summary ---
    print("\n" + "=" * 78)
    print("EXCEEDANCE RE-SCORING on 2025 held-out test")
    print("-" * 78)
    for thr in _THRESHOLDS_UGM3:
        tkey = str(thr)
        print(f"\nthreshold = {thr} ug/m3 (base_rate_trainval={base_rate_trainval[thr]:.4f})")
        header = f"{'':<12}{'':<14}" + "".join(f"{f'{h}h':>10}" for h in horizons)
        print(header)
        for label in ("model (primary)", "model (secondary)", "persistence"):
            if label.startswith("model"):
                ck_label = "primary" if "primary" in label else "secondary"
                if ck_label not in checkpoints_out:
                    continue
                table = checkpoints_out[ck_label]["model"]
            else:
                table = persistence_table
            row_pod = "".join(
                f"{(table[f'{h}h'][tkey]['pod'] or float('nan')):>10.3f}" for h in horizons
            )
            print(f"{label:<26}POD  {row_pod}")
    print("=" * 78 + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonify(result), fh, indent=2, ensure_ascii=False)
    logger.info("Saved exceedance re-scoring to %s", output_path)


if __name__ == "__main__":
    main()
