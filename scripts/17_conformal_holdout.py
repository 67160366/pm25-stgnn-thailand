# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Measure HELD-OUT empirical coverage of split-conformal intervals on 2025 test.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

``scripts/15_conformal_calibrate.py`` calibrates split-conformal half-widths on
the 2024 validation split and reports *in-sample* coverage by construction (the
same residuals both set the quantile and are scored against it). This script
does NOT recalibrate — it loads the frozen quantiles from
``outputs/conformal/conformal_intervals.json``, runs the identical checkpoint
over the 2025 TEST split (never touched during calibration), and reports the
honest held-out coverage: the fraction of true test targets that actually fall
inside the ``pred ± half-width`` band.

Half-width resolution mirrors ``app.lib.inference.conformal_halfwidths``
exactly: per-station quantile when the calibration served in ``per_station``
mode and the station has a finite quantile for that horizon, otherwise the
pooled quantile for that horizon. That logic is small and pure, so it is
reimplemented here (as a vectorised per-row lookup) rather than importing
``app.lib.inference``, which pulls in Streamlit/session-state machinery not
needed for a batch script.

Usage:
    uv run python scripts/17_conformal_holdout.py
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

# Must match the checkpoint the intervals were calibrated on (script 15) and the
# checkpoint the dashboard serves (app.lib.data_access.CHECKPOINT_PATH).
_CHECKPOINT = "checkpoints/mtgnn/best_model.pt"
_INTERVALS_PATH = "outputs/conformal/conformal_intervals.json"
_DEFAULT_OUTPUT = "outputs/conformal/holdout_coverage_test2025.json"


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        return "cpu"
    return requested


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys (device, output, intervals) from Hydra overrides."""
    local_keys = {"device", "output", "intervals"}
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
    """Recursively convert ``inf``/``nan`` floats to null so the JSON stays valid."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _resolve_halfwidths(conf: dict, horizons: list[int], row_station_ids: np.ndarray) -> np.ndarray:
    """Resolve the per-row, per-horizon half-width actually served by the app.

    Mirrors ``app.lib.inference.conformal_halfwidths``: prefer the per-station
    quantile when the calibration ran in ``per_station`` mode and that station
    has a finite quantile at the given horizon; otherwise fall back to the
    pooled quantile for that horizon.

    Args:
        conf: Parsed ``conformal_intervals.json``.
        horizons: Forecast horizons (hours), column order.
        row_station_ids: ``(M,)`` station id per row.

    Returns:
        ``(M, H)`` array of half-widths in µg/m³ (``nan`` where unresolved).
    """
    pooled: dict = conf.get("pooled", {})
    mode = conf.get("meta", {}).get("mode", "pooled")
    per_station: dict = conf.get("per_station", {}) if mode == "per_station" else {}

    n_rows = row_station_ids.shape[0]
    out = np.full((n_rows, len(horizons)), np.nan, dtype=np.float64)
    for j, h in enumerate(horizons):
        key = f"{h}h"
        pooled_q = pooled.get(key)
        pooled_val = float(pooled_q) if pooled_q is not None else np.nan
        col = np.full(n_rows, pooled_val, dtype=np.float64)
        if per_station:
            for sid in np.unique(row_station_ids):
                station_q = per_station.get(str(int(sid)), {}).get(key)
                if station_q is None:
                    continue
                col[row_station_ids == sid] = float(station_q)
        out[:, j] = col
    return out


def _coverage_and_width(
    pred: np.ndarray,
    target: np.ndarray,
    halfwidths: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, int]:
    """Empirical coverage, mean applied half-width, and scored count for one slice.

    Args:
        pred: ``(k,)`` point forecasts.
        target: ``(k,)`` ground-truth values.
        halfwidths: ``(k,)`` half-widths actually applied per row.
        mask: ``(k,)`` bool; ``True`` where the target is valid.

    Returns:
        ``(coverage, mean_halfwidth, n_scored)``; ``coverage``/``mean_halfwidth``
        are ``nan`` if nothing is scored.
    """
    valid = mask & np.isfinite(halfwidths)
    if not np.any(valid):
        return float("nan"), float("nan"), 0
    p, t, w = pred[valid], target[valid], halfwidths[valid]
    lower = np.maximum(p - w, 0.0)
    upper = p + w
    covered = (t >= lower) & (t <= upper)
    return float(np.mean(covered)), float(np.mean(w)), int(valid.sum())


def main() -> None:
    """Score frozen conformal intervals against the 2025 test split (held-out)."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=mtgnn", *overrides])

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    intervals_path = Path(local_args.get("intervals", _INTERVALS_PATH))
    output_path = Path(local_args.get("output", _DEFAULT_OUTPUT))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "from_field")

    with open(_PROJECT_ROOT / intervals_path, encoding="utf-8") as fh:
        conf = json.load(fh)
    alpha = float(conf.get("meta", {}).get("alpha", 0.1))
    mode = conf.get("meta", {}).get("mode", "pooled")
    logger.info(
        "Loaded frozen intervals from %s (calibration mode=%s, alpha=%.3f, target=%.1f%%)",
        intervals_path,
        mode,
        alpha,
        (1 - alpha) * 100,
    )

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

    model = instantiate(cfg.model, n_stations=ds.n_stations, horizons=horizons)
    ckpt = torch.load(_PROJECT_ROOT / _CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt)
    centers, scales = station_scalers(ds)
    pred_ug = denorm_pred(predict(model, loader, device), centers, scales, ds.n_stations)

    target_ug, mask, _pers = build_ground_truth(ds, horizons)

    n_samples = len(ds)
    row_station_ids = np.tile(ds._station_ids, n_samples)
    assert row_station_ids.shape[0] == pred_ug.shape[0], "row/station alignment mismatch"

    halfwidths = _resolve_halfwidths(conf, horizons, row_station_ids)

    t_start, t_end = _SPLIT_BOUNDS["test"]

    pooled_cov: dict[str, float] = {}
    pooled_width: dict[str, float] = {}
    pooled_n: dict[str, int] = {}
    station_keys = [str(int(sid)) for sid in np.unique(row_station_ids)]
    per_station_cov: dict[str, dict[str, float]] = {sid: {} for sid in station_keys}
    per_station_width: dict[str, dict[str, float]] = {sid: {} for sid in station_keys}
    per_station_n: dict[str, dict[str, int]] = {sid: {} for sid in station_keys}

    for j, h in enumerate(horizons):
        key = f"{h}h"
        cov, width, n = _coverage_and_width(
            pred_ug[:, j], target_ug[:, j], halfwidths[:, j], mask[:, j]
        )
        pooled_cov[key] = cov
        pooled_width[key] = width
        pooled_n[key] = n

        for sid in np.unique(row_station_ids):
            sel = row_station_ids == sid
            s_cov, s_width, s_n = _coverage_and_width(
                pred_ug[sel, j], target_ug[sel, j], halfwidths[sel, j], mask[sel, j]
            )
            skey = str(int(sid))
            per_station_cov[skey][key] = s_cov
            per_station_width[skey][key] = s_width
            per_station_n[skey][key] = s_n

    # Best/worst covered station per horizon (finite coverage only).
    extremes: dict[str, dict[str, object]] = {}
    for h in horizons:
        key = f"{h}h"
        finite = {sid: c[key] for sid, c in per_station_cov.items() if math.isfinite(c[key])}
        if not finite:
            continue
        worst_sid = min(finite, key=finite.get)
        best_sid = max(finite, key=finite.get)
        extremes[key] = {
            "worst_station": worst_sid,
            "worst_coverage": round(finite[worst_sid], 4),
            "best_station": best_sid,
            "best_coverage": round(finite[best_sid], 4),
        }

    result: dict[str, object] = {
        "meta": {
            "checkpoint": _CHECKPOINT,
            "split": "test",
            "split_range": f"{t_start.date()} to {t_end.date()}",
            "wind_mode": wind_mode,
            "alpha": alpha,
            "coverage_target": round(1.0 - alpha, 3),
            "calibration_source": intervals_path.as_posix(),
            "calibration_mode": mode,
            "method": (
                "held-out empirical coverage of frozen split-conformal half-widths "
                "(no recalibration); scored on the 2025 test split, unseen during "
                "calibration (Lei et al. 2018 split-conformal quantiles from "
                "scripts/15_conformal_calibrate.py)"
            ),
            "unit": "ug/m3",
            "n_stations": ds.n_stations,
            "n_test_samples": n_samples,
            "horizons_h": horizons,
            "lower_clip": 0.0,
        },
        "pooled_coverage": pooled_cov,
        "pooled_mean_halfwidth": pooled_width,
        "pooled_n": pooled_n,
        "per_station_coverage": per_station_cov,
        "per_station_mean_halfwidth": per_station_width,
        "per_station_n": per_station_n,
        "extremes_per_horizon": extremes,
    }

    print("\n" + "=" * 70)
    print(f"HELD-OUT coverage on 2025 test (target {(1 - alpha) * 100:.0f}%)")
    print("-" * 70)
    print(f"{'horizon':<10}{'coverage':>12}{'mean halfwidth':>18}{'n':>10}")
    for h in horizons:
        key = f"{h}h"
        print(f"{key:<10}{pooled_cov[key]:>12.4f}" f"{pooled_width[key]:>18.2f}{pooled_n[key]:>10}")
    print("-" * 70)
    for h in horizons:
        key = f"{h}h"
        ext = extremes.get(key)
        if ext is None:
            continue
        print(
            f"{key}: worst station {ext['worst_station']} "
            f"({ext['worst_coverage']:.3f}), best station {ext['best_station']} "
            f"({ext['best_coverage']:.3f})"
        )
    print("=" * 70 + "\n")

    output_path = _PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonify(result), fh, indent=2, ensure_ascii=False)
    logger.info("Saved held-out coverage report to %s", output_path)


if __name__ == "__main__":
    main()
