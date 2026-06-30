# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Non-graph ML baseline: does a plain gradient-boosted tree match the STGNN?

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Only persistence has been compared so far, so nothing yet justifies the *graph* machinery
specifically (critical review finding #7). Here we train a `HistGradientBoostingRegressor`
per horizon on exactly the same inputs the MTGNN sees - the flattened 24h x 10-feature
window per (station, sample) - but with NO graph, NO wind edges, NO hotspot nodes. Targets
are the normalised `pm25_scaled` value h hours ahead (identical to the GNN's target); we
denormalise predictions and score on the SAME validity mask via `src/training/evaluation.py`.

Inputs are NaN-filled with 0.0 to mirror the loader's `np.nan_to_num` exactly, so the only
difference from the GNN is the model class. If this pooled-station tree matches MTGNN, the
graph adds little; if MTGNN is clearly better, the spatial structure earns its complexity.

Reproduces `outputs/baseline_ml.json`. Uses sklearn (already a dependency) - no new deps.

Usage:
    uv run python scripts/09_ml_baseline.py
    uv run python scripts/09_ml_baseline.py split=test output=outputs/baseline_ml_test.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from sklearn.ensemble import HistGradientBoostingRegressor
from torch_geometric.loader import DataLoader

from src.data.loader import PM25GraphDataset
from src.training.evaluation import (
    build_ground_truth,
    denorm_pred,
    predict,
    rmse_block,
    station_scalers,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_MTGNN_CKPT = "checkpoints/mtgnn/best_model.pt"
_LOCAL_KEYS = {"device", "output", "split", "seed", "ckpt"}

# Wide per-(T_full, N) feature arrays, in the same order the loader stacks them.
_FEATURE_ATTRS = (
    "_pm25_scaled",
    "_hour_sin",
    "_hour_cos",
    "_doy_sin",
    "_doy_cos",
    "_u10",
    "_v10",
    "_t2m",
    "_d2m",
    "_blh",
)


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available - falling back to CPU.")
        return "cpu"
    return requested


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys (device/output/split/seed) from Hydra overrides."""
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


def _build_features(ds: PM25GraphDataset) -> np.ndarray:
    """Flattened input windows per (sample, station) in sample-major, station-minor order.

    Returns an (S*N, window_in*F) float32 array, NaN-filled with 0.0 to mirror the loader.
    """
    anchors = ds._anchor_indices
    offsets = anchors[:, None] + np.arange(-(ds.window_in - 1), 1)[None, :]  # (S, W)
    cols = [np.transpose(getattr(ds, name)[offsets], (0, 2, 1)) for name in _FEATURE_ATTRS]
    feat = np.stack(cols, axis=-1)  # (S, N, W, F)
    s, n = feat.shape[0], feat.shape[1]
    x = feat.reshape(s * n, ds.window_in * len(_FEATURE_ATTRS))
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def _build_scaled_targets(
    ds: PM25GraphDataset, horizons: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised pm25_scaled targets and their validity mask, (S*N, H) each."""
    anchors = ds._anchor_indices
    y = np.stack([ds._pm25_scaled[anchors + h, :] for h in horizons], axis=2)  # (S, N, H)
    m = np.stack(
        [
            (~ds._mask_in_loss[anchors + h, :])
            & (~ds._exclude[anchors + h, :])
            & np.isfinite(ds._pm25_raw[anchors + h, :])
            for h in horizons
        ],
        axis=2,
    )
    s_n = len(anchors) * ds.n_stations
    return y.reshape(s_n, len(horizons)), m.reshape(s_n, len(horizons))


def _train_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    m_train: np.ndarray,
    x_val: np.ndarray,
    horizons: list[int],
    seed: int,
) -> np.ndarray:
    """Fit one HistGBR per horizon on valid train targets; return (S_val*N, H) scaled preds."""
    pred = np.zeros((x_val.shape[0], len(horizons)), dtype=np.float64)
    for h_idx, h in enumerate(horizons):
        vt = m_train[:, h_idx]
        reg = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
        reg.fit(x_train[vt], y_train[vt, h_idx])
        pred[:, h_idx] = reg.predict(x_val)
        logger.info("HistGBR %dh: trained on %d rows, %d iters", h, int(vt.sum()), reg.n_iter_)
    return pred


def _mtgnn_rmse(
    ds_val: PM25GraphDataset,
    loader: DataLoader,
    device: str,
    overrides: list[str],
    horizons: list[int],
    centers: np.ndarray,
    scales: np.ndarray,
    y_ug: np.ndarray,
    mask: np.ndarray,
    ckpt_path: str,
) -> dict[str, float]:
    """Score the trained MTGNN checkpoint on the same mask, for side-by-side comparison."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        mcfg = compose(config_name="config", overrides=["model=mtgnn", *overrides])
    model = instantiate(mcfg.model, n_stations=ds_val.n_stations, horizons=horizons)
    ckpt = torch.load(_PROJECT_ROOT / ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt)
    pred_ug = denorm_pred(predict(model, loader, device), centers, scales, ds_val.n_stations)
    return rmse_block(pred_ug, y_ug, mask, horizons)


def _make_dataset(cfg: object, split: str, horizons: list[int], wind_mode: str) -> PM25GraphDataset:
    """Construct a PM25GraphDataset for a split from the composed config."""
    return PM25GraphDataset(
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


def main() -> None:
    """Train the non-graph baseline, compare to persistence + MTGNN, print + save."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    device = _resolve_device(local_args.get("device", cfg.trainer.device))
    split = local_args.get("split", "val")
    seed = int(local_args.get("seed", 42))
    ckpt = local_args.get("ckpt", _MTGNN_CKPT)
    output_path = Path(local_args.get("output", "outputs/baseline_ml.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")

    ds_train = _make_dataset(cfg, "train", horizons, wind_mode)
    ds_val = _make_dataset(cfg, split, horizons, wind_mode)
    logger.info("train samples=%d  %s samples=%d", len(ds_train), split, len(ds_val))

    x_train = _build_features(ds_train)
    y_train, m_train = _build_scaled_targets(ds_train, horizons)
    x_val = _build_features(ds_val)

    centers, scales = station_scalers(ds_val)
    y_ug, mask, pers_ug = build_ground_truth(ds_val, horizons)
    persistence_rmse = rmse_block(pers_ug, y_ug, mask, horizons)

    pred_scaled = _train_and_predict(x_train, y_train, m_train, x_val, horizons, seed)
    pred_ug = denorm_pred(pred_scaled, centers, scales, ds_val.n_stations)
    baseline_rmse = rmse_block(pred_ug, y_ug, mask, horizons)

    loader = DataLoader(ds_val, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    mtgnn_rmse = _mtgnn_rmse(
        ds_val, loader, device, overrides, horizons, centers, scales, y_ug, mask, ckpt
    )

    results: dict[str, object] = {
        "split": split,
        "n_stations": ds_val.n_stations,
        "n_samples": len(ds_val),
        "horizons_h": horizons,
        "seed": seed,
        "model": "HistGradientBoostingRegressor per horizon, flattened 24h x 10-feature window",
        "note": (
            "Non-graph baseline on the SAME inputs/targets/mask as MTGNN (see "
            "src/training/evaluation.py). NaN-filled with 0.0 to mirror the loader. RMSE in ug/m3."
        ),
        "persistence_rmse_ug_m3": persistence_rmse,
        "baseline_ml_rmse_ug_m3": baseline_rmse,
        "mtgnn_rmse_ug_m3": mtgnn_rmse,
        "baseline_beats_persistence": {
            f"{h}h": baseline_rmse[f"{h}h"] < persistence_rmse[f"{h}h"] for h in horizons
        },
        "baseline_beats_mtgnn": {
            f"{h}h": baseline_rmse[f"{h}h"] < mtgnn_rmse[f"{h}h"] for h in horizons
        },
    }

    print("\n" + "=" * 64)
    print(f"{'Method':<22}" + "".join(f"{f'{h}h':>10}" for h in horizons))
    print("-" * 64)
    for label, r in (
        ("persistence", persistence_rmse),
        ("baseline (HistGBR)", baseline_rmse),
        ("mtgnn", mtgnn_rmse),
    ):
        print(f"{label:<22}" + "".join(f"{r[f'{h}h']:>10.2f}" for h in horizons))
    print("=" * 64 + "  (RMSE ug/m3, lower is better)\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Saved ML baseline results to %s", output_path)


if __name__ == "__main__":
    main()
