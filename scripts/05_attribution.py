# [TODO: NSC Disclaimer — see booklet page 44]

"""Run source attribution on peak haze event (March 2024) for Chiang Mai station.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Usage:
    uv run python scripts/05_attribution.py
    uv run python scripts/05_attribution.py checkpoint=checkpoints/mtgnn/best_model.pt
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.loader import PM25GraphDataset, _FULL_INDEX
from src.explain.attribution import batch_attribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chiang Mai station: locate by name in metadata
_CHIANG_MAI_NAMES = {"chiang mai", "เชียงใหม่", "chiangmai"}
_PEAK_MONTH = 3   # March
_PEAK_YEAR = 2024
_HORIZON_IDX = 2  # index 2 = 24h (horizons=[6,12,24,48])
_N_SAMPLES = 10   # top-N peak PM2.5 windows to aggregate
_N_IG_STEPS = 50


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Load MTGNN checkpoint and run attribution on March 2024 peak haze."""
    from omegaconf import OmegaConf
    checkpoint_path = Path(OmegaConf.select(cfg, "checkpoint", default="checkpoints/mtgnn/best_model.pt"))
    device = "cpu"  # attribution runs on CPU

    # ── Load checkpoint ──────────────────────────────────────────────────────
    if not checkpoint_path.exists():
        # Fall back to root checkpoint if mtgnn subdir not found
        checkpoint_path = Path("checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run scripts/03_train.py first."
        )
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ── Instantiate model from MTGNN config ──────────────────────────────────
    model = instantiate(
        cfg.model,
        n_stations=18,
        horizons=list(cfg.data.horizons),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded: epoch=%d  val_rmse=%.4f", ckpt["epoch"], ckpt["val_rmse_24h"])

    # ── Load train dataset (March 2024 is in train split) ────────────────────
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")
    ds = PM25GraphDataset(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=Path(cfg.data.hotspots_path),
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        split="train",
        window_in=cfg.data.window_in,
        horizons=list(cfg.data.horizons),
        graph_config={"wind_mode": wind_mode},
    )
    logger.info("Train dataset: %d samples", len(ds))

    # ── Find Chiang Mai station index ─────────────────────────────────────────
    meta = pd.read_parquet(cfg.data.metadata_path).rename(
        columns={"location_id": "station_id"}
    )
    meta = meta.sort_values("station_id").reset_index(drop=True)
    name_lower = meta["name"].str.lower() if "name" in meta.columns else pd.Series([""] * len(meta))
    cmai_mask = name_lower.apply(lambda n: any(k in n for k in _CHIANG_MAI_NAMES))
    if not cmai_mask.any():
        # Fall back to closest station to Chiang Mai city centre (18.788, 98.985)
        dists = ((meta["lat"] - 18.788) ** 2 + (meta["lon"] - 98.985) ** 2) ** 0.5
        station_idx = int(dists.idxmin())
        station_name = meta.iloc[station_idx].get("name", f"station_{meta.iloc[station_idx]['station_id']}")
    else:
        station_idx = int(cmai_mask.idxmax())
        station_name = meta.iloc[station_idx].get("name", "Chiang Mai")
    logger.info("Target station: %s (idx=%d)", station_name, station_idx)

    # ── Find peak March 2024 anchor timestamps ────────────────────────────────
    anchor_indices = ds._anchor_indices
    anchor_timestamps = _FULL_INDEX[anchor_indices]
    march_2024_mask = (anchor_timestamps.year == _PEAK_YEAR) & (anchor_timestamps.month == _PEAK_MONTH)
    march_indices_pos = np.where(march_2024_mask)[0]

    if len(march_indices_pos) == 0:
        raise RuntimeError("No March 2024 samples found in train dataset. Check split bounds.")
    logger.info("March 2024 anchors available: %d", len(march_indices_pos))

    # Sort by Chiang Mai PM2.5 at anchor time (highest pollution first)
    pm25_at_anchor = ds._pm25_raw[anchor_indices[march_indices_pos], station_idx]
    valid = np.isfinite(pm25_at_anchor)
    if valid.sum() == 0:
        raise RuntimeError("No valid (non-NaN) PM2.5 values for Chiang Mai in March 2024.")

    top_n_pos = march_indices_pos[valid][np.argsort(pm25_at_anchor[valid])[::-1][:_N_SAMPLES]]
    peak_pm25 = pm25_at_anchor[valid][np.argsort(pm25_at_anchor[valid])[::-1][:_N_SAMPLES]]
    logger.info(
        "Top-%d peak PM2.5 values (ug/m3): %s",
        _N_SAMPLES,
        ", ".join(f"{v:.1f}" for v in peak_pm25),
    )

    # ── Collect samples ───────────────────────────────────────────────────────
    samples = [ds[int(pos)] for pos in top_n_pos]

    # ── Run batch attribution ─────────────────────────────────────────────────
    logger.info("Running attribution (n_ig_steps=%d, n_samples=%d)…", _N_IG_STEPS, len(samples))
    reports = batch_attribution(
        model=model,
        samples=samples,
        station_idx=station_idx,
        horizon_idx=_HORIZON_IDX,
        station_name=str(station_name),
        n_ig_steps=_N_IG_STEPS,
        device=device,
    )

    # ── Aggregate across samples ──────────────────────────────────────────────
    all_countries: set[str] = set()
    for r in reports:
        all_countries.update(r["country_attribution"].keys())

    # Mean country attribution across samples
    country_means: dict[str, float] = {}
    for country in all_countries:
        scores = [r["country_attribution"].get(country, 0.0) for r in reports]
        country_means[country] = float(np.mean(scores))

    # Re-normalize to sum = 1
    total = sum(country_means.values())
    if total > 0:
        country_means = {k: v / total for k, v in country_means.items()}
    country_means = dict(sorted(country_means.items(), key=lambda x: -x[1]))

    # Mean IG feature importance
    ig_means: dict[str, float] = {}
    for feat in reports[0]["ig_feature_importance"]:
        ig_means[feat] = float(np.mean([r["ig_feature_importance"][feat] for r in reports]))
    ig_means = dict(sorted(ig_means.items(), key=lambda x: -x[1]))

    # ── Print results ─────────────────────────────────────────────────────────
    horizons = list(cfg.data.horizons)
    horizon_h = horizons[_HORIZON_IDX]
    print(f"\n{'='*60}")
    print(f"Attribution: {station_name}  |  Horizon: {horizon_h}h  |  March {_PEAK_YEAR}")
    print(f"Top-{_N_SAMPLES} peak PM2.5 samples aggregated")
    print(f"{'='*60}")

    print("\n--- Country attribution (transboundary source scores) ---")
    for country, score in country_means.items():
        bar = "#" * int(score * 40)
        print(f"  {country:<15} {score*100:5.1f}%  {bar}")

    print("\n--- IG feature importance (mean |attribution|) ---")
    for feat, val in ig_means.items():
        bar = "#" * int(val / max(ig_means.values()) * 30)
        print(f"  {feat:<15} {val:.4f}  {bar}")

    # ── Save to JSON ──────────────────────────────────────────────────────────
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "station": str(station_name),
        "station_idx": station_idx,
        "horizon_h": horizon_h,
        "year_month": f"{_PEAK_YEAR}-{_PEAK_MONTH:02d}",
        "n_samples": len(samples),
        "peak_pm25_ug_m3": [float(v) for v in peak_pm25],
        "country_attribution": country_means,
        "ig_feature_importance": ig_means,
    }
    out_path = output_dir / "attribution_march2024.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
