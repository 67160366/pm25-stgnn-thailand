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
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData

from src.data.loader import _FULL_INDEX, PM25GraphDataset
from src.explain.attribution import batch_attribution
from src.models.base import PM25ModelBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _hotspot_occlusion_delta(
    model: PM25ModelBase,
    data: HeteroData,
    station_idx: int,
    horizon_idx: int,
    device: str = "cpu",
) -> float | None:
    """Return the normalized prediction drop when ALL hotspots are zeroed.

    Computes ``pred_full - pred_with_all_hotspots_zeroed`` at the target
    (station, horizon) in NORMALIZED units. Zeroing all hotspots means setting
    ``data["hotspot"].x[:] = 0.0`` across all three columns (frp, lat, lon),
    mirroring ``occlusion_country_attribution`` but for every node at once.

    The returned delta can be negative (if zeroing hotspots raises the
    prediction), which would indicate no positive fire contribution for that
    sample.

    Args:
        model: Trained model in eval mode.
        data: Single-sample HeteroData.
        station_idx: Station index to explain.
        horizon_idx: Forecast horizon index to explain.
        device: Torch device.

    Returns:
        Normalized delta (full minus all-hotspots-zeroed), or None if the
        sample has no hotspot nodes.
    """
    device_t = torch.device(device)
    model = model.eval().to(device_t)
    data = data.to(device_t)

    orig_hx = data["hotspot"].x.clone()
    if orig_hx.numel() == 0:
        return None

    with torch.no_grad():
        pred_full = model(data)[station_idx, horizon_idx].item()
        data["hotspot"].x = torch.zeros_like(orig_hx)
        pred_zeroed = model(data)[station_idx, horizon_idx].item()
        data["hotspot"].x = orig_hx  # restore

    return pred_full - pred_zeroed


def _read_station_scale(scalers_path: Path, station_id: int) -> float:
    """Read a station's RobustScaler ``scale_`` (IQR) from scalers.json.

    Args:
        scalers_path: Path to data/processed/scalers.json.
        station_id: Target station id (keyed as a string in the JSON).

    Returns:
        The station's ``scale_`` value used to denormalize µg/m³ deltas.

    Raises:
        KeyError: If the station id is absent from the scalers file.
    """
    with open(scalers_path, encoding="utf-8") as fh:
        scalers_json = json.load(fh)
    if str(station_id) not in scalers_json:
        raise KeyError(
            f"station_id {station_id} not found in {scalers_path}. "
            "Cannot denormalize hotspot impact."
        )
    scale = float(scalers_json[str(station_id)]["scale_"])
    return scale


def _aggregate_frp_summary(samples: list[HeteroData]) -> dict[str, dict[str, float]]:
    """Aggregate per-country hotspot FRP statistics across peak samples.

    Reads ``data["hotspot"].x`` column 0 (total_frp) and the per-node
    ``data["hotspot"].country`` labels, summing node counts and FRP per country.

    Args:
        samples: List of single-sample HeteroData objects.

    Returns:
        Dict mapping country to {n_nodes_total, mean_frp, total_frp}, sorted by
        descending total_frp.
    """
    accum: dict[str, dict[str, float]] = {}
    for sample in samples:
        hx = sample["hotspot"].x
        countries = list(getattr(sample["hotspot"], "country", []))
        if hx.numel() == 0 or not countries:
            continue
        frp_vals = hx[:, 0].tolist()
        for country, frp in zip(countries, frp_vals, strict=False):
            entry = accum.setdefault(country, {"n_nodes_total": 0, "total_frp": 0.0})
            entry["n_nodes_total"] += 1
            entry["total_frp"] += float(frp)

    summary: dict[str, dict[str, float]] = {}
    for country, entry in sorted(accum.items(), key=lambda kv: -kv[1]["total_frp"]):
        n_nodes = int(entry["n_nodes_total"])
        summary[country] = {
            "n_nodes_total": n_nodes,
            "mean_frp": entry["total_frp"] / n_nodes if n_nodes else 0.0,
            "total_frp": entry["total_frp"],
        }
    return summary


def _aggregate_hotspot_impact(
    model: PM25ModelBase,
    samples: list[HeteroData],
    station_idx: int,
    horizon_idx: int,
    scale: float,
    device: str = "cpu",
) -> dict[str, float]:
    """Aggregate the µg/m³ hotspot impact across peak samples.

    For each sample, occludes ALL hotspots and reads the normalized prediction
    drop, then converts to µg/m³ by multiplying by the target station's
    RobustScaler ``scale_`` (IQR; the additive ``center_`` cancels in a delta).

    Args:
        model: Trained model.
        samples: List of single-sample HeteroData objects.
        station_idx: Station index to explain.
        horizon_idx: Forecast horizon index to explain.
        scale: Target station RobustScaler ``scale_`` value.
        device: Torch device.

    Returns:
        Dict with mean/min/max of the per-sample µg/m³ impact.
    """
    deltas_ug = [
        delta * scale
        for sample in samples
        if (delta := _hotspot_occlusion_delta(model, sample, station_idx, horizon_idx, device))
        is not None
    ]
    if not deltas_ug:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(deltas_ug)),
        "min": float(np.min(deltas_ug)),
        "max": float(np.max(deltas_ug)),
    }


# Chiang Mai station: locate by name in metadata
_CHIANG_MAI_NAMES = {"chiang mai", "เชียงใหม่", "chiangmai"}
_PEAK_MONTH = 3  # March
_PEAK_YEAR = 2024
_HORIZON_IDX = 2  # index 2 = 24h (horizons=[6,12,24,48])
_N_SAMPLES = 10  # top-N peak PM2.5 windows to aggregate
_N_IG_STEPS = 50


def _load_attribution_model(cfg: DictConfig, device: str) -> PM25ModelBase:
    """Resolve checkpoint, load weights, and return model in eval mode.

    Args:
        cfg: Hydra config.
        device: Torch device string (model is always loaded to CPU for attribution).

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If no checkpoint is found at either path.
    """
    checkpoint_path = Path(
        OmegaConf.select(cfg, "checkpoint", default="checkpoints/mtgnn/best_model.pt")
    )
    if not checkpoint_path.exists():
        # Fall back to root checkpoint if mtgnn subdir not found
        checkpoint_path = Path("checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run scripts/03_train.py first."
        )
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = instantiate(
        cfg.model,
        n_stations=18,
        horizons=list(cfg.data.horizons),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded: epoch=%d  val_rmse=%.4f", ckpt["epoch"], ckpt["val_rmse_24h"])
    return model  # type: ignore[return-value]


def _find_target_station(meta: pd.DataFrame) -> tuple[int, str]:
    """Return (station_idx, station_name) for Chiang Mai station.

    Searches by name using ``_CHIANG_MAI_NAMES``; falls back to nearest station
    to Chiang Mai city centre (18.788°N, 98.985°E) if no name match is found.

    Args:
        meta: Station metadata DataFrame with columns station_id, name, lat, lon.

    Returns:
        Tuple of (station_idx, station_name).
    """
    name_lower = meta["name"].str.lower() if "name" in meta.columns else pd.Series([""] * len(meta))
    cmai_mask = name_lower.apply(lambda n: any(k in n for k in _CHIANG_MAI_NAMES))
    if not cmai_mask.any():
        # Fall back to closest station to Chiang Mai city centre (18.788, 98.985)
        dists = ((meta["lat"] - 18.788) ** 2 + (meta["lon"] - 98.985) ** 2) ** 0.5
        station_idx = int(dists.idxmin())
        station_name = meta.iloc[station_idx].get(
            "name", f"station_{meta.iloc[station_idx]['station_id']}"
        )
    else:
        station_idx = int(cmai_mask.idxmax())
        station_name = meta.iloc[station_idx].get("name", "Chiang Mai")
    logger.info("Target station: %s (idx=%d)", station_name, station_idx)
    return (station_idx, str(station_name))


def _select_peak_samples(
    ds: PM25GraphDataset,
    station_idx: int,
) -> tuple[list[HeteroData], np.ndarray]:
    """Return top-N March 2024 peak-PM2.5 samples and their raw PM2.5 values.

    Args:
        ds: PM25GraphDataset (train split expected to contain March 2024).
        station_idx: Station index to rank by PM2.5.

    Returns:
        Tuple of (samples, peak_pm25) where samples is a list of HeteroData
        and peak_pm25 is a 1-D ndarray of raw PM2.5 values in descending order.

    Raises:
        RuntimeError: If no March 2024 samples or no valid PM2.5 values found.
    """
    anchor_indices = ds._anchor_indices
    anchor_timestamps = _FULL_INDEX[anchor_indices]
    march_2024_mask = (anchor_timestamps.year == _PEAK_YEAR) & (
        anchor_timestamps.month == _PEAK_MONTH
    )
    march_indices_pos = np.where(march_2024_mask)[0]

    if len(march_indices_pos) == 0:
        raise RuntimeError("No March 2024 samples found in train dataset. Check split bounds.")
    logger.info("March 2024 anchors available: %d", len(march_indices_pos))

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

    samples = [ds[int(pos)] for pos in top_n_pos]
    return (samples, peak_pm25)


def _aggregate_reports(
    reports: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate country attributions and IG feature importances across samples.

    Args:
        reports: List of attribution report dicts from ``batch_attribution``.

    Returns:
        Tuple of (country_means, ig_means), both re-normalized and sorted
        descending by score.
    """
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

    return (country_means, ig_means)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Load MTGNN checkpoint and run attribution on March 2024 peak haze."""
    device = "cpu"  # attribution runs on CPU

    model = _load_attribution_model(cfg, device)

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
    meta = pd.read_parquet(cfg.data.metadata_path).rename(columns={"location_id": "station_id"})
    meta = meta.sort_values("station_id").reset_index(drop=True)
    station_idx, station_name = _find_target_station(meta)

    # ── Read this station's RobustScaler scale_ (IQR) for µg/m³ denorm ────────
    # delta in normalized units → µg/m³ via multiply by scale_ (center_ cancels).
    target_station_id = int(meta.iloc[station_idx]["station_id"])
    target_scale = _read_station_scale(Path(cfg.data.scalers_path), target_station_id)
    logger.info(
        "Target station_id=%d scale_=%.6f (used to convert normalized delta → µg/m³)",
        target_station_id,
        target_scale,
    )

    # ── Find peak March 2024 samples ──────────────────────────────────────────
    samples, peak_pm25 = _select_peak_samples(ds, station_idx)

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
    country_means, ig_means = _aggregate_reports(reports)

    # ── Hotspot FRP summary (per-country, aggregated over the peak samples) ────
    hotspot_frp_summary = _aggregate_frp_summary(samples)

    # ── Hotspot impact in µg/m³ (occlude ALL hotspots, denorm by scale_) ──────
    hotspot_impact_ug_m3 = _aggregate_hotspot_impact(
        model, samples, station_idx, _HORIZON_IDX, target_scale, device=device
    )
    logger.info(
        "Hotspot impact (24h, µg/m³): mean=%.3f min=%.3f max=%.3f (scale_=%.4f)",
        hotspot_impact_ug_m3["mean"],
        hotspot_impact_ug_m3["min"],
        hotspot_impact_ug_m3["max"],
        target_scale,
    )

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

    # ── Build interpretation summary (regenerated with corrected µg/m³) ───────
    thai = hotspot_frp_summary.get("Thailand", {})
    mya = hotspot_frp_summary.get("Myanmar", {})
    thai_per = thai.get("n_nodes_total", 0) / len(samples)
    mya_per = mya.get("n_nodes_total", 0) / len(samples)
    thai_frp = thai.get("total_frp", 0.0)
    mya_frp = mya.get("total_frp", 0.0)
    ratio = thai_frp / mya_frp if mya_frp > 0 else float("inf")
    interpretation = (
        f"Local Thai biomass burning dominates ({thai_per:.0f} Thai clusters vs "
        f"{mya_per:.0f} Myanmar cluster per peak sample). Thai FRP sum = {thai_frp:.0f}, "
        f"Myanmar FRP sum = {mya_frp:.0f} (ratio {ratio:.0f}x). Total hotspot "
        f"contribution to {horizon_h}h PM2.5 prediction: "
        f"{hotspot_impact_ug_m3['mean']:.1f} ug/m3 above no-fire baseline."
    )

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
        "hotspot_frp_summary": hotspot_frp_summary,
        "hotspot_impact_ug_m3_24h": hotspot_impact_ug_m3,
        "interpretation": interpretation,
    }
    out_path = output_dir / "attribution_march2024.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
