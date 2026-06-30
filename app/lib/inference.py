# [TODO: NSC Disclaimer - see booklet page 44]
"""Cached model loading + live (hindcast) forecasting for the dashboard.

Live inference uses the pitch checkpoint ``checkpoints/mtgnn/best_model.pt`` (the only
trained model checked into the tree). Every ug/m3 number flows through
``src.training.evaluation`` so it matches ``scripts/04_evaluate.py`` exactly. Because the
weather inputs are ERA5 reanalysis (not available in real time), this is a HINDCAST/replay
over stored windows, not operational real-time forecasting — the UI must say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader

from app.lib import data_access as da
from src.data.loader import PM25GraphDataset
from src.training import evaluation

HORIZONS: list[int] = [6, 12, 24, 48]
_DEVICE = "cpu"

# User-facing period label -> dataset split (years per loader._SPLIT_BOUNDS).
PERIOD_TO_SPLIT: dict[str, str] = {
    "2025 (ปีล่าสุด)": "test",
    "2024": "val",
    "2022–2023": "train",
}
DEFAULT_PERIOD = "2025 (ปีล่าสุด)"


@st.cache_resource(show_spinner="กำลังโหลดโมเดล MTGNN…")
def load_model() -> torch.nn.Module:
    """Instantiate MTGNN from Hydra config and load the pitch checkpoint (CPU, eval)."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(da.CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["model=mtgnn"])
    model = instantiate(cfg.model, n_stations=18, horizons=HORIZONS)
    ckpt = torch.load(da.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@st.cache_resource(show_spinner="กำลังเตรียมกราฟข้อมูล…")
def get_dataset(split: str) -> PM25GraphDataset:
    """Build (and cache) the graph dataset for a split, matching the pitch wind mode."""
    return PM25GraphDataset(
        dataset_path=da.DATASET_PATH,
        hotspots_path=da.HOTSPOTS_PATH,
        metadata_path=da.METADATA_PATH,
        split=split,
        window_in=24,
        horizons=HORIZONS,
        graph_config={"wind_mode": "from_field"},
        scalers_path=da.SCALERS_PATH,
    )


def anchor_timestamps(split: str) -> pd.DatetimeIndex:
    """Forecast-origin timestamps available in this split (one per valid sample)."""
    ds = get_dataset(split)
    return ds._timestamps[ds._anchor_indices]


def pos_for_timestamp(ds: PM25GraphDataset, ts: pd.Timestamp) -> int:
    """Sample index whose anchor is nearest to ``ts`` (tz-aware UTC)."""
    anchors = ds._timestamps[ds._anchor_indices]
    return int(np.argmin(np.abs(anchors.values - np.datetime64(ts))))


@st.cache_data(show_spinner=False)
def default_anchor_iso(split: str) -> str:
    """ISO timestamp of the anchor with the highest cross-station mean PM2.5.

    Used as the landing default so the first glance shows a real haze episode.
    """
    ds = get_dataset(split)
    obs = ds._pm25_raw[ds._anchor_indices, :]  # (S, N)
    with np.errstate(all="ignore"):
        mean_obs = np.nanmean(obs, axis=1)
    pos = int(np.nanargmax(mean_obs))
    return pd.Timestamp(ds._timestamps[ds._anchor_indices[pos]]).isoformat()


def date_bounds(split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(min, max) calendar date that has a valid forecast origin in this split."""
    ats = anchor_timestamps(split)
    return ats.min().date(), ats.max().date()


def resolve_anchor(split: str, day) -> str:
    """ISO timestamp of the anchor nearest to noon on ``day`` for this split."""
    ds = get_dataset(split)
    ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=12)
    pos = pos_for_timestamp(ds, ts)
    return pd.Timestamp(ds._timestamps[ds._anchor_indices[pos]]).isoformat()


@st.cache_data(show_spinner="กำลังพยากรณ์…")
def forecast_at(split: str, anchor_iso: str) -> dict:
    """Run the model for one forecast origin and return display-ready arrays.

    Returns a dict with: ``anchor_iso`` (resolved), ``station_ids`` (list),
    ``pred_ug`` (N,H), ``observed`` (N,) current value at the origin, ``persistence``
    (N,), ``actual_future`` {h: (N,)} (may contain NaN), ``horizons``.
    All PM2.5 arrays are ug/m3 via ``src.training.evaluation``.
    """
    ds = get_dataset(split)
    model = load_model()
    pos = pos_for_timestamp(ds, pd.Timestamp(anchor_iso))
    sample = ds[pos]

    loader = DataLoader([sample], batch_size=1, shuffle=False)
    pred_norm = evaluation.predict(model, loader, _DEVICE)  # (N, H) normalised
    centers, scales = evaluation.station_scalers(ds)
    pred_ug = evaluation.denorm_pred(pred_norm, centers, scales, ds.n_stations)  # (N, H)

    anchor_idx = int(ds._anchor_indices[pos])
    raw = ds._pm25_raw
    observed = raw[anchor_idx, :].astype(np.float64)  # (N,) at the origin

    # Persistence = last finite observation within the input window (matches evaluation).
    pers = np.full(ds.n_stations, np.nan, dtype=np.float64)
    for k in range(ds.window_in):
        cand = raw[anchor_idx - k, :]
        fill = ~np.isfinite(pers) & np.isfinite(cand)
        pers[fill] = cand[fill]

    actual_future = {}
    for h in HORIZONS:
        fut_idx = anchor_idx + h
        actual_future[h] = (
            raw[fut_idx, :].astype(np.float64)
            if fut_idx < raw.shape[0]
            else np.full(ds.n_stations, np.nan)
        )

    return {
        "anchor_iso": pd.Timestamp(ds._timestamps[anchor_idx]).isoformat(),
        "station_ids": ds._station_ids.tolist(),
        "pred_ug": pred_ug,
        "observed": observed,
        "persistence": pers,
        "actual_future": actual_future,
        "horizons": list(HORIZONS),
    }


def station_history(split: str, station_id: int, anchor_iso: str, hours_back: int = 168) -> pd.DataFrame:
    """Observed PM2.5 for one station over [anchor - hours_back, anchor] (DataFrame)."""
    ds = get_dataset(split)
    anchor = pd.Timestamp(anchor_iso)
    sidx = int(np.where(ds._station_ids == station_id)[0][0])
    pos = pos_for_timestamp(ds, anchor)
    anchor_idx = int(ds._anchor_indices[pos])
    start_idx = max(0, anchor_idx - hours_back)
    ts = ds._timestamps[start_idx : anchor_idx + 1]
    vals = ds._pm25_raw[start_idx : anchor_idx + 1, sidx]
    return pd.DataFrame({"timestamp": ts, "pm25": vals})
