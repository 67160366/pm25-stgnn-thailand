# [TODO: NSC Disclaimer - see booklet page 44]
"""Cached data-access helpers for the dashboard.

Centralises every read of ``data/processed`` and ``outputs`` so pages never touch
the filesystem directly and every read is cached. Province is parsed from the station
``name`` (there is no province column in stations_metadata).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data" / "processed"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

DATASET_PATH = _DATA_DIR / "dataset.parquet"
HOTSPOTS_PATH = _DATA_DIR / "hotspots.parquet"
METADATA_PATH = _DATA_DIR / "stations_metadata.parquet"
SCALERS_PATH = _DATA_DIR / "scalers.json"
CONFIGS_DIR = _PROJECT_ROOT / "configs"
CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints" / "mtgnn" / "best_model.pt"


def _province_from_name(name: str) -> str:
    """Best-effort province from a station name (text after the last comma)."""
    if isinstance(name, str) and "," in name:
        return name.rsplit(",", 1)[-1].strip()
    return name or ""


@st.cache_data(show_spinner=False)
def load_stations_meta() -> pd.DataFrame:
    """18 stations with station_id, lat, lon, name, province (sorted by station_id)."""
    meta = pd.read_parquet(METADATA_PATH).rename(columns={"location_id": "station_id"})
    meta = meta.sort_values("station_id").reset_index(drop=True)
    meta["province"] = meta["name"].map(_province_from_name)
    return meta[["station_id", "lat", "lon", "name", "province"]]


@st.cache_data(show_spinner="กำลังโหลดข้อมูลย้อนหลัง…")
def load_pm25_long() -> pd.DataFrame:
    """Long-format observed PM2.5 (timestamp, station_id, pm25_raw) for history views."""
    return pd.read_parquet(DATASET_PATH, columns=["timestamp", "station_id", "pm25_raw"])


@st.cache_data(show_spinner=False)
def load_hotspots() -> pd.DataFrame:
    """FIRMS hotspot clusters (date, centroid_lat, centroid_lon, total_frp, country)."""
    return pd.read_parquet(HOTSPOTS_PATH)


@st.cache_data(show_spinner=False)
def hotspots_for_date(date_str: str) -> pd.DataFrame:
    """Hotspots on a given 'YYYY-MM-DD' renamed to station_map's columns.

    Returns columns ``latitude``, ``longitude``, ``frp``, ``country`` (possibly empty).
    """
    hs = load_hotspots()
    day = hs[hs["date"].astype(str) == date_str]  # 'date' column holds datetime.date objects
    return day.rename(
        columns={"centroid_lat": "latitude", "centroid_lon": "longitude", "total_frp": "frp"}
    )[["latitude", "longitude", "frp", "country"]]


@st.cache_data(show_spinner=False)
def load_output_json(name: str) -> dict:
    """Load an ``outputs/<name>`` JSON file; returns {} if it does not exist."""
    path = _OUTPUTS_DIR / name
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_scalers() -> dict:
    """Per-station RobustScaler params keyed by station_id (string)."""
    with SCALERS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)
