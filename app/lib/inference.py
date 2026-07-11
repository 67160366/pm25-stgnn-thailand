# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Cached model loading + live (hindcast) forecasting for the dashboard.

Live inference uses the pitch checkpoint ``checkpoints/mtgnn/best_model.pt`` (the only
trained model checked into the tree). Every ug/m3 number flows through
``src.training.evaluation`` so it matches ``scripts/04_evaluate.py`` exactly. Because the
weather inputs are ERA5 reanalysis (not available in real time), this is a HINDCAST/replay
over stored windows, not operational real-time forecasting — the UI must say so.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from app.lib import air4thai
from app.lib import data_access as da
from app.lib import nwp as nwp_lib
from src.data.graph_builder import build_graph
from src.data.loader import EMPTY_HOTSPOT_DF, PM25GraphDataset
from src.training import evaluation

HORIZONS: list[int] = [6, 12, 24, 48]
_DEVICE = "cpu"

# Live-mode input window (matches the pitch dataset's window_in) and gap policy.
WINDOW_IN: int = 24
LIVE_MIN_COVERAGE: float = 0.70


class LiveDataError(RuntimeError):
    """Raised when live inputs cannot be fetched/assembled (shown as a Thai error)."""


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


def resolve_anchor(split: str, day: date | str) -> str:
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


def station_history(
    split: str, station_id: int, anchor_iso: str, hours_back: int = 168
) -> pd.DataFrame:
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


def conformal_halfwidths(conf: dict, station_id: int, horizons: list[int]) -> list[float] | None:
    """Resolve split-conformal half-widths (µg/m³) per horizon for one station.

    Prefers per-station quantiles when the calibration served in ``per_station``
    mode and the station is present; otherwise falls back to the pooled quantile.
    Returns ``None`` if any horizon has no usable (finite) quantile, so the caller
    can simply omit the band.

    Args:
        conf: Parsed ``conformal_intervals.json`` (see ``data_access.load_conformal``).
        station_id: OpenAQ station id.
        horizons: Forecast horizons in hours.

    Returns:
        List of half-widths aligned to ``horizons``, or ``None`` if unavailable.
    """
    if not conf:
        return None
    pooled: dict = conf.get("pooled", {})
    mode = conf.get("meta", {}).get("mode", "pooled")
    per_station: dict = conf.get("per_station", {}) if mode == "per_station" else {}
    station_q: dict = per_station.get(str(station_id), {})
    out: list[float] = []
    for h in horizons:
        key = f"{h}h"
        q = station_q.get(key)
        if q is None:
            q = pooled.get(key)
        if q is None:
            return None
        out.append(float(q))
    return out


# ---------------------------------------------------------------------------
# Live mode: forecast-from-now using air4thai PM2.5 + Open-Meteo NWP
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_air4thai_map() -> dict[int, str]:
    """Cached OpenAQ station_id -> air4thai code map (empty if the file is absent)."""
    return air4thai.load_station_code_map()


def _scalers_in_order(station_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(centers, scales) arrays aligned to ``station_ids`` from scalers.json."""
    raw = da.load_scalers()
    centers = np.array([float(raw[str(s)]["center_"]) for s in station_ids], dtype=np.float64)
    scales = np.array([float(raw[str(s)]["scale_"]) for s in station_ids], dtype=np.float64)
    return centers, scales


def _build_live_sample(
    lw: nwp_lib.LiveWindow, station_ids: list[int], lats: np.ndarray, lons: np.ndarray
) -> HeteroData:
    """HeteroData for one live window: wind-aware graph + empty hotspot nodes.

    Mirrors ``loader._build_graph_full`` in ``from_field`` mode (wind read from
    per-station anchor arrays) but with zero hotspot nodes — FIRMS NRT is not
    ingested in live v1, so attribution is unavailable and fires are absent.
    """
    cfg = {
        "wind_mode": "from_arrays",
        "_u10_per_station": lw.anchor_u,
        "_v10_per_station": lw.anchor_v,
        "_station_lats": lats,
        "_station_lons": lons,
    }
    df_stations = pd.DataFrame({"station_id": station_ids, "lat": lats, "lon": lons})
    data = build_graph(df_stations, EMPTY_HOTSPOT_DF, wind_field=None, config=cfg)
    data["hotspot"].country = []
    data["station"].x = torch.from_numpy(lw.x).contiguous().float()
    return data


@st.cache_data(show_spinner="กำลังดึงข้อมูลสดและพยากรณ์…", ttl=1800)
def live_forecast() -> dict:
    """Forecast from *now* using live air4thai PM2.5 + Open-Meteo NWP.

    Fetches a 24 h PM2.5 window (air4thai history) and a past+future NWP window
    (Open-Meteo), assembles the model input, runs the pitch checkpoint, and
    returns the 6/12/24/48 h forecast from the current hour.

    Returns:
        Dict with ``anchor_iso``, ``station_ids``, ``pred_ug`` (N,H),
        ``observed`` (N,), ``horizons``, ``excluded_ids`` (center-filled
        stations), ``coverage`` {sid: frac}, ``slots`` (list of ISO strings),
        ``window_raw`` (N,T µg/m³), ``hotspots_empty`` (always True in v1).

    Raises:
        LiveDataError: If inputs cannot be fetched or coverage is too low
            (already Thai-worded for direct display).
    """
    meta = da.load_stations_meta()
    station_ids = meta["station_id"].tolist()
    lats = meta["lat"].to_numpy(dtype=np.float64)
    lons = meta["lon"].to_numpy(dtype=np.float64)

    code_map = load_air4thai_map()
    if not code_map:
        raise LiveDataError(
            "ไม่พบตารางรหัสสถานี air4thai (configs/air4thai_station_codes.json) — "
            "โหมดสดใช้ไม่ได้ กรุณารัน scripts/14_air4thai_station_map.py ก่อน"
        )

    anchor_ts = pd.Timestamp.now(tz="UTC").floor("h")
    bkk_today = pd.Timestamp.now(tz="Asia/Bangkok").date()
    bkk_start = bkk_today - timedelta(days=3)

    try:
        pm25_hist = air4thai.fetch_history_stations(code_map, bkk_start, bkk_today)
        nwp_frames = []
        for row in meta.itertuples(index=False):
            nwp_df = nwp_lib.fetch_forecast_window(
                lat=float(row.lat), lon=float(row.lon), past_days=3, forecast_days=3
            )
            nwp_df.insert(0, "station_id", int(row.station_id))
            nwp_frames.append(nwp_df)
    except requests.RequestException as exc:
        raise LiveDataError(
            "ดึงข้อมูลสดไม่สำเร็จ (เครือข่ายหรือบริการต้นทางขัดข้อง) — โปรดลองใหม่ภายหลัง"
        ) from exc

    nwp = pd.concat(nwp_frames, ignore_index=True)
    raw_scalers = da.load_scalers()
    scalers = {int(s): raw_scalers[str(s)] for s in station_ids}

    lw = nwp_lib.build_live_window(
        pm25_hist,
        nwp,
        scalers,
        station_ids,
        anchor_ts,
        window_in=WINDOW_IN,
        min_coverage=LIVE_MIN_COVERAGE,
    )

    if len(lw.excluded) == len(station_ids):
        raise LiveDataError(
            "ข้อมูล PM2.5 ย้อนหลังไม่พอสำหรับทุกสถานี (ต่ำกว่าเกณฑ์ 70% ของหน้าต่าง 24 ชม.) — "
            "โปรดลองใหม่ภายหลัง"
        )

    model = load_model()
    sample = _build_live_sample(lw, station_ids, lats, lons)
    loader = DataLoader([sample], batch_size=1, shuffle=False)
    pred_norm = evaluation.predict(model, loader, _DEVICE)  # (N, H)
    centers, scales = _scalers_in_order(station_ids)
    pred_ug = evaluation.denorm_pred(pred_norm, centers, scales, len(station_ids))  # (N, H)

    return {
        "anchor_iso": anchor_ts.isoformat(),
        "station_ids": station_ids,
        "pred_ug": pred_ug,
        "observed": lw.observed,
        "horizons": list(HORIZONS),
        "excluded_ids": lw.excluded,
        "coverage": lw.coverage,
        "slots": [pd.Timestamp(s).isoformat() for s in lw.slots],
        "window_raw": lw.window_raw,
        "hotspots_empty": True,
    }
