# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Live input-window assembly for the dashboard's NWP forecast mode.

Two pure, network-light building blocks used by ``app.lib.inference.live_forecast``:

* :func:`fetch_forecast_window` extends the committed Open-Meteo scraper with a
  ``past_days`` parameter (which its public API cannot pass) by calling the same
  URL and reusing :func:`src.data.scrapers.openmeteo._parse_hourly`, so one call
  per station covers both the 24 h input window and the 48 h horizon.
* :func:`build_live_window` turns fetched air4thai PM2.5 + Open-Meteo NWP into
  the exact ``(N, T_in, F)`` tensor the model expects, scaling PM2.5 with the
  same per-station RobustScaler params and cyclic-feature formulas used at
  training time (``src.data.preprocessing._add_cyclic_features``, all in UTC).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from src.data.scrapers.openmeteo import HOURLY_VARS, OPENMETEO_URL, _parse_hourly

logger = logging.getLogger(__name__)

# Feature order MUST match src.data.loader.PM25GraphDataset.__getitem__.
FEATURE_ORDER: tuple[str, ...] = (
    "pm25_scaled",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "u10",
    "v10",
    "t2m",
    "d2m",
    "blh",
)
_NWP_FEATURES: tuple[str, ...] = ("u10", "v10", "t2m", "d2m", "blh")


@dataclass
class LiveWindow:
    """Assembled live input window and the metadata needed to build the graph.

    Attributes:
        x: (N, T_in, F) float32 model input, F == len(FEATURE_ORDER).
        anchor_u: (N,) eastward wind at the anchor hour, per station (for edges).
        anchor_v: (N,) northward wind at the anchor hour, per station.
        observed: (N,) most recent finite PM2.5 (µg/m³) per station, NaN if none.
        window_raw: (N, T_in) raw PM2.5 (µg/m³) over the window, NaN preserved.
        coverage: station_id -> fraction of finite PM2.5 hours in the window.
        excluded: station_ids center-filled because coverage < ``min_coverage``.
        slots: the T_in UTC hourly timestamps, oldest first, ending at the anchor.
    """

    x: np.ndarray
    anchor_u: np.ndarray
    anchor_v: np.ndarray
    observed: np.ndarray
    window_raw: np.ndarray
    coverage: dict[int, float]
    excluded: list[int]
    slots: pd.DatetimeIndex


def fetch_forecast_window(
    lat: float,
    lon: float,
    *,
    past_days: int,
    forecast_days: int,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch a past+future hourly NWP window for one point from Open-Meteo.

    Mirrors ``openmeteo.fetch_forecast`` but adds ``past_days`` (max 92) so the
    same response covers the input window and the forecast horizon.

    Args:
        lat: Latitude (WGS84 degrees).
        lon: Longitude (WGS84 degrees).
        past_days: Number of past days to include (Open-Meteo max 92).
        forecast_days: Number of forecast days to include (Open-Meteo max 16).
        timeout: Request timeout in seconds.

    Returns:
        DataFrame with columns ``time`` (UTC tz-aware), ``u10``, ``v10``,
        ``t2m`` (K), ``d2m`` (K), ``blh`` (m).

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    logger.info(
        "Open-Meteo window lat=%.4f lon=%.4f past=%d fcst=%d", lat, lon, past_days, forecast_days
    )
    response = requests.get(OPENMETEO_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return _parse_hourly(response.json()["hourly"])


def _cyclic_features(slots: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Hour/day-of-year sin-cos features for UTC ``slots`` (matches preprocessing)."""
    hour = slots.hour.to_numpy(dtype=np.float64)
    doy = slots.dayofyear.to_numpy(dtype=np.float64)
    return {
        "hour_sin": np.sin(2 * math.pi * hour / 24),
        "hour_cos": np.cos(2 * math.pi * hour / 24),
        "doy_sin": np.sin(2 * math.pi * doy / 365),
        "doy_cos": np.cos(2 * math.pi * doy / 365),
    }


def _pivot(
    df: pd.DataFrame, value: str, slots: pd.DatetimeIndex, station_ids: Sequence[int]
) -> pd.DataFrame:
    """Pivot a long (station_id, time, value) frame to a (T, N) frame on ``slots``."""
    if df.empty:
        return pd.DataFrame(np.nan, index=slots, columns=list(station_ids))
    wide = df.pivot_table(index="time", columns="station_id", values=value, aggfunc="mean")
    return wide.reindex(index=slots, columns=list(station_ids))


def build_live_window(
    pm25_hist: pd.DataFrame,
    nwp: pd.DataFrame,
    scalers: dict[int, dict[str, float]],
    station_ids: Sequence[int],
    anchor_ts: pd.Timestamp,
    *,
    window_in: int = 24,
    min_coverage: float = 0.7,
) -> LiveWindow:
    """Assemble the live model input window from fetched observations + NWP.

    Args:
        pm25_hist: Long frame (station_id, time[UTC], pm25) of recent PM2.5.
        nwp: Long frame (station_id, time[UTC], u10, v10, t2m, d2m, blh).
        scalers: Per-station RobustScaler params ({station_id: {center_, scale_}}).
        station_ids: Ordered station ids (ascending; must match model node order).
        anchor_ts: Forecast origin (UTC). The window ends at this hour.
        window_in: Input window length in hours.
        min_coverage: Minimum finite-PM2.5 fraction; stations below this are
            center-filled (scaled 0) and reported in ``excluded``.

    Returns:
        A populated :class:`LiveWindow`.

    Note:
        Gap policy here diverges from training preprocessing. Live mode
        forward-fills PM2.5 in RAW space then scales, and center-fills any
        leading gap or a below-threshold station (scaled 0 == the station
        median). Training used the richer interpolate/ffill/mask policy in
        ``src.data.preprocessing`` — acceptable because that path needs a
        complete historical grid, unavailable in real time.
    """
    station_ids = [int(s) for s in station_ids]
    anchor_ts = pd.Timestamp(anchor_ts)
    if anchor_ts.tzinfo is None:
        anchor_ts = anchor_ts.tz_localize("UTC")
    else:
        anchor_ts = anchor_ts.tz_convert("UTC")
    anchor_ts = anchor_ts.floor("h")
    slots = pd.date_range(end=anchor_ts, periods=window_in, freq="1h")

    n = len(station_ids)
    t = window_in

    pm_wide = _pivot(pm25_hist, "pm25", slots, station_ids)  # (T, N)
    window_raw = pm_wide.to_numpy(dtype=np.float64).T  # (N, T), NaN preserved

    pm_scaled = np.zeros((t, n), dtype=np.float64)
    observed = np.full(n, np.nan, dtype=np.float64)
    coverage: dict[int, float] = {}
    excluded: list[int] = []

    for i, sid in enumerate(station_ids):
        col = pm_wide.iloc[:, i]
        cov = float(col.notna().mean()) if t else 0.0
        coverage[sid] = cov

        finite = col.dropna()
        if not finite.empty:
            observed[i] = float(finite.iloc[-1])

        params = scalers.get(sid, {"center_": 0.0, "scale_": 1.0})
        center = float(params["center_"])
        scale = float(params["scale_"]) or 1.0

        if cov < min_coverage:
            excluded.append(sid)  # leave pm_scaled column at 0 (== center in raw space)
            continue

        raw = col.ffill()  # forward-fill in raw space (see Note)
        scaled = ((raw - center) / scale).fillna(0.0)  # leading gap -> center (scaled 0)
        pm_scaled[:, i] = scaled.to_numpy(dtype=np.float64)

    cyc = _cyclic_features(slots)
    layers: list[np.ndarray] = [pm_scaled]
    for name in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
        layers.append(np.repeat(cyc[name][:, None], n, axis=1))  # (T, N)

    nwp_wide: dict[str, np.ndarray] = {}
    for feat in _NWP_FEATURES:
        w = _pivot(nwp, feat, slots, station_ids).ffill().bfill()
        nwp_wide[feat] = np.nan_to_num(w.to_numpy(dtype=np.float64), nan=0.0)  # (T, N)
        layers.append(nwp_wide[feat])

    stacked = np.stack(layers, axis=-1)  # (T, N, F)
    x = np.nan_to_num(stacked.transpose(1, 0, 2), nan=0.0).astype(np.float32)  # (N, T, F)

    anchor_u = nwp_wide["u10"][-1, :].astype(np.float32)
    anchor_v = nwp_wide["v10"][-1, :].astype(np.float32)

    return LiveWindow(
        x=x,
        anchor_u=anchor_u,
        anchor_v=anchor_v,
        observed=observed,
        window_raw=window_raw,
        coverage=coverage,
        excluded=excluded,
        slots=slots,
    )
