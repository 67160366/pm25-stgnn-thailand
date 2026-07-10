"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Fetches 7-day hourly NWP forecasts from the free Open-Meteo API (no API key
required) and converts them to the ERA5-style feature schema (``u10``,
``v10``, ``t2m``, ``d2m``, ``blh``) expected by the model.  Used for
live/demo-time forecasting where ERA5 reanalysis (which lags by several
days) is not yet available.
"""

import logging
import math

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo hourly variables requested in every forecast call.
HOURLY_VARS: tuple[str, ...] = (
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
)


def _to_kelvin(celsius: float | None) -> float:
    """Convert a Celsius value to Kelvin, propagating None as NaN.

    Args:
        celsius: Temperature in degrees Celsius, or None.

    Returns:
        Temperature in Kelvin, or NaN if ``celsius`` is None.
    """
    if celsius is None:
        return math.nan
    return celsius + 273.15


def _wind_components(speed: float | None, direction_deg: float | None) -> tuple[float, float]:
    """Convert meteorological wind speed/direction to (u10, v10) components.

    Open-Meteo reports the direction wind is blowing *from* (meteorological
    convention). This converts to eastward (u) and northward (v) components.

    Args:
        speed: Wind speed in m/s, or None.
        direction_deg: Wind direction in degrees (0 = from north), or None.

    Returns:
        (u10, v10) tuple in m/s. NaN if either input is None.
    """
    if speed is None or direction_deg is None:
        return math.nan, math.nan
    rad = math.radians(direction_deg)
    u10 = -speed * math.sin(rad)
    v10 = -speed * math.cos(rad)
    return u10, v10


def _parse_hourly(hourly: dict) -> pd.DataFrame:
    """Parse the Open-Meteo ``hourly`` response block into ERA5-style columns.

    Args:
        hourly: The ``hourly`` dict from the Open-Meteo JSON response, with
            keys ``time`` and each entry in :data:`HOURLY_VARS`.

    Returns:
        DataFrame with columns ``time``, ``u10``, ``v10``, ``t2m``, ``d2m``,
        ``blh``. ``time`` is UTC tz-aware. Missing (None) source values
        become NaN.
    """
    times = pd.to_datetime(hourly["time"], utc=True)
    temperature = hourly["temperature_2m"]
    dew_point = hourly["dew_point_2m"]
    wind_speed = hourly["wind_speed_10m"]
    wind_direction = hourly["wind_direction_10m"]
    blh = hourly["boundary_layer_height"]

    n = len(times)
    t2m = [math.nan] * n
    d2m = [math.nan] * n
    u10 = [math.nan] * n
    v10 = [math.nan] * n

    for i in range(n):
        t2m[i] = _to_kelvin(temperature[i])
        d2m[i] = _to_kelvin(dew_point[i])
        u10[i], v10[i] = _wind_components(wind_speed[i], wind_direction[i])

    return pd.DataFrame(
        {
            "time": times,
            "u10": u10,
            "v10": v10,
            "t2m": t2m,
            "d2m": d2m,
            "blh": [b if b is not None else math.nan for b in blh],
        }
    )


def fetch_forecast(
    lat: float,
    lon: float,
    *,
    forecast_days: int = 7,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch an hourly NWP forecast for a single point from Open-Meteo.

    Args:
        lat: Latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees.
        forecast_days: Number of forecast days to request (Open-Meteo max 16).
        timeout: Request timeout in seconds.

    Returns:
        DataFrame with columns ``time`` (UTC tz-aware), ``u10``, ``v10``,
        ``t2m`` (K), ``d2m`` (K), ``blh`` (m).

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    logger.info("Fetching Open-Meteo forecast lat=%.4f lon=%.4f days=%d", lat, lon, forecast_days)
    response = requests.get(OPENMETEO_URL, params=params, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    df = _parse_hourly(payload["hourly"])
    logger.info("Open-Meteo forecast lat=%.4f lon=%.4f: %d hourly rows", lat, lon, len(df))
    return df


def fetch_forecast_stations(
    stations: pd.DataFrame,
    *,
    forecast_days: int = 7,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch hourly NWP forecasts for every station in a metadata DataFrame.

    Loops over ``stations`` and calls :func:`fetch_forecast` per row,
    concatenating results with a ``station_id`` column.

    Args:
        stations: DataFrame with columns ``station_id``, ``lat``, ``lon``
            (same schema as ``data/processed/stations_metadata.parquet``,
            see :mod:`src.data.loader`).
        forecast_days: Number of forecast days to request per station.
        timeout: Request timeout in seconds, per station.

    Returns:
        Concatenated DataFrame with columns ``station_id``, ``time``,
        ``u10``, ``v10``, ``t2m``, ``d2m``, ``blh``.

    Raises:
        requests.HTTPError: If any station's API call returns a non-2xx
            status code.
    """
    frames: list[pd.DataFrame] = []
    for row in stations.itertuples(index=False):
        df_station = fetch_forecast(
            lat=row.lat, lon=row.lon, forecast_days=forecast_days, timeout=timeout
        )
        df_station.insert(0, "station_id", row.station_id)
        frames.append(df_station)
        logger.info(
            "fetch_forecast_stations: station_id=%s lat=%.4f lon=%.4f -> %d rows",
            row.station_id,
            row.lat,
            row.lon,
            len(df_station),
        )

    if not frames:
        return pd.DataFrame(columns=["station_id", "time", "u10", "v10", "t2m", "d2m", "blh"])

    return pd.concat(frames, ignore_index=True)
