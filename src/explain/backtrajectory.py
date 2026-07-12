"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Kinematic ERA5 backward-trajectory analysis: an independent witness for the model's
transboundary (Myanmar/Laos) source attribution at a border station (roadmap item 11).

**Honest limitation (read before trusting any output of this module):** this uses
**surface (10 m) winds only, a single launch time, no vertical transport, and no
turbulent mixing** -- it is *indicative corroboration*, NOT equivalent to a full HYSPLIT
Lagrangian dispersion run. The agreement metric it computes is a *binary direction*
comparison (foreign vs. not-foreign) against the model's attribution -- it never compares
magnitudes.

Method: launch a parcel at a station's peak-PM2.5 anchor hour (the same anchor the model's
attribution used) and integrate its position backward in time using ERA5 10 m wind
(``u10``/``v10``), bilinear in space and linear in time, with RK2 (midpoint) backward
integration. Every trajectory point is labelled by country
(:func:`src.data.geocode.country_of`) and checked against FIRMS hotspot clusters within a
fixed radius, to answer: "did the air that arrived at the station come from -- and pass
near active fires in -- Myanmar or Laos?"
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.geocode import DEFAULT_BORDERS_PATH, country_of
from src.data.scrapers.era5 import _ERA5_ALIASES, _find_var, _normalise_coords, _open_nc
from src.explain.transboundary_events import _haversine_km

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_M_PER_DEG_LAT: float = 110_574.0  # mean meters per degree latitude
_M_PER_DEG_LON_EQ: float = 111_320.0  # meters per degree longitude at equator
_SEC_PER_HOUR: float = 3600.0
ERA5_BBOX: tuple[float, float, float, float] = (97.0, 16.0, 101.5, 21.0)  # W,S,E,N
_MODEL_FOREIGN_TAU: float = 0.05  # model "says foreign" threshold

_COUNTRY_KEYS: tuple[str, str, str, str] = ("Thailand", "Myanmar", "Laos", "other")
_FOREIGN_COUNTRIES: tuple[str, str] = ("Myanmar", "Laos")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindField:
    """Regular-grid hourly wind field, coords ascending, tz-naive UTC times.

    Attributes:
        times: ``datetime64[ns]`` ascending, hourly.
        lats: float64 ascending.
        lons: float64 ascending.
        u: float64 array shape ``(T, nlat, nlon)``, m/s (eastward). NaN allowed (missing hour).
        v: float64 array shape ``(T, nlat, nlon)``, m/s (northward). NaN allowed (missing hour).
    """

    times: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class BackTrajectory:
    """Result of one backward integration.

    Attributes:
        times: ``datetime64[ns]`` length K (K = ``hours_integrated`` + 1); ``times[0]``
            is the launch time.
        lats: float64 length K.
        lons: float64 length K.
        terminated_reason: ``"completed"`` | ``"left_domain"`` | ``"missing_wind"``.
        hours_integrated: Number of successful backward steps taken.
    """

    times: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    terminated_reason: str
    hours_integrated: int


def make_wind_field(
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> WindField:
    """Build a :class:`WindField`, normalising coordinate order and dtypes.

    ERA5 ships latitude descending (21.0 -> 16.0); this factory sorts both lat and lon
    ascending (flipping ``u``/``v`` rows/cols to match) so downstream bilinear/RK2 code can
    assume ascending coordinates unconditionally.

    Args:
        times: Length-T array of timestamps (any datetime-like dtype).
        lats: Length-``nlat`` latitude array (any order).
        lons: Length-``nlon`` longitude array (any order).
        u: Eastward wind, shape ``(T, nlat, nlon)``, m/s.
        v: Northward wind, shape ``(T, nlat, nlon)``, m/s.

    Returns:
        A :class:`WindField` with ascending ``lats``/``lons`` and float64/``datetime64[ns]``
        dtypes.

    Raises:
        ValueError: If ``u``/``v`` shapes do not match ``(len(times), len(lats), len(lons))``.
    """
    n_t, n_lat, n_lon = len(times), len(lats), len(lons)
    expected_shape = (n_t, n_lat, n_lon)
    if u.shape != expected_shape or v.shape != expected_shape:
        raise ValueError(
            f"u/v shape {u.shape}/{v.shape} does not match expected {expected_shape} "
            f"derived from (times, lats, lons)"
        )

    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    times_arr = np.asarray(times, dtype="datetime64[ns]")

    if lats_arr[0] > lats_arr[-1]:
        lats_arr = lats_arr[::-1]
        u_arr = u_arr[:, ::-1, :]
        v_arr = v_arr[:, ::-1, :]
    if lons_arr[0] > lons_arr[-1]:
        lons_arr = lons_arr[::-1]
        u_arr = u_arr[:, :, ::-1]
        v_arr = v_arr[:, :, ::-1]

    return WindField(
        times=times_arr,
        lats=np.ascontiguousarray(lats_arr),
        lons=np.ascontiguousarray(lons_arr),
        u=np.ascontiguousarray(u_arr),
        v=np.ascontiguousarray(v_arr),
    )


# ---------------------------------------------------------------------------
# Pure numeric functions
# ---------------------------------------------------------------------------


def _bilinear(
    grid2d: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float
) -> float:
    """Bilinear interpolate one ``(nlat, nlon)`` grid at ``(lat, lon)``.

    Ascending coordinates assumed. Caller guarantees ``(lat, lon)`` is in-bounds (use
    :func:`in_domain` first).

    Args:
        grid2d: Values, shape ``(nlat, nlon)``.
        lats: Ascending latitude coordinates, length ``nlat``.
        lons: Ascending longitude coordinates, length ``nlon``.
        lat: Query latitude.
        lon: Query longitude.

    Returns:
        Interpolated value, or NaN if any of the 4 surrounding corners is NaN.
    """
    lat_idx = int(np.clip(np.searchsorted(lats, lat, side="left") - 1, 0, len(lats) - 2))
    lon_idx = int(np.clip(np.searchsorted(lons, lon, side="left") - 1, 0, len(lons) - 2))

    lat0, lat1 = lats[lat_idx], lats[lat_idx + 1]
    lon0, lon1 = lons[lon_idx], lons[lon_idx + 1]
    t = (lat - lat0) / (lat1 - lat0) if lat1 != lat0 else 0.0
    s = (lon - lon0) / (lon1 - lon0) if lon1 != lon0 else 0.0

    v00 = grid2d[lat_idx, lon_idx]
    v01 = grid2d[lat_idx, lon_idx + 1]
    v10 = grid2d[lat_idx + 1, lon_idx]
    v11 = grid2d[lat_idx + 1, lon_idx + 1]

    return float(
        (1.0 - t) * (1.0 - s) * v00 + (1.0 - t) * s * v01 + t * (1.0 - s) * v10 + t * s * v11
    )


def in_domain(wind: WindField, lat: float, lon: float) -> bool:
    """True iff ``(lat, lon)`` is within ``[lats[0], lats[-1]] x [lons[0], lons[-1]]``.

    Args:
        wind: The wind field defining the domain.
        lat: Latitude to test.
        lon: Longitude to test.

    Returns:
        Whether the point is inside the (inclusive) domain bounds.
    """
    return bool(wind.lats[0] <= lat <= wind.lats[-1] and wind.lons[0] <= lon <= wind.lons[-1])


def sample_uv(wind: WindField, when: np.datetime64, lat: float, lon: float) -> tuple[float, float]:
    """Sample ``(u, v)`` m/s at ``(when, lat, lon)``.

    Bilinear in space, linear in time between the two bracketing hourly slices.

    Args:
        wind: Source wind field.
        when: Query time; must fall within ``[wind.times[0], wind.times[-1]]``.
        lat: Query latitude.
        lon: Query longitude.

    Returns:
        ``(u, v)`` in m/s. ``(nan, nan)`` if the interpolation touches a NaN grid cell
        (missing hour).

    Raises:
        ValueError: If ``when`` is outside the wind field's time range.
    """
    when64 = np.datetime64(when, "ns")
    if when64 < wind.times[0] or when64 > wind.times[-1]:
        raise ValueError(
            f"when={when64} outside wind time range [{wind.times[0]}, {wind.times[-1]}]"
        )

    idx = int(
        np.clip(np.searchsorted(wind.times, when64, side="right") - 1, 0, len(wind.times) - 2)
    )
    t0, t1 = wind.times[idx], wind.times[idx + 1]
    frac = 0.0 if t1 == t0 else float((when64 - t0) / (t1 - t0))

    u0 = _bilinear(wind.u[idx], wind.lats, wind.lons, lat, lon)
    u1 = _bilinear(wind.u[idx + 1], wind.lats, wind.lons, lat, lon)
    v0 = _bilinear(wind.v[idx], wind.lats, wind.lons, lat, lon)
    v1 = _bilinear(wind.v[idx + 1], wind.lats, wind.lons, lat, lon)

    u = (1.0 - frac) * u0 + frac * u1
    v = (1.0 - frac) * v0 + frac * v1
    return float(u), float(v)


def _deg_per_hour(u_ms: float, v_ms: float, lat_deg: float) -> tuple[float, float]:
    """Convert ``(u, v)`` m/s to ``(dlon, dlat)`` degrees per hour at latitude ``lat_deg``.

    Args:
        u_ms: Eastward wind speed, m/s.
        v_ms: Northward wind speed, m/s.
        lat_deg: Latitude at which the longitude scale factor is evaluated.

    Returns:
        ``(dlon_per_hour, dlat_per_hour)`` in degrees/hour.
    """
    dlat = v_ms * _SEC_PER_HOUR / _M_PER_DEG_LAT
    dlon = u_ms * _SEC_PER_HOUR / (_M_PER_DEG_LON_EQ * np.cos(np.radians(lat_deg)))
    return float(dlon), float(dlat)


def _hours_timedelta(hours: float) -> np.timedelta64:
    """Convert a (possibly fractional) hour count to a nanosecond-precision timedelta64."""
    return np.timedelta64(round(hours * _SEC_PER_HOUR * 1e9), "ns")


def _rk2_back_step_detail(
    wind: WindField, lat: float, lon: float, when: np.datetime64, dt_hours: float
) -> tuple[tuple[float, float, np.datetime64] | None, str]:
    """Implementation shared by :func:`rk2_back_step` and :func:`integrate_backtrajectory`.

    Returns both the stepped position (or ``None``) and a diagnostic reason
    (``"completed"`` | ``"left_domain"`` | ``"missing_wind"``) so the caller can record why a
    trajectory stopped without re-deriving the failure cause.
    """
    when64 = np.datetime64(when, "ns")
    try:
        u1, v1 = sample_uv(wind, when64, lat, lon)
    except ValueError:
        return None, "missing_wind"
    if not (np.isfinite(u1) and np.isfinite(v1)):
        return None, "missing_wind"

    dlon1, dlat1 = _deg_per_hour(u1, v1, lat)
    half_dt = dt_hours / 2.0
    mid_lat = lat - dlat1 * half_dt
    mid_lon = lon - dlon1 * half_dt
    mid_time = when64 - _hours_timedelta(half_dt)

    if not in_domain(wind, mid_lat, mid_lon):
        return None, "left_domain"

    try:
        u2, v2 = sample_uv(wind, mid_time, mid_lat, mid_lon)
    except ValueError:
        return None, "missing_wind"
    if not (np.isfinite(u2) and np.isfinite(v2)):
        return None, "missing_wind"

    dlon2, dlat2 = _deg_per_hour(u2, v2, mid_lat)
    lat_prev = lat - dlat2 * dt_hours
    lon_prev = lon - dlon2 * dt_hours
    when_prev = when64 - _hours_timedelta(dt_hours)

    if not in_domain(wind, lat_prev, lon_prev):
        return None, "left_domain"

    return (lat_prev, lon_prev, when_prev), "completed"


def rk2_back_step(
    wind: WindField, lat: float, lon: float, when: np.datetime64, dt_hours: float
) -> tuple[float, float, np.datetime64] | None:
    """One RK2 (midpoint) backward step.

    The parcel arrives at ``(lat, lon)`` at time ``when``; this returns where it was
    ``dt_hours`` earlier::

        k1 = uv(x_t,   t)
        x_mid = x_t - deg(k1, lat_t) * dt/2   at time t - dt/2
        k2 = uv(x_mid, t - dt/2)
        x_{t-dt} = x_t - deg(k2, lat_mid) * dt

    Args:
        wind: Source wind field.
        lat: Current latitude.
        lon: Current longitude.
        when: Current time.
        dt_hours: Step size in hours.

    Returns:
        ``(lat_prev, lon_prev, when_prev)``, or ``None`` if the step leaves the spatial
        domain or hits missing wind (NaN) at either stage.
    """
    result, _reason = _rk2_back_step_detail(wind, lat, lon, when, dt_hours)
    return result


def integrate_backtrajectory(
    wind: WindField,
    start_lat: float,
    start_lon: float,
    start_time: np.datetime64,
    hours_back: int,
    dt_hours: float = 1.0,
) -> BackTrajectory:
    """Integrate backward ``hours_back`` hours from the launch point.

    Stops early and records ``terminated_reason`` when a step leaves the domain
    (``"left_domain"``) or hits missing wind (``"missing_wind"``); otherwise ``"completed"``.

    Args:
        wind: Source wind field.
        start_lat: Launch latitude.
        start_lon: Launch longitude.
        start_time: Launch time (offset 0 of the returned trajectory).
        hours_back: Number of hours to integrate backward.
        dt_hours: Step size in hours (default 1.0, matching ERA5's native resolution).

    Returns:
        A :class:`BackTrajectory` of length ``hours_integrated + 1``.

    Raises:
        ValueError: If the launch point itself is outside the wind field's spatial domain.
    """
    if not in_domain(wind, start_lat, start_lon):
        raise ValueError(
            f"Launch point (lat={start_lat}, lon={start_lon}) is outside the wind domain "
            f"[{wind.lats[0]}, {wind.lats[-1]}] x [{wind.lons[0]}, {wind.lons[-1]}]"
        )

    n_steps = round(hours_back / dt_hours)
    lats = [float(start_lat)]
    lons = [float(start_lon)]
    times = [np.datetime64(start_time, "ns")]
    reason = "completed"

    cur_lat, cur_lon, cur_time = lats[0], lons[0], times[0]
    for _ in range(n_steps):
        result, step_reason = _rk2_back_step_detail(wind, cur_lat, cur_lon, cur_time, dt_hours)
        if result is None:
            reason = step_reason
            break
        cur_lat, cur_lon, cur_time = result
        lats.append(cur_lat)
        lons.append(cur_lon)
        times.append(cur_time)

    return BackTrajectory(
        times=np.array(times, dtype="datetime64[ns]"),
        lats=np.array(lats, dtype=np.float64),
        lons=np.array(lons, dtype=np.float64),
        terminated_reason=reason,
        hours_integrated=len(lats) - 1,
    )


# ---------------------------------------------------------------------------
# Labeling / metric functions
# ---------------------------------------------------------------------------


def label_countries(
    lats: np.ndarray, lons: np.ndarray, geojson_path: Path | str = DEFAULT_BORDERS_PATH
) -> list[str]:
    """Country per trajectory point via :func:`src.data.geocode.country_of`.

    Args:
        lats: Trajectory latitudes.
        lons: Trajectory longitudes (same length as ``lats``).
        geojson_path: Path to the borders geojson (injectable for tests).

    Returns:
        List of ``"Thailand"`` | ``"Myanmar"`` | ``"Laos"`` | ``"other"``, one per point.
    """
    return [
        country_of(float(lon), float(lat), geojson_path)
        for lat, lon in zip(lats, lons, strict=True)
    ]


def country_hours_fraction(countries: Sequence[str]) -> dict[str, float]:
    """Fraction of points in each of Thailand/Myanmar/Laos/other.

    Args:
        countries: Per-point country labels.

    Returns:
        Dict with all four keys always present (``0.0`` default), summing to 1.0
        (``{}``-safe: returns all-zero if ``countries`` is empty).
    """
    result = {key: 0.0 for key in _COUNTRY_KEYS}
    n = len(countries)
    if n == 0:
        return result
    for name in countries:
        if name in result:
            result[name] += 1.0
        else:
            result[name] = result.get(name, 0.0) + 1.0
    return {key: value / n for key, value in result.items()}


def corridor_frp_by_country(
    traj: BackTrajectory, hotspots: pd.DataFrame, radius_km: float = 50.0
) -> dict[str, float]:
    """Sum FIRMS ``total_frp`` by cluster country for clusters near the trajectory corridor.

    For each trajectory hour, finds hotspot clusters within ``radius_km`` of that point on
    that point's UTC date, and accumulates ``total_frp`` by the cluster's own ``country``
    label. Deduplicated per ``(date, cluster_id)`` so a cluster loitered-near across several
    hours counts once. Uses haversine distance.

    Args:
        traj: The backward trajectory to check.
        hotspots: FIRMS hotspot clusters with columns ``date``, ``cluster_id``,
            ``centroid_lat``, ``centroid_lon``, ``total_frp``, ``country``.
        radius_km: Corridor radius in km.

    Returns:
        Dict ``{"Thailand": float, "Myanmar": float, "Laos": float, ...}`` (any additional
        country label present in ``hotspots`` is also included); all-zero for the three
        known countries if ``hotspots`` is empty or nothing is nearby.
    """
    result: dict[str, float] = {"Thailand": 0.0, "Myanmar": 0.0, "Laos": 0.0}
    if hotspots.empty:
        return result

    dates = pd.to_datetime(hotspots["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    lats_h = hotspots["centroid_lat"].to_numpy(dtype=np.float64)
    lons_h = hotspots["centroid_lon"].to_numpy(dtype=np.float64)
    frps = hotspots["total_frp"].to_numpy(dtype=np.float64)
    cluster_ids = hotspots["cluster_id"].to_numpy()
    countries = hotspots["country"].to_numpy()

    seen: set[tuple[str, object]] = set()
    for i in range(len(traj.lats)):
        date_str = pd.Timestamp(traj.times[i]).strftime("%Y-%m-%d")
        day_mask = dates == date_str
        if not day_mask.any():
            continue
        day_idx = np.where(day_mask)[0]
        dist = _haversine_km(
            float(traj.lats[i]), float(traj.lons[i]), lats_h[day_idx], lons_h[day_idx]
        )
        for j in day_idx[dist <= radius_km]:
            key = (date_str, cluster_ids[j])
            if key in seen:
                continue
            seen.add(key)
            country = str(countries[j])
            result[country] = result.get(country, 0.0) + float(frps[j])
    return result


def _iso_utc_z(when: np.datetime64) -> str:
    """Format a tz-naive UTC ``datetime64`` as an ISO 8601 string with a ``Z`` suffix."""
    return pd.Timestamp(when).strftime("%Y-%m-%dT%H:%M:%SZ")


def assess_event(
    traj: BackTrajectory,
    countries: Sequence[str],
    model_foreign_attr: float | None,
    model_country_attr: dict[str, float] | None,
    corridor_frp: dict[str, float],
    foreign_hours_threshold: float = 0.15,
) -> dict[str, object]:
    """Assemble one event dict (schema section 5.2 of the design spec).

    Computes ``foreign_hours_fraction``, ``trajectory_foreign_plausible`` and
    ``agrees_with_model`` (binary direction agreement only -- never a magnitude comparison).
    Does not include ``date``/``launch_time_utc``/``peak_pm25_ug_m3`` (those are
    event-selection metadata the caller already has); the caller merges them in.

    Args:
        traj: The backward trajectory for this event.
        countries: Per-point country labels (same length as ``traj.lats``), from
            :func:`label_countries`.
        model_foreign_attr: The model's foreign (Myanmar+Laos) attribution for this event,
            or ``None`` if the event date is absent from the frozen model-attribution JSON.
        model_country_attr: The model's per-country attribution dict, or ``None``.
        corridor_frp: Output of :func:`corridor_frp_by_country` for this event.
        foreign_hours_threshold: Minimum ``foreign_hours_fraction`` to call the trajectory
            "foreign plausible" (independent of any corridor fire).

    Returns:
        Dict with keys ``model_foreign_attribution``, ``model_country_attribution``,
        ``trajectory`` (``hours_integrated``, ``terminated_reason``, ``points``),
        ``country_hours_fraction``, ``foreign_hours_fraction``, ``corridor_frp_by_country``,
        ``corridor_foreign_frp_fraction``, ``trajectory_foreign_plausible``,
        ``agrees_with_model`` (``None`` if ``model_foreign_attr`` is ``None``).
    """
    chf = country_hours_fraction(countries)
    foreign_hours_fraction = chf["Myanmar"] + chf["Laos"]

    corridor_total = sum(corridor_frp.values())
    corridor_foreign = sum(corridor_frp.get(c, 0.0) for c in _FOREIGN_COUNTRIES)
    corridor_foreign_frp_fraction = corridor_foreign / corridor_total if corridor_total > 0 else 0.0

    trajectory_foreign_plausible = bool(
        foreign_hours_fraction >= foreign_hours_threshold or corridor_foreign > 0.0
    )

    if model_foreign_attr is None:
        agrees_with_model: bool | None = None
    else:
        model_says_foreign = model_foreign_attr >= _MODEL_FOREIGN_TAU
        agrees_with_model = model_says_foreign == trajectory_foreign_plausible

    points = [
        {
            "hour_offset": i,
            "time_utc": _iso_utc_z(traj.times[i]),
            "lat": round(float(traj.lats[i]), 4),
            "lon": round(float(traj.lons[i]), 4),
            "country": countries[i],
        }
        for i in range(len(traj.lats))
    ]

    return {
        "model_foreign_attribution": (
            round(float(model_foreign_attr), 4) if model_foreign_attr is not None else None
        ),
        "model_country_attribution": (
            {k: round(float(v), 4) for k, v in model_country_attr.items()}
            if model_country_attr is not None
            else None
        ),
        "trajectory": {
            "hours_integrated": traj.hours_integrated,
            "terminated_reason": traj.terminated_reason,
            "points": points,
        },
        "country_hours_fraction": {k: round(v, 4) for k, v in chf.items()},
        "foreign_hours_fraction": round(foreign_hours_fraction, 4),
        "corridor_frp_by_country": {k: round(v, 1) for k, v in corridor_frp.items()},
        "corridor_foreign_frp_fraction": round(corridor_foreign_frp_fraction, 4),
        "trajectory_foreign_plausible": trajectory_foreign_plausible,
        "agrees_with_model": agrees_with_model,
    }


# ---------------------------------------------------------------------------
# IO helper (xarray; exercised only by scripts/22_backtrajectory.py, not unit-tested)
# ---------------------------------------------------------------------------


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """Inclusive list of ``(year, month)`` calendar months spanned by ``[start, end]``."""
    months: list[tuple[int, int]] = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1)
    while cur <= end_month:
        months.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return months


def load_era5_window(
    era5_dir: Path,
    start_time: pd.Timestamp,
    hours_back: int,
    bbox: tuple[float, float, float, float] = ERA5_BBOX,
) -> WindField:
    """Load the ERA5 monthly file(s) covering the backward-integration window.

    Reuses :func:`src.data.scrapers.era5._open_nc` (ASCII-temp copy -- the Thai project cwd
    breaks plain ``xr.open_dataset``), ``_normalise_coords``, ``_find_var``/``_ERA5_ALIASES``.
    If the window spans a month boundary, opens both monthly files, concatenates on time,
    drops duplicate timestamps, and sorts.

    Args:
        era5_dir: Directory containing ``era5_{year}_{month:02d}.nc`` monthly files.
        start_time: Launch time (the trajectory's ``hour_offset=0``). Any timezone is
            stripped before comparing against ERA5's tz-naive UTC time coordinate.
        hours_back: Number of hours the trajectory will integrate backward; enough lead
            time is loaded so :func:`sample_uv` never runs out of range for this request.
        bbox: ``(west, south, east, north)`` bounding box to select from the file.

    Returns:
        A :class:`WindField` covering ``[start_time - hours_back - 1h, start_time]``.

    Raises:
        FileNotFoundError: If a monthly file needed to cover the window is absent.
        KeyError: If ``u10``/``v10`` cannot be resolved in the loaded file(s).
    """
    if start_time.tzinfo is not None:
        start_time = start_time.tz_localize(None)

    window_start = start_time - pd.Timedelta(hours=hours_back + 1)
    months = _month_range(window_start, start_time)

    datasets: list[xr.Dataset] = []
    for year, month in months:
        nc_path = era5_dir / f"era5_{year}_{month:02d}.nc"
        if not nc_path.exists():
            raise FileNotFoundError(
                f"ERA5 monthly file not found: {nc_path} (needed to cover launch="
                f"{start_time} back {hours_back}h)"
            )
        datasets.append(_normalise_coords(_open_nc(nc_path)))

    ds_all = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    ds_all = ds_all.sortby("time")
    _, unique_idx = np.unique(ds_all["time"].values, return_index=True)
    ds_all = ds_all.isel(time=np.sort(unique_idx))

    west, south, east, north = bbox
    ds_all = ds_all.sortby("lat").sel(lat=slice(south, north), lon=slice(west, east))

    u_name = _find_var(ds_all, _ERA5_ALIASES["u10"])
    v_name = _find_var(ds_all, _ERA5_ALIASES["v10"])
    if u_name is None or v_name is None:
        month_names = [f"era5_{y}_{m:02d}.nc" for y, m in months]
        raise KeyError(
            f"u10/v10 not found in ERA5 window files {month_names}: "
            f"available data_vars={list(ds_all.data_vars)}"
        )

    times = ds_all["time"].values
    lats = ds_all["lat"].values.astype(np.float64)
    lons = ds_all["lon"].values.astype(np.float64)
    u = ds_all[u_name].values.astype(np.float64)
    v = ds_all[v_name].values.astype(np.float64)

    for ds in datasets:
        ds.close()

    logger.info(
        "load_era5_window: %d months, %d timesteps, %dx%d grid, window ending %s",
        len(months),
        len(times),
        len(lats),
        len(lons),
        start_time,
    )
    return make_wind_field(times, lats, lons, u, v)
