"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# Default thresholds from DESIGN.md §5.2.
_DEFAULT_CONFIG: dict[str, Any] = {
    "wind_mode": "constant_ne",  # "constant_ne" | "random" | "from_field"
    "type_a_max_km": 100.0,
    "type_b_max_km": 200.0,
    "type_b_min_alignment": 0.3,
    "type_c_max_km": 500.0,
    "type_c_min_alignment": 0.4,
    # constant_ne synthetic wind components (u=eastward, v=northward, m/s)
    "synthetic_u": 3.5,
    "synthetic_v": 3.5,
    # feature columns to read from df_stations for station node features
    "station_features": ["pm25_scaled", "hour_sin", "hour_cos", "doy_sin", "doy_cos"],
}

_EARTH_RADIUS_KM: float = 6371.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (haversine formula).

    Args:
        lat1: Source latitude (degrees).
        lon1: Source longitude (degrees).
        lat2: Target latitude (degrees).
        lon2: Target longitude (degrees).

    Returns:
        Distance in km.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_rad(src_lat: float, src_lon: float, tgt_lat: float, tgt_lon: float) -> float:
    """Equirectangular bearing from source to target in radians.

    Uses the equirectangular approximation:
        bearing = atan2(dlon * cos(src_lat), dlat)

    At our bbox extremes (~500 km, lat 16-21 deg), deviation from great-circle
    bearing is <1.5 deg, which maps to <0.02 alignment error — within tolerance.

    Returns:
        Bearing in radians. 0 = north, pi/2 = east, -pi/2 = west.
    """
    dlat = math.radians(tgt_lat - src_lat)
    dlon = math.radians(tgt_lon - src_lon)
    return math.atan2(dlon * math.cos(math.radians(src_lat)), dlat)


def _wind_alignment(
    u: float, v: float, src_lat: float, src_lon: float, tgt_lat: float, tgt_lon: float
) -> float:
    """Wind alignment = cos(theta), where theta = bearing - wind_direction.

    Positive (close to 1) when wind blows from source toward target.
    Zero for perpendicular wind. Negative when wind opposes the direction.

    Wind direction is the direction the wind IS BLOWING TO (matches ERA5/ECMWF
    u/v convention: u = eastward component, v = northward component).
        wind_dir = atan2(u, v)   # 0=north, pi/2=east

    Args:
        u: Eastward wind component (m/s).
        v: Northward wind component (m/s).
        src_lat: Source latitude (degrees).
        src_lon: Source longitude (degrees).
        tgt_lat: Target latitude (degrees).
        tgt_lon: Target longitude (degrees).

    Returns:
        Alignment in [-1, 1].
    """
    bearing = _bearing_rad(src_lat, src_lon, tgt_lat, tgt_lon)
    wind_dir = math.atan2(u, v)
    return math.cos(bearing - wind_dir)


# ---------------------------------------------------------------------------
# Wind field access
# ---------------------------------------------------------------------------


def _get_wind_uv(
    wind_field: xr.DataArray | None,
    config: dict[str, Any],
    lat: float,
    lon: float,
) -> tuple[float, float]:
    """Retrieve (u, v) wind components for a spatial location.

    Args:
        wind_field: ERA5 DataArray (time, lat, lon, component). Ignored unless
            wind_mode is "from_field".
        config: Graph config dict (see ``_DEFAULT_CONFIG``).
        lat: Latitude of the query point (degrees).
        lon: Longitude of the query point (degrees).

    Returns:
        (u, v) in m/s.

    Raises:
        ValueError: If wind_mode is unknown.
        RuntimeError: If wind_mode is "from_field" but wind_field is None.
    """
    mode = config.get("wind_mode", "constant_ne")
    if mode == "constant_ne":
        return float(config.get("synthetic_u", 3.5)), float(config.get("synthetic_v", 3.5))
    if mode == "random":
        rng: np.random.Generator = config.setdefault("_rng", np.random.default_rng())
        return float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))
    if mode == "from_field":
        if wind_field is None:
            raise RuntimeError("wind_mode='from_field' requires a non-None wind_field DataArray")
        ts = config.get("timestamp")
        t_slice = (
            wind_field.sel(time=ts, method="nearest") if ts is not None else wind_field.isel(time=0)
        )
        nearest = t_slice.sel(lat=lat, lon=lon, method="nearest")
        u = float(nearest.isel(component=0).values)
        v = float(nearest.isel(component=1).values)
        return u, v
    if mode == "from_arrays":
        # Pre-interpolated per-station ERA5 arrays stored in config by the loader.
        # Nearest-station lookup serves both station locations (exact match) and
        # hotspot locations (close enough for wind direction estimation).
        u_arr: np.ndarray = config["_u10_per_station"]
        v_arr: np.ndarray = config["_v10_per_station"]
        s_lats: np.ndarray = config["_station_lats"]
        s_lons: np.ndarray = config["_station_lons"]
        dists = (s_lats - lat) ** 2 + (s_lons - lon) ** 2
        idx = int(np.argmin(dists))
        return float(u_arr[idx]), float(v_arr[idx])
    raise ValueError(
        f"Unknown wind_mode: {mode!r}. "
        "Expected 'constant_ne', 'random', 'from_field', or 'from_arrays'."
    )


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------


def _build_type_a_edges(
    df_stations: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Type A (static spatial) edges.

    Symmetric (undirected): adds both (i->j) and (j->i) for all pairs
    within ``type_a_max_km``. Weight = exp(-d / 50).

    Args:
        df_stations: Stations with at least ``lat`` and ``lon`` columns.
            Row order determines node index.
        config: Graph config dict.

    Returns:
        edge_index: LongTensor [2, E_a].
        edge_attr: FloatTensor [E_a, 1] (weight).
    """
    max_km = float(config.get("type_a_max_km", _DEFAULT_CONFIG["type_a_max_km"]))
    lats = df_stations["lat"].values
    lons = df_stations["lon"].values
    n = len(df_stations)

    src_idx: list[int] = []
    tgt_idx: list[int] = []
    weights: list[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_km(lats[i], lons[i], lats[j], lons[j])
            if d <= max_km:
                w = math.exp(-d / 50.0)
                src_idx += [i, j]
                tgt_idx += [j, i]
                weights += [w, w]

    if not src_idx:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 1), dtype=torch.float32),
        )
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    edge_attr = torch.tensor([[w] for w in weights], dtype=torch.float32)
    return edge_index, edge_attr


def _build_type_b_edges(
    df_stations: pd.DataFrame,
    wind_field: xr.DataArray | None,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Type B (wind-aware dynamic) edges between stations.

    Directed: i -> j only when wind blows from i toward j.
    Condition: distance <= type_b_max_km AND alignment > type_b_min_alignment.
    Weight = alignment * wind_speed * exp(-d / 100).

    Args:
        df_stations: Stations with ``lat`` and ``lon`` columns.
        wind_field: ERA5 DataArray or None (synthetic modes ignore it).
        config: Graph config dict.

    Returns:
        edge_index: LongTensor [2, E_b].
        edge_attr: FloatTensor [E_b, 3] (weight, alignment, wind_speed).
    """
    max_km = float(config.get("type_b_max_km", _DEFAULT_CONFIG["type_b_max_km"]))
    min_align = float(config.get("type_b_min_alignment", _DEFAULT_CONFIG["type_b_min_alignment"]))
    lats = df_stations["lat"].values
    lons = df_stations["lon"].values
    n = len(df_stations)

    src_idx: list[int] = []
    tgt_idx: list[int] = []
    attrs: list[list[float]] = []

    for i in range(n):
        u, v = _get_wind_uv(wind_field, config, lats[i], lons[i])
        wind_speed = math.sqrt(u**2 + v**2)
        for j in range(n):
            if i == j:
                continue
            d = _haversine_km(lats[i], lons[i], lats[j], lons[j])
            if d > max_km:
                continue
            alignment = _wind_alignment(u, v, lats[i], lons[i], lats[j], lons[j])
            if alignment <= min_align:
                continue
            w = alignment * wind_speed * math.exp(-d / 100.0)
            src_idx.append(i)
            tgt_idx.append(j)
            attrs.append([w, alignment, wind_speed])

    if not src_idx:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 3), dtype=torch.float32),
        )
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def _build_type_c_edges(
    df_stations: pd.DataFrame,
    df_hotspots: pd.DataFrame,
    wind_field: xr.DataArray | None,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Type C (hotspot influence) edges from hotspot nodes to station nodes.

    Directed: hotspot h -> station s when wind blows from h toward s.
    Condition: distance <= type_c_max_km AND alignment > type_c_min_alignment.
    Weight = alignment * total_frp * exp(-d / 200).

    Args:
        df_stations: Stations with ``lat`` and ``lon`` columns.
        df_hotspots: Daily clusters with ``centroid_lat``, ``centroid_lon``,
            ``total_frp``. Row order determines hotspot node index.
        wind_field: ERA5 DataArray or None.
        config: Graph config dict.

    Returns:
        edge_index: LongTensor [2, E_c] where row 0 = hotspot index, row 1 = station index.
        edge_attr: FloatTensor [E_c, 3] (weight, alignment, frp).
    """
    if df_hotspots.empty:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 3), dtype=torch.float32),
        )

    max_km = float(config.get("type_c_max_km", _DEFAULT_CONFIG["type_c_max_km"]))
    min_align = float(config.get("type_c_min_alignment", _DEFAULT_CONFIG["type_c_min_alignment"]))
    s_lats = df_stations["lat"].values
    s_lons = df_stations["lon"].values
    h_lats = df_hotspots["centroid_lat"].values
    h_lons = df_hotspots["centroid_lon"].values
    frp_vals = df_hotspots["total_frp"].values

    src_idx: list[int] = []
    tgt_idx: list[int] = []
    attrs: list[list[float]] = []

    for h_idx in range(len(df_hotspots)):
        u, v = _get_wind_uv(wind_field, config, float(h_lats[h_idx]), float(h_lons[h_idx]))
        for s_idx in range(len(df_stations)):
            d = _haversine_km(
                float(h_lats[h_idx]),
                float(h_lons[h_idx]),
                float(s_lats[s_idx]),
                float(s_lons[s_idx]),
            )
            if d > max_km:
                continue
            alignment = _wind_alignment(
                u,
                v,
                float(h_lats[h_idx]),
                float(h_lons[h_idx]),
                float(s_lats[s_idx]),
                float(s_lons[s_idx]),
            )
            if alignment <= min_align:
                continue
            frp = float(frp_vals[h_idx])
            w = alignment * frp * math.exp(-d / 200.0)
            src_idx.append(h_idx)
            tgt_idx.append(s_idx)
            attrs.append([w, alignment, frp])

    if not src_idx:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 3), dtype=torch.float32),
        )
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_graph(
    df_stations: pd.DataFrame,
    df_hotspots: pd.DataFrame,
    wind_field: xr.DataArray | None,
    config: dict[str, Any],
) -> HeteroData:
    """Build a heterogeneous graph for a single timestep.

    Builds graph for a single timestep. Time-series stacking is the
    responsibility of the dataset loader (Session 3).

    Three edge types (DESIGN.md §5.2):
        Type A: static spatial (station <-> station, distance <= 100 km)
        Type B: wind-aware dynamic (station -> station, distance <= 200 km, alignment > 0.3)
        Type C: hotspot influence (hotspot -> station, distance <= 500 km, alignment > 0.4)

    Args:
        df_stations: DataFrame with at least ``station_id``, ``lat``, ``lon``
            columns. Feature columns listed in ``config['station_features']``
            are used as ``data['station'].x``. Row order determines node index.
        df_hotspots: Daily hotspot clusters for the target date.
            Required: ``centroid_lat``, ``centroid_lon``, ``total_frp``.
            Pass an empty DataFrame for days with no hotspot data.
        wind_field: ERA5 wind DataArray with dims (time, lat, lon, component)
            where component 0 = u (east) and component 1 = v (north).
            Pass None when wind_mode is 'constant_ne' or 'random'.
        config: Graph configuration. Keys and defaults defined in
            ``_DEFAULT_CONFIG``. Unset keys fall back to defaults.

    Returns:
        HeteroData with node types 'station' and 'hotspot' and edge types
        ('station','type_a','station'), ('station','type_b','station'),
        ('hotspot','type_c','station').
    """
    cfg = {**_DEFAULT_CONFIG, **config}

    # ---- Station node features ----
    feat_cols = cfg.get("station_features", [])
    available = [c for c in feat_cols if c in df_stations.columns]
    if available:
        station_x = torch.tensor(
            df_stations[available].values.astype("float32"), dtype=torch.float32
        )
    else:
        station_x = torch.zeros((len(df_stations), 0), dtype=torch.float32)

    # ---- Hotspot node features ----
    if df_hotspots.empty:
        hotspot_x = torch.zeros((0, 3), dtype=torch.float32)
    else:
        hotspot_x = torch.tensor(
            df_hotspots[["total_frp", "centroid_lat", "centroid_lon"]].values.astype("float32"),
            dtype=torch.float32,
        )

    # ---- Edge construction ----
    ei_a, ea_a = _build_type_a_edges(df_stations, cfg)
    ei_b, ea_b = _build_type_b_edges(df_stations, wind_field, cfg)
    ei_c, ea_c = _build_type_c_edges(df_stations, df_hotspots, wind_field, cfg)

    logger.debug(
        "build_graph: stations=%d hotspots=%d edges: A=%d B=%d C=%d",
        len(df_stations),
        len(df_hotspots),
        ei_a.shape[1],
        ei_b.shape[1],
        ei_c.shape[1],
    )

    # ---- Assemble HeteroData ----
    data = HeteroData()
    data["station"].x = station_x
    data["hotspot"].x = hotspot_x
    data["station", "type_a", "station"].edge_index = ei_a
    data["station", "type_a", "station"].edge_attr = ea_a
    data["station", "type_b", "station"].edge_index = ei_b
    data["station", "type_b", "station"].edge_attr = ea_b
    data["hotspot", "type_c", "station"].edge_index = ei_c
    data["hotspot", "type_c", "station"].edge_attr = ea_c
    return data


# Public wrappers for use in loader.py fast-path precompute.
build_type_a_edges = _build_type_a_edges
build_type_b_edges = _build_type_b_edges
build_type_c_edges = _build_type_c_edges
