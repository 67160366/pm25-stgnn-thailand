"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Shared orchestration for the transboundary-attribution demo scripts
(``scripts/20_transboundary_matrix.py`` and ``scripts/21_transboundary_uncertainty.py``).

``scripts/10_transboundary_attr.py`` produced the frozen
``outputs/transboundary_attr_test_split2.json`` that the report depends on, so it is
deliberately left untouched (editing it before the SIMS submission risks a reviewer
re-run producing different bytes). Its reusable pure functions
(``_candidate_foreign_dates``, ``_peak_anchor_for_date``, ``_connected_frp_by_country``,
``_summarize_event``, ``_load_mtgnn``) are instead loaded by file path via
``importlib.util.spec_from_file_location`` (the same pattern
``tests/test_eval_2026.py::_load_eval_module`` uses for other digit-prefixed scripts) and
re-exposed here. This is a mild ``src`` -> ``scripts`` layering inversion, accepted because
not modifying script 10 is the higher-priority constraint; do not "fix" it by editing
script 10 or by copy-pasting its logic into this module.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
from datetime import date as _date
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from src.data.geocode import DEFAULT_BORDERS_PATH

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT10_PATH = _PROJECT_ROOT / "scripts" / "10_transboundary_attr.py"

# Myanmar/Laos ISO3 -> display name, matching src.data.geocode._ISO_TO_NAME.
_FOREIGN_ISO_TO_NAME: dict[str, str] = {"MMR": "Myanmar", "LAO": "Laos"}
_EARTH_RADIUS_KM = 6371.0088


@functools.lru_cache(maxsize=1)
def load_script10() -> ModuleType:
    """Import ``scripts/10_transboundary_attr.py`` by file path and return the module.

    Digit-prefixed scripts are not importable via normal ``import`` statements, so this
    loads the module by path (cached: the file is only executed once per process). Re-exposes
    ``_candidate_foreign_dates``, ``_peak_anchor_for_date``, ``_connected_frp_by_country``,
    ``_summarize_event``, ``_load_mtgnn``, ``_FOREIGN`` and ``_DEFAULT_STATION_ID`` without
    duplicating their logic.

    Returns:
        The executed ``scripts/10_transboundary_attr.py`` module object.

    Raises:
        ImportError: If the module spec/loader cannot be created for the script path.
    """
    spec = importlib.util.spec_from_file_location("script10_transboundary_attr", _SCRIPT10_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {_SCRIPT10_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_connected_foreign_events(
    ds: object,
    station_idx: int,
    hotspots_path: Path,
    split: str,
    n_candidates: int = 15,
    top_k: int = 5,
) -> list[tuple[float, int, _date, float]]:
    """Rank candidate events at a station by connected-foreign FRP (script-10 event loop).

    Mirrors the event-ranking loop inside ``scripts/10_transboundary_attr.py::main()``, built
    on top of the reused script-10 pure functions. Event selection is MODEL-INDEPENDENT: it
    depends only on FIRMS FRP ranking (``_candidate_foreign_dates``), per-station peak PM2.5
    (``_peak_anchor_for_date``), and the wind-deterministic type_c graph baked into ``ds`` at
    construction time (``_connected_frp_by_country``) -- never on a trained model. This
    property is load-bearing for item 4 (per-event mean +/- spread across model seeds): the
    identical event set is reused across every checkpoint/seed, so seed-to-seed variation can
    only come from attribution magnitude, not from a different set of events being compared.

    Args:
        ds: PM25GraphDataset (or a duck-typed equivalent exposing ``_timestamps``,
            ``_anchor_indices``, ``_pm25_raw``, ``_station_ids`` and ``__getitem__``).
        station_idx: Index of the target station within ``ds``.
        hotspots_path: Path to the FIRMS hotspots parquet used to rank candidate dates.
        split: Dataset split name (``"train"`` | ``"val"`` | ``"test"``).
        n_candidates: Number of foreign-FRP-ranked candidate dates to examine.
        top_k: Number of top events (by connected-foreign FRP) to return.

    Returns:
        Up to ``top_k`` tuples ``(connected_foreign_frp, anchor_pos, date, peak_pm25)``,
        sorted by ``connected_foreign_frp`` descending.
    """
    mod = load_script10()
    cand_dates = mod._candidate_foreign_dates(hotspots_path, split, n_candidates)
    events: list[tuple[float, int, _date, float]] = []
    for d in cand_dates:
        found = mod._peak_anchor_for_date(ds, d, station_idx)
        if found is None:
            continue
        pos, peak = found
        conn = mod._connected_frp_by_country(ds[pos], station_idx)
        foreign = sum(v for k, v in conn.items() if k in mod._FOREIGN)
        events.append((foreign, pos, d, peak))

    events.sort(key=lambda e: -e[0])
    return events[:top_k]


def _load_foreign_vertices(borders_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten every Myanmar/Laos polygon vertex out of a borders geojson.

    Args:
        borders_path: Path to a geojson with per-feature ``id`` in {"THA","MMR","LAO"} and
            ``Polygon``/``MultiPolygon`` geometry in (lon, lat) coordinate order.

    Returns:
        Tuple ``(lons, lats, country_names)`` of equal-length float/str arrays, one entry
        per Myanmar-or-Laos polygon vertex (Thailand vertices are excluded).
    """
    with Path(borders_path).open(encoding="utf-8") as fh:
        gj = json.load(fh)

    lons: list[float] = []
    lats: list[float] = []
    countries: list[str] = []
    for feat in gj["features"]:
        name = _FOREIGN_ISO_TO_NAME.get(feat.get("id", ""))
        if name is None:
            continue
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for rings in polys:
            for ring in rings:
                for lon, lat in ring:
                    lons.append(float(lon))
                    lats.append(float(lat))
                    countries.append(name)

    return (
        np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
        np.asarray(countries),
    )


@functools.lru_cache(maxsize=4)
def _cached_foreign_vertices(borders_path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached wrapper around ``_load_foreign_vertices`` keyed by path string."""
    return _load_foreign_vertices(borders_path_str)


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized great-circle distance in km from one point to an array of points."""
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return _EARTH_RADIUS_KM * c


def nearest_foreign_distance_km(
    lat: float, lon: float, borders_path: Path | str = DEFAULT_BORDERS_PATH
) -> tuple[float, str]:
    """Great-circle distance from (lat, lon) to the nearest Myanmar-or-Laos polygon vertex.

    Uses the same committed geojson the map and geocoder use (single source of truth), so
    border-relevance is reproducible from data already in the repo. Vertex-based, so this is
    a documented monotonic approximation of true polygon-edge border distance (a point just
    outside the vertex-to-vertex chord can be nominally closer to the true boundary than to
    any single vertex); adequate for a coarse >=50km border-relevance filter.

    Args:
        lat: Station latitude.
        lon: Station longitude.
        borders_path: Path to the borders geojson (injectable for tests).

    Returns:
        Tuple ``(distance_km, nearest_country)`` where ``nearest_country`` is
        ``"Myanmar"`` or ``"Laos"``.

    Raises:
        ValueError: If the geojson contains no Myanmar/Laos geometry.
    """
    lons, lats, countries = _cached_foreign_vertices(str(borders_path))
    if lons.size == 0:
        raise ValueError(f"No Myanmar/Laos polygon vertices found in {borders_path}")
    dist = _haversine_km(lat, lon, lats, lons)
    i = int(np.argmin(dist))
    return float(dist[i]), str(countries[i])


def select_border_stations(
    metadata: pd.DataFrame,
    max_dist_km: float = 50.0,
    borders_path: Path | str = DEFAULT_BORDERS_PATH,
) -> list[dict]:
    """Select stations near the Myanmar/Laos border, ranked by distance.

    Accepts either ``location_id`` (the raw ``stations_metadata.parquet`` column name) or
    ``station_id`` (script-10's renamed column) so callers don't need to pre-rename.

    Args:
        metadata: Station metadata with columns ``location_id``/``station_id``, ``name``,
            ``lat``, ``lon``.
        max_dist_km: Inclusive border-relevance threshold in km.
        borders_path: Path to the borders geojson (injectable for tests).

    Returns:
        List of dicts ``{station_id, name, lat, lon, dist_km_nearest_foreign,
        nearest_country}``, one per selected station, sorted by distance ascending.
    """
    df = metadata.copy()
    if "station_id" not in df.columns and "location_id" in df.columns:
        df = df.rename(columns={"location_id": "station_id"})

    rows: list[dict] = []
    for _, r in df.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        dist_km, country = nearest_foreign_distance_km(lat, lon, borders_path=borders_path)
        if dist_km <= max_dist_km:
            rows.append(
                {
                    "station_id": int(r["station_id"]),
                    "name": str(r["name"]),
                    "lat": lat,
                    "lon": lon,
                    "dist_km_nearest_foreign": round(dist_km, 1),
                    "nearest_country": country,
                }
            )

    rows.sort(key=lambda d: d["dist_km_nearest_foreign"])
    logger.info(
        "select_border_stations: %d/%d stations within %.1f km", len(rows), len(df), max_dist_km
    )
    return rows
