# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Country geocoding from real boundary polygons (no shapely dependency).

Replaces the coarse overlapping bounding-box lookup in
``src/data/hotspot_clustering.py`` for *display* purposes in the dashboard.
Uses a ray-casting point-in-polygon test against Natural-Earth-derived
country outlines (Thailand, Myanmar, Laos) stored in ``app/assets``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ASSET = Path(__file__).resolve().parent.parent / "assets" / "borders_th_mm_la.geojson"

_ISO_TO_NAME = {"THA": "Thailand", "MMR": "Myanmar", "LAO": "Laos"}


@lru_cache(maxsize=1)
def _polygons() -> list[tuple[str, list[list[list[float]]]]]:
    """Return [(country_name, polygon_rings)] where each polygon is a list of rings.

    A ring is a list of [lon, lat] points; ring[0] is the exterior, the rest holes.
    Handles both GeoJSON ``Polygon`` and ``MultiPolygon`` geometries.
    """
    with _ASSET.open(encoding="utf-8") as fh:
        gj = json.load(fh)
    out: list[tuple[str, list[list[list[float]]]]] = []
    for feat in gj["features"]:
        name = _ISO_TO_NAME.get(feat.get("id", ""), feat.get("id", ""))
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            out.append((name, geom["coordinates"]))
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                out.append((name, poly))
    return out


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    """Ray-casting test: is (lon, lat) inside this ring?"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_polygon(lon: float, lat: float, rings: list[list[list[float]]]) -> bool:
    """Inside the exterior ring and outside every hole."""
    if not rings or not _point_in_ring(lon, lat, rings[0]):
        return False
    return not any(_point_in_ring(lon, lat, hole) for hole in rings[1:])


@lru_cache(maxsize=1)
def border_geojson() -> dict:
    """Raw GeoJSON FeatureCollection of the three country outlines (for map layers)."""
    with _ASSET.open(encoding="utf-8") as fh:
        return json.load(fh)


def country_of(lon: float, lat: float) -> str:
    """Return 'Thailand' | 'Myanmar' | 'Laos' | 'other' for a coordinate.

    Uses real country outlines (point-in-polygon). Falls back to 'other'
    when the point is outside all three countries.
    """
    for name, rings in _polygons():
        if _point_in_polygon(lon, lat, rings):
            return name
    return "other"
