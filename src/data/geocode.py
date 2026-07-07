"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Country geocoding from real boundary polygons (no shapely dependency).

Replaces the coarse overlapping bounding-box lookup previously used by
``hotspot_clustering`` for country attribution labels. Uses a vectorized
ray-casting point-in-polygon test against geoBoundaries-derived country
outlines (Thailand, Myanmar, Laos) stored in ``app/assets``.

The bbox method undercounted Myanmar/Laos fires near the border (verified
2026-07-02, see docs/SESSION_HANDOFF_pitch_review.md section 3), which biased
the country grouping used by ``occlusion_country_attribution``. Labels are
attribution/display metadata only — they are NOT model input features, so
relabeling requires no retraining.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The geojson ships with the Streamlit app (also used to draw map borders).
DEFAULT_BORDERS_PATH = (
    Path(__file__).resolve().parents[2] / "app" / "assets" / "borders_th_mm_la.geojson"
)

_ISO_TO_NAME = {"THA": "Thailand", "MMR": "Myanmar", "LAO": "Laos"}

# One prepared polygon: country name, per-ring vertex arrays, and the
# exterior ring's bounding box for a cheap reject test.
_Polygon = tuple[str, list[tuple[np.ndarray, np.ndarray]], tuple[float, float, float, float]]


@lru_cache(maxsize=2)
def _prepared_polygons(geojson_path: str) -> list[_Polygon]:
    """Load and prepare country polygons for fast point-in-polygon tests.

    Handles both ``Polygon`` and ``MultiPolygon`` geometries. Each polygon's
    rings become (xs, ys) float64 arrays; ring[0] is the exterior, the rest
    are holes.
    """
    with Path(geojson_path).open(encoding="utf-8") as fh:
        gj = json.load(fh)

    polygons: list[_Polygon] = []
    for feat in gj["features"]:
        name = _ISO_TO_NAME.get(feat.get("id", ""), feat.get("id", ""))
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for rings in polys:
            prepared = [
                (
                    np.asarray([p[0] for p in ring], dtype=np.float64),
                    np.asarray([p[1] for p in ring], dtype=np.float64),
                )
                for ring in rings
            ]
            xs, ys = prepared[0]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            polygons.append((name, prepared, bbox))
    logger.debug("Prepared %d polygons from %s", len(polygons), geojson_path)
    return polygons


def _point_in_ring(lon: float, lat: float, xs: np.ndarray, ys: np.ndarray) -> bool:
    """Vectorized ray-casting test: is (lon, lat) inside this ring?"""
    x1, y1 = xs, ys
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (y1 > lat) != (y2 > lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_lat = (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
    return int(np.count_nonzero(crosses & (lon < x_at_lat))) % 2 == 1


def _point_in_polygon(lon: float, lat: float, rings: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Inside the exterior ring and outside every hole."""
    if not _point_in_ring(lon, lat, *rings[0]):
        return False
    return not any(_point_in_ring(lon, lat, xs, ys) for xs, ys in rings[1:])


def country_of(lon: float, lat: float, geojson_path: Path | str = DEFAULT_BORDERS_PATH) -> str:
    """Return 'Thailand' | 'Myanmar' | 'Laos' | 'other' for a coordinate.

    Uses real country outlines (point-in-polygon with a bounding-box
    pre-filter). Falls back to 'other' when the point is outside all three
    countries.
    """
    for name, rings, (lon_min, lat_min, lon_max, lat_max) in _prepared_polygons(str(geojson_path)):
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        if _point_in_polygon(lon, lat, rings):
            return name
    return "other"
