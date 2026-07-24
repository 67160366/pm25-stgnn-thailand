# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Country geocoding + border shapes for the dashboard maps.

The point-in-polygon geocoder now lives in ``src.data.geocode`` — the single
source of truth also used by the data pipeline's hotspot country labels
(hotspots.parquet), so the map and the attribution numbers can no longer
disagree. This module re-exports it and adds the raw GeoJSON accessor used
to draw border outlines on maps.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

from src.data.geocode import DEFAULT_BORDERS_PATH, country_of

__all__ = ["arrow_lines", "border_geojson", "country_of"]


@lru_cache(maxsize=1)
def border_geojson() -> dict:
    """Raw GeoJSON FeatureCollection of the three country outlines (for map layers)."""
    with DEFAULT_BORDERS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def arrow_lines(
    lat: float,
    lon: float,
    u: float,
    v: float,
    shaft_deg: float = 0.10,
    head_frac: float = 0.4,
    head_deg: float = 28.0,
) -> tuple[list[float | None], list[float | None]]:
    """Line coordinates for one arrow at ``(lat, lon)`` pointing along ``(u, v)``.

    ``u``/``v`` are eastward/northward components (the ERA5 convention: the arrow
    points to *where the air is going*). Length is fixed at ``shaft_deg`` so the
    message is direction, not speed. Longitude steps are divided by cos(lat) so
    the arrow keeps its bearing on a Mercator map.

    Returns ``(lats, lons)`` with ``None`` separators, ready to concatenate into a
    single ``mode="lines"`` trace; ``([], [])`` when the vector has no direction.
    """
    speed = math.hypot(u, v)
    if speed < 1e-9:
        return [], []
    coslat = math.cos(math.radians(lat)) or 1.0
    dlon = (u / speed) * shaft_deg / coslat
    dlat = (v / speed) * shaft_deg
    lat1, lon1 = lat + dlat, lon + dlon
    bx, by = -dlon * head_frac, -dlat * head_frac  # barbs point back from the tip
    a = math.radians(head_deg)
    b1x, b1y = bx * math.cos(a) - by * math.sin(a), bx * math.sin(a) + by * math.cos(a)
    b2x, b2y = bx * math.cos(-a) - by * math.sin(-a), bx * math.sin(-a) + by * math.cos(-a)
    lats: list[float | None] = [lat, lat1, None, lat1, lat1 + b1y, None, lat1, lat1 + b2y, None]
    lons: list[float | None] = [lon, lon1, None, lon1, lon1 + b1x, None, lon1, lon1 + b2x, None]
    return lats, lons
