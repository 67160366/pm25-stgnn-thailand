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

__all__ = ["arrow_lines", "border_geojson", "country_of", "fit_view"]


@lru_cache(maxsize=1)
def border_geojson() -> dict:
    """Raw GeoJSON FeatureCollection of the three country outlines (for map layers)."""
    with DEFAULT_BORDERS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def fit_view(
    points: list[tuple[float, float]],
    height: int,
    width: int = 1000,
    pad_deg: float = 0.7,
    zoom_range: tuple[float, float] = (5.0, 7.2),
) -> tuple[dict, float]:
    """Map centre + zoom that frames ``points`` inside a ``height`` x ``width`` canvas.

    Web Mercator shows ``360 / 2**z`` degrees of longitude per 512 px, so the zoom
    that fits a span is ``log2(360 * px / 512 / span)``; the tighter of the two
    axes wins. Callers pass the event's own points (station, fires, air path) and
    get a view that never depends on a hand-tuned constant per event.

    Args:
        points: ``(lat, lon)`` pairs that must all be visible.
        height: Canvas height in pixels.
        width: Assumed container width in pixels (Streamlit does not report it).
        pad_deg: Degrees of breathing room added to each span.
        zoom_range: Clamp applied to the computed zoom.

    Returns:
        Tuple ``(center, zoom)`` ready for ``fig.update_layout(mapbox=...)``.
    """
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    lat_c, lon_c = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    lat_span = max(max(lats) - min(lats), 0.5) + pad_deg
    lon_span = max(max(lons) - min(lons), 0.5) + pad_deg
    z_lat = math.log2(360 * (height / 512) / lat_span)
    z_lon = math.log2(360 * (width / 512) / lon_span)
    lo, hi = zoom_range
    return dict(lat=lat_c, lon=lon_c), min(max(min(z_lat, z_lon), lo), hi)


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
