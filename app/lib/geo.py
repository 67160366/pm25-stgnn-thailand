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
from functools import lru_cache

from src.data.geocode import DEFAULT_BORDERS_PATH, country_of

__all__ = ["border_geojson", "country_of"]


@lru_cache(maxsize=1)
def border_geojson() -> dict:
    """Raw GeoJSON FeatureCollection of the three country outlines (for map layers)."""
    with DEFAULT_BORDERS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)
