# [TODO: NSC Disclaimer - see booklet page 44]
"""Thai PCD PM2.5 AQI helpers: category, colour, emoji, and health advice (Thai).

Single source of truth for AQI banding in the dashboard so the KPI cards, the legend,
and chart colours stay consistent with ``src/viz`` (which encodes the same Thai PCD
breakpoints). Breakpoints in ug/m3: 25 / 37.5 / 50 / 90 / 150.
"""

from __future__ import annotations

import math

# (lo, hi, name_th, hex_colour, emoji, advice_th) — Thai PCD PM2.5 bands.
_BANDS: list[tuple[float, float, str, str, str, str]] = [
    (0.0, 25.0, "ดี", "#00e400", "🟢", "คุณภาพอากาศดี ทำกิจกรรมกลางแจ้งได้ตามปกติ"),
    (25.0, 37.5, "ปานกลาง", "#ffd400", "🟡", "คุณภาพอากาศพอใช้ ผู้ที่ไวต่อมลพิษควรสังเกตอาการ"),
    (
        37.5,
        50.0,
        "เริ่มมีผลต่อกลุ่มเสี่ยง",
        "#ff7e00",
        "🟠",
        "กลุ่มเสี่ยง (เด็ก ผู้สูงอายุ ผู้มีโรคระบบทางเดินหายใจ/หัวใจ) ควรลดกิจกรรมกลางแจ้ง",
    ),
    (
        50.0,
        90.0,
        "มีผลต่อสุขภาพ",
        "#ff0000",
        "🔴",
        "ประชาชนทั่วไปควรลดกิจกรรมกลางแจ้ง กลุ่มเสี่ยงควรงดและสวมหน้ากากกันฝุ่น PM2.5",
    ),
    (
        90.0,
        150.0,
        "มีผลต่อสุขภาพมาก",
        "#8f3f97",
        "🟣",
        "ทุกคนควรงดกิจกรรมกลางแจ้ง สวมหน้ากาก N95 และปิดประตูหน้าต่างให้มิดชิด",
    ),
    (
        150.0,
        float("inf"),
        "อันตราย",
        "#7e0023",
        "🟤",
        "อันตรายต่อสุขภาพ ควรอยู่ในอาคาร ใช้เครื่องฟอกอากาศ และติดตามประกาศของทางราชการ",
    ),
]

# (name, colour, emoji, advice) shown when a station has no measurement.
_UNKNOWN: tuple[str, str, str, str] = (
    "ไม่มีข้อมูล",
    "#9e9e9e",
    "⚪",
    "ไม่มีข้อมูลการตรวจวัดในช่วงเวลานี้",
)


def _is_missing(pm25: float | None) -> bool:
    return pm25 is None or (isinstance(pm25, float) and math.isnan(pm25))


def _band(pm25: float | None) -> tuple[float, float, str, str, str, str] | None:
    """Return the matching AQI band tuple, or None if the value is missing."""
    if _is_missing(pm25):
        return None
    for band in _BANDS:
        if band[0] <= pm25 < band[1]:
            return band
    return _BANDS[-1]  # >= 150 ug/m3


def category(pm25: float | None) -> str:
    """Thai AQI category name for a PM2.5 value (ug/m3)."""
    band = _band(pm25)
    return _UNKNOWN[0] if band is None else band[2]


def color(pm25: float | None) -> str:
    """Hex colour for a PM2.5 value, matching the src/viz AQI scale."""
    band = _band(pm25)
    return _UNKNOWN[1] if band is None else band[3]


def emoji(pm25: float | None) -> str:
    """Coloured-circle emoji for a PM2.5 value (for compact KPI labels)."""
    band = _band(pm25)
    return _UNKNOWN[2] if band is None else band[4]


def health_advice(pm25: float | None) -> str:
    """Plain-Thai health guidance for a PM2.5 value."""
    band = _band(pm25)
    return _UNKNOWN[3] if band is None else band[5]


def legend() -> list[tuple[str, str, str]]:
    """Return ``(emoji, 'name (range ug/m3)', hex_colour)`` for each AQI band."""
    rows: list[tuple[str, str, str]] = []
    for lo, hi, name, hex_colour, emoji_, _advice in _BANDS:
        rng = f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+"
        rows.append((emoji_, f"{name} ({rng} µg/m³)", hex_colour))
    return rows
