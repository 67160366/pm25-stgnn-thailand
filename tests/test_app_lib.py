# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Unit tests for dashboard lib helpers (AQI banding, province parsing)."""

from __future__ import annotations

import math

from app.lib import aqi
from app.lib.data_access import _province_from_name
from app.lib.inference import conformal_halfwidths


def test_aqi_category_bands():
    assert aqi.category(10) == "ดี"
    assert aqi.category(30) == "ปานกลาง"
    assert aqi.category(45) == "เริ่มมีผลต่อกลุ่มเสี่ยง"
    assert aqi.category(70) == "มีผลต่อสุขภาพ"
    assert aqi.category(120) == "มีผลต่อสุขภาพมาก"
    assert aqi.category(200) == "อันตราย"


def test_aqi_boundaries_go_to_upper_band():
    # Bands are [lo, hi); an exact boundary belongs to the higher band.
    assert aqi.category(25) == "ปานกลาง"
    assert aqi.category(150) == "อันตราย"


def test_aqi_missing_values():
    assert aqi.category(None) == "ไม่มีข้อมูล"
    assert aqi.category(math.nan) == "ไม่มีข้อมูล"
    assert aqi.color(None) == "#9e9e9e"
    assert aqi.emoji(None) == "⚪"


def test_aqi_color_and_emoji():
    assert aqi.color(120) == "#8f3f97"
    assert aqi.emoji(10) == "🟢"
    assert aqi.emoji(200) == "🟤"


def test_aqi_legend_shape():
    legend = aqi.legend()
    assert len(legend) == 6
    assert all(len(row) == 3 for row in legend)


def test_province_parsing():
    office = "Natural Resources and Environment Office, Mae Hongson"
    assert _province_from_name("City Hall, Chiangmai") == "Chiangmai"
    assert _province_from_name(office) == "Mae Hongson"
    assert _province_from_name("NoComma") == "NoComma"
    assert _province_from_name("") == ""


_CONF = {
    "meta": {"mode": "per_station"},
    "pooled": {"6h": 6.0, "12h": 8.0, "24h": 11.0, "48h": 15.0},
    "per_station": {"101": {"6h": 4.0, "12h": 5.0, "24h": 9.0, "48h": 13.0}},
}


def test_conformal_halfwidths_per_station_preferred():
    hw = conformal_halfwidths(_CONF, 101, [6, 12, 24, 48])
    assert hw == [4.0, 5.0, 9.0, 13.0]


def test_conformal_halfwidths_falls_back_to_pooled():
    # Station 999 absent from per_station -> pooled quantiles used.
    hw = conformal_halfwidths(_CONF, 999, [6, 12, 24, 48])
    assert hw == [6.0, 8.0, 11.0, 15.0]


def test_conformal_halfwidths_pooled_mode_ignores_per_station():
    conf = {**_CONF, "meta": {"mode": "pooled"}}
    hw = conformal_halfwidths(conf, 101, [6, 12])
    assert hw == [6.0, 8.0]


def test_conformal_halfwidths_missing_returns_none():
    assert conformal_halfwidths({}, 101, [6, 12]) is None
    # Horizon 72h has no quantile anywhere -> None so the caller omits the band.
    assert conformal_halfwidths(_CONF, 101, [6, 72]) is None
