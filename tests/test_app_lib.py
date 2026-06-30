# [TODO: NSC Disclaimer - see booklet page 44]
"""Unit tests for dashboard lib helpers (AQI banding, province parsing)."""

from __future__ import annotations

import math

from app.lib import aqi
from app.lib.data_access import _province_from_name


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
