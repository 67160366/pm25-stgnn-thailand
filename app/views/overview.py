# [TODO: NSC Disclaimer - see booklet page 44]
"""Overview (landing) page: current AQI snapshot, map, 48h outlook, health guidance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.lib import aqi, ui
from app.lib import data_access as da
from app.lib import inference as inf
from src.viz.maps import station_map


def render() -> None:
    """Render the landing overview for the current scenario (split + anchor)."""
    split = st.session_state["split"]
    anchor_iso = st.session_state["anchor_iso"]

    ui.page_title(
        "ภาพรวมสถานการณ์ฝุ่น PM2.5",
        "9 จังหวัดภาคเหนือ · พยากรณ์ล่วงหน้า + ระบุแหล่งกำเนิดที่อธิบายได้",
    )
    ui.hindcast_note(anchor_iso)

    meta = da.load_stations_meta()
    fc = inf.forecast_at(split, anchor_iso)
    sids: list[int] = fc["station_ids"]
    observed = np.asarray(fc["observed"], dtype=float)
    pred = np.asarray(fc["pred_ug"], dtype=float)
    obs_series = pd.Series(observed, index=sids)
    name_by_id = meta.set_index("station_id")["name"]

    has_obs = bool(np.isfinite(observed).any())
    worst_i = int(np.nanargmax(observed)) if has_obs else 0
    worst_val = float(observed[worst_i]) if has_obs else float("nan")
    worst_name = str(name_by_id.get(sids[worst_i], sids[worst_i]))
    n_over = int(np.nansum(observed > 50.0))
    regional = float(np.nanmean(observed)) if has_obs else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("สถานีที่ค่าฝุ่นสูงสุดตอนนี้", f"{worst_val:.0f} µg/m³")
    c1.markdown(f"{ui.aqi_badge_md(worst_val)} · {worst_name}")
    c2.metric("สถานีที่เกินเกณฑ์มีผลต่อสุขภาพ (>50)", f"{n_over} / {len(sids)}")
    c3.metric("ค่าเฉลี่ยทั้งภูมิภาค", f"{regional:.0f} µg/m³")
    c3.markdown(ui.aqi_badge_md(regional))

    st.warning(
        f"**คำแนะนำสุขภาพ (จุดที่แย่ที่สุด — {aqi.category(worst_val)}):** "
        f"{aqi.health_advice(worst_val)}"
    )

    day = pd.Timestamp(anchor_iso).strftime("%Y-%m-%d")
    hotspots = da.hotspots_for_date(day)
    fig = station_map(
        meta,
        pm25_values=obs_series,
        title=f"PM2.5 รายสถานี — {pd.Timestamp(anchor_iso):%d %b %Y %H:%M} UTC (จุดส้ม = จุดความร้อนไฟ)",
        hotspots=hotspots,
    )
    st.plotly_chart(fig, width="stretch")

    pred48 = float(np.nanmean(pred[:, 3]))
    delta = pred48 - regional if np.isfinite(regional) else 0.0
    trend = "▲ มีแนวโน้มแย่ลง" if delta > 2 else ("▼ มีแนวโน้มดีขึ้น" if delta < -2 else "▬ ทรงตัว")
    st.subheader("แนวโน้ม 48 ชั่วโมงข้างหน้า")
    st.markdown(
        f"ค่าเฉลี่ยภูมิภาคที่พยากรณ์ไว้ใน 48 ชม. ≈ **{pred48:.0f} µg/m³** "
        f"({trend} เทียบกับปัจจุบัน {regional:.0f}) — {ui.aqi_badge_md(pred48)}"
    )
    st.caption("ดูพยากรณ์รายสถานีและช่วงเวลาอื่นได้ที่หน้า ‘พยากรณ์รายสถานี’")

    st.divider()
    ui.aqi_legend()
