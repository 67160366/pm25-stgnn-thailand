# [TODO: NSC Disclaimer - see booklet page 44]
"""Entry point for the PM2.5 STGNN dashboard (Streamlit native multipage).

Run: ``uv run streamlit run app/streamlit_app.py``

Public-facing dashboard for 9 Northern Thai provinces. A sidebar picks the scenario
(period + forecast-origin date); pages read it from ``st.session_state``. Live forecasts
are a hindcast on historical ERA5 data using the pitch checkpoint (not real-time).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import inference as inf
from app.lib import ui
from app.views import about, attribution, forecast, overview, performance, transboundary

st.set_page_config(page_title="PM2.5 ภาคเหนือ — STGNN", page_icon="🌫️", layout="wide")


def _scenario_controls() -> None:
    """Sidebar controls that set the shared scenario (split + anchor) in session_state."""
    with st.sidebar:
        st.title("🌫️ PM2.5 ภาคเหนือ")
        st.caption("Explainable STGNN · NSC 2026 หมวด 14")
        st.divider()
        st.markdown("**เลือกสถานการณ์**")
        periods = list(inf.PERIOD_TO_SPLIT.keys())
        period = st.selectbox("ช่วงข้อมูล", periods, index=periods.index(inf.DEFAULT_PERIOD))
        split = inf.PERIOD_TO_SPLIT[period]
        lo, hi = inf.date_bounds(split)
        default_day = pd.Timestamp(inf.default_anchor_iso(split)).date()
        day = st.date_input(
            "วันที่ (จุดเริ่มพยากรณ์)",
            value=default_day,
            min_value=lo,
            max_value=hi,
            key=f"day_{split}",
        )
        st.session_state["split"] = split
        st.session_state["anchor_iso"] = inf.resolve_anchor(split, day)
        st.caption("ค่าเริ่มต้น = วันที่ค่าฝุ่นเฉลี่ยสูงสุดในช่วงที่เลือก")


def main() -> None:
    """Build navigation, render the selected page, and append the shared footer."""
    pages = [
        st.Page(overview.render, title="ภาพรวม", icon="🏠", url_path="overview", default=True),
        st.Page(forecast.render, title="พยากรณ์รายสถานี", icon="📈", url_path="forecast"),
        st.Page(attribution.render, title="แหล่งกำเนิด (XAI)", icon="🧭", url_path="attribution"),
        st.Page(transboundary.render, title="หมอกควันข้ามแดน", icon="🌏", url_path="transboundary"),
        st.Page(
            performance.render, title="ผลทดสอบ & ความซื่อสัตย์", icon="📊", url_path="performance"
        ),
        st.Page(about.render, title="เกี่ยวกับ", icon="ℹ️", url_path="about"),
    ]
    pg = st.navigation(pages)
    _scenario_controls()
    pg.run()
    ui.footer()


main()
