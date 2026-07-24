# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
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
from app.views import (
    about,
    attribution,
    cases,
    forecast,
    overview,
    performance,
    transboundary,
    whatif,
)

st.set_page_config(page_title="PM2.5 ภาคเหนือ — STGNN", page_icon="🌫️", layout="wide")


def _scenario_controls() -> None:
    """Scenario widgets (split + anchor) shared by the deep-dive pages via session_state.

    Rendered inside the collapsed "deep dive" expander so the main landing page stays
    clean; the widgets still run each rerun, so ``session_state`` is always populated.
    """
    st.markdown("**เลือกสถานการณ์** (สำหรับหน้าพยากรณ์/แหล่งกำเนิด)")
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


def main() -> None:
    """Show the single landing page; tuck the deep-dive pages into a sidebar dropdown."""
    landing = st.Page(
        cases.render, title="ฝุ่นวันนั้นมาจากไหน", icon="🔎", url_path="cases", default=True
    )
    # Second headline page, not a deep dive: the click-a-fire counterfactual is the
    # demo's "show me it works" moment and must be one click from the landing page.
    lab = st.Page(whatif.render, title="ปิดสวิตช์ไฟ (ทดลองเอง)", icon="🔥", url_path="whatif")
    deep = [
        st.Page(overview.render, title="ภาพรวม", icon="🏠", url_path="overview"),
        st.Page(forecast.render, title="พยากรณ์รายสถานี", icon="📈", url_path="forecast"),
        st.Page(attribution.render, title="แหล่งกำเนิด (XAI)", icon="🧭", url_path="attribution"),
        st.Page(transboundary.render, title="หมอกควันข้ามแดน", icon="🌏", url_path="transboundary"),
        st.Page(
            performance.render, title="ผลทดสอบ & ความซื่อสัตย์", icon="📊", url_path="performance"
        ),
        st.Page(about.render, title="เกี่ยวกับ", icon="ℹ️", url_path="about"),
    ]
    # position="hidden" suppresses the auto page list; we build a compact sidebar instead.
    pg = st.navigation([landing, lab, *deep], position="hidden")
    # Pages are callable-based, so st.page_link needs the Page object itself; stash the
    # ones other views link to rather than hard-coding a url_path string in each view.
    st.session_state["_pages"] = {"landing": landing, "lab": lab}

    with st.sidebar:
        st.title("🌫️ PM2.5 ภาคเหนือ")
        st.caption("Explainable STGNN · NSC 2026 หมวด 14")
        st.divider()
        st.page_link(landing)
        st.page_link(lab)
        with st.expander("🔬 หน้าวิเคราะห์เชิงลึก"):
            for p in deep:
                st.page_link(p)
            st.divider()
            _scenario_controls()

    pg.run()
    ui.footer()


main()
