# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Source Attribution (XAI): per-country occlusion + per-feature IG for one station/date."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.lib import data_access as da
from app.lib import inference as inf
from app.lib import ui
from src.explain.attribution import station_source_report
from src.viz.maps import attribution_bar_map, feature_importance_bar

_H_LABELS = ["6h", "12h", "24h", "48h"]


def render() -> None:
    """Render the source-attribution (XAI) page."""
    split = st.session_state["split"]
    anchor_iso = st.session_state["anchor_iso"]

    ui.page_title(
        "แหล่งกำเนิดฝุ่น (Explainable AI)",
        "ระบบประเมินว่าฝุ่นที่สถานีนี้มาจากประเทศใด และปัจจัยใดมีผลต่อการพยากรณ์",
    )
    ui.hindcast_note(anchor_iso)

    meta = da.load_stations_meta()
    col_a, col_b = st.columns(2)
    sel_name = col_a.selectbox("เลือกสถานี", meta["name"].tolist(), key="attr_station")
    h_label = col_b.selectbox("ช่วงเวลาพยากรณ์", _H_LABELS, index=2, key="attr_horizon")
    horizon_idx = _H_LABELS.index(h_label)
    sid = int(meta[meta["name"] == sel_name].iloc[0]["station_id"])

    ds = inf.get_dataset(split)
    model = inf.load_model()
    pos = inf.pos_for_timestamp(ds, pd.Timestamp(anchor_iso))
    sample = ds[pos]
    sidx = int(np.where(ds._station_ids == sid)[0][0])

    with st.spinner("กำลังคำนวณ Integrated Gradients + Occlusion Attribution…"):
        report = station_source_report(
            model, sample, sidx, horizon_idx, station_name=sel_name, n_ig_steps=30, device="cpu"
        )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("สัดส่วนแหล่งกำเนิดรายประเทศ")
        st.caption("จากเทคนิค Occlusion: ปิดไฟแต่ละประเทศแล้ววัดว่าค่าพยากรณ์ลดลงเท่าใด")
        st.plotly_chart(
            attribution_bar_map(meta, report["country_attribution"], station_id=sid),
            width="stretch",
        )
    with c2:
        st.subheader("ปัจจัยที่มีผลต่อการพยากรณ์")
        st.caption("จากเทคนิค Integrated Gradients: ความสำคัญเฉลี่ยของแต่ละปัจจัย")
        st.plotly_chart(feature_importance_bar(report["ig_feature_importance"]), width="stretch")

    st.info(
        "ℹ️ ตัวเลขเหล่านี้เป็น **การประมาณจากโมเดล** (occlusion/IG) ตรวจสอบเทียบกับข้อมูลไฟจริง FIRMS "
        "ไม่ใช่การวัดโดยตรงในอากาศ จึงควรสื่อสารพร้อมความไม่แน่นอน และ**ไม่ควรใช้กล่าวโทษเชิงการทูต** "
        "— เป็นเครื่องมือสนับสนุนการตัดสินใจเชิงนโยบาย"
    )
    with st.expander("ข้อมูลดิบ (raw attribution report)"):
        st.json(report)
