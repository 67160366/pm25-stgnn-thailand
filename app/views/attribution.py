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

    _render_counterfactual(split, anchor_iso, meta, sel_name, sid, h_label, horizon_idx)


def _render_counterfactual(
    split: str,
    anchor_iso: str,
    meta: pd.DataFrame,
    sel_name: str,
    sid: int,
    h_label: str,
    horizon_idx: int,
) -> None:
    """Interactive 'what-if we put out this country's fires' policy simulator."""
    st.divider()
    st.subheader("จำลองนโยบาย: ถ้าดับไฟของประเทศต้นทาง ฝุ่นจะลดเท่าไร")
    st.caption(
        "เลือกประเทศต้นทางเพื่อ 'ดับไฟ' (occlude) ของประเทศนั้นในกราฟ แล้วเทียบค่าพยากรณ์ "
        "ก่อน–หลัง เป็น µg/m³ รายสถานี — คันโยกนโยบาย ‘มาตรการในประเทศ vs การทูตข้ามพรมแดน’"
    )

    countries = inf.hotspot_countries_at(split, anchor_iso)
    if not countries:
        st.warning(
            "วันนี้ไม่มีจุดไฟ (hotspot) ในกราฟ จึงจำลองการดับไฟไม่ได้ — ลองเลือกวันในช่วงฤดูหมอกควัน"
        )
        return

    country = st.selectbox("เลือกประเทศต้นทางที่จะ ‘ดับไฟ’", countries, key="cf_country")
    cf = inf.counterfactual_at(split, anchor_iso, country)

    if cf["n_occluded"] == 0:
        st.info(f"ไม่พบจุดไฟของ ‘{country}’ ในตัวอย่างนี้ — ไม่มีการเปลี่ยนแปลง")
        return

    sids: list[int] = cf["station_ids"]
    delta = np.asarray(cf["delta_ug"], dtype=float)[:, horizon_idx]
    full = np.asarray(cf["pred_full_ug"], dtype=float)[:, horizon_idx]
    occ = np.asarray(cf["pred_occluded_ug"], dtype=float)[:, horizon_idx]

    st.caption(
        f"ดับไฟของ **{country}** ({cf['n_occluded']} จุด) · ช่วงพยากรณ์ **{h_label}** · "
        "ค่าบวก = ฝุ่นลดลงเมื่อดับไฟประเทศนี้"
    )

    i = sids.index(sid)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{sel_name} · ก่อนดับไฟ", f"{full[i]:.0f} µg/m³")
    m2.metric("หลังดับไฟ", f"{occ[i]:.0f} µg/m³", delta=f"{-delta[i]:.1f}", delta_color="inverse")
    m3.metric("ฝุ่นที่ลดได้", f"{delta[i]:.1f} µg/m³")

    name_by_sid = dict(zip(meta["station_id"], meta["name"], strict=False))
    df = pd.DataFrame(
        {
            "สถานี": [name_by_sid.get(s, str(s)) for s in sids],
            "ก่อนดับไฟ (µg/m³)": np.round(full, 1),
            "หลังดับไฟ (µg/m³)": np.round(occ, 1),
            "ลดลง (µg/m³)": np.round(delta, 1),
        }
    ).sort_values("ลดลง (µg/m³)", ascending=False, ignore_index=True)

    st.dataframe(df, width="stretch", hide_index=True)
    st.bar_chart(df.set_index("สถานี")["ลดลง (µg/m³)"])

    st.info(
        "ℹ️ นี่คือการจำลองแบบ **what-if ภายใต้โมเดล** (occlusion) ไม่ใช่ข้อสรุปเชิงสาเหตุ (causal) ที่ "
        "ผ่านการพิสูจน์ — ขนาดของผลขึ้นกับความไวที่โมเดลเรียนรู้ และ**ไม่ควรใช้กล่าวโทษเชิงการทูต** "
        "ใช้เป็นเครื่องมือสำรวจเชิงนโยบายเท่านั้น"
    )
