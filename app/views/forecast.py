# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Forecast Explorer: per-station observed history + MTGNN multi-horizon forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.lib import data_access as da
from app.lib import inference as inf
from app.lib import ui


def render() -> None:
    """Render the per-station forecast explorer."""
    split = st.session_state["split"]
    anchor_iso = st.session_state["anchor_iso"]

    ui.page_title(
        "พยากรณ์รายสถานี", "เลือกสถานีเพื่อดูค่าจริงย้อนหลังและพยากรณ์ 6 / 12 / 24 / 48 ชม."
    )
    ui.hindcast_note(anchor_iso)

    meta = da.load_stations_meta()
    names = meta["name"].tolist()
    sel_name = st.selectbox("เลือกสถานี", names)
    sid = int(meta[meta["name"] == sel_name].iloc[0]["station_id"])

    fc = inf.forecast_at(split, anchor_iso)
    sids: list[int] = fc["station_ids"]
    i = sids.index(sid)
    pred = np.asarray(fc["pred_ug"], dtype=float)[i]
    pers = float(fc["persistence"][i])
    horizons: list[int] = fc["horizons"]
    anchor_ts = pd.Timestamp(fc["anchor_iso"])

    hist = inf.station_history(split, sid, anchor_iso, hours_back=168)
    fig = ui.forecast_overlay_figure(hist, pred.tolist(), pers, anchor_ts, horizons, sel_name)
    st.plotly_chart(fig, width="stretch")

    st.subheader("ค่าพยากรณ์รายช่วงเวลา")
    cols = st.columns(len(horizons))
    for col, h, val in zip(cols, horizons, pred, strict=True):
        col.metric(f"+{h} ชม.", f"{val:.0f} µg/m³")
        col.markdown(ui.aqi_badge_md(float(val)))

    st.caption(
        "หมายเหตุ (ตามจริง): ในช่วงสั้น 6–12 ชม. ค่า baseline persistence มักแม่นกว่าโมเดล "
        "เพราะ PM2.5 มี autocorrelation สูง — จุดเด่นเชิงพยากรณ์ของระบบอยู่ที่ 48 ชม. "
        "สำหรับการเตือนภัยล่วงหน้า เราจึงเสนอใช้แบบ hybrid (persistence ช่วงสั้น + MTGNN ช่วงยาว)"
    )
