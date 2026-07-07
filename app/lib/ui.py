# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Shared UI building blocks for the dashboard (header, AQI legend, badges, footer).

Keeps look-and-feel and the honesty captions (hindcast note, NSC disclaimer) in one place.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib import aqi
from src.viz.timeseries import forecast_vs_actual

# Short notice shown in the footer on every page; full text lives in the About page below.
DISCLAIMER = (
    'ผลงานภายใต้ NSC 2026 / สวทช. — เผยแพร่ตาม "ต้นฉบับ" '
    "ไม่รับประกันความถูกต้องหรือความเสียหายใด ๆ (ฉบับเต็มดูหน้า ‘เกี่ยวกับโครงการ’)"
)

# Full NSC software disclaimer (booklet p.44), verbatim Thai + English, shown on the About page.
DISCLAIMER_FULL_TH = (
    "ซอฟต์แวร์นี้เป็นผลงานที่พัฒนาขึ้นโดย นายรณชัย ขาวสะอาด จาก มหาวิทยาลัยบูรพา ภายใต้การดูแลของ "
    "ดร.วัชรพงศ์ อยู่ขวัญ ภายใต้โครงการ “ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วย"
    "โครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลาแบบอธิบายได้ สำหรับภาคเหนือของประเทศไทย” ซึ่งสนับสนุนโดย"
    "สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ โดยมีวัตถุประสงค์เพื่อส่งเสริมให้นักเรียนและนักศึกษา"
    "ได้เรียนรู้และฝึกทักษะในการพัฒนาซอฟต์แวร์ ลิขสิทธิ์ของซอฟต์แวร์นี้จึงเป็นของผู้พัฒนา ซึ่งผู้พัฒนา"
    "ได้อนุญาตให้สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติเผยแพร่ซอฟต์แวร์นี้ตาม “ต้นฉบับ” โดยไม่มี"
    "การแก้ไขดัดแปลงใด ๆ ทั้งสิ้น ให้แก่บุคคลทั่วไปได้ใช้เพื่อประโยชน์ส่วนบุคคลหรือประโยชน์ทางการศึกษา"
    "ที่ไม่มีวัตถุประสงค์ในเชิงพาณิชย์ โดยไม่คิดค่าตอบแทนการใช้ซอฟต์แวร์ ดังนั้น สำนักงานพัฒนา"
    "วิทยาศาสตร์และเทคโนโลยีแห่งชาติจึงไม่มีหน้าที่ในการดูแล บำรุงรักษา จัดการอบรมการใช้งาน หรือพัฒนา"
    "ประสิทธิภาพซอฟต์แวร์ รวมทั้งไม่รับรองความถูกต้องหรือประสิทธิภาพการทำงานของซอฟต์แวร์ ตลอดจนไม่"
    "รับประกันความเสียหายต่าง ๆ อันเกิดจากการใช้ซอฟต์แวร์นี้ทั้งสิ้น"
)
DISCLAIMER_FULL_EN = (
    "License Agreement. This software is a work developed by Mr. Ronnachai Khaosa-ard from Burapha "
    "University under the provision of Dr. Watcharapong Yookwan under the project “Explainable "
    "Spatio-Temporal Graph Neural Network for PM2.5 Forecasting and Source Attribution in Northern "
    "Thailand”, which has been supported by the National Science and Technology Development Agency "
    "(NSTDA), in order to encourage pupils and students to learn and practice their skills in "
    "developing software. Therefore, the intellectual property of this software shall belong to the "
    "developer and the developer gives NSTDA a permission to distribute this software as an “as is” "
    "and non-modified software for a temporary and non-exclusive use without remuneration to anyone "
    "for his or her own purpose or academic purpose, which are not commercial purposes. In this "
    "connection, NSTDA shall not be responsible to the user for taking care, maintaining, training, "
    "or developing the efficiency of this software. Moreover, NSTDA shall not be liable for any "
    "error, software efficiency and damages in connection with or arising out of the use of the "
    "software."
)


def page_title(title: str, subtitle: str = "") -> None:
    """Standard page heading + optional subtitle."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def aqi_legend() -> None:
    """Render the Thai PCD AQI colour legend as inline chips."""
    chips = "  ".join(
        f"<span style='background:{c};color:#111;padding:2px 9px;border-radius:11px;"
        f"font-size:0.8rem;white-space:nowrap'>{e} {label}</span>"
        for e, label, c in aqi.legend()
    )
    st.markdown(
        f"<div style='line-height:2.1'><b>เกณฑ์ดัชนีคุณภาพอากาศ (PM2.5):</b><br>{chips}</div>",
        unsafe_allow_html=True,
    )


def aqi_badge_md(pm25: float | None) -> str:
    """Compact markdown badge: emoji + Thai category for a PM2.5 value."""
    return f"{aqi.emoji(pm25)} **{aqi.category(pm25)}**"


def hindcast_note(anchor_iso: str) -> None:
    """Honest banner: this is a hindcast on ERA5 data, not real-time, with the origin time."""
    ts = pd.Timestamp(anchor_iso)
    st.info(
        f"จุดเริ่มพยากรณ์ (forecast origin): **{ts:%d %b %Y, %H:%M} UTC** — "
        "ใช้ข้อมูลสภาพอากาศ ERA5 (ข้อมูลย้อนหลัง) จึงเป็นการพยากรณ์ย้อนหลังบนข้อมูลจริง "
        "เพื่อสาธิตความสามารถ ไม่ใช่ระบบเรียลไทม์",
        icon="🕒",
    )


def forecast_overlay_figure(
    history: pd.DataFrame,
    pred_station: list[float],
    persistence_value: float,
    anchor_ts: pd.Timestamp,
    horizons: list[int],
    station_name: str,
) -> go.Figure:
    """Observed history (with AQI bands) plus the multi-horizon MTGNN forecast.

    Reuses ``forecast_vs_actual`` for the observed line + AQI bands, then overlays the
    forecast points at origin+h and a flat persistence reference.
    """
    fig = forecast_vs_actual(
        timestamps=history["timestamp"].tolist(),
        actual=history["pm25"].tolist(),
        forecasts={},
        station_name=station_name,
        show_aqi_bands=True,
    )
    last_obs = next(
        (v for v in reversed(history["pm25"].tolist()) if pd.notna(v)), persistence_value
    )
    fx = [anchor_ts] + [anchor_ts + pd.Timedelta(hours=h) for h in horizons]
    fy = [last_obs, *pred_station]
    fig.add_trace(
        go.Scatter(
            x=fx,
            y=fy,
            mode="lines+markers",
            name="พยากรณ์ MTGNN",
            line=dict(color="#d62728", dash="dash", width=2),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[anchor_ts, anchor_ts + pd.Timedelta(hours=max(horizons))],
            y=[persistence_value, persistence_value],
            mode="lines",
            name="Persistence (ค่าคงที่)",
            line=dict(color="#888", dash="dot", width=1.5),
        )
    )
    return fig


def footer() -> None:
    """Shared footer with branding + the short NSC disclaimer pointer."""
    st.divider()
    st.caption(
        "ระบบพยากรณ์ PM2.5 และวิเคราะห์แหล่งกำเนิด (Explainable STGNN) · "
        f"NSC 2026 หมวด 14 · {DISCLAIMER}"
    )
