# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Transboundary haze: the held-out Mae Hong Son cross-border attribution result."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib import data_access as da
from app.lib import ui

_COUNTRY_COLOR = {"Thailand": "#2196F3", "Myanmar": "#FF5722", "Laos": "#4CAF50"}


def _fire_map(event_date: str, station_lat: float, station_lon: float) -> go.Figure:
    """Map of FIRMS hotspots on an event date, coloured by country, plus the station."""
    hs = da.hotspots_for_date(event_date)
    fig = go.Figure()
    for country, colour in _COUNTRY_COLOR.items():
        sub = hs[hs["country"] == country]
        if len(sub) > 0:
            fig.add_trace(
                go.Scattermapbox(
                    lat=sub["latitude"].tolist(),
                    lon=sub["longitude"].tolist(),
                    mode="markers",
                    marker=go.scattermapbox.Marker(size=8, color=colour, opacity=0.7),
                    name=f"ไฟ {country}",
                    text=[f"{country} · FRP {f:.0f}" for f in sub["frp"]],
                    hoverinfo="text",
                )
            )
    fig.add_trace(
        go.Scattermapbox(
            lat=[station_lat],
            lon=[station_lon],
            mode="markers",
            marker=go.scattermapbox.Marker(size=16, color="black"),
            name="สถานีแม่ฮ่องสอน",
            hoverinfo="name",
        )
    )
    fig.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lat=station_lat, lon=station_lon), zoom=6),
        margin=dict(l=0, r=0, t=10, b=0),
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    return fig


def render() -> None:
    """Render the transboundary attribution page."""
    ui.page_title(
        "หมอกควันข้ามแดน",
        "การระบุแหล่งข้ามพรมแดนบนข้อมูล held-out ปี 2025 ที่สถานีชายแดนแม่ฮ่องสอน",
    )
    data = da.load_output_json("transboundary_attr_test_split2.json")
    events = data.get("events", []) if data else []
    if not events:
        st.warning("ไม่พบผลลัพธ์ transboundary_attr_test_split2.json")
        return

    st.markdown(
        f"**สถานี:** {data.get('station', '-')} · **ชุดข้อมูล:** held-out test 2025 · "
        f"**ช่วงพยากรณ์:** {data.get('horizon_h', '-')} ชม. "
        "(โมเดลไม่เคยเห็นข้อมูลปี 2025 ตอนเทรน)"
    )
    st.success(
        "✅ กรณีที่สอบเทียบได้ดี — 2025-02-16: ไฟที่เชื่อมถึงสถานีเป็นของเมียนมา ~37% "
        "และโมเดลชี้ว่ามาจากเมียนมา ~37% (สอดคล้องกับสัดส่วนไฟจริง)"
    )

    rows = [
        {
            "วันที่": e["date"],
            "PM2.5 สูงสุด (µg/m³)": round(e["peak_pm25_ug_m3"], 1),
            "% ไฟต่างชาติเชื่อมถึง": round(e["connected_foreign_fraction"] * 100, 1),
            "% โมเดลชี้ต่างชาติ": round(e["foreign_attribution"] * 100, 1),
            "เมียนมา (attr %)": round(e["country_attribution"].get("Myanmar", 0.0) * 100, 1),
        }
        for e in events
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    dates = [e["date"] for e in events]
    fig = go.Figure()
    fig.add_bar(
        x=dates,
        y=[e["connected_foreign_fraction"] * 100 for e in events],
        name="% ไฟต่างชาติ (FRP) ที่เชื่อมถึงสถานี",
        marker_color="#FF9800",
    )
    fig.add_bar(
        x=dates,
        y=[e["foreign_attribution"] * 100 for e in events],
        name="% ที่โมเดลชี้ว่ามาจากต่างชาติ",
        marker_color="#FF5722",
    )
    fig.update_layout(
        barmode="group",
        title="สัดส่วนไฟต่างชาติจริง เทียบกับที่โมเดลระบุ (รายเหตุการณ์)",
        yaxis_title="ร้อยละ (%)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.06),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    default_i = dates.index("2025-02-16") if "2025-02-16" in dates else 0
    sel = st.selectbox("ดูแผนที่จุดไฟของเหตุการณ์", dates, index=default_i)
    meta = da.load_stations_meta()
    st_row = meta[meta["station_id"] == data.get("station_id", 225648)]
    s_lat = float(st_row.iloc[0]["lat"]) if len(st_row) else 19.3
    s_lon = float(st_row.iloc[0]["lon"]) if len(st_row) else 97.97
    st.plotly_chart(_fire_map(sel, s_lat, s_lon), width="stretch")

    cm = da.load_output_json("attribution_march2024.json")
    th_pct = round(cm.get("country_attribution", {}).get("Thailand", 1.0) * 100) if cm else 100
    st.info(
        "ℹ️ **ความซื่อสัตย์:** 'ขนาด' ของสัดส่วนขึ้นกับโมเดลและเหตุการณ์ (เช่น 2025-03-25 โมเดล pitch "
        "ขยายเป็น 100% ขณะที่โมเดลสำหรับรายงานให้ ~4%) แต่ **ความสามารถระบุข้ามแดนเกิดในทั้งสองโมเดล** "
        "ตามความใกล้ของไฟ (proximity-tracking) — ตรงข้ามกับเชียงใหม่กลางเมือง (มี.ค. 2024) ที่ผลชี้ไฟ"
        f"**ในไทยเป็นหลัก ~{th_pct}%** (FRP ไทยสูงกว่าเมียนมา ~128 เท่า) ซึ่งเป็นเมืองภายในแผ่นดิน"
    )
