# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Transboundary haze: the held-out Mae Hong Son cross-border attribution result."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib import data_access as da
from app.lib import geo, ui

_COUNTRY_COLOR = {
    "Thailand": "#2196F3",
    "Myanmar": "#FF5722",
    "Laos": "#4CAF50",
    "other": "#9E9E9E",
}
_COUNTRY_TH = {"Thailand": "ไทย", "Myanmar": "เมียนมา", "Laos": "ลาว", "other": "อื่นๆ"}


def _wind_arrow_trace(wind: pd.DataFrame) -> go.Scattermapbox | None:
    """Fixed-length ERA5 wind-direction arrows (one per station); points to where wind blows."""
    if wind is None or len(wind) == 0:
        return None
    lats: list[float | None] = []
    lons: list[float | None] = []
    shaft = 0.10  # degrees — fixed length so direction (not speed) is the message
    for _, r in wind.iterrows():
        u, v = float(r["u10"]), float(r["v10"])
        sp = math.hypot(u, v)
        if sp < 1e-6:
            continue
        lat0, lon0 = float(r["lat"]), float(r["lon"])
        coslat = math.cos(math.radians(lat0)) or 1.0
        dlon = (u / sp) * shaft / coslat  # u = eastward, v = northward
        dlat = (v / sp) * shaft
        lat1, lon1 = lat0 + dlat, lon0 + dlon
        bx, by = -dlon * 0.4, -dlat * 0.4  # arrowhead barbs point back from the tip
        a = math.radians(28)
        b1x, b1y = bx * math.cos(a) - by * math.sin(a), bx * math.sin(a) + by * math.cos(a)
        b2x, b2y = bx * math.cos(-a) - by * math.sin(-a), bx * math.sin(-a) + by * math.cos(-a)
        lats += [lat0, lat1, None, lat1, lat1 + b1y, None, lat1, lat1 + b2y, None]
        lons += [lon0, lon1, None, lon1, lon1 + b1x, None, lon1, lon1 + b2x, None]
    if not lats:
        return None
    return go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=2, color="rgba(25,80,190,0.8)"),
        name="ทิศลม (ERA5)",
        hoverinfo="skip",
    )


def _fire_map(event_date: str, station_lat: float, station_lon: float) -> go.Figure:
    """FIRMS hotspots by country + national borders + ERA5 wind arrows + the station."""
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
                    marker=go.scattermapbox.Marker(size=9, color=colour, opacity=0.75),
                    name=f"ไฟ {_COUNTRY_TH[country]}",
                    text=[f"{_COUNTRY_TH[country]} · FRP {f:.0f}" for f in sub["frp"]],
                    hoverinfo="text",
                )
            )
    wind_trace = _wind_arrow_trace(da.wind_for_date(event_date))
    if wind_trace is not None:
        fig.add_trace(wind_trace)
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
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=station_lat, lon=station_lon),
            zoom=6,
            layers=[
                dict(
                    sourcetype="geojson",
                    source=geo.border_geojson(),
                    type="line",
                    color="rgba(80,80,80,0.55)",
                    line=dict(width=1.5),
                )
            ],
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.75)",
        ),
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
        "✅ กรณีเด่น — 2025-03-18: ไฟที่เชื่อมถึงสถานีเป็นไฟต่างชาติ ~72% (เมียนมา+ลาว) "
        "และโมเดลชี้ว่ามาจากต่างชาติ ~63% — สัดส่วนที่โมเดลระบุไล่ตามสัดส่วนไฟจริงทั้ง 5 เหตุการณ์"
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
        name="ไฟต่างชาติจริง (FRP)",
        marker_color="#FF9800",
    )
    fig.add_bar(
        x=dates,
        y=[e["foreign_attribution"] * 100 for e in events],
        name="ที่โมเดลชี้ว่าต่างชาติ",
        marker_color="#FF5722",
    )
    fig.update_layout(
        barmode="group",
        title=dict(
            text="สัดส่วนไฟต่างชาติจริง เทียบกับที่โมเดลระบุ (รายเหตุการณ์)",
            x=0,
            xanchor="left",
        ),
        yaxis_title="ร้อยละ (%)",
        xaxis_title="วันที่เหตุการณ์",
        height=430,
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=44, b=112),
    )
    st.plotly_chart(fig, width="stretch")

    default_i = dates.index("2025-03-18") if "2025-03-18" in dates else 0
    sel = st.selectbox("ดูแผนที่จุดไฟของเหตุการณ์", dates, index=default_i)
    meta = da.load_stations_meta()
    st_row = meta[meta["station_id"] == data.get("station_id", 225648)]
    s_lat = float(st_row.iloc[0]["lat"]) if len(st_row) else 19.3
    s_lon = float(st_row.iloc[0]["lon"]) if len(st_row) else 97.97
    st.plotly_chart(_fire_map(sel, s_lat, s_lon), width="stretch")
    st.caption(
        "🗺️ เส้นเทา = พรมแดนประเทศจริง (point-in-polygon) · จุดสี = ไฟแยกตามประเทศ · "
        "ลูกศรน้ำเงิน = ทิศลม ERA5 (ชี้ไปทางที่ลมพัดพาควันไป) — ดูว่าลมพัดจากไฟฝั่งใดเข้าหาสถานี. "
        "ทั้งแผนที่และตัวเลข % ใช้ป้ายประเทศจากขอบเขตจริงชุดเดียวกัน"
    )

    cm = da.load_output_json("attribution_march2024.json")
    th_pct = round(cm.get("country_attribution", {}).get("Thailand", 1.0) * 100) if cm else 100
    st.info(
        "ℹ️ **ความซื่อสัตย์:** 'ขนาด' ของสัดส่วนขึ้นกับโมเดล — โมเดลรายงาน (split เข้มงวด) ไล่ตาม"
        "สัดส่วนไฟจริงได้ดี ขณะที่โมเดล pitch ระบุต่างชาติเพียง ~0-6% ในเหตุการณ์ชุดเดียวกัน "
        "จึงต้องรายงานพร้อม caveat เสมอ — ตรงข้ามกับเชียงใหม่กลางเมือง (มี.ค. 2024) ที่ผลชี้ไฟ"
        f"**ในไทยเป็นหลัก ~{th_pct}%** (FRP ไทยสูงกว่าไฟต่างชาติรวม ~59 เท่า) ซึ่งเป็นเมืองภายในแผ่นดิน"
    )
