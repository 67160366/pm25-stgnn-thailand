# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Transboundary haze: the held-out Mae Hong Son cross-border attribution result."""

from __future__ import annotations

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
    for _, r in wind.iterrows():
        # fixed shaft length so direction (not speed) is the message
        a_lat, a_lon = geo.arrow_lines(
            float(r["lat"]), float(r["lon"]), float(r["u10"]), float(r["v10"]), shaft_deg=0.10
        )
        lats += a_lat
        lons += a_lon
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


_CKPT_LABEL_TH = {"demo": "demo (โมเดล pitch)", "report": "report (split2 เข้มงวด)"}


def _matrix_section(data: dict) -> None:
    """Item-2 section: station x event attribution matrix, checkpoint-selectable.

    Reads ``outputs/transboundary_matrix.json`` (Package A artifact, read-only).
    Degrades to ``st.info`` when the file is absent so the page never crashes.
    """
    st.divider()
    st.markdown("#### เมทริกซ์แหล่งกำเนิดข้ามแดน — หลายสถานีชายแดน × หลาย checkpoint")
    if not data or not data.get("results"):
        st.info(
            "ℹ️ ยังไม่มีผลลัพธ์ transboundary_matrix.json (ยังไม่ได้รัน "
            "scripts/20_transboundary_matrix.py) — ข้ามส่วนนี้ไปก่อน"
        )
        return

    st.caption(
        f"สถานีชายแดน {len(data.get('stations', []))} แห่ง (เกณฑ์: ระยะทางถึงพรมแดนพม่า/ลาวใกล้สุด "
        f"≤ {data.get('max_dist_km', 50):.0f} กม.) × เหตุการณ์ไฟเชื่อมถึงสถานีสูงสุด "
        f"{data.get('top_k', 5)} เหตุการณ์ต่อสถานี บนชุด held-out {data.get('split', 'test')} "
        f"(ช่วงพยากรณ์ {data.get('horizon_h', '-')} ชม.)"
    )

    checkpoints = data.get("checkpoints", [])
    labels = [c["label"] for c in checkpoints]
    chosen = (
        st.radio(
            "เลือก checkpoint",
            labels,
            format_func=lambda label: _CKPT_LABEL_TH.get(label, label),
            horizontal=True,
            key="tb_matrix_checkpoint",
        )
        if labels
        else None
    )

    results = [r for r in data.get("results", []) if r.get("checkpoint_label") == chosen]
    if not results:
        st.warning("ไม่พบผลลัพธ์สำหรับ checkpoint ที่เลือก")
        return

    station_names = [r["station_name"] for r in results]
    all_dates = sorted({e["date"] for r in results for e in r.get("events", [])})
    z: list[list[float | None]] = []
    hover: list[list[str]] = []
    for r in results:
        by_date = {e["date"]: e for e in r.get("events", [])}
        row_z: list[float | None] = []
        row_hover: list[str] = []
        for d in all_dates:
            e = by_date.get(d)
            if e is None:
                row_z.append(None)
                row_hover.append("")
            else:
                row_z.append(round(e["foreign_attribution"] * 100, 1))
                row_hover.append(
                    f"{r['station_name']}<br>{d}<br>"
                    f"โมเดลชี้ต่างชาติ: {e['foreign_attribution'] * 100:.1f}%<br>"
                    f"ไฟต่างชาติเชื่อมถึงจริง: {e['connected_foreign_fraction'] * 100:.1f}%"
                )
        z.append(row_z)
        hover.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=all_dates,
            y=station_names,
            colorscale="Oranges",
            zmin=0,
            zmax=100,
            text=hover,
            hoverinfo="text",
            colorbar=dict(title="% ต่างชาติ"),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"% ที่โมเดลชี้ว่าเป็นต่างชาติ — checkpoint: {_CKPT_LABEL_TH.get(chosen, chosen)}",
            x=0,
            xanchor="left",
        ),
        xaxis_title="วันที่เหตุการณ์",
        height=380,
        margin=dict(l=10, r=10, t=44, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "ระยะทางสถานีถึงพรมแดนต่างชาติที่ใกล้ที่สุด (เกณฑ์การคัดเลือก ไม่ใช่การเลือกเฉพาะที่สวย):"
    )
    station_rows = [
        {
            "สถานี": s["name"],
            "ระยะถึงพรมแดนใกล้สุด (กม.)": s["dist_km_nearest_foreign"],
            "ประเทศใกล้สุด": _COUNTRY_TH.get(s["nearest_country"], s["nearest_country"]),
        }
        for s in data.get("stations", [])
    ]
    st.dataframe(pd.DataFrame(station_rows), width="stretch", hide_index=True)

    for caveat in data.get("caveats", []):
        st.caption(f"⚠️ {caveat}")


def _uncertainty_section(data: dict) -> None:
    """Item-4 section: per-event mean +/- seed spread for the flagship station.

    Reads ``outputs/transboundary_uncertainty.json`` (Package B artifact, read-only).
    Degrades to ``st.info`` when the file is absent so the page never crashes.
    """
    st.divider()
    st.markdown("#### ความไม่แน่นอนของขนาด attribution ข้าม seed")
    if not data or not data.get("events"):
        st.info(
            "ℹ️ ยังไม่มีผลลัพธ์ transboundary_uncertainty.json (ยังไม่ได้รัน "
            "scripts/21_transboundary_uncertainty.py) — ข้ามส่วนนี้ไปก่อน"
        )
        return

    seeds = data.get("seeds", [])
    seed_labels = [s["label"] for s in seeds]
    st.caption(
        f"สถานี: {data.get('station', '-')} · เทรนซ้ำ variant เดียวกัน ({len(seeds)} seeds) แล้ววัด "
        "foreign_attribution ที่เหตุการณ์เดียวกันทุก seed (การเลือกเหตุการณ์ไม่ขึ้นกับโมเดล — ขับด้วยข้อมูล "
        "FIRMS/PM2.5 ล้วน ๆ) เพื่อรายงานค่าเฉลี่ยพร้อมช่วงที่วัดจริง แทนคำเตือนลอย ๆ ว่า 'ขึ้นกับโมเดล'"
    )

    events = data.get("events", [])
    dates = [e["date"] for e in events]
    means = [e["foreign_attribution_mean"] * 100 for e in events]
    mins = [e["foreign_attribution_min"] * 100 for e in events]
    maxs = [e["foreign_attribution_max"] * 100 for e in events]
    err_plus = [mx - m for mx, m in zip(maxs, means, strict=True)]
    err_minus = [m - mn for m, mn in zip(means, mins, strict=True)]

    fig = go.Figure()
    fig.add_bar(
        x=dates,
        y=means,
        name="% ต่างชาติเฉลี่ย (ข้าม seeds)",
        marker_color="#FF5722",
        error_y=dict(
            type="data",
            symmetric=False,
            array=err_plus,
            arrayminus=err_minus,
            visible=True,
            color="#333",
        ),
    )
    fig.update_layout(
        title=dict(
            text="ค่าเฉลี่ย % ต่างชาติต่อเหตุการณ์ พร้อมช่วง min–max ข้าม seed (error bar = ช่วงจริง ไม่ใช่ std)",
            x=0,
            xanchor="left",
        ),
        yaxis_title="ร้อยละ (%)",
        xaxis_title="วันที่เหตุการณ์",
        height=400,
        margin=dict(l=50, r=20, t=56, b=60),
    )
    st.plotly_chart(fig, width="stretch")

    rows = []
    for e in events:
        per_seed = e.get("per_seed", {})
        row = {
            "วันที่": e["date"],
            "% ไฟต่างชาติเชื่อมถึง": round(e["connected_foreign_fraction"] * 100, 1),
            "% ต่างชาติเฉลี่ย": round(e["foreign_attribution_mean"] * 100, 1),
            "ช่วง min–max (%)": (
                f"{e['foreign_attribution_min'] * 100:.1f}–{e['foreign_attribution_max'] * 100:.1f}"
            ),
        }
        for label in seed_labels:
            seed_val = per_seed.get(label, {}).get("foreign_attribution", 0.0)
            row[f"seed {label} (%)"] = round(seed_val * 100, 1)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    summary = data.get("summary", {})
    st.caption(
        "สรุป: ค่าเฉลี่ยข้าม seed ของ foreign_attribution ทุกเหตุการณ์ = "
        f"{summary.get('mean_foreign_attribution', 0.0) * 100:.1f}% · "
        "ช่วง (max−min) เฉลี่ยต่อเหตุการณ์ = "
        f"{summary.get('mean_event_spread_minmax', 0.0) * 100:.1f} จุดเปอร์เซ็นต์"
    )

    for caveat in data.get("caveats", []):
        st.caption(f"⚠️ {caveat}")


_TRAJ_COLORS: list[str] = ["#E91E63", "#3F51B5", "#009688", "#FF9800", "#795548"]


def _trajectory_map(events: list[dict], station_lat: float, station_lon: float) -> go.Figure:
    """One coloured backward-trajectory line per event, plus the launch station + borders."""
    fig = go.Figure()
    for i, e in enumerate(events):
        points = e.get("trajectory", {}).get("points", [])
        if not points:
            continue
        colour = _TRAJ_COLORS[i % len(_TRAJ_COLORS)]
        fig.add_trace(
            go.Scattermapbox(
                lat=[p["lat"] for p in points],
                lon=[p["lon"] for p in points],
                mode="lines+markers",
                line=dict(width=2, color=colour),
                marker=dict(size=4, color=colour),
                name=e.get("date", f"event {i}"),
                text=[
                    f"{e.get('date', '-')}<br>ย้อนหลัง {p['hour_offset']} ชม.<br>"
                    f"{p['time_utc']}<br>ประเทศ: {_COUNTRY_TH.get(p['country'], p['country'])}"
                    for p in points
                ],
                hoverinfo="text",
            )
        )
    fig.add_trace(
        go.Scattermapbox(
            lat=[station_lat],
            lon=[station_lon],
            mode="markers",
            marker=go.scattermapbox.Marker(size=14, color="black"),
            name="สถานีแม่ฮ่องสอน (จุดปล่อยวิถี)",
            hoverinfo="name",
        )
    )
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=station_lat, lon=station_lon),
            zoom=5.5,
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
        height=520,
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


def _backtrajectory_section(data: dict) -> None:
    """Item-11 section: ERA5 kinematic back-trajectory as an independent witness.

    Reads ``outputs/backtrajectory_test2025.json`` (Package A artifact, read-only).
    Degrades to ``st.info`` when the file is absent so the page never crashes. Presents the
    binary direction-agreement result honestly (2/5 under the pre-registered metric as
    measured) with the corridor-FRP explanation for the 3 disagreements -- never spins the
    number and never re-derives or discusses tuning the underlying thresholds.
    """
    st.divider()
    st.markdown("#### วิถีลมย้อนหลัง (ERA5) — พยานอิสระของทิศทางการระบุแหล่งข้ามแดน")
    events = data.get("events", []) if data else []
    if not events:
        st.info(
            "ℹ️ ยังไม่มีผลลัพธ์ backtrajectory_test2025.json (ยังไม่ได้รัน "
            "scripts/22_backtrajectory.py) — ข้ามส่วนนี้ไปก่อน"
        )
        return

    hours_back = data.get("trajectory_params", {}).get("hours_back", 48)
    st.caption(
        f"สถานี: {data.get('station', '-')} · วิถีย้อนหลัง {hours_back} ชม. จากลมผิวพื้น ERA5 "
        "(u10/v10) เท่านั้น เริ่มที่ชั่วโมง PM2.5 สูงสุดของแต่ละเหตุการณ์ (anchor เดียวกับที่โมเดลใช้ระบุแหล่ง) "
        "— เป็นหลักฐานสนับสนุนเชิงบ่งชี้ ไม่ใช่ HYSPLIT เต็มรูปแบบ"
    )

    meta = da.load_stations_meta()
    st_row = meta[meta["station_id"] == data.get("station_id", 225648)]
    s_lat = float(st_row.iloc[0]["lat"]) if len(st_row) else 19.30455
    s_lon = float(st_row.iloc[0]["lon"]) if len(st_row) else 97.97165

    st.plotly_chart(_trajectory_map(events, s_lat, s_lon), width="stretch")
    st.caption(
        "🗺️ เส้นสี = วิถีลมย้อนหลังแต่ละเหตุการณ์ (จุดดำ = สถานีแม่ฮ่องสอน จุดปล่อยวิถี) · "
        "เส้นเทา = พรมแดนประเทศจริง — ดูว่าวิถีผ่านประเทศใดก่อนย้อนไปถึงสถานี"
    )

    rows = [
        {
            "วันที่": e["date"],
            "PM2.5 สูงสุด (µg/m³)": round(e.get("peak_pm25_ug_m3", 0.0), 1),
            "โมเดลชี้ต่างชาติ (%)": (
                round(e["model_foreign_attribution"] * 100, 1)
                if e.get("model_foreign_attribution") is not None
                else None
            ),
            "% ชม.วิถีอยู่ต่างชาติ": round(e.get("foreign_hours_fraction", 0.0) * 100, 1),
            "FRP ทางเดินวิถี เมียนมา": round(
                e.get("corridor_frp_by_country", {}).get("Myanmar", 0.0), 1
            ),
            "FRP ทางเดินวิถี ไทย": round(
                e.get("corridor_frp_by_country", {}).get("Thailand", 0.0), 1
            ),
            "FRP ทางเดินวิถี ลาว": round(e.get("corridor_frp_by_country", {}).get("Laos", 0.0), 1),
            "เห็นตรงกันกับโมเดล": (
                "✅"
                if e.get("agrees_with_model") is True
                else ("❌" if e.get("agrees_with_model") is False else "—")
            ),
        }
        for e in events
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    summary = data.get("agreement_summary", {})
    n_agree = summary.get("n_agree", 0)
    n_with_model = summary.get("n_events_with_model", len(events))
    st.markdown(
        f"**สรุปแบบตรงไปตรงมา:** พยานอิสระ (วิถีลมย้อนหลังจาก ERA5) เห็นตรงกับทิศทางที่โมเดลระบุ "
        f"**{n_agree}/{n_with_model}** เหตุการณ์ (ตัวชี้วัดไบนารี ต่างชาติ/ในประเทศ ที่กำหนดไว้ก่อนดูผล) — "
        "ทั้ง 2 เหตุการณ์ที่โมเดลชี้ต่างชาติจริง (2025-03-18, 2025-03-13) มีวิถีลมยืนยัน คือผ่านเมียนมาและอยู่ใกล้"
        "ไฟต่างชาติจริงตามทางเดินวิถี ส่วนอีก 3 เหตุการณ์ วิถีลมยังคงผ่านเมียนมาเช่นเดียวกัน (ปกติของสถานีชายแดน"
        "แม่ฮ่องสอน) แต่ไม่พบไฟต่างชาติที่มีนัยสำคัญตามทางเดินวิถีในวันเหล่านั้น — สอดคล้องกับที่โมเดลรายงาน "
        "attribution ต่างชาติใกล้ศูนย์ ไม่ใช่ข้อขัดแย้งที่อธิบายไม่ได้"
    )

    for caveat in data.get("caveats", []):
        st.caption(f"⚠️ {caveat}")


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

    _matrix_section(da.load_transboundary_matrix())
    _uncertainty_section(da.load_transboundary_uncertainty())
    _backtrajectory_section(da.load_backtrajectory())
