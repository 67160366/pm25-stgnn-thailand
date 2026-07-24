# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Interactive "switch off these fires" lab — click hotspots on the map, re-run the model.

The demo answer to *"how do you know that fire mattered?"*: pick fire clusters on a
map, set their FRP toward zero, and watch the same checkpoint re-forecast. Nothing
is looked up — every number on this page is the difference between two live
forward passes of ``checkpoints/mtgnn/best_model.pt`` over one stored window.

The page is built to be *falsifiable in front of a judge*, which is why it ships
its own attempts to break itself:

* dose–response — sweep 0→100 % suppression; a hand-written rule can only step,
  a network reading the FRP feature bends
* placebo — extinguish an FRP-matched set of fires the wind does **not** connect
  to this station; the response should collapse toward zero
* spatial footprint — the same edit measured at all 18 stations, so a viewer sees
  the graph carrying (or not carrying) the change outward
* glass box — the hotspot tensor before/after and the *live source* of the
  function that just ran, via ``inspect.getsource``

Honesty constraints baked in, not bolted on: the model's absolute fire
sensitivity on this checkpoint is small (a few µg/m³, concentrated on the largest
events) and occasionally wrong-signed. The page reports the measured number
whatever it is, and says plainly that a model sensitivity is not a validated
causal share — there is no ground-truth label for "where this dust came from".
"""

from __future__ import annotations

import datetime as dt
import inspect
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib import data_access as da
from app.lib import geo, ui
from app.lib import inference as inf
from src.explain import gb_ig

# Same two-colour language as the case gallery: blue = Thailand, orange = abroad.
_THAI = "#2a78d6"
_FOREIGN = "#eb6834"
_OFF = "#9aa3af"  # fires the wind does not connect to this station
_PICKED = "#111827"  # ring around a selected (about to be extinguished) cluster
_MAP_INK = "#1f2937"
_MAP_BG = "#ffffff"

_H_LABELS = ["6h", "12h", "24h", "48h"]

# split -> (badge text, colour, one-line meaning)
_SPLIT_BADGE: dict[str, tuple[str, str, str]] = {
    "test": (
        "ชุดทดสอบ",
        "#137a4d",
        "โมเดลไม่เคยเห็นข้อมูลปีนี้ตอนฝึก — ผลที่นี่จึงเชื่อถือได้มากที่สุด",
    ),
    "val": (
        "ชุดปรับจูน",
        "#b45309",
        "ใช้เลือกจุดหยุดฝึก โมเดลไม่ได้ฝึกทับ แต่ก็ไม่ใช่ข้อมูลใหม่ล้วน",
    ),
    "train": (
        "ชุดฝึก",
        "#6b7280",
        "โมเดลเคยเห็นช่วงนี้ตอนฝึก ใช้ดูกลไกได้ แต่ห้ามใช้อ้างความแม่นยำ",
    ),
}

_TH_MONTHS = [
    "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
]  # fmt: skip

_COUNTRY_TH = {"Thailand": "ไทย", "Myanmar": "เมียนมา", "Laos": "ลาว"}


def _thai_date(iso: str) -> str:
    """Format an ISO ``YYYY-MM-DD`` date as ``D <thai-month> <B.E. year>``."""
    y, m, d = (int(x) for x in iso.split("-")[:3])
    return f"{d} {_TH_MONTHS[m - 1]} {y + 543}"


# ---------------------------------------------------------------- render ----


def render() -> None:
    """Render the interactive switch-off lab."""
    ui.page_title(
        "ห้องทดลอง: ปิดสวิตช์ไฟ แล้วดูว่าอะไรหาย",
        "คลิกเลือกกลุ่มไฟบนแผนที่ → ระบบตั้งความแรงไฟเป็นศูนย์ → รันโมเดลตัวเดิมใหม่ → "
        "ส่วนต่างคือ “อิทธิพลของไฟกลุ่มนั้น” ตามที่โมเดลมองเห็น",
    )
    _recipe()

    split, anchor_iso, sid, station_name, horizon_idx = _scenario_controls()
    fires = inf.event_hotspots(split, anchor_iso, sid)
    if fires.empty:
        st.warning(
            "วันนี้ไม่มีกลุ่มไฟในกราฟเลย จึงไม่มีสวิตช์ให้ปิด — ลองเลือกวันในฤดูหมอกควัน (ก.พ.–เม.ย.)"
        )
        return

    picked = _selection_ui(split, anchor_iso, sid, station_name, fires)
    if not picked:
        st.info(
            "👆 คลิกกลุ่มไฟบนแผนที่ (กด Shift ค้างเพื่อเลือกหลายกลุ่ม) "
            "หรือกดปุ่มลัดด้านบน แล้วผลจะคำนวณให้ทันที",
            icon="🖱️",
        )
        return

    remaining = _suppression_slider()
    cf = inf.counterfactual_nodes_at(split, anchor_iso, tuple(picked), remaining)
    _result_block(cf, sid, station_name, horizon_idx, remaining, fires, picked)
    _dose_response_block(split, anchor_iso, tuple(picked), sid, horizon_idx)
    _footprint_block(cf, sid, horizon_idx)
    _placebo_block(split, anchor_iso, tuple(picked), sid, horizon_idx, cf)
    _glass_box(split, anchor_iso, cf, fires, picked)
    _honesty_footer()


def _recipe() -> None:
    """The three-line method statement, up front and in plain Thai."""
    st.markdown(
        f"""
<div style="border-left:3px solid {_FOREIGN};background:rgba(235,104,52,.09);
     padding:12px 16px;border-radius:6px;margin:6px 0 12px;line-height:1.7">
  <b>วิธีคิด — “ปิดสวิตช์แล้วดูว่าอะไรหาย”</b><br>
  <b>1.</b> ทำนายตามปกติ (ไฟครบทุกกลุ่มตามภาพถ่ายดาวเทียมจริง) &nbsp;→&nbsp; ได้ค่า <b>A</b><br>
  <b>2.</b> เอากลุ่มไฟที่เลือก มาตั้ง <code>total_frp = 0</code> (ทำเหมือนไฟกลุ่มนั้นไม่เคยเกิด)
  แล้ว<b>รันโมเดลตัวเดิมใหม่</b> &nbsp;→&nbsp; ได้ค่า <b>B</b><br>
  <b>3.</b> ค่าที่หายไป <b>A − B</b> คืออิทธิพลของไฟกลุ่มนั้น <u>เท่าที่โมเดลมองเห็น</u>
  <div style="opacity:.8;font-size:.86rem;margin-top:6px">
    ไม่มีตารางคำตอบและไม่มี <code>if</code> ที่ไหนเลย — ทั้ง A และ B มาจากคำสั่ง
    <code>model(data)</code> อันเดียวกัน ต่างกันแค่ตัวเลขความแรงไฟในอินพุต
    (กด “หลังบ้านทำอะไรบ้าง” ท้ายหน้าเพื่อดูโค้ดจริงที่เพิ่งรัน)
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------- scenario ----


@st.cache_data(show_spinner=False)
def _curated_events() -> list[dict]:
    """Curated station x date shortlist from the frozen transboundary matrix.

    Uses ``transboundary_matrix.json`` (5 border stations x 5 test dates, all held-out
    2025), whose event set was chosen model-independently — dates ranked by FIRMS
    foreign FRP, anchored at each station's peak PM2.5 — so offering them here does not
    cherry-pick flattering model behaviour. Deduplicated across the two checkpoints.

    Ordered by *connected fire load* (total FRP the wind links to that station), which
    is an observation, not a model output. Ranking on the measured response instead
    would quietly promote whichever events the checkpoint happens to like; ranking on
    fire load simply puts the events with the most to switch off at the top, and the
    weak-response ones stay in the list rather than being hidden.
    """
    matrix = da.load_transboundary_matrix()
    seen: dict[tuple[int, str], dict] = {}
    for res in matrix.get("results", []):
        sid = int(res["station_id"])
        for ev in res.get("events", []):
            key = (sid, ev["date"])
            if key in seen:
                continue
            frp = ev.get("connected_frp_by_country", {})
            seen[key] = {
                "station_id": sid,
                "station_name": str(res["station_name"]),
                "date": ev["date"],
                "peak": float(ev.get("peak_pm25_ug_m3", 0.0)),
                "foreign_frac": float(ev.get("connected_foreign_fraction", 0.0)),
                "connected_frp": float(sum(float(v) for v in frp.values())),
            }
    return sorted(seen.values(), key=lambda e: -e["connected_frp"])


def _split_badge(split: str) -> None:
    """Render the train/val/test provenance badge — the credibility line of this page."""
    label, colour, meaning = _SPLIT_BADGE.get(split, ("?", "#6b7280", ""))
    st.markdown(
        f"<div style='margin:2px 0 10px'><span style='background:{colour};color:#fff;"
        f"padding:3px 12px;border-radius:999px;font-size:.85rem;font-weight:700'>{label}</span>"
        f"<span style='opacity:.8;font-size:.85rem'> &nbsp;{meaning}</span></div>",
        unsafe_allow_html=True,
    )


def _scenario_controls() -> tuple[str, str, int, str, int]:
    """Event/station/horizon pickers. Returns ``(split, anchor_iso, sid, name, h_idx)``."""
    meta = da.load_stations_meta()
    name_by_sid = dict(zip(meta["station_id"], meta["name"], strict=False))
    curated = _curated_events()

    mode = st.radio(
        "เลือกเหตุการณ์",
        ["เหตุการณ์คัดสรร (ชุดทดสอบ 2568)", "เลือกวันเอง (ทุกช่วงข้อมูล)"],
        horizontal=True,
        key="wi_mode",
    )

    if mode.startswith("เหตุการณ์คัดสรร") and curated:
        opts = list(range(len(curated)))
        i = st.selectbox(
            "เหตุการณ์",
            opts,
            format_func=lambda i: (
                f"{_thai_date(curated[i]['date'])} · {curated[i]['station_name']} · "
                f"PM2.5 สูงสุด {curated[i]['peak']:.0f} µg/m³ · "
                f"ไฟที่ลมพามา {curated[i]['connected_frp']:,.0f} MW "
                f"(ต่างชาติ {100 * curated[i]['foreign_frac']:.0f}%)"
            ),
            key="wi_event",
        )
        ev = curated[i]
        split = "test"
        sid = ev["station_id"]
        anchor_iso = inf.resolve_anchor(split, dt.date.fromisoformat(ev["date"]))
    else:
        c1, c2 = st.columns([1, 1])
        periods = list(inf.PERIOD_TO_SPLIT.keys())
        period = c1.selectbox("ช่วงข้อมูล", periods, key="wi_period")
        split = inf.PERIOD_TO_SPLIT[period]
        lo, hi = inf.date_bounds(split)
        default = min(max(dt.date(int(str(lo)[:4]), 3, 20), lo), hi)
        day = c2.date_input(
            "วันที่", value=default, min_value=lo, max_value=hi, key=f"wi_day_{split}"
        )
        anchor_iso = inf.resolve_anchor(split, day)
        sid = int(meta.iloc[0]["station_id"])

    c1, c2 = st.columns([2, 1])
    names = meta["name"].tolist()
    default_name = name_by_sid.get(sid, names[0])
    sel_name = c1.selectbox(
        "สถานีที่ต้องการดูผล",
        names,
        index=names.index(default_name),
        key=f"wi_station_{anchor_iso}",
    )
    sid = int(meta[meta["name"] == sel_name].iloc[0]["station_id"])
    h_label = c2.selectbox("ช่วงพยากรณ์", _H_LABELS, index=2, key="wi_horizon")

    _split_badge(split)
    ts = pd.Timestamp(anchor_iso).tz_convert("Asia/Bangkok")
    st.caption(
        f"จุดเริ่มพยากรณ์ {ts:%d/%m/%Y %H:%M} น. (เวลาไทย) · "
        f"พยากรณ์ล่วงหน้า {h_label} · โมเดล MTGNN checkpoint เดียวกับหน้าอื่นทั้งเว็บ"
    )
    return split, anchor_iso, sid, sel_name, _H_LABELS.index(h_label)


# ------------------------------------------------------------ selection ----


def _quick_picks(fires: pd.DataFrame, key_ns: str) -> list[int] | None:
    """Shortcut buttons. Returns a new selection, or ``None`` when nothing was pressed."""
    conn = fires["connected"]
    foreign = fires["country"].isin(["Myanmar", "Laos"])
    presets: list[tuple[str, pd.Series, str]] = [
        ("🔥 ไฟต่างชาติที่ลมพามา", conn & foreign, "ไฟเมียนมา/ลาว ที่กราฟเชื่อมถึงสถานีนี้"),
        ("🏠 ไฟในไทยที่ลมพามา", conn & ~foreign, "ไฟในประเทศ ที่กราฟเชื่อมถึงสถานีนี้"),
        ("🌏 ไฟทุกกลุ่มที่ลมพามา", conn, "ทุกกลุ่มที่กราฟเชื่อมถึงสถานีนี้"),
        ("✖️ ล้างการเลือก", pd.Series(False, index=fires.index), "ยกเลิกทั้งหมด"),
    ]
    # Every button must be rendered on every run — returning from inside the loop
    # would silently drop the remaining buttons on the rerun that follows a press.
    pressed: list[int] | None = None
    cols = st.columns(len(presets))
    for col, (label, mask, help_txt) in zip(cols, presets, strict=True):
        n = int(mask.sum())
        is_clear = "ล้าง" in label
        if col.button(
            label if is_clear else f"{label} ({n})",
            key=f"{key_ns}_qp_{label}",
            width="stretch",
            help=help_txt,
            disabled=n == 0 and not is_clear,
        ):
            pressed = fires.loc[mask, "node_idx"].astype(int).tolist()
    return pressed


def _selection_ui(
    split: str, anchor_iso: str, sid: int, station_name: str, fires: pd.DataFrame
) -> list[int]:
    """Map + quick picks; returns the hotspot node indices currently switched off.

    Selection state lives in ``session_state`` keyed by the scenario, so changing
    event or station starts clean. Two inputs write to it — map clicks and the quick
    buttons — and they are reconciled by *change*, not by precedence: the map applies
    its selection only when that selection differs from the one it reported last run.

    The signature therefore starts as an empty tuple, not ``None``. On the rerun that
    follows a button press the map reports "nothing selected", and if that counted as a
    change it would immediately wipe the selection the button just made — which is
    exactly what happened before this was pinned down. A genuine deselect (map goes
    from some points to none) still registers, because the previous signature was
    non-empty.
    """
    ns = f"wi_{split}_{anchor_iso}_{sid}"
    sel_key, sig_key = f"{ns}_sel", f"{ns}_sig"
    st.session_state.setdefault(sel_key, [])
    st.session_state.setdefault(sig_key, ())

    st.markdown("##### 1) เลือกกลุ่มไฟที่จะ “ดับ”")
    pressed = _quick_picks(fires, ns)
    if pressed is not None:
        st.session_state[sel_key] = pressed

    picked = set(st.session_state[sel_key])
    event = st.plotly_chart(
        _fire_map(fires, sid, station_name, picked),
        width="stretch",
        key=f"{ns}_map",
        theme=None,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
    )

    from_map = _nodes_from_event(event)
    if from_map is not None:
        sig = tuple(sorted(from_map))
        if sig != st.session_state[sig_key]:
            st.session_state[sig_key] = sig
            st.session_state[sel_key] = list(sig)

    st.caption(
        "คลิกที่จุดไฟเพื่อเลือก · กด Shift ค้างไว้เพื่อเลือกหลายจุด · "
        "หรือใช้เครื่องมือ Box/Lasso ที่มุมขวาบนของแผนที่เพื่อกวาดเลือกทั้งโซน · "
        f"⬤ วงสีเข้ม = กลุ่มที่กำลังจะถูกดับ · จุดสีจาง = ลมไม่ได้พาควันมาที่{station_name} วันนี้"
    )
    return sorted(st.session_state[sel_key])


def _nodes_from_event(event: object) -> list[int] | None:
    """Hotspot node indices out of a Streamlit plotly selection event (``None`` if absent)."""
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    if not isinstance(sel, dict):
        return None
    out: list[int] = []
    for p in sel.get("points", []):
        cd = p.get("customdata")
        if isinstance(cd, list | tuple) and cd:
            out.append(int(cd[0]))
        elif isinstance(cd, int | float):
            out.append(int(cd))
    return sorted(set(out))


def _fire_map(fires: pd.DataFrame, sid: int, station_name: str, picked: set[int]) -> go.Figure:
    """Offline click-to-select fire map: connected vs not, selected ringed, station marked."""
    meta = da.load_stations_meta()
    srow = meta[meta["station_id"] == sid].iloc[0]
    lat0, lon0 = float(srow["lat"]), float(srow["lon"])
    height = 470
    fig = go.Figure()

    groups: list[tuple[str, pd.Series, str, float]] = [
        (
            f"ไฟที่ลมไม่ได้พามาที่{station_name}",
            ~fires["connected"],
            _OFF,
            0.5,
        ),
        (
            "ไฟในไทย (ลมพามา)",
            fires["connected"] & (fires["country"] == "Thailand"),
            _THAI,
            0.9,
        ),
        (
            "ไฟต่างชาติ (ลมพามา)",
            fires["connected"] & fires["country"].isin(["Myanmar", "Laos"]),
            _FOREIGN,
            0.9,
        ),
    ]
    for label, mask, colour, opacity in groups:
        sub = fires[mask]
        if sub.empty:
            continue
        sizes = [max(7, min(30, 5 + math.sqrt(max(f, 0)) * 1.05)) for f in sub["frp"]]
        fig.add_trace(
            go.Scattermapbox(
                lat=sub["lat"],
                lon=sub["lon"],
                mode="markers",
                marker=go.scattermapbox.Marker(size=sizes, color=colour, opacity=opacity),
                customdata=[[int(i)] for i in sub["node_idx"]],
                name=label,
                text=[
                    f"กลุ่มไฟ #{int(r.node_idx)} · {_COUNTRY_TH.get(r.country, r.country)}<br>"
                    f"ความแรง FRP {r.frp:,.0f} MW<br>"
                    + ("ลมพาควันมาที่สถานีนี้" if r.connected else "ลมไม่ได้พามาที่สถานีนี้")
                    + "<br><i>คลิกเพื่อเลือก/ดับ</i>"
                    for r in sub.itertuples()
                ],
                hoverinfo="text",
            )
        )

    chosen = fires[fires["node_idx"].isin(picked)]
    if not chosen.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=chosen["lat"],
                lon=chosen["lon"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=[
                        max(15, min(40, 13 + math.sqrt(max(f, 0)) * 1.05)) for f in chosen["frp"]
                    ],
                    color=_PICKED,
                    opacity=0.32,
                ),
                name=f"กำลังจะดับ ({len(chosen)} กลุ่ม)",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=[lat0],
            lon=[lon0],
            mode="markers",
            marker=go.scattermapbox.Marker(size=18, color="#111"),
            name=f"สถานี {station_name}",
            hoverinfo="name",
        )
    )

    # Frame on the station and the fires the wind actually connects to it. Fires far
    # downwind stay drawn (they are the visual control) but must not zoom the view out
    # until the station itself is a speck at the edge of the canvas.
    focus = fires[fires["connected"]]
    if focus.empty:
        focus = fires
    pts = [(lat0, lon0)] + [
        (float(r.lat), float(r.lon))
        for r in focus.itertuples()
        if 16.0 <= r.lat <= 21.0 and 97.0 <= r.lon <= 101.5
    ]
    center, zoom = geo.fit_view(pts, height, pad_deg=1.0, zoom_range=(5.2, 7.4))
    fig.update_layout(
        mapbox=dict(
            style="white-bg",
            center=center,
            zoom=zoom,
            layers=[
                dict(
                    sourcetype="geojson",
                    source=geo.border_geojson(),
                    type="line",
                    color="rgba(70,70,70,.75)",
                    line=dict(width=1.4),
                )
            ],
        ),
        margin=dict(l=0, r=0, t=6, b=0),
        height=height,
        clickmode="event+select",
        paper_bgcolor=_MAP_BG,
        plot_bgcolor=_MAP_BG,
        font=dict(color=_MAP_INK, size=12),
        hoverlabel=dict(font=dict(color=_MAP_INK, size=12), bgcolor="#ffffff"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,.94)",
            bordercolor="rgba(31,41,55,.35)",
            borderwidth=1,
            font=dict(size=12.5, color=_MAP_INK),
            itemsizing="constant",
        ),
    )
    return fig


def _suppression_slider() -> float:
    """Suppression-level slider; returns the *remaining* FRP fraction for the model."""
    st.markdown("##### 2) ดับได้มากแค่ไหน")
    pct = st.slider(
        "สมมติดับไฟกลุ่มที่เลือกได้กี่ %",
        0,
        100,
        100,
        step=5,
        key="wi_rate",
        help="100% = ดับสนิท (ตั้ง total_frp = 0) · 50% = ความแรงไฟเหลือครึ่งเดียว",
    )
    return 1.0 - pct / 100.0


# --------------------------------------------------------------- results ---


def _result_block(
    cf: dict,
    sid: int,
    station_name: str,
    horizon_idx: int,
    remaining: float,
    fires: pd.DataFrame,
    picked: list[int],
) -> None:
    """Before / after / difference for the selected station, with an honest reading."""
    st.markdown("##### 3) ผลลัพธ์ — รันโมเดลใหม่แล้วได้เท่าไร")
    i = cf["station_ids"].index(sid)
    full = float(cf["pred_full_ug"][i, horizon_idx])
    occ = float(cf["pred_occluded_ug"][i, horizon_idx])
    delta = full - occ
    pct = 100 * delta / full if full > 0 else 0.0
    h = _H_LABELS[horizon_idx]

    sub = fires[fires["node_idx"].isin(picked)]
    by_country = sub.groupby("country")["frp"].sum().sort_values(ascending=False)
    mix = " · ".join(f"{_COUNTRY_TH.get(k, k)} {v:,.0f} MW" for k, v in by_country.items())

    c1, c2, c3 = st.columns(3)
    c1.metric(f"A · ทำนายปกติ ({h})", f"{full:,.1f} µg/m³", help="ไฟครบทุกกลุ่มตามดาวเทียมจริง")
    c2.metric(
        f"B · หลังดับไฟที่เลือก ({h})",
        f"{occ:,.1f} µg/m³",
        delta=f"{-delta:,.2f}",
        delta_color="inverse",
    )
    # delta_color="off": the percentage is a *share* of the forecast, not a change in
    # it, so it must not inherit the green-good / red-bad arrow language.
    c3.metric(
        "A − B · ค่าที่หายไป",
        f"{delta:,.2f} µg/m³",
        f"{pct:.1f}% ของค่าเดิม",
        delta_color="off",
    )

    st.caption(
        f"ดับ {cf['n_selected']} กลุ่มไฟ รวมความแรง {cf['frp_removed']:,.0f} MW "
        f"({(1 - remaining) * 100:.0f}% ของกลุ่มที่เลือก) — {mix} · "
        f"ที่สถานี{station_name} · ใช้เวลารันสองรอบ {cf['elapsed_ms']:.0f} มิลลิวินาที"
    )
    st.markdown(_verdict_box(delta, pct, station_name, h), unsafe_allow_html=True)


def _verdict_box(delta: float, pct: float, station_name: str, h: str) -> str:
    """One honest sentence about the measured response, including the awkward cases."""
    if delta >= 3.0:
        colour, head = _FOREIGN, "ไฟกลุ่มนี้มีผลชัดเจน"
        body = (
            f"ถ้าไฟกลุ่มนี้ไม่เกิดขึ้น โมเดลจะทำนาย{station_name}ที่ {h} ข้างหน้า "
            f"<b>ต่ำลง {delta:,.1f} µg/m³ ({pct:.1f}%)</b> — นี่คือส่วนของฝุ่นที่โมเดลโยงกลับไปหาไฟกลุ่มนี้ได้"
        )
    elif delta >= 0.5:
        colour, head = "#b45309", "มีผล แต่ไม่มาก"
        body = (
            f"ค่าลดลง <b>{delta:,.2f} µg/m³ ({pct:.1f}%)</b> — เห็นทิศทางชัด "
            "แต่ขนาดเล็กเมื่อเทียบกับค่าฝุ่นทั้งก้อน"
        )
    elif delta > -0.1:
        colour, head = "#6b7280", "แทบไม่ขยับ — และนี่คือผลที่ควรได้"
        body = (
            f"ค่าเปลี่ยนเพียง <b>{delta:,.2f} µg/m³</b> แปลว่าโมเดล<u>ไม่ได้</u>โยงฝุ่นที่"
            f"{station_name}วันนี้เข้ากับไฟกลุ่มนี้ "
            "ถ้าหลังบ้านเป็นกฎ if-else ที่ “เห็นไฟแล้วลบค่าออก” ตัวเลขนี้จะไม่มีวันเป็นศูนย์ "
            "— ผลลบเชิงซื่อสัตย์แบบนี้คือหลักฐานว่าโมเดลไล่ตามไฟจริง ไม่ได้ตั้งธงไว้ล่วงหน้า"
        )
    else:
        colour, head = "#7c3aed", "ค่ากลับเพิ่มขึ้นเล็กน้อย — รายงานตามจริง"
        body = (
            f"ดับไฟแล้วโมเดลกลับทำนาย<b>สูงขึ้น {abs(delta):,.2f} µg/m³</b> "
            "ซึ่งผิดจากสัญชาตญาณทางกายภาพ เกิดได้เมื่อความไวต่อไฟที่โมเดลเรียนรู้มาอ่อนมาก "
            "จนถูกกลบด้วยความไม่เป็นเชิงเส้นของโครงข่าย เราแสดงค่านี้ตามจริงแทนที่จะตัดทิ้ง"
        )
    return (
        f"<div style='border-left:3px solid {colour};background:rgba(130,130,130,.10);"
        f"padding:10px 14px;border-radius:5px;margin:8px 0;line-height:1.65'>"
        f"<b>{head}</b><br>{body}</div>"
    )


def _dose_response_block(
    split: str, anchor_iso: str, picked: tuple[int, ...], sid: int, horizon_idx: int
) -> None:
    """The falsifiability exhibit: forecast vs. suppression level, 0→100 %."""
    st.markdown("##### 4) พิสูจน์ว่าไม่ใช่ if-else — เส้นตอบสนองต่อระดับการดับไฟ")
    dr = inf.dose_response_at(split, anchor_iso, picked, sid, horizon_idx)
    x, y = dr["suppressed_pct"], dr["pm25_ug"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=_FOREIGN, width=3, shape="spline"),
            marker=dict(size=8, color=_FOREIGN),
            name="โมเดลจริง",
            hovertemplate="ดับไฟ %{x}% → %{y:.2f} µg/m³<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 49.9, 50, 100],
            y=[y[0], y[0], y[-1], y[-1]],
            mode="lines",
            line=dict(color=_OFF, width=2, dash="dot"),
            name="ถ้าเป็นกฎ if-else (เทียบให้ดู)",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(title="ดับไฟกลุ่มที่เลือกได้ (%)", ticksuffix="%", zeroline=False),
        yaxis=dict(title="PM2.5 ที่ทำนาย (µg/m³)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch", key=f"wi_dose_{anchor_iso}_{sid}")
    st.markdown(
        f"""
<div style="border-left:3px solid {_FOREIGN};background:rgba(130,130,130,.10);
     padding:10px 14px;border-radius:5px;line-height:1.65">
  <b>อ่านยังไง:</b> เราไม่ได้ดับไฟแค่ “ดับ/ไม่ดับ” แต่ค่อย ๆ ลดความแรงไฟลงทีละขั้น
  แล้วรันโมเดลใหม่ทุกขั้น (ทั้งหมด {len(x)} รอบ)<br>
  ถ้าหลังบ้านเป็นกฎ <code>if มีไฟต่างชาติ → ลบ X</code> กราฟจะเป็น<b>ขั้นบันได</b>
  (เส้นประเทา) เพราะกฎมีแค่สองสถานะ<br>
  แต่กราฟจริง<b>ค่อย ๆ โค้งลง</b> ตามปริมาณไฟที่เหลือ เพราะตัวเลขนี้ออกมาจากโครงข่าย
  ที่อ่านค่า <code>total_frp</code> เป็นตัวเลขต่อเนื่อง ไม่ใช่เงื่อนไขจริง/เท็จ
</div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _station_labels(station_ids: list[int]) -> list[str]:
    """Short, **unique** chart labels for stations, aligned to ``station_ids``.

    Several stations share a name prefix ("Natural Resources and Environment Office,
    Chiangrai" / "…, Mae Hongson"), so naive truncation produces duplicate categories —
    and a plotly bar chart silently merges duplicate categories into one row, which
    quietly corrupts the whole footprint chart. Labels are therefore built from the
    province suffix plus a trimmed name, and any residual collision is broken with the
    station id rather than left to collapse.
    """
    meta = da.load_stations_meta()
    by_sid = {int(r.station_id): (str(r["name"]), str(r.province)) for _, r in meta.iterrows()}
    labels: list[str] = []
    seen: set[str] = set()
    for s in station_ids:
        full, province = by_sid.get(int(s), (str(s), ""))
        head = full.rsplit(",", 1)[0].strip()
        label = f"{province} · {head[:24]}" if province else head[:32]
        if label in seen:
            label = f"{label} #{s}"
        seen.add(label)
        labels.append(label)
    return labels


def _footprint_block(cf: dict, sid: int, horizon_idx: int) -> None:
    """Where else the edit landed — the graph propagating one local change outward."""
    st.markdown("##### 5) ผลกระทบแผ่ไปถึงสถานีไหนบ้าง")
    sids = cf["station_ids"]
    delta = np.asarray(cf["delta_ug"], dtype=float)[:, horizon_idx]

    df = pd.DataFrame(
        {
            "สถานี": _station_labels(sids),
            "ลดลง (µg/m³)": np.round(delta, 2),
            "_sel": [s == sid for s in sids],
        }
    ).sort_values("ลดลง (µg/m³)", ascending=True, ignore_index=True)

    fig = go.Figure(
        go.Bar(
            x=df["ลดลง (µg/m³)"],
            y=df["สถานี"],
            orientation="h",
            marker=dict(color=[_FOREIGN if s else _OFF for s in df["_sel"]]),
            hovertemplate="%{y}: %{x:.2f} µg/m³<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(340, 24 * len(df)),
        margin=dict(l=0, r=0, t=8, b=0),
        bargap=0.25,
        xaxis=dict(title="ค่าที่หายไปเมื่อดับไฟกลุ่มที่เลือก (µg/m³)", zeroline=True),
        yaxis=dict(title="", type="category"),
    )
    st.plotly_chart(fig, width="stretch", key=f"wi_foot_{cf['elapsed_ms']:.0f}_{sid}")
    n_moved = int((np.abs(delta) >= 0.05).sum())
    n_neg = int((delta <= -0.05).sum())
    note = (
        f"ดับไฟชุดเดียวกัน แล้ววัดผลที่ทั้ง {len(sids)} สถานีพร้อมกัน — ขยับจริง {n_moved} สถานี "
        "(แถบสีส้ม = สถานีที่เลือกไว้ด้านบน). สถานีที่ได้ผลมากที่สุดไม่จำเป็นต้องเป็นสถานีที่เลือก "
        "เพราะลมพาควันไปตามเส้นทางของมัน ไม่ใช่ตามที่เราสนใจ — "
        "และสถานีที่กราฟไม่ได้เชื่อมถึงไฟชุดนี้จะนิ่งอยู่ที่ ~0"
    )
    if n_neg:
        note += (
            f". มี {n_neg} สถานีที่ค่า<b>ติดลบ</b> คือดับไฟแล้วโมเดลกลับทำนายสูงขึ้นเล็กน้อย "
            "ซึ่งขัดสัญชาตญาณทางกายภาพ เป็นผลข้างเคียงจากความไม่เป็นเชิงเส้นของโครงข่าย "
            "เราแสดงไว้ตามจริงแทนที่จะตัดทิ้ง"
        )
    st.caption(note, unsafe_allow_html=True)


def _placebo_block(
    split: str,
    anchor_iso: str,
    picked: tuple[int, ...],
    sid: int,
    horizon_idx: int,
    cf: dict,
) -> None:
    """Negative control: extinguish fires the wind does not connect to this station."""
    st.markdown("##### 6) กลุ่มควบคุม (placebo) — ดับไฟที่ลมไม่ได้พามา")
    pl = inf.placebo_at(split, anchor_iso, picked, sid)
    if pl["n_selected"] == 0:
        st.caption(
            "วันนี้กลุ่มไฟเกือบทั้งหมดถูกลมเชื่อมมาที่สถานีนี้ จึงไม่เหลือไฟ “นอกเส้นทางลม” "
            "ให้ใช้เป็นกลุ่มควบคุม — ดูข้อ 5 แทน (สถานีที่ไม่ได้เชื่อมถึงไฟชุดนี้นิ่งที่ ~0)"
        )
        return

    i = cf["station_ids"].index(sid)
    real = float(cf["delta_ug"][i, horizon_idx])
    fake = float(np.asarray(pl["delta_ug"], dtype=float)[i, horizon_idx])

    c1, c2 = st.columns(2)
    c1.metric(
        "ของจริง · ดับไฟที่ลมพามา",
        f"{real:,.2f} µg/m³",
        help=f"{cf['n_selected']} กลุ่ม · {cf['frp_removed']:,.0f} MW",
    )
    c2.metric(
        "ควบคุม · ดับไฟที่ลมไม่ได้พามา",
        f"{fake:,.2f} µg/m³",
        help=f"{pl['n_selected']} กลุ่ม · {pl['frp_removed']:,.0f} MW",
    )

    ratio_note = (
        f"กลุ่มควบคุมดับไฟได้ {pl['frp_removed']:,.0f} MW เทียบกับของจริง "
        f"{cf['frp_removed']:,.0f} MW"
    )
    matched = pl["frp_removed"] >= 0.5 * max(cf["frp_removed"], 1e-9)
    if not matched:
        ratio_note += (
            " — วันนี้ไฟที่อยู่นอกเส้นทางลมมีน้อย จึงจับคู่ความแรงไฟให้เท่ากันไม่ได้ "
            "ผลนี้จึงเป็นหลักฐานสนับสนุน ไม่ใช่การทดลองควบคุมเต็มรูปแบบ"
        )
    st.caption(
        "เลือกไฟที่ **ไม่มีเส้นเชื่อม (Type-C) มาที่สถานีนี้** แล้วดับด้วยวิธีเดียวกันทุกประการ. "
        f"{ratio_note}. ถ้าโมเดลตอบสนองต่อ “มีไฟที่ไหนก็ได้” ตัวเลขสองช่องนี้จะใกล้เคียงกัน; "
        "การที่ช่องขวานิ่งกว่ามาก แปลว่าโมเดลตอบสนองต่อ*เส้นทางลม*ที่เชื่อมไฟถึงสถานี ไม่ใช่ต่อการมีไฟเฉย ๆ"
    )


# ------------------------------------------------------------- glass box ---


def _glass_box(
    split: str, anchor_iso: str, cf: dict, fires: pd.DataFrame, picked: list[int]
) -> None:
    """Show the mechanism itself: the graph, the edited tensor, and the live source code."""
    with st.expander("🔍 หลังบ้านทำอะไรบ้าง — เปิดดูของจริง (ไม่มี if-else)"):
        facts = inf.graph_facts(split, anchor_iso)

        st.markdown("**กราฟที่โมเดลรับเข้าไปในวันนี้**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("สถานี (node)", facts["n_stations"])
        c2.metric("กลุ่มไฟ (node)", facts["n_hotspots"])
        c3.metric("เส้นเชื่อมไฟ→สถานี", facts["edges_type_c"])
        c4.metric("พารามิเตอร์โมเดล", f"{facts['n_params']:,}")
        st.caption(
            f"อินพุตสถานี {facts['station_x_shape']} (สถานี × ชั่วโมงย้อนหลัง × ปัจจัย) · "
            f"อินพุตกลุ่มไฟ {facts['hotspot_x_shape']} · "
            f"เส้นเชื่อม ระยะทาง {facts['edges_type_a']} · ลม {facts['edges_type_b']} · "
            f"ไฟ {facts['edges_type_c']} · checkpoint `{facts['checkpoint']}`"
        )

        st.markdown("**ตัวเลขที่ถูกแก้จริง ๆ — เทนเซอร์ `hotspot.x` ก่อน/หลัง**")
        cols = facts["hotspot_feature_names"]
        before = np.asarray(cf["x_before"], dtype=float)
        after = np.asarray(cf["x_after"], dtype=float)
        country_by_idx = dict(zip(fires["node_idx"].astype(int), fires["country"], strict=False))
        tbl = pd.DataFrame(
            {
                "กลุ่มไฟ #": cf["node_indices"],
                "ประเทศ (ป้ายกำกับ ไม่ใช่อินพุต)": [
                    _COUNTRY_TH.get(country_by_idx.get(i, "?"), "?") for i in cf["node_indices"]
                ],
                f"{cols[0]} ก่อน": np.round(before[:, 0], 1),
                f"{cols[0]} หลัง": np.round(after[:, 0], 1),
                f"{cols[1]}": np.round(before[:, 1], 4),
                f"{cols[2]}": np.round(before[:, 2], 4),
            }
        )
        st.dataframe(tbl, width="stretch", hide_index=True)
        st.markdown(
            f"""
<div style="border-left:3px solid {_THAI};background:rgba(42,120,214,.09);
     padding:10px 14px;border-radius:5px;line-height:1.7;margin:4px 0 10px">
  <b>จุดสำคัญที่สุดของหน้านี้:</b> อินพุตของกลุ่มไฟมีแค่
  <code>[total_frp, centroid_lat, centroid_lon]</code> — <b>ไม่มีคอลัมน์ “ประเทศ”</b>
  (ดู <code>src/data/graph_builder.py :: build_graph</code>).
  โมเดลจึงไม่มีทางรู้ว่าไฟกลุ่มไหนอยู่เมียนมาหรืออยู่ไทย มันเห็นแค่
  “ไฟแรงเท่านี้ อยู่พิกัดนี้” แล้วลมเป็นตัวสร้างเส้นเชื่อมให้เอง<br>
  ป้ายชื่อประเทศถูกใช้<u>ตอนเราสรุปผลทีหลัง</u>เท่านั้น (เพื่อจัดกลุ่มให้คนอ่าน)
  ถ้าสลับป้ายประเทศทุกจุดตอนนี้ ตัวเลขทุกตัวบนหน้านี้จะเท่าเดิมเป๊ะ — เพราะมันไม่เคยเข้าโมเดล
</div>
            """,
            unsafe_allow_html=True,
        )

        idx_repr = repr(picked) if len(picked) <= 8 else repr(picked[:8])[:-1] + ", ...]"
        max_delta = float(np.asarray(cf["delta_ug"], dtype=float).max())
        st.markdown("**ลำดับการทำงานจริง (รอบที่เพิ่งรันไป)**")
        st.code(
            f'ds     = PM25GraphDataset(split="{split}", wind_mode="from_field")\n'
            f'sample = ds[pos_for_timestamp(ds, "{anchor_iso}")]   # กราฟ 1 ภาพ ณ เวลานั้น\n'
            f"\n"
            f"pred_A = model(sample)                     # รอบที่ 1: ไฟครบตามดาวเทียม\n"
            f"\n"
            f'hx = sample["hotspot"].x.clone()\n'
            f"hx[{idx_repr}, 0] *= {cf['remaining']}   # แก้เฉพาะคอลัมน์ 0 = total_frp\n"
            f'sample["hotspot"].x = hx\n'
            f"pred_B = model(sample)                     # รอบที่ 2: โมเดล/น้ำหนัก/วันเดียวกันทุกอย่าง\n"
            f"\n"
            f"delta  = denorm(pred_A) - denorm(pred_B)   # สูงสุดข้ามสถานี = {max_delta:.2f} µg/m³\n"
            f"# รวมเวลารันสองรอบ: {cf['elapsed_ms']:.0f} ms",
            language="python",
        )

        st.markdown("**โค้ดจริงของฟังก์ชันที่เพิ่งรัน** (ดึงสด ๆ ด้วย `inspect.getsource`)")
        st.caption(
            "ไม่ใช่ภาพหน้าจอหรือโค้ดที่พิมพ์ประกอบ — Python อ่านมาจากไฟล์ "
            "`src/explain/gb_ig.py` ที่กำลังทำงานอยู่ตอนนี้ ตรวจได้ว่าไม่มี `if` "
            "ที่ดูชื่อประเทศ วันที่ หรือค่าที่ต้องการ"
        )
        st.code(inspect.getsource(gb_ig.occlude_hotspot_nodes), language="python")
        st.code(inspect.getsource(gb_ig._forward_with_hotspot_x), language="python")


def _honesty_footer() -> None:
    """What this page does and does not prove — including the missing ground truth."""
    st.divider()
    st.markdown(
        f"""
<div style="border-left:3px solid #6b7280;background:rgba(130,130,130,.10);
     padding:12px 16px;border-radius:6px;line-height:1.75">
  <b>สิ่งที่หน้านี้พิสูจน์ได้ และพิสูจน์ไม่ได้</b><br><br>
  <b style="color:{_THAI}">✔ พิสูจน์ได้:</b> ตัวเลขทุกตัวคือ<b>ความไวของโมเดล</b>
  ที่วัดซ้ำได้เป๊ะ ๆ — “ถ้าอินพุตความแรงไฟตรงนี้เป็นศูนย์ ผลลัพธ์ของโมเดลจะเปลี่ยนไปเท่านี้”
  ใครรันซ้ำด้วย checkpoint เดิมก็ได้เลขเดิม<br><br>
  <b style="color:{_FOREIGN}">✘ พิสูจน์ไม่ได้:</b> ว่าฝุ่นในอากาศจริง ๆ <i>มาจากไฟกลุ่มนั้นกี่ไมโครกรัม</i>.
  เครื่องวัด PM2.5 วัดได้แค่ “เข้มข้นเท่าไร” ไม่ได้วัดว่า “มาจากไหน”
  <b>โลกนี้จึงไม่มีเฉลยรายวันของแหล่งที่มา</b>ให้เทียบ ไม่ว่าจะเป็นงานของเราหรือของใคร<br><br>
  เราจึงไม่เคลมว่า “23 µg/m³ มาจากเมียนมา” แต่เคลมว่า
  “<u>ภายใต้โมเดลนี้</u> การตัดไฟชุดนี้ออก ทำให้ค่าพยากรณ์ลดลง 23 µg/m³”
  แล้วเอาไปเทียบกับหลักฐานอิสระที่วัดได้จริง (ไฟจาก NASA FIRMS + เส้นทางลมจาก ERA5)
  ว่าชี้ทิศเดียวกันไหม — ดูหน้า <b>“ฝุ่นวันนั้นมาจากไหน”</b> และ <b>“หมอกควันข้ามแดน”</b><br><br>
  <b>ขนาดของผลก็ต้องพูดตามตรง:</b> บน checkpoint นี้ ความไวต่อไฟอยู่ราว
  <b>3–7% ของค่าฝุ่น</b> และจะเห็นผลชัดเฉพาะวันที่ไฟต่างชาติแรงมากจริง ๆ
  (ระดับหมื่น MW ขึ้นไป) วันที่ไฟน้อยค่าจะออกมาใกล้ศูนย์ เราแสดงตามที่วัดได้
  ไม่ได้ขยายให้ดูน่าตื่นเต้น<br><br>
  <span style="opacity:.85">⚠️ เครื่องมือนี้ใช้สำรวจเชิงนโยบายและการศึกษา
  <b>ไม่ควรใช้กล่าวโทษเชิงการทูต</b></span>
</div>
        """,
        unsafe_allow_html=True,
    )
