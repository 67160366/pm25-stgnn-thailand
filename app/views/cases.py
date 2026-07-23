# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Plain-language 'where did the dust come from?' case view.

The demo-facing page: each haze event is one self-contained "case card" a
non-expert reads top-to-bottom. Every verdict is backed by two *observable*
signals that do not depend on the (seed-sensitive) neural attribution:

* fire  — connected FIRMS FRP per country (where is it burning?)
* wind  — hours the 48h ERA5 back-trajectory spent over Myanmar (where from?)

The AI model's country attribution is a *third, corroborating witness* with its
honest cross-seed range, never the headline number; where it cannot pin the
proportion the card says so plainly and defers to the evidence.

Design follows the dataviz skill: one two-colour language throughout — blue =
Thailand, orange = abroad (validated CVD-safe pair) — minimal iconography, and
every number carries its real source (NASA FIRMS / ERA5 / two model checkpoints)
with a raw-data expander so a judge can verify nothing was invented. All values
are computed at render time from the frozen result JSONs.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from app.lib import aqi, geo, ui
from app.lib import data_access as da

_STATION_LABEL = "แม่ฮ่องสอน"

# One colour language, reused everywhere the entity appears (dataviz slots 1 & 2,
# validated CVD-safe in both light and dark). Borderline is a quiet neutral, not
# a third loud hue.
_THAI = "#2a78d6"
_FOREIGN = "#eb6834"
_MIXED = "#6b7280"

# verdict -> (accent, banner label)
_VERDICT = {
    "domestic": (_THAI, "ฝุ่นในประเทศ"),
    "foreign": (_FOREIGN, "ฝุ่นข้ามแดน"),
    "borderline": (_MIXED, "ก้ำกึ่ง — มีทั้งสองฝั่ง"),
}

_TH_MONTHS = [
    "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
]  # fmt: skip


def _thai_date(iso: str) -> str:
    """Format an ISO ``YYYY-MM-DD`` date as ``D <thai-month> <B.E. year>``."""
    y, m, d = (int(x) for x in iso.split("-"))
    return f"{d} {_TH_MONTHS[m - 1]} {y + 543}"


# ---------------------------------------------------------------- render ----


def render() -> None:
    """Render the plain-language source-attribution case gallery."""
    ui.page_title(
        "ฝุ่นวันนั้นมาจากไหน?",
        "เหตุการณ์ฝุ่นจริงในภาคเหนือ — ระบุว่ามาจากในประเทศหรือข้ามแดน พร้อมหลักฐาน",
    )
    _intro()
    cases = _build_cases()
    for case in cases:
        _render_case(case)
    _policy_simulator(cases)
    _closing_honesty()


def _intro() -> None:
    """Two-signal explainer plus a provenance banner (numbers are real)."""
    st.markdown(
        """
<div style="border-left:3px solid #2a78d6;background:rgba(42,120,214,.08);
     padding:12px 16px;border-radius:6px;margin:6px 0 10px">
  <b>วิธีอ่าน:</b> ที่มาของฝุ่นตัดสินจาก <b>2 หลักฐานที่วัดได้จริง</b> —
  <b style="color:#eb6834">ไฟ</b> (กำลังไหม้ที่ไหน) และ
  <b style="color:#2a78d6">ลม</b> (พัดมาจากไหน).
  ลมพาฝุ่นจากเมียนมาได้ก็จริง แต่ถ้าไม่มีไฟฝั่งนั้นก็ไม่มีฝุ่นให้พามา —
  จึงเป็น “ข้ามแดน” ต่อเมื่อ <u>ลมมาจากนอกประเทศ และ มีไฟไหม้อยู่ฝั่งนั้น</u>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "ตัวเลขทุกตัวคำนวณจากข้อมูลสาธารณะจริง ตรวจย้อนได้ — ไฟจากดาวเทียม NASA FIRMS · "
        "ลม/อากาศจาก ERA5 (ECMWF/Copernicus) · ค่าฝุ่นจาก Air4Thai (กรมควบคุมมลพิษ). "
        "กด “ตัวเลขนี้มาจากไหน” ใต้แต่ละเหตุการณ์เพื่อดูค่าดิบ"
    )


# --------------------------------------------------------------- builders --


def _wind_hours(bt_event: dict | None) -> tuple[int, int] | None:
    """Return ``(myanmar_hours, total_in_domain_hours)`` from a trajectory event."""
    if not bt_event:
        return None
    pts = bt_event.get("trajectory", {}).get("points", [])
    if not pts:
        return None
    myanmar = sum(1 for p in pts if p.get("country") == "Myanmar")
    return myanmar, len(pts)


def _build_cases() -> list[dict]:
    """Assemble ordered case dicts from the frozen result JSONs.

    Order tells the demo story: the domestic monster event first (proves the
    system does not blindly blame abroad), then the flagship transboundary case,
    the borderline case, two more domestic cases, and the robust 2024 contrast.
    """
    unc = da.load_transboundary_uncertainty()
    bt = da.load_backtrajectory()
    bt_by_date = {e["date"]: e for e in bt.get("events", [])}
    unc_by_date = {e["date"]: e for e in unc.get("events", [])}

    order = ["2025-03-30", "2025-03-18", "2025-03-13", "2025-03-11", "2025-03-17"]
    cases = [_case_2025(unc_by_date[d], bt_by_date.get(d)) for d in order if d in unc_by_date]
    cases.append(_case_chiangmai_2024())
    return cases


def _case_2025(unc_event: dict, bt_event: dict | None) -> dict:
    """Build a case dict for a 2025 Mae Hong Son event, with raw provenance."""
    frp = unc_event.get("connected_frp_by_country", {})
    thai = float(frp.get("Thailand", 0.0))
    mmr = float(frp.get("Myanmar", 0.0))
    lao = float(frp.get("Laos", 0.0))
    foreign_frac = float(unc_event.get("connected_foreign_fraction", 0.0))
    if 0.4 <= foreign_frac <= 0.6:
        verdict = "borderline"
    elif foreign_frac > 0.6:
        verdict = "foreign"
    else:
        verdict = "domestic"

    mean = 100 * float(unc_event.get("foreign_attribution_mean", 0.0))
    lo = 100 * float(unc_event.get("foreign_attribution_min", 0.0))
    hi = 100 * float(unc_event.get("foreign_attribution_max", 0.0))
    per_seed = {
        lbl: 100 * float(v.get("foreign_attribution", 0.0))
        for lbl, v in unc_event.get("per_seed", {}).items()
    }
    traj = [(p["lat"], p["lon"]) for p in (bt_event or {}).get("trajectory", {}).get("points", [])]
    station_lat, station_lon = traj[0] if traj else (19.30, 97.97)

    return {
        "key": unc_event["date"],
        "date_iso": unc_event["date"],
        "date_th": _thai_date(unc_event["date"]),
        "station": _STATION_LABEL,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "traj": traj,
        "peak": float(unc_event.get("peak_pm25_ug_m3", 0.0)),
        "verdict": verdict,
        "thai_frp": thai,
        "foreign_frp": mmr + lao,
        "frp_detail": {"ไทย": thai, "เมียนมา": mmr, "ลาว": lao},
        "wind_hours": _wind_hours(bt_event),
        "model_mean": mean,
        "model_lo": lo,
        "model_hi": hi,
        "model_unstable": (hi - lo) >= 40.0,
        "per_seed": per_seed,
        "src_file": "transboundary_uncertainty.json · backtrajectory_test2025.json",
    }


def _case_chiangmai_2024() -> dict:
    """Build the robust 2024 Chiang Mai domestic contrast case.

    Two independent checkpoints (test + val) agree to ~99.7% domestic, so this
    card carries a *stable* model reading — the opposite of the seed-sensitive
    2025 flagship event.
    """
    test = da.load_output_json("attribution_march2024.json")
    val = da.load_output_json("attribution_march2024_val.json")
    frp = test.get("hotspot_frp_summary", {})
    thai = float(frp.get("Thailand", {}).get("total_frp", 0.0))
    mmr = float(frp.get("Myanmar", {}).get("total_frp", 0.0))
    lao = float(frp.get("Laos", {}).get("total_frp", 0.0))
    dom_test = 100 * float(test.get("country_attribution", {}).get("Thailand", 0.0))
    dom_val = 100 * float(val.get("country_attribution", {}).get("Thailand", 0.0))
    peaks = test.get("peak_pm25_ug_m3", [0.0])

    return {
        "key": "2024-03",
        "date_iso": None,  # month aggregate — no single-day map
        "date_th": "มี.ค. 2567",
        "station": "เชียงใหม่",
        "station_lat": None,
        "station_lon": None,
        "traj": [],
        "peak": float(max(peaks)),
        "verdict": "domestic",
        "thai_frp": thai,
        "foreign_frp": mmr + lao,
        "frp_detail": {"ไทย": thai, "เมียนมา": mmr, "ลาว": lao},
        "wind_hours": None,  # no back-trajectory computed for the 2024 case
        "model_mean": 100 - (dom_test + dom_val) / 2,
        "model_lo": 100 - max(dom_test, dom_val),
        "model_hi": 100 - min(dom_test, dom_val),
        "model_unstable": False,
        "model_note": f"2 โมเดลอิสระเห็นตรงกัน (ในประเทศ {dom_test:.1f}% และ {dom_val:.1f}%)",
        "per_seed": {"checkpoint test": 100 - dom_test, "checkpoint val": 100 - dom_val},
        "src_file": "attribution_march2024.json · attribution_march2024_val.json",
    }


# ---------------------------------------------------------------- cards ----


def _render_case(case: dict) -> None:
    """Render one case card: header, verdict pill, evidence, AI, provenance."""
    accent, verdict_label = _VERDICT[case["verdict"]]
    cat = aqi.category(case["peak"])

    pm_color = aqi.color(case["peak"])
    with st.container(border=True):
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:flex-end;"
            f"gap:12px;flex-wrap:wrap'>"
            f"<div style='font-size:1.5rem;font-weight:800;line-height:1.1'>{case['date_th']}"
            f"<span style='font-size:1.05rem;font-weight:600;opacity:.75'> · "
            f"{case['station']}</span></div>"
            f"<div style='text-align:right;line-height:1.15'>"
            f"<span style='font-size:.8rem;opacity:.7'>PM2.5 สูงสุด</span><br>"
            f"<span style='font-size:1.6rem;font-weight:800'>{case['peak']:.0f}</span>"
            f"<span style='font-size:.9rem;opacity:.7'> µg/m³</span> &nbsp;"
            f"<span style='background:{pm_color};color:#111;padding:1px 8px;border-radius:6px;"
            f"font-size:.8rem;font-weight:700;white-space:nowrap'>{cat}</span></div></div>"
            f"<div style='margin-top:10px'><span style='background:{accent};color:#fff;"
            f"padding:5px 16px;border-radius:999px;font-size:1.1rem;font-weight:700'>"
            f"ที่มา: {verdict_label}</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='margin:12px 0 2px;font-weight:600;opacity:.85'>หลักฐาน</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_fire_block(case), unsafe_allow_html=True)
        if case["wind_hours"]:
            mm, tot = case["wind_hours"]
            pct = round(100 * mm / tot)
            st.markdown(
                f"<div style='margin-top:6px'><b style='color:{_THAI}'>ลม</b> "
                f"<span style='opacity:.6;font-size:.8rem'>· ERA5</span> &nbsp; "
                f"48 ชม.ก่อนหน้า ลมพัดผ่านเมียนมา <b>{pct}%</b> ของเวลา</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='border-left:3px solid {accent};background:rgba(130,130,130,.10);"
            f"padding:8px 12px;margin:10px 0;border-radius:4px'>{_why_line(case)}</div>",
            unsafe_allow_html=True,
        )

        _ai_row(case)
        if case.get("date_iso"):
            st.plotly_chart(_event_map(case), width="stretch", key=f"map_{case['key']}")
        _provenance(case)


def _event_map(case: dict) -> go.Figure:
    """Offline map: fire hotspots (blue TH / orange foreign) + wind path + station.

    Uses ``style="white-bg"`` (no map tiles, so it works without internet at the
    venue) with national borders drawn from the committed geojson. Marker area
    grows with FRP so the eye lands on the strongest fires.
    """
    hs = da.hotspots_for_date(case["date_iso"])
    lat0, lon0 = case["station_lat"], case["station_lon"]
    fig = go.Figure()

    groups = [("ไฟในไทย", {"Thailand"}, _THAI), ("ไฟต่างชาติ", {"Myanmar", "Laos"}, _FOREIGN)]
    for name, countries, colour in groups:
        sub = hs[hs["country"].isin(countries)]
        if len(sub) == 0:
            continue
        fig.add_trace(
            go.Scattermapbox(
                lat=sub["latitude"].tolist(),
                lon=sub["longitude"].tolist(),
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=[max(5, min(26, 4 + math.sqrt(max(f, 0)) * 1.1)) for f in sub["frp"]],
                    color=colour,
                    opacity=0.8,
                ),
                name=name,
                text=[f"FRP {f:.0f} MW" for f in sub["frp"]],
                hoverinfo="text",
            )
        )

    if case["traj"]:
        fig.add_trace(
            go.Scattermapbox(
                lat=[p[0] for p in case["traj"]],
                lon=[p[1] for p in case["traj"]],
                mode="lines",
                line=dict(width=2.5, color="#555"),
                name="เส้นทางลม 48 ชม. (ฝุ่นลอยมาตามนี้)",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scattermapbox(
            lat=[lat0],
            lon=[lon0],
            mode="markers",
            marker=go.scattermapbox.Marker(size=15, color="#111"),
            name=f"สถานี {case['station']}",
            hoverinfo="name",
        )
    )
    fig.update_layout(
        mapbox=dict(
            style="white-bg",
            center=dict(lat=lat0, lon=lon0 - 0.3),
            zoom=5.6,
            layers=[
                dict(
                    sourcetype="geojson",
                    source=geo.border_geojson(),
                    type="line",
                    color="rgba(90,90,90,.6)",
                    line=dict(width=1.4),
                )
            ],
        ),
        margin=dict(l=0, r=0, t=6, b=0),
        height=360,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,.8)",
            font=dict(size=11),
        ),
    )
    return fig


def _fire_block(case: dict) -> str:
    """Fire evidence: a headline plus a two-colour proportion bar (2px gap)."""
    thai, foreign = case["thai_frp"], case["foreign_frp"]
    total = thai + foreign
    if total <= 0:
        return f"<b style='color:{_FOREIGN}'>ไฟ</b> · ไม่มีจุดไฟที่เชื่อมโยงในเหตุการณ์นี้"
    thai_pct, foreign_pct = 100 * thai / total, 100 * foreign / total
    if max(thai, foreign) / max(min(thai, foreign), 1e-9) < 1.3:
        headline = "ไฟทั้งสองฝั่ง<b>พอ ๆ กัน</b>"
    else:
        side = "ต่างชาติ" if foreign > thai else "ในไทย"
        ratio = max(thai, foreign) / max(min(thai, foreign), 1e-9)
        headline = f"ไฟฝั่ง<b>{side} แรงกว่า ~{ratio:.0f} เท่า</b>"
    thai_txt = f"ไทย {thai:.0f}" if thai_pct >= 14 else ""
    foreign_txt = f"ต่างชาติ {foreign:.0f}" if foreign_pct >= 14 else ""
    seg = (
        "display:flex;align-items:center;justify-content:center;color:#fff;"
        "font-size:.75rem;font-weight:600;white-space:nowrap;overflow:hidden;border-radius:3px"
    )
    return (
        f"<div><b style='color:{_FOREIGN}'>ไฟ</b> "
        f"<span style='opacity:.6;font-size:.8rem'>· NASA FIRMS</span> &nbsp; {headline}</div>"
        f"<div style='display:flex;gap:2px;height:20px;margin:5px 0'>"
        f"<div style='width:{thai_pct:.0f}%;background:{_THAI};{seg}'>{thai_txt}</div>"
        f"<div style='width:{foreign_pct:.0f}%;background:{_FOREIGN};{seg}'>{foreign_txt}</div></div>"
        f"<div style='font-size:.72rem;opacity:.55'>ความแรงไฟ (FRP, เมกะวัตต์) "
        f"ของจุดไฟที่ลมเชื่อมมาถึงสถานี</div>"
    )


def _why_line(case: dict) -> str:
    """One-sentence physical explanation combining wind + fire."""
    v = case["verdict"]
    wind = case["wind_hours"]
    wind_pct = round(100 * wind[0] / wind[1]) if wind else None
    if v == "foreign":
        return "<b>สรุป:</b> ลมมาจากเมียนมา และ ไฟฝั่งต่างประเทศแรงกว่า → ฝุ่นก้อนนี้มาจากต่างประเทศเป็นหลัก"
    if v == "borderline":
        return (
            "<b>สรุป:</b> ไฟสองฝั่งพอ ๆ กัน และลมพัดผ่านทั้งสองฝั่ง → "
            "ชี้ชัดไม่ได้ว่ามาจากในหรือนอกประเทศ (เหตุการณ์ผสม)"
        )
    if wind_pct is not None and wind_pct >= 45:
        return (
            f"<b>สรุป:</b> แม้ลมพัดมาจากเมียนมา {wind_pct}% แต่ไฟอยู่ในไทย ไม่ได้อยู่เมียนมา → "
            "ฝุ่นก้อนนี้เป็นของในประเทศ (ลมอย่างเดียวไม่พอ ต้องมีไฟด้วย)"
        )
    return "<b>สรุป:</b> ไฟกระจุกอยู่ในไทยเป็นหลัก → ฝุ่นก้อนนี้เป็นของในประเทศ"


def _ai_row(case: dict) -> None:
    """AI as a corroborating witness — a quiet chip plus a plain sentence.

    Distinguishes *source attribution* (this) from forecast accuracy, and frames
    instability as "agrees on direction, can't pin the exact share" so a reader
    never mistakes it for a bad forecast.
    """
    mean = case["model_mean"]

    if case["model_unstable"]:
        chip = "AI ยังไม่ฟันธงสัดส่วน"
        dir_word = "ต่างประเทศ" if mean >= 50 else "เหตุการณ์ผสม"
        body = (
            f"โมเดลก็มองว่าเป็น{dir_word}เช่นกัน แต่ยังบอก “สัดส่วนเป๊ะ” ไม่ได้ "
            f"(โมเดล 3 ตัวให้ค่าต่างกันมาก {case['model_lo']:.0f}–{case['model_hi']:.0f}%) — "
            "จึงยึดไฟและลมเป็นตัวตัดสิน ใช้ AI เป็นเสียงสนับสนุน"
        )
    else:
        chip = "AI เห็นตรงกัน"
        direction = "ต่างประเทศ" if mean >= 50 else "ในประเทศ"
        shown = mean if mean >= 50 else 100 - mean
        detail = case.get("model_note") or "โมเดลทุกตัวเห็นตรงกัน"
        body = f"โมเดลก็ชี้ว่าเป็นฝุ่น{direction} {shown:.0f}% — {detail}"

    st.markdown(
        f"<div style='margin-top:4px'>"
        f"<span style='border:1px solid rgba(130,130,130,.5);border-radius:999px;"
        f"padding:1px 9px;font-size:.78rem;font-weight:600;opacity:.85'>{chip}</span> "
        f"<span style='opacity:.85'>{body}</span></div>",
        unsafe_allow_html=True,
    )


def _provenance(case: dict) -> None:
    """Expandable raw-numbers block so a judge can verify nothing was invented."""
    with st.expander("ตัวเลขนี้มาจากไหน?"):
        frp = case["frp_detail"]
        st.markdown(
            "**ไฟ — NASA FIRMS (ดาวเทียม VIIRS/MODIS):** ความแรงไฟที่ลมเชื่อมมาถึงสถานี  \n"
            + " · ".join(f"{k} {v:,.0f} MW" for k, v in frp.items())
        )
        if case["wind_hours"]:
            mm, tot = case["wind_hours"]
            st.markdown(
                f"**ลม — ERA5 (ECMWF):** วิถีลมย้อนหลัง 48 ชม. อยู่เหนือเมียนมา "
                f"{mm} จาก {tot} ชม. ({round(100 * mm / tot)}%)"
            )
        seeds = " · ".join(f"{lbl}: {v:.0f}%" for lbl, v in case["per_seed"].items())
        st.markdown(f"**โมเดล AI — สัดส่วนต่างประเทศต่อ checkpoint:** {seeds}")
        st.caption(f"ไฟล์ผลลัพธ์: {case['src_file']}")


def _policy_simulator(cases: list[dict]) -> None:
    """Interactive 'which lever helps' panel — transparent arithmetic on fire load.

    Not a dispersion model: it scales the *connected fire load* (FRP that the
    wind links to the station) by a hypothetical suppression rate, to show which
    source a policy should target. Framed honestly as a first-order estimate.
    """
    st.markdown("### 🧪 จำลองนโยบาย: ถ้าลดไฟที่ต้นตอ จะช่วยได้แค่ไหน?")
    st.caption(
        "เลื่อนดูว่า “ถ้าดับไฟฝั่งใดฝั่งหนึ่งได้” จะตัดต้นตอฝุ่นที่ลอยมาถึงสถานีลงเท่าไร — "
        "ช่วยให้เห็นว่ามาตรการควรมุ่งไปที่ไหน"
    )

    labels = {c["key"]: f"{c['date_th']} · {c['station']}" for c in cases}
    by_key = {c["key"]: c for c in cases}
    pick = st.selectbox(
        "เลือกเหตุการณ์", list(labels), format_func=lambda k: labels[k], key="policy_event"
    )
    case = by_key[pick]
    thai, foreign = case["thai_frp"], case["foreign_frp"]
    total = thai + foreign

    col1, col2 = st.columns([1, 1])
    lever = col1.radio(
        "มาตรการ",
        ["ควบคุมไฟในประเทศ (ไทย)", "เจรจา/ควบคุมไฟข้ามแดน (เมียนมา/ลาว)"],
        key="policy_lever",
    )
    rate = col2.slider("สมมติดับไฟฝั่งนี้ได้ (%)", 0, 100, 70, step=5, key="policy_rate")

    removed = (thai if "ในประเทศ" in lever else foreign) * rate / 100
    reduction = 100 * removed / total if total > 0 else 0.0
    col2.metric("ต้นตอฝุ่นที่เชื่อมโยง (FRP) ลดลง", f"{reduction:.0f}%")

    best = "ต่างประเทศ" if foreign > thai else "ในประเทศ"
    st.info(
        f"สำหรับเหตุการณ์นี้ ต้นตอไฟส่วนใหญ่อยู่ฝั่ง **{best}** — "
        f"มาตรการที่ได้ผลมากที่สุดคือการลดไฟฝั่งนั้น "
        f"(ดับไฟฝั่ง{best}หมด = ตัดต้นตอ ~{100 * (foreign if best == 'ต่างประเทศ' else thai) / total:.0f}%). "
        "นี่คือคุณค่าของการรู้แหล่งที่มา: ทุ่มทรัพยากรให้ตรงจุด",
        icon="🎯",
    )
    st.caption(
        "⚠️ ประมาณการเบื้องต้นจากปริมาณไฟ (FRP) ที่ลมเชื่อมมาถึงสถานี — "
        "ไม่ใช่การจำลองการฟุ้งกระจาย/เคมีบรรยากาศเต็มรูปแบบ"
    )


def _closing_honesty() -> None:
    """Footer note explaining the model's role and why evidence leads."""
    st.info(
        "**“โมเดล AI ยังไม่ฟันธงสัดส่วน” หมายความว่าอะไร?** ส่วนนี้คือการให้ AI ช่วย"
        "*ระบุแหล่งที่มา* (คนละเรื่องกับการพยากรณ์ค่าฝุ่น). เมื่อ AI ไม่ฟันธง แปลว่ามันบอก"
        "*ทิศทาง*ได้ (ในหรือนอกประเทศ) แต่ยังไม่แน่ใจว่ากี่เปอร์เซ็นต์พอดี — ไม่ได้แปลว่า"
        "พยากรณ์ผิด. เราจึงให้หลักฐานที่วัดได้จริง (ไฟจากดาวเทียม + ลมจากอากาศจริง) เป็นตัวตัดสิน "
        "และรายงานความไม่แน่นอนของ AI ตามจริง ไม่กลบไว้",
        icon="💡",
    )
