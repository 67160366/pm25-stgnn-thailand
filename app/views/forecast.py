# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Forecast Explorer: per-station observed history + MTGNN multi-horizon forecast.

Two modes: a hindcast replay on stored ERA5 windows, and a live "forecast from
now" mode driven by air4thai PM2.5 + Open-Meteo NWP (honest caveats attached).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.lib import data_access as da
from app.lib import inference as inf
from app.lib import ui

_HINDCAST = "ย้อนหลัง (ERA5 reanalysis)"
_LIVE = "พยากรณ์วันนี้ (NWP สด)"


def render() -> None:
    """Render the per-station forecast explorer with a mode toggle."""
    ui.page_title(
        "พยากรณ์รายสถานี", "เลือกสถานีเพื่อดูค่าจริงย้อนหลังและพยากรณ์ 6 / 12 / 24 / 48 ชม."
    )
    mode = st.radio("โหมดพยากรณ์", [_HINDCAST, _LIVE], horizontal=True)
    if mode == _LIVE:
        _render_live()
    else:
        _render_hindcast()


def _render_hindcast() -> None:
    """Hindcast replay over stored ERA5 windows (the original behaviour)."""
    split = st.session_state["split"]
    anchor_iso = st.session_state["anchor_iso"]
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

    hw = inf.conformal_halfwidths(da.load_conformal(), sid, horizons)
    interval = _interval_from_halfwidths(pred, hw)
    hist = inf.station_history(split, sid, anchor_iso, hours_back=168)
    fig = ui.forecast_overlay_figure(
        hist, pred.tolist(), pers, anchor_ts, horizons, sel_name, interval=interval
    )
    st.plotly_chart(fig, width="stretch")

    _forecast_metrics(horizons, pred, hw)
    _conformal_caption()

    st.caption(
        "หมายเหตุ (ตามจริง): ในช่วงสั้น 6–12 ชม. ค่า baseline persistence มักแม่นกว่าโมเดล "
        "เพราะ PM2.5 มี autocorrelation สูง — จุดเด่นเชิงพยากรณ์ของระบบอยู่ที่ 48 ชม. "
        "สำหรับการเตือนภัยล่วงหน้า เราจึงเสนอใช้แบบ hybrid (persistence ช่วงสั้น + MTGNN ช่วงยาว)"
    )


def _render_live() -> None:
    """Live forecast-from-now using air4thai + Open-Meteo (honest NWP caveats)."""
    st.info(
        "โหมดสด: ใช้ค่าฝุ่นจริงล่าสุดจาก air4thai และพยากรณ์อากาศ (NWP) จาก Open-Meteo "
        "**ไม่ใช่ ERA5 reanalysis** ที่ใช้ตอนเทรน — โมเดลถูกฝึกด้วย ERA5 จึงมีความคลาดเคลื่อน "
        "ของ input ต่างจากตอนวัดผล ดูหลักฐานความทนต่อ noise ของ NWP ได้ที่หน้า "
        "‘ผลทดสอบ & ความซื่อสัตย์’ (nwp_sensitivity)",
        icon="🛰️",
    )
    st.caption(
        "หน้าต่าง input = 24 ชม. ล่าสุด; หากชั่วโมงหายจะเติมแบบ forward-fill และปฏิเสธสถานี "
        "ที่มีข้อมูลครบน้อยกว่า 70% (เติมด้วยค่ากลางของสถานีแทน). "
        "ไฟป่า (FIRMS) ไม่ถูกนำเข้ากราฟในโหมดนี้ → attribution ไม่พร้อมใช้ในโหมดสด"
    )

    try:
        fc = inf.live_forecast()
    except inf.LiveDataError as exc:
        st.error(str(exc))
        return

    anchor_ts = pd.Timestamp(fc["anchor_iso"])
    st.success(f"จุดเริ่มพยากรณ์ (ตอนนี้): **{anchor_ts:%d %b %Y, %H:%M} UTC**")

    meta = da.load_stations_meta()
    sids: list[int] = fc["station_ids"]
    excluded = set(fc["excluded_ids"])
    if excluded:
        excl_names = meta[meta["station_id"].isin(excluded)]["name"].tolist()
        st.warning(
            "สถานีต่อไปนี้ไม่อยู่ในฟีด air4thai หรือมีข้อมูลสดไม่พอ จึงไม่แสดงพยากรณ์โหมดสด "
            "(ใช้ค่ากลางแทนในกราฟเท่านั้น): " + ", ".join(excl_names)
        )

    # Excluded stations are dropped from the selector entirely: their whole 24h
    # PM2.5 input is synthetic median-fill, so a "forecast" for them would mislead
    # even with a caption (own-station history is the dominant signal).
    available = meta[~meta["station_id"].isin(excluded)]
    if available.empty:
        st.error("ไม่มีสถานีที่มีข้อมูลสดเพียงพอในขณะนี้ — โปรดลองใหม่ภายหลัง")
        return

    sel_name = st.selectbox("เลือกสถานี", available["name"].tolist())
    sid = int(available[available["name"] == sel_name].iloc[0]["station_id"])
    i = sids.index(sid)

    pred = np.asarray(fc["pred_ug"], dtype=float)[i]
    horizons: list[int] = fc["horizons"]
    observed = float(fc["observed"][i]) if np.isfinite(fc["observed"][i]) else float("nan")

    slots = pd.to_datetime(fc["slots"])
    window_raw = np.asarray(fc["window_raw"], dtype=float)[i]
    hist = pd.DataFrame({"timestamp": slots, "pm25": window_raw})
    pers = observed if np.isfinite(observed) else 0.0
    hw = inf.conformal_halfwidths(da.load_conformal(), sid, horizons)
    interval = _interval_from_halfwidths(pred, hw)
    fig = ui.forecast_overlay_figure(
        hist, pred.tolist(), pers, anchor_ts, horizons, sel_name, interval=interval
    )
    st.plotly_chart(fig, width="stretch")

    cov = fc["coverage"].get(sid, fc["coverage"].get(str(sid), 0.0))
    st.caption(f"ความครบของข้อมูล PM2.5 ในหน้าต่าง 24 ชม.: {cov * 100:.0f}%")

    _forecast_metrics(horizons, pred, hw)
    _conformal_caption(live=True)


def _interval_from_halfwidths(
    pred: np.ndarray, halfwidths: list[float] | None
) -> tuple[list[float], list[float]] | None:
    """Build (lower, upper) bounds from point forecasts and conformal half-widths.

    Lower bound is clipped at 0 (PM2.5 cannot be negative). Returns ``None`` when
    no half-widths are available so the band is simply omitted.
    """
    if halfwidths is None:
        return None
    pred_list = np.asarray(pred, dtype=float).tolist()
    lower = [max(0.0, p - q) for p, q in zip(pred_list, halfwidths, strict=True)]
    upper = [p + q for p, q in zip(pred_list, halfwidths, strict=True)]
    return lower, upper


def _conformal_caption(live: bool = False) -> None:
    """Explain the 90% conformal band and keep the honest coverage caveat visible."""
    msg = (
        "แถบสีแดงอ่อน = ช่วงพยากรณ์ 90% แบบ split-conformal ปรับเทียบจากชุด validation ปี 2024 "
        "(residual จริงหน่วย µg/m³ แยกตาม horizon และรายสถานี) — การันตี marginal coverage "
        "บน distribution เดียวกับ val; วัดจริงบนชุด test ปี 2025 (held-out) ได้ 0.87–0.88 "
        "ต่ำกว่า nominal 0.90 เล็กน้อย"
    )
    if live:
        msg += (
            ". หมายเหตุโหมดสด: ช่วงนี้ปรับเทียบบน input ERA5 ปี 2024 แต่โหมดสดใช้ NWP "
            "จึงอาจ cover คลาดเคลื่อน (mis-cover) ได้"
        )
    st.caption(msg)


def _forecast_metrics(
    horizons: list[int], pred: np.ndarray, halfwidths: list[float] | None = None
) -> None:
    """Per-horizon metric cards + AQI badges + optional 90% range, shared by both modes."""
    st.subheader("ค่าพยากรณ์รายช่วงเวลา")
    cols = st.columns(len(horizons))
    for idx, (col, h, val) in enumerate(zip(cols, horizons, pred, strict=True)):
        col.metric(f"+{h} ชม.", f"{val:.0f} µg/m³")
        col.markdown(ui.aqi_badge_md(float(val)))
        if halfwidths is not None:
            q = halfwidths[idx]
            lo = max(0.0, float(val) - q)
            hi = float(val) + q
            col.caption(f"90%: {lo:.0f}–{hi:.0f}")
