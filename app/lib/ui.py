# [TODO: NSC Disclaimer - see booklet page 44]
"""Shared UI building blocks for the dashboard (header, AQI legend, badges, footer).

Keeps look-and-feel and the honesty captions (hindcast note, NSC disclaimer) in one place.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib import aqi
from src.viz.timeseries import forecast_vs_actual

DISCLAIMER = "[TODO: NSC Disclaimer - see booklet page 44]"


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
    fy = [last_obs] + list(pred_station)
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
    """Shared footer with branding + NSC disclaimer placeholder."""
    st.divider()
    st.caption(
        "ระบบพยากรณ์ PM2.5 และวิเคราะห์แหล่งกำเนิด (Explainable STGNN) · "
        f"NSC 2026 หมวด 14 · {DISCLAIMER}"
    )
