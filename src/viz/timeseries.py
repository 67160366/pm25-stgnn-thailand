# [TODO: NSC Disclaimer — see booklet page 44]

"""Time-series plotting utilities for PM2.5 STGNN visualisation.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

All functions return Plotly ``go.Figure`` objects for use with Streamlit.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Thai PCD PM2.5 AQI thresholds (µg/m³)
_AQI_BANDS = [
    (0, 25, "Good", "rgba(0,228,0,0.08)"),
    (25, 37.5, "Moderate", "rgba(255,255,0,0.08)"),
    (37.5, 50, "Unhealthy (Sensitive)", "rgba(255,126,0,0.08)"),
    (50, 90, "Unhealthy", "rgba(255,0,0,0.08)"),
    (90, 150, "Very Unhealthy", "rgba(143,63,151,0.08)"),
]


def forecast_vs_actual(
    timestamps: list,
    actual: list[float],
    forecasts: dict[str, list[float]],
    station_name: str = "",
    show_aqi_bands: bool = True,
) -> go.Figure:
    """Line chart comparing actual PM2.5 measurements against model forecasts.

    Args:
        timestamps: List of datetime-like objects for the x-axis.
        actual: Observed PM2.5 values (µg/m³) aligned with ``timestamps``.
        forecasts: Dict mapping horizon label (e.g. ``"6h"``) to list of
            predicted values aligned with ``timestamps``.
        station_name: Optional station name for the title.
        show_aqi_bands: If True, add horizontal Thai PCD AQI threshold bands.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    if show_aqi_bands:
        y_max = max(
            (max(actual) if actual else 0),
            *(max(v) for v in forecasts.values() if v),
            50.0,
        )
        for lo, hi, label, colour in _AQI_BANDS:
            if lo > y_max * 1.1:
                break
            fig.add_hrect(
                y0=lo,
                y1=min(hi, y_max * 1.2),
                fillcolor=colour,
                line_width=0,
                annotation_text=label,
                annotation_position="right",
                annotation_font_size=9,
            )

    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=actual,
            mode="lines",
            name="Observed",
            line=dict(color="black", width=2),
        )
    )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (label, values) in enumerate(forecasts.items()):
        fig.add_trace(
            go.Scatter(
                x=list(timestamps),
                y=values,
                mode="lines",
                name=f"Forecast {label}",
                line=dict(color=palette[i % len(palette)], dash="dash"),
            )
        )

    title = "PM2.5 Forecast vs Actual"
    if station_name:
        title += f" — {station_name}"

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="PM2.5 (µg/m³)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def multi_station_heatmap(
    df: pd.DataFrame,
    station_col: str = "station_id",
    time_col: str = "timestamp",
    value_col: str = "pm25",
    title: str = "PM2.5 Heatmap — All Stations",
) -> go.Figure:
    """Heatmap of PM2.5 across all stations and time.

    Rows are stations, columns are timestamps. Useful for spotting
    simultaneous pollution events (haze episodes).

    Args:
        df: Long-format DataFrame with station, time, and PM2.5 columns.
        station_col: Column name for station identifier.
        time_col: Column name for timestamp.
        value_col: Column name for PM2.5 concentration (µg/m³).
        title: Figure title.

    Returns:
        Plotly heatmap figure.
    """
    pivot = df.pivot_table(index=station_col, columns=time_col, values=value_col, aggfunc="mean")

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=_AQI_BANDS[-1][1],
            colorbar=dict(title="PM2.5 (µg/m³)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Station",
        height=max(300, 30 * len(pivot) + 100),
        margin=dict(l=120, r=20, t=50, b=60),
    )
    return fig


def error_metrics_table(
    metrics: dict[str, dict[str, float]],
) -> go.Figure:
    """Render a table of RMSE/MAE/R2 per forecast horizon.

    Args:
        metrics: Dict mapping horizon label (e.g. ``"6h"``) to inner dict
            with keys ``rmse``, ``mae``, ``r2``.

    Returns:
        Plotly table figure.
    """
    horizons = list(metrics.keys())
    rmse_vals = [f"{metrics[h].get('rmse', float('nan')):.3f}" for h in horizons]
    mae_vals = [f"{metrics[h].get('mae', float('nan')):.3f}" for h in horizons]
    r2_vals = [f"{metrics[h].get('r2', float('nan')):.3f}" for h in horizons]

    fig = go.Figure(
        go.Table(
            header=dict(
                values=["Horizon", "RMSE", "MAE", "R²"],
                fill_color="steelblue",
                font=dict(color="white"),
                align="center",
            ),
            cells=dict(
                values=[horizons, rmse_vals, mae_vals, r2_vals],
                fill_color="white",
                align="center",
            ),
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=180)
    return fig
