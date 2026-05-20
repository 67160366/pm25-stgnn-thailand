# [TODO: NSC Disclaimer — see booklet page 44]

"""Choropleth and scatter map utilities for PM2.5 STGNN visualisation.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

All functions return Plotly ``go.Figure`` objects so they can be embedded
directly in Streamlit (``st.plotly_chart``) without extra rendering code.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# AQI colour scale aligned with Thai PCD breakpoints (µg/m³)
_PM25_COLORSCALE = [
    [0.00, "#00e400"],  # Good         0-25
    [0.20, "#ffff00"],  # Moderate     25-37.5
    [0.40, "#ff7e00"],  # Unhealthy (sensitive)  37.5-50
    [0.60, "#ff0000"],  # Unhealthy    50-90
    [0.80, "#8f3f97"],  # Very unhealthy  90-150
    [1.00, "#7e0023"],  # Hazardous    150+
]

_MAX_COLORSCALE_PM25 = 150.0  # µg/m³ ceiling for colour mapping


def station_map(
    stations_meta: pd.DataFrame,
    pm25_values: pd.Series | None = None,
    title: str = "PM2.5 Station Map — Northern Thailand",
    hotspots: pd.DataFrame | None = None,
) -> go.Figure:
    """Render station locations as a scatter mapbox with optional PM2.5 colour.

    Args:
        stations_meta: DataFrame with columns ``station_id``, ``lat``, ``lon``,
            ``name``, ``province``.
        pm25_values: Optional Series indexed by ``station_id`` with PM2.5 values
            in µg/m³. If None, all markers are drawn in a neutral colour.
        title: Figure title.
        hotspots: Optional DataFrame with columns ``latitude``, ``longitude``,
            ``frp`` for FIRMS hotspot overlay. Drawn as small orange markers.

    Returns:
        Plotly figure ready for ``st.plotly_chart``.
    """
    fig = go.Figure()

    if pm25_values is not None:
        merged = stations_meta.set_index("station_id").join(pm25_values.rename("pm25"), how="left")
        marker_color = merged["pm25"].fillna(0.0).tolist()
        cmax = _MAX_COLORSCALE_PM25
    else:
        merged = stations_meta.set_index("station_id")
        marker_color = "steelblue"
        cmax = None

    hover_texts = [
        f"{row.get('name', sid)}<br>{row.get('province', '')}<br>"
        + (f"PM2.5: {marker_color[i]:.1f} µg/m³" if pm25_values is not None else "")
        for i, (sid, row) in enumerate(merged.iterrows())
    ]

    fig.add_trace(
        go.Scattermapbox(
            lat=merged["lat"].tolist(),
            lon=merged["lon"].tolist(),
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=14,
                color=marker_color,
                colorscale=_PM25_COLORSCALE if pm25_values is not None else None,
                cmin=0,
                cmax=cmax,
                colorbar=dict(title="PM2.5 (µg/m³)") if pm25_values is not None else None,
            ),
            text=hover_texts,
            hoverinfo="text",
            name="Stations",
        )
    )

    if hotspots is not None and len(hotspots) > 0:
        fig.add_trace(
            go.Scattermapbox(
                lat=hotspots["latitude"].tolist(),
                lon=hotspots["longitude"].tolist(),
                mode="markers",
                marker=go.scattermapbox.Marker(size=6, color="orange", opacity=0.6),
                text=[f"FRP: {f:.0f}" for f in hotspots.get("frp", [0] * len(hotspots))],
                hoverinfo="text",
                name="Fire Hotspots",
            )
        )

    fig.update_layout(
        title=title,
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=18.8, lon=99.0),
            zoom=6,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    return fig


def attribution_bar_map(
    stations_meta: pd.DataFrame,
    attribution_scores: dict[str, float],
    station_id: str,
    title: str = "Source Country Attribution",
) -> go.Figure:
    """Bar chart of country attribution scores for a single station.

    Args:
        stations_meta: Used only to retrieve the station name for the subtitle.
        attribution_scores: Dict mapping country name to score in [0, 1].
        station_id: Station being explained.
        title: Chart title prefix.

    Returns:
        Plotly horizontal bar chart figure.
    """
    if not attribution_scores:
        fig = go.Figure()
        fig.add_annotation(text="No hotspot attribution available", showarrow=False)
        return fig

    station_name = ""
    if "name" in stations_meta.columns and "station_id" in stations_meta.columns:
        row = stations_meta[stations_meta["station_id"] == station_id]
        if len(row) > 0:
            station_name = row.iloc[0].get("name", "")

    countries = list(attribution_scores.keys())
    scores = [attribution_scores[c] * 100 for c in countries]  # percent

    colour_map = {
        "Thailand": "#2196F3",
        "Myanmar": "#FF5722",
        "Laos": "#4CAF50",
        "Cambodia": "#FF9800",
    }
    colours = [colour_map.get(c, "#9E9E9E") for c in countries]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=countries,
            orientation="h",
            marker_color=colours,
            text=[f"{s:.1f}%" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{title} — {station_name or station_id}",
        xaxis_title="Attribution (%)",
        xaxis=dict(range=[0, 110]),
        height=300,
        margin=dict(l=80, r=40, t=50, b=40),
    )
    return fig


def feature_importance_bar(
    ig_feature_importance: dict[str, float],
    title: str = "Feature Importance (IG)",
) -> go.Figure:
    """Horizontal bar chart of mean absolute IG attributions per feature.

    Args:
        ig_feature_importance: Dict from ``attribution.station_source_report``
            mapping feature name to mean |attr| value.
        title: Chart title.

    Returns:
        Plotly horizontal bar chart figure.
    """
    if not ig_feature_importance:
        fig = go.Figure()
        fig.add_annotation(text="No IG data available", showarrow=False)
        return fig

    sorted_items = sorted(ig_feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_items, strict=False)

    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=list(features),
            orientation="h",
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Mean |Attribution|",
        height=max(200, 40 * len(features) + 80),
        margin=dict(l=100, r=40, t=50, b=40),
    )
    return fig
