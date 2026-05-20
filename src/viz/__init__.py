# [TODO: NSC Disclaimer — see booklet page 44]

"""Visualisation module for PM2.5 STGNN.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Public API:
    maps.station_map               — scatter mapbox with PM2.5 colour scale
    maps.attribution_bar_map       — country attribution horizontal bar chart
    maps.feature_importance_bar    — IG feature importance bar chart
    timeseries.forecast_vs_actual  — line chart with AQI threshold bands
    timeseries.multi_station_heatmap — heatmap across stations and time
    timeseries.error_metrics_table — RMSE/MAE/R2 table by horizon
"""

from src.viz.maps import attribution_bar_map, feature_importance_bar, station_map
from src.viz.timeseries import error_metrics_table, forecast_vs_actual, multi_station_heatmap

__all__ = [
    "attribution_bar_map",
    "error_metrics_table",
    "feature_importance_bar",
    "forecast_vs_actual",
    "multi_station_heatmap",
    "station_map",
]
