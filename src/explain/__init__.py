# [TODO: NSC Disclaimer — see booklet page 44]

"""Explainability module for PM2.5 STGNN.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Public API:
    gb_ig.integrated_gradients        — IG path-integral over station features
    gb_ig.occlusion_country_attribution — model-agnostic country source scores
    gnn_explainer.gradient_x_input    — fast single-pass Gradient x Input
    attribution.station_source_report  — combined IG + occlusion report
    attribution.batch_attribution      — run reports over multiple samples
"""

from src.explain.attribution import batch_attribution, station_source_report
from src.explain.gb_ig import integrated_gradients, occlusion_country_attribution
from src.explain.gnn_explainer import gradient_x_input

__all__ = [
    "batch_attribution",
    "gradient_x_input",
    "integrated_gradients",
    "occlusion_country_attribution",
    "station_source_report",
]
