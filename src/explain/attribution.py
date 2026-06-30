# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""High-level attribution helpers for PM2.5 source analysis.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Thin wrapper around ``gb_ig`` and ``gnn_explainer`` that handles:
- Loading per-hotspot country labels from HeteroData
- Summarising IG attributions into a human-readable station report
- Batch attribution over multiple timestamps
"""

from __future__ import annotations

import logging

import torch
from torch_geometric.data import HeteroData

from src.explain.gb_ig import integrated_gradients, occlusion_country_attribution
from src.models.base import PM25ModelBase

logger = logging.getLogger(__name__)

_FEATURE_NAMES: list[str] = [
    "pm25_scaled",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "u10",
    "v10",
    "t2m",
    "d2m",
    "blh",
]


def load_hotspot_countries(data: HeteroData) -> list[str]:
    """Extract country labels for each hotspot node from HeteroData.

    The loader stores country labels as ``data["hotspot"].country`` — a plain
    Python list set during graph construction from the FIRMS hotspot DataFrame.

    Args:
        data: HeteroData from PM25GraphDataset; must have ``data["hotspot"].country``.

    Returns:
        List of country strings, one per hotspot node. Empty list if no hotspots.
    """
    countries = getattr(data["hotspot"], "country", [])
    return list(countries)


def station_source_report(
    model: PM25ModelBase,
    data: HeteroData,
    station_idx: int,
    horizon_idx: int,
    station_name: str = "",
    n_ig_steps: int = 50,
    device: str | torch.device = "cpu",
) -> dict:
    """Generate a combined IG + occlusion attribution report for one station.

    Runs both Integrated Gradients (feature importance) and occlusion-based
    country attribution, then summarises into a single report dict.

    Args:
        model: Trained PM25ModelBase.
        data: Single-sample HeteroData.
        station_idx: Station index to explain.
        horizon_idx: Forecast horizon index to explain.
        station_name: Optional human-readable station name for the report.
        n_ig_steps: Riemann sum steps for IG (50 default, 100 for publication).
        device: Torch device.

    Returns:
        Dict with keys:
            - ``"station_idx"``: int
            - ``"station_name"``: str
            - ``"horizon_idx"``: int
            - ``"ig_feature_importance"``: dict mapping feature name to mean |attr|
            - ``"country_attribution"``: dict mapping country to normalized score in [0, 1]
    """
    ig_attrs = integrated_gradients(
        model=model,
        data=data,
        target_station_idx=station_idx,
        target_horizon_idx=horizon_idx,
        n_steps=n_ig_steps,
        device=device,
    )  # {"station_x": (N, T_in, F)}

    station_x_attr: torch.Tensor = ig_attrs["station_x"]  # (N, T_in, F)
    mean_abs = station_x_attr.abs().mean(dim=(0, 1))  # (F,)
    n_feat = mean_abs.shape[0]
    feat_names = _FEATURE_NAMES[:n_feat]
    ig_summary = {feat_names[i]: float(mean_abs[i]) for i in range(n_feat)}

    hotspot_countries = load_hotspot_countries(data)
    country_attr = occlusion_country_attribution(
        model=model,
        data=data,
        target_station_idx=station_idx,
        target_horizon_idx=horizon_idx,
        hotspot_countries=hotspot_countries,
        device=device,
    )

    return {
        "station_idx": station_idx,
        "station_name": station_name,
        "horizon_idx": horizon_idx,
        "ig_feature_importance": ig_summary,
        "country_attribution": country_attr,
    }


def batch_attribution(
    model: PM25ModelBase,
    samples: list[HeteroData],
    station_idx: int,
    horizon_idx: int,
    station_name: str = "",
    n_ig_steps: int = 50,
    device: str | torch.device = "cpu",
) -> list[dict]:
    """Run ``station_source_report`` over a list of samples.

    Useful for aggregating attribution scores across multiple forecast windows
    to identify persistent cross-border contributions.

    Args:
        model: Trained PM25ModelBase.
        samples: List of single-sample HeteroData objects.
        station_idx: Station index to explain.
        horizon_idx: Forecast horizon index.
        station_name: Optional human-readable station name.
        n_ig_steps: IG steps per sample.
        device: Torch device.

    Returns:
        List of report dicts (one per sample), same structure as
        ``station_source_report``.
    """
    reports: list[dict] = []
    for i, sample in enumerate(samples):
        logger.debug("batch_attribution: sample %d / %d", i + 1, len(samples))
        report = station_source_report(
            model=model,
            data=sample,
            station_idx=station_idx,
            horizon_idx=horizon_idx,
            station_name=station_name,
            n_ig_steps=n_ig_steps,
            device=device,
        )
        reports.append(report)
    return reports
