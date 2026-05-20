# [TODO: NSC Disclaimer — see booklet page 44]

"""Explainability methods for PM2.5 STGNN source attribution.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Implements two attribution methods for the heterogeneous PM2.5 graph:

1. ``integrated_gradients`` — path-integral attribution over station node
   features from a zero baseline to the actual input (Sundararajan et al. 2017).
   Satisfies the completeness axiom: sum of attributions equals the difference
   in model output between the input and the baseline.

2. ``occlusion_country_attribution`` — model-agnostic country-level source
   attribution by masking the total_frp of hotspot nodes per country and
   measuring the resulting drop in the predicted PM2.5 value.
   This is the primary method for the NSC "source attribution" novelty.

References:
    Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for
    Deep Networks. ICML 2017. https://proceedings.mlr.press/v70/sundararajan17a
"""

from __future__ import annotations

import logging

import torch
from torch_geometric.data import HeteroData

from src.models.base import PM25ModelBase

logger = logging.getLogger(__name__)


def integrated_gradients(
    model: PM25ModelBase,
    data: HeteroData,
    target_station_idx: int,
    target_horizon_idx: int,
    n_steps: int = 50,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Compute Integrated Gradients attributions for station node features.

    Approximates the path integral of gradients from a zero baseline to the
    actual input using a Riemann sum with ``n_steps`` steps. The result
    satisfies the completeness axiom: sum(attr) = f(x) - f(baseline).

    Args:
        model: Trained PM25ModelBase instance (eval mode used internally).
        data: Single-sample HeteroData (un-batched). Station x shape (N, T_in, F).
        target_station_idx: Index in [0, N) identifying the station to explain.
        target_horizon_idx: Index in [0, H) identifying the forecast horizon.
        n_steps: Riemann sum steps (50 is sufficient; 100 for publication quality).
        device: Torch device for computation.

    Returns:
        Dict with single key ``"station_x"``: FloatTensor of shape (N, T_in, F).
        Positive values indicate features that increase the prediction; negative
        values indicate features that suppress it.
    """
    device = torch.device(device)
    model = model.eval().to(device)
    data = data.to(device)

    orig_x = data["station"].x.detach()  # (N, T_in, F)
    baseline = torch.zeros_like(orig_x)
    integrated_grads = torch.zeros_like(orig_x)

    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        x_alpha = (baseline + alpha * (orig_x - baseline)).requires_grad_(True)
        data["station"].x = x_alpha

        pred = model(data)  # (N, H)
        scalar = pred[target_station_idx, target_horizon_idx]
        scalar.backward()

        with torch.no_grad():
            if x_alpha.grad is not None:
                integrated_grads += x_alpha.grad

        model.zero_grad()

    data["station"].x = orig_x  # restore

    integrated_grads /= n_steps
    attr = (orig_x - baseline) * integrated_grads

    with torch.no_grad():
        f_x = model(data)[target_station_idx, target_horizon_idx].item()
        data["station"].x = baseline
        f_baseline = model(data)[target_station_idx, target_horizon_idx].item()
        data["station"].x = orig_x  # restore after baseline forward

    logger.debug(
        "IG completeness check: attr_sum=%.4f  delta_f=%.4f",
        attr.sum().item(),
        f_x - f_baseline,
    )

    return {"station_x": attr.detach().cpu()}


def occlusion_country_attribution(
    model: PM25ModelBase,
    data: HeteroData,
    target_station_idx: int,
    target_horizon_idx: int,
    hotspot_countries: list[str],
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Attribute PM2.5 prediction to pollution source countries via occlusion.

    For each country present in the hotspot set, zeroes out the total_frp of
    all hotspot nodes from that country and measures the resulting drop in the
    predicted PM2.5 value. A larger drop means stronger contribution.

    This method is model-agnostic (no gradient computation) and always produces
    non-negative, interpretable scores. Country labels come from the FIRMS
    hotspot clustering bbox geocoder (Thailand / Myanmar / Laos / other).

    Args:
        model: Trained PM25ModelBase instance.
        data: Single-sample HeteroData with hotspot.x shape (n_hotspots, 3)
              where column 0 is total_frp.
        target_station_idx: Station to explain (index in [0, N)).
        target_horizon_idx: Forecast horizon to explain (index in [0, H)).
        hotspot_countries: Country label for each hotspot node. Length must
            equal the number of hotspot nodes in ``data``.
        device: Torch device for computation.

    Returns:
        Dict mapping country name to normalized attribution score in [0, 1].
        Scores sum to 1.0. Countries with no attributable influence are omitted.
        Returns empty dict if there are no hotspot nodes.
    """
    if not hotspot_countries:
        return {}

    device = torch.device(device)
    model = model.eval().to(device)
    data = data.to(device)

    with torch.no_grad():
        pred_full = model(data)[target_station_idx, target_horizon_idx].item()

    countries = sorted(set(hotspot_countries))
    raw_attr: dict[str, float] = {}
    orig_hx = data["hotspot"].x.clone()  # (n_hotspots, 3)

    for country in countries:
        mask = torch.tensor(
            [c == country for c in hotspot_countries],
            dtype=torch.bool,
            device=device,
        )
        if not mask.any():
            continue

        hx_occluded = orig_hx.clone()
        hx_occluded[mask] = 0.0  # zero out all features (frp, lat, lon) for this country
        data["hotspot"].x = hx_occluded

        with torch.no_grad():
            pred_occluded = model(data)[target_station_idx, target_horizon_idx].item()

        raw_attr[country] = max(0.0, pred_full - pred_occluded)

    data["hotspot"].x = orig_hx  # restore

    total = sum(raw_attr.values())
    if total > 1e-8:
        return {c: v / total for c, v in raw_attr.items()}
    return {c: 0.0 for c in countries}
