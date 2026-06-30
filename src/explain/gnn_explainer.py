# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Gradient x Input baseline explainer for PM2.5 STGNN.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Implements a lightweight Gradient x Input saliency method as a fast
alternative to Integrated Gradients when runtime is constrained.
This is a single-point approximation: attr = grad(f) * x, evaluated at
the actual input rather than along a path. It does NOT satisfy the
completeness axiom but is orders of magnitude faster than IG.

Reference:
    Baehrens et al. (2010) "How to Explain Individual Classification Decisions"
    Gradient x Input is reviewed in Ancona et al. (2018) arXiv:1711.06104.
"""

from __future__ import annotations

import logging

import torch
from torch_geometric.data import HeteroData

from src.models.base import PM25ModelBase

logger = logging.getLogger(__name__)


def gradient_x_input(
    model: PM25ModelBase,
    data: HeteroData,
    target_station_idx: int,
    target_horizon_idx: int,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Compute Gradient x Input saliency for station node features.

    Single forward+backward pass. Fast but does not satisfy completeness.
    Use Integrated Gradients (``gb_ig.integrated_gradients``) for
    publication-quality attributions.

    Args:
        model: Trained PM25ModelBase instance (eval mode used internally).
        data: Single-sample HeteroData (un-batched). Station x shape (N, T_in, F).
        target_station_idx: Index in [0, N) identifying the station to explain.
        target_horizon_idx: Index in [0, H) identifying the forecast horizon.
        device: Torch device for computation.

    Returns:
        Dict with single key ``"station_x"``: FloatTensor of shape (N, T_in, F).
        Values are grad * input; magnitude indicates per-feature importance.
    """
    device = torch.device(device)
    model = model.eval().to(device)
    data = data.to(device)

    x = data["station"].x.detach().requires_grad_(True)
    data["station"].x = x

    pred = model(data)
    scalar = pred[target_station_idx, target_horizon_idx]
    scalar.backward()

    attr: torch.Tensor | None = None
    with torch.no_grad():
        if x.grad is not None:
            attr = (x.grad * x).detach().cpu()
        else:
            logger.warning("gradient_x_input: grad is None — returning zeros")
            attr = torch.zeros_like(x).cpu()

    model.zero_grad()
    data["station"].x = x.detach()

    return {"station_x": attr}
