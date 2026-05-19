# [TODO: NSC Disclaimer — see booklet page 44]

"""Loss functions for PM2.5 multi-horizon forecasting.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

from __future__ import annotations

import torch


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over valid (non-masked) positions only.

    Args:
        pred: Predictions of shape (B*N, H), normalized scale (pm25_scaled).
        target: Ground-truth values of shape (B*N, H), normalized scale.
        mask: Boolean validity mask of shape (B*N, H). True = valid.

    Returns:
        Scalar MSE tensor. Returns 0.0 if no valid positions exist.
    """
    if not mask.any():
        return pred.sum() * 0.0  # differentiable zero
    return ((pred - target) ** 2)[mask].mean()


def masked_smape(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """Symmetric mean absolute percentage error over valid positions.

    Uses the stabilised form: 2 * |y - ŷ| / (|y| + |ŷ| + eps) to avoid
    division by zero when PM2.5 scaled values are near zero.

    Args:
        pred: Predictions of shape (B*N, H), normalized scale (pm25_scaled).
        target: Ground-truth values of shape (B*N, H), normalized scale.
        mask: Boolean validity mask of shape (B*N, H). True = valid.
        eps: Denominator stability constant. Default 1.0.

    Returns:
        Scalar sMAPE tensor in [0, 2]. Returns 0.0 if no valid positions.
    """
    if not mask.any():
        return pred.sum() * 0.0
    denom = torch.abs(target) + torch.abs(pred) + eps
    smape = 2.0 * torch.abs(pred - target) / denom
    return smape[mask].mean()
