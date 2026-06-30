# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Evaluation metrics for PM2.5 multi-horizon forecasting.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    horizons: list[int] = [6, 12, 24, 48],  # noqa: B006
) -> dict[str, float]:
    """Compute per-horizon RMSE, MAE, and MAPE over valid positions.

    Inputs are in the same scale as the training targets (normalized/scaled
    by default). To obtain metrics in raw µg/m³, inverse-transform pred and
    target before calling. Division by zero in MAPE is avoided by skipping
    positions where |target| < 1.0 (in whatever scale is passed).

    Args:
        pred: Predictions of shape (total_N, H), float32.
        target: Ground-truth of shape (total_N, H), float32.
        mask: Boolean validity mask of shape (total_N, H). True = valid.
        horizons: Horizon offsets in hours, one per column of pred/target.

    Returns:
        Dict with keys like ``"rmse_6h"``, ``"mae_24h"``, ``"mape_24h"``.
        Each value is a Python float. NaN is returned for horizons where no
        valid positions exist.
    """
    if len(horizons) != pred.shape[1]:
        raise ValueError(
            f"len(horizons)={len(horizons)} does not match pred.shape[1]={pred.shape[1]}"
        )

    results: dict[str, float] = {}

    for h_idx, h in enumerate(horizons):
        m = mask[:, h_idx]  # (total_N,) bool
        p = pred[:, h_idx][m]
        t = target[:, h_idx][m]

        if len(p) == 0:
            results[f"rmse_{h}h"] = float("nan")
            results[f"mae_{h}h"] = float("nan")
            results[f"mape_{h}h"] = float("nan")
            continue

        err = p - t
        results[f"rmse_{h}h"] = float(np.sqrt(np.mean(err**2)))
        results[f"mae_{h}h"] = float(np.mean(np.abs(err)))

        # MAPE: skip where |target| < 1 µg/m³ to avoid near-zero division
        valid_mape = np.abs(t) >= 1.0
        if valid_mape.any():
            results[f"mape_{h}h"] = float(
                np.mean(np.abs(err[valid_mape]) / np.abs(t[valid_mape])) * 100.0
            )
        else:
            results[f"mape_{h}h"] = float("nan")

    return results
