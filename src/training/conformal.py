# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Split-conformal prediction intervals for PM2.5 forecasts (post-hoc).

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Wraps a trained model's point forecasts with distribution-free prediction
intervals. Given a held-out calibration set (the 2024 validation split), we take
the absolute residual ``s = |y - ŷ|`` in real units (µg/m³, denormalised) as the
nonconformity score and read off the finite-sample conformal quantile

    q = the k-th smallest score,  k = ceil((n + 1) * (1 - alpha)),

so that the symmetric interval ``[ŷ - q, ŷ + q]`` has *marginal* coverage
``≥ 1 - alpha`` for a fresh, exchangeable test point. Quantiles are computed
separately per forecast horizon (error grows with horizon) and, when the
per-station calibration count is large enough, per station (heteroscedastic
noise across the network).

References:
    Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L. (2018).
    "Distribution-Free Predictive Inference for Regression." JASA 113(523).
    (Split / inductive conformal — Theorem 2.1, symmetric absolute-residual score.)
    Vovk, V., Gammerman, A., Shafer, G. (2005). "Algorithmic Learning in a
    Random World." Springer.

Caveat: coverage is *marginal* over the calibration distribution. The dashboard
live mode feeds NWP forecast weather (not the ERA5 reanalysis the intervals were
calibrated on), which breaks exchangeability and may cause mis-coverage — this
must stay stated in the UI.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of nonconformity scores.

    Returns the ``k``-th smallest score where ``k = ceil((n + 1) * (1 - alpha))``
    over the ``n`` finite scores (Lei et al. 2018). When ``k > n`` the sample is
    too small to guarantee ``1 - alpha`` coverage and ``+inf`` is returned.

    Args:
        scores: 1-D array of nonconformity scores (non-finite entries dropped).
        alpha: Miscoverage level in (0, 1); e.g. 0.1 targets 90% coverage.

    Returns:
        The conformal quantile as a float (``math.inf`` if under-sampled).

    Raises:
        ValueError: If ``alpha`` is outside (0, 1) or no finite scores remain.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")
    arr = np.asarray(scores, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        raise ValueError("need at least one finite score to calibrate")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(np.partition(arr, k - 1)[k - 1])


def absolute_residual_scores(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Nonconformity scores ``|target - pred|`` (elementwise, float64)."""
    return np.abs(np.asarray(target, dtype=np.float64) - np.asarray(pred, dtype=np.float64))


def interval_from_quantile(
    pred: np.ndarray, q: float, lower_clip: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric interval ``[pred - q, pred + q]`` with a lower clip.

    Args:
        pred: Point forecasts (any shape).
        q: Conformal half-width (µg/m³).
        lower_clip: Floor for the lower bound (0 — PM2.5 cannot be negative).

    Returns:
        Tuple ``(lower, upper)`` broadcast to ``pred``'s shape.
    """
    pred = np.asarray(pred, dtype=np.float64)
    lower = np.maximum(pred - q, lower_clip)
    upper = pred + q
    return lower, upper


def empirical_coverage(
    pred: np.ndarray, target: np.ndarray, q: float, mask: np.ndarray | None = None
) -> float:
    """Fraction of targets falling within ``pred ± q``.

    The lower clip at 0 does not change coverage because targets are
    non-negative: ``target ≥ 0 ≥ pred - q`` whenever ``pred - q`` is clipped.

    Args:
        pred: Point forecasts.
        target: Ground-truth values (same shape as ``pred``).
        q: Conformal half-width.
        mask: Optional boolean array; only ``True`` positions are scored.

    Returns:
        Empirical coverage in [0, 1]; ``nan`` if no positions are scored.
    """
    scores = absolute_residual_scores(pred, target)
    if mask is not None:
        scores = scores[np.asarray(mask, dtype=bool)]
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores <= q))


def calibrate_conformal(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    horizons: list[int],
    alpha: float,
    row_station_ids: np.ndarray | None = None,
    min_per_station: int = 100,
) -> dict[str, object]:
    """Calibrate pooled (and optionally per-station) conformal quantiles.

    Rows are (sample, station) observations; columns are forecast horizons.
    A pooled quantile is always computed per horizon. When ``row_station_ids`` is
    given, a per-station quantile is also computed per horizon and reported
    together with the calibration count and in-sample coverage.

    Args:
        pred: ``(M, H)`` point forecasts in µg/m³.
        target: ``(M, H)`` ground-truth PM2.5 in µg/m³.
        mask: ``(M, H)`` bool; ``True`` where the residual is valid.
        horizons: Forecast horizons (hours); length ``H``, order matches columns.
        alpha: Miscoverage level (0.1 → 90% target coverage).
        row_station_ids: ``(M,)`` station id per row; enables per-station output.
        min_per_station: Minimum valid residuals a station needs at *every*
            horizon for its per-station quantiles to be considered reliable
            (recorded in the ``min_station_count`` field for the caller's mode
            decision; per-station output is emitted regardless).

    Returns:
        Dict with keys ``pooled``, ``pooled_coverage``, ``pooled_n`` and (if
        station ids supplied) ``per_station``, ``per_station_coverage``,
        ``per_station_n``, ``min_station_count``. Quantile/coverage values are
        keyed ``"6h"``…; ``inf`` quantiles are serialised as the JSON-invalid
        ``float('inf')`` and must be handled by the writer.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    scores = absolute_residual_scores(pred, target)

    pooled: dict[str, float] = {}
    pooled_cov: dict[str, float] = {}
    pooled_n: dict[str, int] = {}
    for j, h in enumerate(horizons):
        col = scores[:, j][mask[:, j]]
        q = conformal_quantile(col, alpha) if col.size else math.inf
        pooled[f"{h}h"] = q
        pooled_cov[f"{h}h"] = empirical_coverage(pred[:, j], target[:, j], q, mask[:, j])
        pooled_n[f"{h}h"] = int(col.size)

    result: dict[str, object] = {
        "pooled": pooled,
        "pooled_coverage": pooled_cov,
        "pooled_n": pooled_n,
    }

    if row_station_ids is None:
        return result

    row_station_ids = np.asarray(row_station_ids)
    per_station: dict[str, dict[str, float]] = {}
    per_station_cov: dict[str, dict[str, float]] = {}
    per_station_n: dict[str, dict[str, int]] = {}
    min_count = math.inf
    for sid in np.unique(row_station_ids):
        rows = row_station_ids == sid
        key = str(int(sid))
        q_h: dict[str, float] = {}
        cov_h: dict[str, float] = {}
        n_h: dict[str, int] = {}
        for j, h in enumerate(horizons):
            sel = rows & mask[:, j]
            col = scores[sel, j]
            n_h[f"{h}h"] = int(col.size)
            min_count = min(min_count, col.size)
            q_h[f"{h}h"] = conformal_quantile(col, alpha) if col.size else math.inf
            cov_h[f"{h}h"] = empirical_coverage(pred[sel, j], target[sel, j], q_h[f"{h}h"])
        per_station[key] = q_h
        per_station_cov[key] = cov_h
        per_station_n[key] = n_h

    result["per_station"] = per_station
    result["per_station_coverage"] = per_station_cov
    result["per_station_n"] = per_station_n
    result["min_station_count"] = int(min_count) if math.isfinite(min_count) else 0
    result["min_per_station_threshold"] = int(min_per_station)
    return result
