# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Exceedance (binary event) re-scoring of PM2.5 point forecasts.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Persistence is a strong RMSE baseline exactly because PM2.5 is autocorrelated —
it wins whenever the level does *not* change. It is weakest at the moments a
forecaster actually cares about: when PM2.5 crosses a health-relevant
threshold. This module re-scores the same point forecasts already produced by
``scripts/04_evaluate.py`` / ``scripts/11_ablation_eval.py`` as a binary
classification problem ("will PM2.5 exceed X µg/m³ in H hours?"), which is a
fairer test of whether the fire+wind features earn their keep.

All functions here are pure numpy (no I/O, no torch) so they are directly
unit-testable; ``scripts/18_exceedance_eval.py`` wires them to the trained
checkpoints and the existing ``src/training/evaluation.py`` loading pattern.

Event convention:
    An "event" is defined as ``value > threshold`` (strict). This is a scoring
    convention for verification statistics (Wilks 2011) and is independent of
    the *operational* alert trigger in ``app/lib/telegram.py``, which uses
    ``>=`` for its cut-in (a deliberately conservative "alert on the boundary"
    policy). The two need not match; do not conflate them.

Deterministic vs probabilistic scoring:
    The model and persistence are deterministic point forecasters: for
    contingency-table statistics (POD/FAR/CSI/accuracy) their "forecast" is
    simply the binary event derived from the point prediction. For the Brier
    score, a deterministic forecast is treated as a degenerate probabilistic
    forecast with ``p in {0, 1}`` (the standard way to compare deterministic
    forecasts to a probabilistic reference on the same score — see Brier 1950;
    Wilks 2011, section 8.4.2). Climatology is a genuinely probabilistic
    forecast: a constant probability equal to the historical base rate.

No-leakage climatology:
    The climatology reference probability MUST be estimated from years that do
    not overlap the scored (test) period — otherwise "climatology" would
    silently know the test-set event frequency and its Brier score would be
    optimistic by construction. ``scripts/18_exceedance_eval.py`` estimates it
    from the train+val (2022-2024) portion of the split and applies it as a
    constant to the held-out 2025 test years; this module only provides the
    pure ``climatology_base_rate`` estimator, the caller is responsible for
    supplying disjoint value arrays.

References:
    Brier, G. W. (1950). "Verification of forecasts expressed in terms of
    probability." Monthly Weather Review, 78(1), 1-3.
    Wilks, D. S. (2011). "Statistical Methods in the Atmospheric Sciences"
    (3rd ed.), chapter 8 (forecast verification: contingency tables, Brier
    score, skill scores).
"""

from __future__ import annotations

import math

import numpy as np


def binary_event(values: np.ndarray, threshold: float) -> np.ndarray:
    """Elementwise exceedance indicator: ``True`` where ``value > threshold``.

    NaN inputs compare ``False`` under numpy semantics (never truthy) — callers
    MUST combine the result with a validity mask before counting, since an
    unmasked NaN would otherwise be silently scored as a non-event rather than
    excluded.

    Args:
        values: Array of any shape, µg/m³ (or forecast probabilities scaled the
            same way — this function only compares against ``threshold``).
        threshold: Exceedance cut-in in the same units as ``values``.

    Returns:
        Boolean array, same shape as ``values``.
    """
    return np.asarray(values, dtype=np.float64) > threshold


def contingency_counts(
    pred_event: np.ndarray, obs_event: np.ndarray, mask: np.ndarray
) -> dict[str, int]:
    """2x2 contingency table counts restricted to valid (``mask``) positions.

    Args:
        pred_event: Boolean forecast event indicator, any shape.
        obs_event: Boolean observed event indicator, same shape as ``pred_event``.
        mask: Boolean validity mask, same shape; positions where ``False`` are
            excluded from every count (invalid/NaN targets never contribute).

    Returns:
        Dict with integer keys ``hits``, ``misses``, ``false_alarms``,
        ``correct_negatives`` (standard verification contingency table; Wilks
        2011 table 8.3). ``hits + misses + false_alarms + correct_negatives``
        equals the number of ``True`` entries in ``mask``.
    """
    pred_event = np.asarray(pred_event, dtype=bool)
    obs_event = np.asarray(obs_event, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    p = pred_event[mask]
    o = obs_event[mask]
    return {
        "hits": int(np.sum(p & o)),
        "misses": int(np.sum(~p & o)),
        "false_alarms": int(np.sum(p & ~o)),
        "correct_negatives": int(np.sum(~p & ~o)),
    }


def contingency_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    """Derive POD, FAR, CSI, accuracy, and base rate from contingency counts.

    Each ratio is ``None`` (never NaN, so the result survives ``json.dump``
    unmodified) when its denominator is zero, i.e. the statistic is undefined
    for this sample rather than accidentally zero.

    Args:
        counts: Output of :func:`contingency_counts` (or an equivalent dict
            with the same four integer keys).

    Returns:
        Dict with keys:
            pod: Probability of detection = hits / (hits + misses). ``None``
                if no observed events occurred (nothing to detect).
            far: False alarm ratio = false_alarms / (hits + false_alarms).
                ``None`` if the forecast never predicted an event.
            csi: Critical success index = hits / (hits + misses + false_alarms).
                ``None`` if neither the forecast nor the observation ever had
                an event.
            accuracy: (hits + correct_negatives) / n. ``None`` if n == 0.
            base_rate: (hits + misses) / n — observed event frequency in this
                sample. ``None`` if n == 0.
            n: Total scored count (int), always present.
    """
    hits = counts["hits"]
    misses = counts["misses"]
    false_alarms = counts["false_alarms"]
    correct_negatives = counts["correct_negatives"]
    n = hits + misses + false_alarms + correct_negatives

    pod_denom = hits + misses
    far_denom = hits + false_alarms
    csi_denom = hits + misses + false_alarms

    return {
        "pod": (hits / pod_denom) if pod_denom > 0 else None,
        "far": (false_alarms / far_denom) if far_denom > 0 else None,
        "csi": (hits / csi_denom) if csi_denom > 0 else None,
        "accuracy": ((hits + correct_negatives) / n) if n > 0 else None,
        "base_rate": (pod_denom / n) if n > 0 else None,
        "n": n,
    }


def climatology_base_rate(values: np.ndarray, threshold: float, mask: np.ndarray) -> float:
    """Historical event frequency ``P(value > threshold)`` over valid positions.

    Intended for estimating the climatology reference probability from a
    period disjoint from the scored period (see module docstring). This
    function itself is agnostic to which period it is called on — the caller
    is responsible for passing only train+val data when the climatology must
    not see the test period.

    Args:
        values: Array of any shape, µg/m³.
        threshold: Exceedance cut-in (µg/m³).
        mask: Boolean validity mask, same shape as ``values``.

    Returns:
        Base rate in [0, 1]; ``nan`` if ``mask`` selects zero positions.
    """
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    event = binary_event(values, threshold)
    return float(np.mean(event[mask]))


def brier_score(prob: np.ndarray, obs_event: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared error between a forecast probability and a binary outcome.

    Deterministic forecasts (model, persistence) are scored by passing
    ``prob = pred_event.astype(float)`` (degenerate p in {0, 1}); climatology
    is scored by passing a constant array of the reference base rate. A
    perfect deterministic forecast (prob == obs_event everywhere scored)
    yields ``BS == 0``.

    Args:
        prob: Forecast probability of the event, any shape, values expected in
            [0, 1] (not enforced — pass degenerate {0, 1} for deterministic
            forecasts).
        obs_event: Boolean observed event indicator, same shape as ``prob``.
        mask: Boolean validity mask, same shape; ``False`` positions excluded.

    Returns:
        Brier score (lower is better, 0 = perfect); ``nan`` if ``mask``
        selects zero positions.
    """
    prob = np.asarray(prob, dtype=np.float64)
    obs = np.asarray(obs_event, dtype=bool).astype(np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    p = prob[mask]
    o = obs[mask]
    return float(np.mean((p - o) ** 2))


def brier_skill_score(bs: float, bs_clim: float) -> float | None:
    """Brier skill score relative to a climatology reference.

    ``BSS = 1 - BS / BS_clim``: positive means the forecast beats climatology,
    0 means it ties climatology, negative means it is worse than always
    forecasting the historical base rate. Scoring climatology against itself
    (``bs == bs_clim``) yields ``BSS == 0`` by construction — a useful sanity
    check.

    Args:
        bs: Brier score of the forecast under test.
        bs_clim: Brier score of the climatology reference (same events, same
            mask).

    Returns:
        BSS as a float, or ``None`` if either score is non-finite or
        ``bs_clim == 0`` (a degenerate reference — e.g. an event that never
        occurs in the reference period — makes the ratio undefined).
    """
    if not math.isfinite(bs) or not math.isfinite(bs_clim) or bs_clim == 0:
        return None
    return 1.0 - bs / bs_clim


def exceedance_scores(
    pred_ug: np.ndarray,
    target_ug: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    climatology_p: float,
) -> dict[str, object]:
    """Full exceedance scoring for one deterministic forecaster at one threshold.

    Convenience wrapper composing :func:`binary_event`, :func:`contingency_counts`,
    :func:`contingency_metrics`, :func:`brier_score`, and
    :func:`brier_skill_score` for a single (forecaster, threshold) pair. The
    forecaster is scored as deterministic (p in {0, 1}) against a constant
    climatology probability supplied by the caller (see module docstring on
    no-leakage climatology).

    Args:
        pred_ug: Point forecast, µg/m³, any shape.
        target_ug: Ground-truth PM2.5, µg/m³, same shape as ``pred_ug``.
        mask: Boolean validity mask, same shape.
        threshold: Exceedance cut-in (µg/m³).
        climatology_p: Constant reference probability for the Brier skill
            score denominator (estimated from a disjoint period).

    Returns:
        Dict with keys ``counts`` (from :func:`contingency_counts`),
        ``pod``, ``far``, ``csi``, ``accuracy``, ``base_rate``, ``n`` (from
        :func:`contingency_metrics`), ``brier``, ``brier_climatology``, ``bss``.
    """
    pred_event = binary_event(pred_ug, threshold)
    obs_event = binary_event(target_ug, threshold)

    counts = contingency_counts(pred_event, obs_event, mask)
    metrics = contingency_metrics(counts)

    bs = brier_score(pred_event.astype(np.float64), obs_event, mask)
    clim_prob = np.full_like(np.asarray(target_ug, dtype=np.float64), climatology_p)
    bs_clim = brier_score(clim_prob, obs_event, mask)
    bss = brier_skill_score(bs, bs_clim)

    return {
        "counts": counts,
        "pod": metrics["pod"],
        "far": metrics["far"],
        "csi": metrics["csi"],
        "accuracy": metrics["accuracy"],
        "base_rate": metrics["base_rate"],
        "n": metrics["n"],
        "brier": bs,
        "brier_climatology": bs_clim,
        "bss": bss,
    }
