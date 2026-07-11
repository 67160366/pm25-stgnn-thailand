# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Unit tests for src/training/conformal.py (split-conformal prediction intervals).

All synthetic — no data files, no network. Verifies the finite-sample quantile
rule, its edge cases, empirical coverage on held-out synthetic data, and the
higher-level pooled/per-station calibration assembler.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.training.conformal import (
    absolute_residual_scores,
    calibrate_conformal,
    conformal_quantile,
    empirical_coverage,
    interval_from_quantile,
)


class TestConformalQuantile:
    def test_kth_smallest_exact(self):
        # n=9, alpha=0.1 -> k = ceil(10 * 0.9) = 9 -> 9th smallest (the max here).
        scores = np.array([9, 1, 8, 2, 7, 3, 6, 4, 5], dtype=float)
        assert conformal_quantile(scores, 0.1) == 9.0

    def test_finite_sample_correction_picks_interior_rank(self):
        # n=19, alpha=0.1 -> k = ceil(20 * 0.9) = 18 -> 18th smallest = value 18.
        scores = np.arange(1, 20, dtype=float)  # 1..19
        assert conformal_quantile(scores, 0.1) == 18.0

    def test_undersampled_returns_inf(self):
        # n=5, alpha=0.1 -> k = ceil(6 * 0.9) = 6 > 5 -> no finite guarantee.
        scores = np.arange(1, 6, dtype=float)
        assert conformal_quantile(scores, 0.1) == math.inf

    def test_exact_boundary_when_product_is_integer(self):
        # n=9, alpha=0.5 -> (n+1)(1-alpha) = 5.0 exactly -> k=5 -> median = 5.
        scores = np.arange(1, 10, dtype=float)
        assert conformal_quantile(scores, 0.5) == 5.0

    def test_non_finite_scores_dropped(self):
        scores = np.array([1, 2, 3, np.nan, np.inf, 4, 5, 6, 7, 8, 9], dtype=float)
        # After dropping nan/inf: 1..9, n=9, alpha=0.1 -> k=9 -> 9.
        assert conformal_quantile(scores, 0.1) == 9.0

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            conformal_quantile(np.arange(10.0), 0.0)
        with pytest.raises(ValueError):
            conformal_quantile(np.arange(10.0), 1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            conformal_quantile(np.array([]), 0.1)


class TestAbsoluteResidualScores:
    def test_elementwise_abs(self):
        pred = np.array([1.0, 5.0, -2.0])
        target = np.array([3.0, 1.0, -2.0])
        np.testing.assert_allclose(
            absolute_residual_scores(pred, target), np.array([2.0, 4.0, 0.0])
        )


class TestIntervalFromQuantile:
    def test_symmetric_with_lower_clip(self):
        pred = np.array([10.0, 3.0, 0.5])
        lower, upper = interval_from_quantile(pred, q=5.0)
        np.testing.assert_allclose(lower, np.array([5.0, 0.0, 0.0]))  # clipped at 0
        np.testing.assert_allclose(upper, np.array([15.0, 8.0, 5.5]))


class TestEmpiricalCoverage:
    def test_counts_within_band(self):
        pred = np.zeros(4)
        target = np.array([1.0, -2.0, 6.0, 3.0])
        # q=3 -> |target|<=3 for [1,-2,3] -> 3/4.
        assert empirical_coverage(pred, target, q=3.0) == 0.75

    def test_mask_applied(self):
        pred = np.zeros(4)
        target = np.array([1.0, 100.0, 2.0, 100.0])
        mask = np.array([True, False, True, False])
        assert empirical_coverage(pred, target, q=3.0, mask=mask) == 1.0

    def test_empty_returns_nan(self):
        assert math.isnan(empirical_coverage(np.zeros(3), np.zeros(3), 1.0, mask=np.zeros(3, bool)))


class TestCoverageGuarantee:
    def test_split_conformal_covers_on_holdout(self):
        # Calibrate q on one exchangeable sample, verify coverage on a fresh one.
        rng = np.random.default_rng(0)
        cal = rng.normal(0.0, 4.0, size=5000)  # residuals = |target - pred|, pred=0
        test = rng.normal(0.0, 4.0, size=20000)
        q = conformal_quantile(np.abs(cal), alpha=0.1)
        cov = empirical_coverage(np.zeros_like(test), test, q)
        assert 0.88 <= cov <= 0.92  # ~0.90 marginal coverage

    def test_heavier_noise_needs_wider_q(self):
        rng = np.random.default_rng(1)
        q_narrow = conformal_quantile(np.abs(rng.normal(0, 2.0, 5000)), 0.1)
        q_wide = conformal_quantile(np.abs(rng.normal(0, 8.0, 5000)), 0.1)
        assert q_wide > q_narrow


class TestCalibrateConformal:
    def _synthetic(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_stations, n_samples, horizons = 3, 2000, [6, 12, 24]
        m = n_stations * n_samples
        pred = np.zeros((m, len(horizons)))
        # Error grows with horizon; station 2 is noisier than 0 and 1.
        sigmas_h = np.array([2.0, 4.0, 8.0])
        station_scale = np.array([1.0, 1.0, 2.0])
        row_station = np.tile(np.array([10, 20, 30]), n_samples)
        target = np.zeros_like(pred)
        for j, sig in enumerate(sigmas_h):
            for r in range(m):
                s_idx = r % n_stations
                target[r, j] = rng.normal(0.0, sig * station_scale[s_idx])
        mask = np.ones_like(pred, dtype=bool)
        return pred, target, mask, horizons, row_station

    def test_pooled_structure_and_monotonic_horizon(self):
        pred, target, mask, horizons, _ = self._synthetic()
        out = calibrate_conformal(pred, target, mask, horizons, alpha=0.1)
        assert set(out["pooled"]) == {"6h", "12h", "24h"}
        q = out["pooled"]
        assert q["6h"] < q["12h"] < q["24h"]  # error grows with horizon
        for h in ("6h", "12h", "24h"):
            assert out["pooled_coverage"][h] >= 0.89

    def test_per_station_present_and_heteroscedastic(self):
        pred, target, mask, horizons, row_station = self._synthetic()
        out = calibrate_conformal(
            pred, target, mask, horizons, alpha=0.1, row_station_ids=row_station
        )
        assert set(out["per_station"]) == {"10", "20", "30"}
        # Station 30 is 2x noisier -> larger half-width at every horizon.
        for h in ("6h", "12h", "24h"):
            assert out["per_station"]["30"][h] > out["per_station"]["10"][h]
        assert out["min_station_count"] > 0

    def test_masked_positions_excluded(self):
        pred, target, mask, horizons, row_station = self._synthetic()
        mask[:, 0] = False  # drop all 6h residuals
        out = calibrate_conformal(
            pred, target, mask, horizons, alpha=0.1, row_station_ids=row_station
        )
        assert out["pooled_n"]["6h"] == 0
        assert out["pooled"]["6h"] == math.inf  # no data -> inf half-width
