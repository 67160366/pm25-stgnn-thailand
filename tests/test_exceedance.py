# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Unit tests for src/training/exceedance.py (exceedance re-scoring).

All synthetic hand-computed cases — no data files, no network, no model.
"""

from __future__ import annotations

import math

import numpy as np

from src.training.exceedance import (
    binary_event,
    brier_score,
    brier_skill_score,
    climatology_base_rate,
    contingency_counts,
    contingency_metrics,
    exceedance_scores,
)


class TestBinaryEvent:
    def test_strict_greater_than(self):
        values = np.array([10.0, 37.5, 37.6, 100.0])
        np.testing.assert_array_equal(
            binary_event(values, 37.5), np.array([False, False, True, True])
        )

    def test_nan_is_false(self):
        values = np.array([np.nan, 50.0])
        np.testing.assert_array_equal(binary_event(values, 37.5), np.array([False, True]))


class TestContingencyCounts:
    def test_hand_computed(self):
        # 8 positions: pred/obs event flags chosen to hit each of the 4 cells twice.
        pred = np.array([True, True, False, False, True, True, False, False])
        obs = np.array([True, False, True, False, True, False, True, False])
        mask = np.ones(8, dtype=bool)
        counts = contingency_counts(pred, obs, mask)
        assert counts == {
            "hits": 2,
            "misses": 2,
            "false_alarms": 2,
            "correct_negatives": 2,
        }

    def test_mask_excludes_positions(self):
        pred = np.array([True, True, False])
        obs = np.array([True, False, True])
        mask = np.array([True, False, True])  # drop the false-alarm position
        counts = contingency_counts(pred, obs, mask)
        assert counts == {"hits": 1, "misses": 1, "false_alarms": 0, "correct_negatives": 0}

    def test_all_masked_out_is_all_zero(self):
        pred = np.array([True, False])
        obs = np.array([True, False])
        mask = np.zeros(2, dtype=bool)
        counts = contingency_counts(pred, obs, mask)
        assert counts == {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0}


class TestContingencyMetrics:
    def test_hand_computed_ratios(self):
        counts = {"hits": 6, "misses": 2, "false_alarms": 3, "correct_negatives": 9}
        m = contingency_metrics(counts)
        assert m["pod"] == 6 / 8
        assert m["far"] == 3 / 9
        assert m["csi"] == 6 / 11
        assert m["accuracy"] == 15 / 20
        assert m["base_rate"] == 8 / 20
        assert m["n"] == 20

    def test_perfect_forecast(self):
        counts = {"hits": 5, "misses": 0, "false_alarms": 0, "correct_negatives": 10}
        m = contingency_metrics(counts)
        assert m["pod"] == 1.0
        assert m["far"] == 0.0
        assert m["csi"] == 1.0
        assert m["accuracy"] == 1.0

    def test_no_observed_events_pod_none(self):
        counts = {"hits": 0, "misses": 0, "false_alarms": 3, "correct_negatives": 7}
        m = contingency_metrics(counts)
        assert m["pod"] is None
        assert m["far"] == 3 / 3
        assert m["csi"] == 0.0  # hits=0 -> CSI is 0 even though FAR is 1 (no hits to reward)
        assert m["accuracy"] == 7 / 10

    def test_no_predicted_events_far_none(self):
        counts = {"hits": 0, "misses": 4, "false_alarms": 0, "correct_negatives": 6}
        m = contingency_metrics(counts)
        assert m["far"] is None
        assert m["pod"] == 0.0
        assert m["csi"] == 0.0

    def test_no_events_anywhere_csi_none(self):
        counts = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 10}
        m = contingency_metrics(counts)
        assert m["pod"] is None
        assert m["far"] is None
        assert m["csi"] is None
        assert m["accuracy"] == 1.0

    def test_empty_sample_all_none(self):
        counts = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0}
        m = contingency_metrics(counts)
        assert m["pod"] is None
        assert m["far"] is None
        assert m["csi"] is None
        assert m["accuracy"] is None
        assert m["base_rate"] is None
        assert m["n"] == 0


class TestClimatologyBaseRate:
    def test_hand_computed(self):
        values = np.array([10.0, 40.0, 80.0, 20.0])
        mask = np.ones(4, dtype=bool)
        # events at 40 and 80 (>37.5) -> 2/4 = 0.5
        assert climatology_base_rate(values, 37.5, mask) == 0.5

    def test_mask_applied(self):
        values = np.array([100.0, 100.0, 1.0, 1.0])
        mask = np.array([False, False, True, True])
        assert climatology_base_rate(values, 37.5, mask) == 0.0

    def test_empty_mask_returns_nan(self):
        values = np.array([1.0, 2.0])
        mask = np.zeros(2, dtype=bool)
        assert math.isnan(climatology_base_rate(values, 37.5, mask))


class TestBrierScore:
    def test_perfect_forecast_is_zero(self):
        prob = np.array([1.0, 0.0, 1.0, 0.0])
        obs = np.array([True, False, True, False])
        mask = np.ones(4, dtype=bool)
        assert brier_score(prob, obs, mask) == 0.0

    def test_always_wrong_is_one(self):
        prob = np.array([1.0, 0.0])
        obs = np.array([False, True])
        mask = np.ones(2, dtype=bool)
        assert brier_score(prob, obs, mask) == 1.0

    def test_constant_probability_hand_computed(self):
        # p=0.3 constant; obs = [T, F, T, F] -> errors: 0.49, 0.09, 0.49, 0.09 -> mean=0.29
        prob = np.full(4, 0.3)
        obs = np.array([True, False, True, False])
        mask = np.ones(4, dtype=bool)
        assert math.isclose(brier_score(prob, obs, mask), 0.29, rel_tol=1e-9)

    def test_mask_excludes_positions(self):
        prob = np.array([1.0, 0.0])
        obs = np.array([False, True])  # both wrong
        mask = np.array([True, False])  # only score the first (wrong -> BS=1.0)
        assert brier_score(prob, obs, mask) == 1.0

    def test_empty_mask_returns_nan(self):
        prob = np.array([0.5, 0.5])
        obs = np.array([True, False])
        mask = np.zeros(2, dtype=bool)
        assert math.isnan(brier_score(prob, obs, mask))


class TestBrierSkillScore:
    def test_perfect_forecast_bss_one(self):
        assert brier_skill_score(bs=0.0, bs_clim=0.25) == 1.0

    def test_climatology_scored_against_itself_is_zero(self):
        assert brier_skill_score(bs=0.21, bs_clim=0.21) == 0.0

    def test_worse_than_climatology_is_negative(self):
        bss = brier_skill_score(bs=0.4, bs_clim=0.2)
        assert bss == -1.0

    def test_zero_reference_returns_none(self):
        assert brier_skill_score(bs=0.1, bs_clim=0.0) is None

    def test_non_finite_returns_none(self):
        assert brier_skill_score(bs=float("nan"), bs_clim=0.2) is None
        assert brier_skill_score(bs=0.1, bs_clim=float("inf")) is None


class TestExceedanceScores:
    def test_perfect_forecast_end_to_end(self):
        target = np.array([10.0, 50.0, 80.0, 20.0])
        pred = target.copy()  # perfect point forecast
        mask = np.ones(4, dtype=bool)
        out = exceedance_scores(pred, target, mask, threshold=37.5, climatology_p=0.5)
        assert out["pod"] == 1.0
        assert out["far"] == 0.0
        assert out["csi"] == 1.0
        assert out["accuracy"] == 1.0
        assert out["brier"] == 0.0
        assert out["bss"] == 1.0  # climatology BS=0.25 here, perfect forecast beats it fully

    def test_masked_nan_targets_excluded_from_all_counts(self):
        target = np.array([10.0, np.nan, 80.0, 20.0])
        pred = np.array([10.0, 60.0, 80.0, 20.0])
        mask = np.array([True, False, True, True])  # NaN position excluded upstream
        out = exceedance_scores(pred, target, mask, threshold=37.5, climatology_p=0.25)
        assert out["n"] == 3
        assert out["counts"]["hits"] == 1
        assert out["counts"]["misses"] == 0
        assert out["counts"]["false_alarms"] == 0
        assert out["counts"]["correct_negatives"] == 2

    def test_climatology_reference_matches_manual_brier(self):
        target = np.array([10.0, 50.0, 80.0, 20.0])  # events at 50, 80 -> base rate 0.5
        pred = np.array([10.0, 10.0, 10.0, 10.0])  # always predicts no event
        mask = np.ones(4, dtype=bool)
        climatology_p = 0.5
        out = exceedance_scores(pred, target, mask, threshold=37.5, climatology_p=climatology_p)
        # climatology constant 0.5 vs obs [F,T,T,F] -> BS = mean((0.5-o)^2) = 0.25 always
        assert math.isclose(out["brier_climatology"], 0.25, rel_tol=1e-9)
        # model never predicts an event -> BS = mean(obs) since pred=0 -> (0-o)^2 = o
        assert math.isclose(out["brier"], 0.5, rel_tol=1e-9)
        assert out["bss"] == -1.0  # worse than climatology here
