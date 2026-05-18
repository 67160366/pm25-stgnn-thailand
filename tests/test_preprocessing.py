"""Unit tests for src/data/preprocessing.py.

All tests are offline — no real files or network calls.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    _add_cyclic_features,
    _apply_gap_policy,
    _fit_scalers,
    _label_gap_runs,
    _normalize,
)


def _hourly_series(n_hours: int = 100, fill_value: float = 10.0) -> pd.Series:
    """Helper: hourly UTC DatetimeIndex series filled with a constant."""
    idx = pd.date_range("2022-01-01", periods=n_hours, freq="1h", tz="UTC")
    return pd.Series(fill_value, index=idx, dtype="float32")


def _series_with_gap(gap_start: int, gap_len: int, n_hours: int = 200) -> pd.Series:
    """Helper: series with one contiguous NaN gap."""
    s = _hourly_series(n_hours, fill_value=5.0)
    s.iloc[gap_start : gap_start + gap_len] = float("nan")
    return s


class TestLabelGapRuns:
    def test_no_gaps_returns_all_zeros(self) -> None:
        s = _hourly_series(10)
        result = _label_gap_runs(s)
        assert (result == 0).all()

    def test_gap_length_matches_run(self) -> None:
        s = _series_with_gap(gap_start=5, gap_len=8)
        result = _label_gap_runs(s)
        assert (result.iloc[5:13] == 8).all()
        assert (result.iloc[:5] == 0).all()
        assert (result.iloc[13:] == 0).all()

    def test_multiple_gaps(self) -> None:
        s = _hourly_series(50)
        s.iloc[2:5] = float("nan")  # gap of 3h
        s.iloc[20:27] = float("nan")  # gap of 7h
        result = _label_gap_runs(s)
        assert (result.iloc[2:5] == 3).all()
        assert (result.iloc[20:27] == 7).all()
        assert result.iloc[0] == 0


class TestApplyGapPolicy:
    def test_no_gaps_returns_unchanged(self) -> None:
        s = _hourly_series(50)
        filled, mask, excl = _apply_gap_policy(s)
        pd.testing.assert_series_equal(filled, s)
        assert not mask.any()
        assert not excl.any()

    def test_short_gap_is_interpolated(self) -> None:
        """Gap < 6h should be filled by linear interpolation (not NaN, not ffill)."""
        s = _hourly_series(20, fill_value=0.0)
        # Set a linear ramp so interpolation produces non-zero values
        s.iloc[0] = 0.0
        s.iloc[5] = 10.0
        s.iloc[1:5] = float("nan")  # 4-hour gap
        filled, mask, _ = _apply_gap_policy(s)
        assert not filled.iloc[1:5].isna().any(), "Short gap should be interpolated"
        assert not mask.any(), "Short gap should NOT set mask_in_loss"

    def test_medium_gap_is_forward_filled(self) -> None:
        """Gap of 10h (6-24h) should be forward-filled from last known value."""
        s = _hourly_series(50, fill_value=10.0)
        s.iloc[5:15] = float("nan")  # 10h gap
        filled, mask, _ = _apply_gap_policy(s)
        assert not filled.iloc[5:15].isna().any(), "Medium gap should be ffilled"
        assert (filled.iloc[5:15] == 10.0).all()
        assert not mask.any()

    def test_medium_gap_no_interpolation_ramp(self) -> None:
        """Medium gap on a ramp series must be ffilled, not interpolated (DESIGN.md §4.3)."""
        # Ramp 0..4, then 10-hour gap (medium), then 15..19.
        # Correct: all 10 gap positions filled with 4.0 (last known value).
        # Buggy (pre-fix): positions 0-4 of gap get interpolated to 5.0..9.0.
        idx = pd.date_range("2022-01-01", periods=20, freq="1h", tz="UTC")
        nans: list[float] = [float("nan")] * 10
        vals = [float(i) for i in range(5)] + nans + [float(i) for i in range(15, 20)]
        s = pd.Series(vals, index=idx, dtype="float32")
        filled, _, _ = _apply_gap_policy(s)
        assert filled.iloc[5:15].notna().all(), "Medium gap should be fully filled"
        assert all(
            v == pytest.approx(4.0) for v in filled.iloc[5:15]
        ), "Medium gap must be ffilled from 4.0, not interpolated"

    def test_long_gap_masked_not_filled(self) -> None:
        """Gap > 24h should remain NaN with mask_in_loss=True."""
        s = _hourly_series(100)
        s.iloc[10:40] = float("nan")  # 30h gap
        filled, mask, _ = _apply_gap_policy(s)
        assert filled.iloc[10:40].isna().all(), "Long gap should remain NaN"
        assert mask.iloc[10:40].all(), "Long gap should set mask_in_loss=True"
        assert not mask.iloc[:10].any()
        assert not mask.iloc[40:].any()

    def test_gap_exactly_24h_is_ffilled(self) -> None:
        """Gap of exactly 24h (boundary) should be forward-filled, not masked."""
        s = _hourly_series(100)
        s.iloc[10:34] = float("nan")  # exactly 24h
        filled, mask, _ = _apply_gap_policy(s)
        assert not filled.iloc[10:34].isna().any()
        assert not mask.any()

    def test_gap_25h_is_masked(self) -> None:
        """Gap of 25h (just over boundary) should be masked."""
        s = _hourly_series(100)
        s.iloc[10:35] = float("nan")  # 25h gap
        filled, mask, _ = _apply_gap_policy(s)
        assert filled.iloc[10:35].isna().all()
        assert mask.iloc[10:35].all()

    def test_exclude_from_training_set_for_7day_gap(self) -> None:
        """A gap > 7 days within a month should mark the entire month as excluded."""
        # Create a 30-day series entirely in January 2022
        idx = pd.date_range("2022-01-01", periods=30 * 24, freq="1h", tz="UTC")
        s = pd.Series(10.0, index=idx, dtype="float32")
        s.iloc[5 : 5 + 8 * 24] = float("nan")  # 8-day gap (>7d)
        _, _, excl = _apply_gap_policy(s)
        # Entire series is January 2022 — the month should be excluded
        assert excl.all(), "Entire month-of-gap should be excluded_from_training"

    def test_no_exclude_for_gap_exactly_7days(self) -> None:
        """Gap of exactly 7 days (168h, not strictly >7d) should NOT be excluded."""
        idx = pd.date_range("2022-01-01", periods=30 * 24, freq="1h", tz="UTC")
        s = pd.Series(10.0, index=idx, dtype="float32")
        s.iloc[5 : 5 + 168] = float("nan")  # exactly 168h = 7 days
        _, _, excl = _apply_gap_policy(s)
        assert not excl.any(), "Gap of exactly 7 days should not trigger exclusion"


class TestFitScalers:
    def _make_df(
        self, station_id: int, values: list[float], year: int, freq: str = "1D"
    ) -> pd.DataFrame:
        """Build a DataFrame with hourly timestamps all in the given year."""
        idx = pd.date_range(f"{year}-01-01", periods=len(values), freq=freq, tz="UTC")
        return pd.DataFrame({"station_id": station_id, "timestamp": idx, "pm25_raw": values})

    def test_scaler_fits_only_on_train_years(self) -> None:
        """Scaler must use only 2022-2023 data; 2024-2025 values must not affect it."""
        # 10 values at 10.0 and 10 values at 20.0 in training years -> median ~15
        train_vals = [10.0] * 10 + [20.0] * 10
        df_2022 = self._make_df(1, train_vals, 2022)
        # Extreme outlier in test year — must not shift the median
        df_2024 = self._make_df(1, [9999.0] * 5, 2024)
        df = pd.concat([df_2022, df_2024], ignore_index=True)
        scalers = _fit_scalers(df)
        # Median of [10]*10 + [20]*10 = 15.0
        assert scalers[1]["center_"] == pytest.approx(15.0, abs=0.1)

    def test_scaler_scale_is_iqr(self) -> None:
        """scale_ should equal Q75 - Q25 of training data."""
        vals = list(range(1, 21))  # 20 distinct values so IQR is well-defined
        df = self._make_df(1, [float(v) for v in vals], year=2022)
        scalers = _fit_scalers(df)
        q75 = float(np.percentile(vals, 75))
        q25 = float(np.percentile(vals, 25))
        assert scalers[1]["scale_"] == pytest.approx(q75 - q25, rel=1e-4)


class TestNormalize:
    def test_normalize_produces_zero_at_median(self) -> None:
        """A value equal to the median should normalize to 0."""
        ts = pd.Timestamp("2022-01-01", tz="UTC")
        df = pd.DataFrame({"station_id": [1], "pm25_raw": [15.0], "timestamp": [ts]})
        scalers = {1: {"center_": 15.0, "scale_": 10.0}}
        result = _normalize(df, scalers)
        assert result["pm25_scaled"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_normalize_nan_stays_nan(self) -> None:
        """NaN pm25_raw should produce NaN pm25_scaled."""
        ts = pd.Timestamp("2022-01-01", tz="UTC")
        df = pd.DataFrame({"station_id": [1], "pm25_raw": [float("nan")], "timestamp": [ts]})
        scalers = {1: {"center_": 10.0, "scale_": 5.0}}
        result = _normalize(df, scalers)
        assert np.isnan(result["pm25_scaled"].iloc[0])


class TestAddCyclicFeatures:
    def test_midnight_hour_sin_zero(self) -> None:
        """Hour 0 → hour_sin = 0."""
        idx = pd.DatetimeIndex([pd.Timestamp("2022-01-01 00:00", tz="UTC")])
        df = pd.DataFrame({"timestamp": idx, "station_id": [1], "pm25_raw": [5.0]})
        result = _add_cyclic_features(df)
        assert result["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_hour_6_sin_cos(self) -> None:
        """Hour 6 → hour_sin = sin(pi/2) = 1, hour_cos = cos(pi/2) = 0."""
        idx = pd.DatetimeIndex([pd.Timestamp("2022-01-01 06:00", tz="UTC")])
        df = pd.DataFrame({"timestamp": idx, "station_id": [1], "pm25_raw": [5.0]})
        result = _add_cyclic_features(df)
        assert result["hour_sin"].iloc[0] == pytest.approx(1.0, abs=1e-6)
        assert result["hour_cos"].iloc[0] == pytest.approx(0.0, abs=1e-5)

    def test_doy_encoding_range(self) -> None:
        """sin/cos values for day-of-year should be in [-1, 1]."""
        idx = pd.date_range("2022-01-01", periods=365, freq="1D", tz="UTC")
        df = pd.DataFrame({"timestamp": idx, "station_id": 1, "pm25_raw": 5.0})
        result = _add_cyclic_features(df)
        assert result["doy_sin"].between(-1, 1).all()
        assert result["doy_cos"].between(-1, 1).all()
