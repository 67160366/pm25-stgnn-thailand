"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

Unit tests for ``src/explain/backtrajectory.py`` (roadmap item 11).

All tests build :class:`~src.explain.backtrajectory.WindField` in-memory via
``make_wind_field`` with tiny synthetic numpy grids -- no real ERA5 file, no network, no
real dataset/checkpoint. ``load_era5_window`` (IO) is exercised only by
``scripts/22_backtrajectory.py``, not here.

Tests 7 and 8 are the sign-convention oracles (design spec section 7.7-7.8): a uniform wind
field with a known direction must produce a backward trajectory displaced in the physically
correct direction. These are written first (TDD) because RK2-backward sign errors are the
classic place to get this kind of integration wrong silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explain.backtrajectory import (
    BackTrajectory,
    WindField,
    _bilinear,
    _deg_per_hour,
    assess_event,
    corridor_frp_by_country,
    country_hours_fraction,
    in_domain,
    integrate_backtrajectory,
    make_wind_field,
    rk2_back_step,
    sample_uv,
)


def _uniform_field(
    u_val: float,
    v_val: float,
    lat_range: tuple[float, float] = (15.0, 25.0),
    lon_range: tuple[float, float] = (50.0, 150.0),
    n_lat: int = 5,
    n_lon: int = 101,
    n_hours: int = 30,
) -> WindField:
    """A uniform (u, v) field, wide and long enough for 10-20h backward integrations."""
    times = pd.date_range("2025-03-18T00:00:00", periods=n_hours, freq="1h").values
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)
    u = np.full((n_hours, n_lat, n_lon), u_val, dtype=np.float64)
    v = np.full((n_hours, n_lat, n_lon), v_val, dtype=np.float64)
    return make_wind_field(times, lats, lons, u, v)


# ---------------------------------------------------------------------------
# 1. make_wind_field
# ---------------------------------------------------------------------------


def test_make_wind_field_sorts_descending_lats() -> None:
    times = pd.date_range("2025-01-01", periods=1, freq="1h").values
    lats = np.array([21.0, 20.5, 20.0])  # descending, ERA5-style
    lons = np.array([97.0, 97.5, 98.0])
    u = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    v = np.zeros_like(u)

    wind = make_wind_field(times, lats, lons, u, v)

    assert list(wind.lats) == [20.0, 20.5, 21.0]
    # Row that was lat=20.0 (originally last row, values 7,8,9) must now be row 0.
    assert wind.u[0, 0, 0] == 7.0
    assert wind.u[0, 2, 0] == 1.0  # lat=21.0 row (originally first) is now last.


# ---------------------------------------------------------------------------
# 2-3. _bilinear
# ---------------------------------------------------------------------------


def test_bilinear_exact_on_node() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert _bilinear(grid, lats, lons, 0.0, 0.0) == pytest.approx(1.0)
    assert _bilinear(grid, lats, lons, 1.0, 1.0) == pytest.approx(4.0)
    assert _bilinear(grid, lats, lons, 0.5, 0.5) == pytest.approx((1.0 + 2.0 + 3.0 + 4.0) / 4.0)


def test_bilinear_nan_propagates() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    grid = np.array([[1.0, 2.0], [3.0, np.nan]])

    assert np.isnan(_bilinear(grid, lats, lons, 0.5, 0.5))


# ---------------------------------------------------------------------------
# 4. in_domain
# ---------------------------------------------------------------------------


def test_in_domain_boundaries() -> None:
    wind = _uniform_field(
        0.0, 0.0, lat_range=(16.0, 21.0), lon_range=(97.0, 101.5), n_lat=21, n_lon=19
    )

    assert in_domain(wind, 16.0, 97.0) is True
    assert in_domain(wind, 21.0, 101.5) is True
    assert in_domain(wind, 19.0, 99.0) is True
    assert in_domain(wind, 15.9, 99.0) is False
    assert in_domain(wind, 19.0, 101.6) is False


# ---------------------------------------------------------------------------
# 5. sample_uv time interpolation
# ---------------------------------------------------------------------------


def test_sample_uv_time_interp() -> None:
    times = pd.to_datetime(["2025-03-18T00:00:00", "2025-03-18T01:00:00"]).values
    lats = np.array([18.0, 19.0])
    lons = np.array([97.0, 98.0])
    u = np.array([np.zeros((2, 2)), np.full((2, 2), 10.0)])
    v = np.zeros_like(u)
    wind = make_wind_field(times, lats, lons, u, v)

    half_hour = times[0] + np.timedelta64(30, "m")
    u_val, v_val = sample_uv(wind, half_hour, 18.5, 97.5)
    assert u_val == pytest.approx(5.0)
    assert v_val == pytest.approx(0.0)


def test_sample_uv_raises_outside_time_range() -> None:
    wind = _uniform_field(1.0, 0.0, n_hours=3)
    too_late = wind.times[-1] + np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="outside wind time range"):
        sample_uv(wind, too_late, 19.0, 99.0)


# ---------------------------------------------------------------------------
# 6. _deg_per_hour
# ---------------------------------------------------------------------------


def test_deg_per_hour_signs() -> None:
    dlon_pos, dlat_zero = _deg_per_hour(5.0, 0.0, 19.0)
    assert dlon_pos > 0.0
    assert dlat_zero == pytest.approx(0.0)

    dlon_zero, dlat_pos = _deg_per_hour(0.0, 5.0, 19.0)
    assert dlon_zero == pytest.approx(0.0)
    assert dlat_pos > 0.0

    # ~0.03-0.04 deg/h per m/s of eastward wind at lat 19 (1/cos(19deg) magnification).
    dlon_unit, _ = _deg_per_hour(1.0, 0.0, 19.0)
    assert 0.03 < dlon_unit < 0.04

    dlon_neg, dlat_neg = _deg_per_hour(-5.0, -5.0, 19.0)
    assert dlon_neg < 0.0
    assert dlat_neg < 0.0


# ---------------------------------------------------------------------------
# 7-8. ORACLE tests -- sign convention (design spec section 10.4/7.7-7.8)
# ---------------------------------------------------------------------------


def test_uniform_easterly_goes_east() -> None:
    """Air moving WEST (u=-5) must have come FROM THE EAST: backward lon strictly increases."""
    wind = _uniform_field(u_val=-5.0, v_val=0.0, n_hours=15)
    launch = wind.times[-1]  # last available hour so 10h back stays in-range

    traj = integrate_backtrajectory(
        wind, start_lat=19.0, start_lon=100.0, start_time=launch, hours_back=10, dt_hours=1.0
    )

    assert traj.terminated_reason in {"completed", "left_domain"}
    assert traj.hours_integrated > 0
    assert np.all(np.diff(traj.lons) > 0.0), "backward longitudes must strictly increase"
    assert np.allclose(traj.lats, traj.lats[0], atol=1e-9)


def test_uniform_southerly_goes_south() -> None:
    """Air moving NORTH (v=5) must have come FROM THE SOUTH: backward lat strictly decreases."""
    wind = _uniform_field(u_val=0.0, v_val=5.0, n_hours=15)
    launch = wind.times[-1]

    traj = integrate_backtrajectory(
        wind, start_lat=20.0, start_lon=100.0, start_time=launch, hours_back=10, dt_hours=1.0
    )

    assert traj.terminated_reason in {"completed", "left_domain"}
    assert traj.hours_integrated > 0
    assert np.all(np.diff(traj.lats) < 0.0), "backward latitudes must strictly decrease"
    assert np.allclose(traj.lons, traj.lons[0], atol=1e-9)


# ---------------------------------------------------------------------------
# 9. Leaves domain
# ---------------------------------------------------------------------------


def test_leaves_domain_terminates() -> None:
    # Narrow lon domain + strong westward wind -> exits the west edge within a few hours.
    wind = _uniform_field(
        u_val=-50.0,
        v_val=0.0,
        lat_range=(18.0, 20.0),
        lon_range=(97.0, 99.0),
        n_lat=3,
        n_lon=9,
        n_hours=10,
    )
    launch = wind.times[-1]

    traj = integrate_backtrajectory(
        wind, start_lat=19.0, start_lon=98.0, start_time=launch, hours_back=8, dt_hours=1.0
    )

    assert traj.terminated_reason == "left_domain"
    assert traj.hours_integrated < 8


# ---------------------------------------------------------------------------
# 10. Missing wind
# ---------------------------------------------------------------------------


def test_missing_wind_terminates() -> None:
    wind = _uniform_field(u_val=-2.0, v_val=0.0, n_hours=15)
    launch = wind.times[-1]
    # NaN-out the entire field a few hours before launch, guaranteeing the backward
    # trajectory reaches a NaN slice before completing 10h.
    nan_wind = WindField(
        times=wind.times,
        lats=wind.lats,
        lons=wind.lons,
        u=wind.u.copy(),
        v=wind.v.copy(),
    )
    nan_wind.u[5, :, :] = np.nan
    nan_wind.v[5, :, :] = np.nan

    traj = integrate_backtrajectory(
        nan_wind, start_lat=19.0, start_lon=100.0, start_time=launch, hours_back=10, dt_hours=1.0
    )

    assert traj.terminated_reason == "missing_wind"
    assert traj.hours_integrated < 10


# ---------------------------------------------------------------------------
# 11, 14. country_hours_fraction
# ---------------------------------------------------------------------------


def test_country_hours_fraction_sums_to_one() -> None:
    countries = ["Thailand", "Thailand", "Myanmar", "other"]
    frac = country_hours_fraction(countries)

    assert set(frac.keys()) == {"Thailand", "Myanmar", "Laos", "other"}
    assert frac["Thailand"] == pytest.approx(0.5)
    assert frac["Myanmar"] == pytest.approx(0.25)
    assert frac["Laos"] == pytest.approx(0.0)
    assert frac["other"] == pytest.approx(0.25)
    assert sum(frac.values()) == pytest.approx(1.0)


def test_country_hours_fraction_empty() -> None:
    frac = country_hours_fraction([])
    assert frac == {"Thailand": 0.0, "Myanmar": 0.0, "Laos": 0.0, "other": 0.0}


# ---------------------------------------------------------------------------
# 12. corridor_frp_by_country dedup
# ---------------------------------------------------------------------------


def test_corridor_frp_dedup() -> None:
    hotspots = pd.DataFrame(
        {
            "date": ["2025-03-18", "2025-03-18", "2025-03-19"],
            "cluster_id": [1, 2, 3],
            "centroid_lat": [19.30, 18.50, 19.30],
            "centroid_lon": [97.95, 99.00, 97.95],
            "total_frp": [500.0, 10.0, 300.0],
            "country": ["Myanmar", "Thailand", "Laos"],
        }
    )
    # Trajectory lingers within 50km of cluster 1 (Myanmar) for 3 hours on 2025-03-18,
    # then moves to near cluster 3 (Laos) on 2025-03-19. Cluster 2 (Thailand) is ~140km away.
    times = pd.to_datetime(
        ["2025-03-18T10:00:00", "2025-03-18T11:00:00", "2025-03-18T12:00:00", "2025-03-19T00:00:00"]
    ).values
    traj = BackTrajectory(
        times=times,
        lats=np.array([19.30, 19.31, 19.29, 19.30]),
        lons=np.array([97.95, 97.96, 97.94, 97.95]),
        terminated_reason="completed",
        hours_integrated=3,
    )

    result = corridor_frp_by_country(traj, hotspots, radius_km=50.0)

    assert result["Myanmar"] == pytest.approx(500.0)  # counted once, not 3x
    assert result["Laos"] == pytest.approx(300.0)
    assert result["Thailand"] == pytest.approx(0.0)  # cluster 2 is >50km from every point


def test_corridor_frp_by_country_empty_hotspots() -> None:
    traj = BackTrajectory(
        times=pd.to_datetime(["2025-03-18T00:00:00"]).values,
        lats=np.array([19.30]),
        lons=np.array([97.95]),
        terminated_reason="completed",
        hours_integrated=0,
    )
    result = corridor_frp_by_country(traj, pd.DataFrame(columns=["date", "cluster_id"]))
    assert result == {"Thailand": 0.0, "Myanmar": 0.0, "Laos": 0.0}


# ---------------------------------------------------------------------------
# 13. assess_event agreement logic
# ---------------------------------------------------------------------------


def _make_traj(country: str) -> tuple[BackTrajectory, list[str]]:
    traj = BackTrajectory(
        times=pd.to_datetime(["2025-03-18T00:00:00"]).values,
        lats=np.array([19.30]),
        lons=np.array([97.95]),
        terminated_reason="completed",
        hours_integrated=0,
    )
    return traj, [country]


class TestAssessEventAgreement:
    def test_model_foreign_and_trajectory_foreign_agree(self) -> None:
        traj, countries = _make_traj("Thailand")  # 0 foreign hours
        corridor = {
            "Myanmar": 50.0,
            "Thailand": 0.0,
            "Laos": 0.0,
        }  # but corridor foreign fire present
        result = assess_event(traj, countries, 0.6, {"Myanmar": 0.6}, corridor)
        assert result["trajectory_foreign_plausible"] is True
        assert result["agrees_with_model"] is True

    def test_model_foreign_but_trajectory_not_foreign_disagree(self) -> None:
        traj, countries = _make_traj("Thailand")
        corridor = {"Myanmar": 0.0, "Thailand": 0.0, "Laos": 0.0}
        result = assess_event(traj, countries, 0.6, {"Myanmar": 0.6}, corridor)
        assert result["trajectory_foreign_plausible"] is False
        assert result["agrees_with_model"] is False

    def test_model_not_foreign_but_trajectory_foreign_disagree(self) -> None:
        traj, countries = _make_traj("Myanmar")  # 100% foreign hours
        corridor = {"Myanmar": 0.0, "Thailand": 0.0, "Laos": 0.0}
        result = assess_event(traj, countries, 0.0, {"Thailand": 1.0}, corridor)
        assert result["trajectory_foreign_plausible"] is True
        assert result["agrees_with_model"] is False

    def test_model_not_foreign_and_trajectory_not_foreign_agree(self) -> None:
        traj, countries = _make_traj("Thailand")
        corridor = {"Myanmar": 0.0, "Thailand": 0.0, "Laos": 0.0}
        result = assess_event(traj, countries, 0.0, {"Thailand": 1.0}, corridor)
        assert result["trajectory_foreign_plausible"] is False
        assert result["agrees_with_model"] is True

    def test_null_model_attribution_gives_null_agreement(self) -> None:
        traj, countries = _make_traj("Myanmar")
        corridor = {"Myanmar": 0.0, "Thailand": 0.0, "Laos": 0.0}
        result = assess_event(traj, countries, None, None, corridor)
        assert result["agrees_with_model"] is None
        assert result["model_foreign_attribution"] is None
        assert result["model_country_attribution"] is None


# ---------------------------------------------------------------------------
# Bonus: rk2_back_step public wrapper sanity
# ---------------------------------------------------------------------------


def test_rk2_back_step_returns_none_when_leaving_domain() -> None:
    # u<0 means air moves west, so a backward step moves EAST (see oracle 7). Starting
    # near the east edge, one full-strength step overshoots past lon=99.
    wind = _uniform_field(
        u_val=-50.0,
        v_val=0.0,
        lat_range=(18.0, 20.0),
        lon_range=(97.0, 99.0),
        n_lat=3,
        n_lon=9,
        n_hours=3,
    )
    launch = wind.times[-1]
    result = rk2_back_step(wind, 19.0, 98.9, launch, dt_hours=1.0)
    assert result is None
