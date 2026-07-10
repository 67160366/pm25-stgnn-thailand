"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

Tests for src/data/scrapers/openmeteo.py.
All tests are offline and use requests-mock for HTTP mocking.
"""

import pandas as pd
import pytest
import requests

from src.data.scrapers import openmeteo


def _payload(
    times: list[str],
    temperature: list,
    dew_point: list,
    wind_speed: list,
    wind_direction: list,
    blh: list,
) -> dict:
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temperature,
            "dew_point_2m": dew_point,
            "wind_speed_10m": wind_speed,
            "wind_direction_10m": wind_direction,
            "boundary_layer_height": blh,
        }
    }


class TestFetchForecast:
    """Tests for fetch_forecast()."""

    def test_columns_and_kelvin_conversion(self, requests_mock):
        """Response should have expected columns and exact Kelvin conversion."""
        payload = _payload(
            times=["2026-05-20T00:00", "2026-05-20T01:00"],
            temperature=[25.0, 20.0],
            dew_point=[18.0, 15.0],
            wind_speed=[2.0, 2.0],
            wind_direction=[0.0, 90.0],
            blh=[500.0, 600.0],
        )
        requests_mock.get(openmeteo.OPENMETEO_URL, json=payload)

        df = openmeteo.fetch_forecast(lat=18.787, lon=98.993)

        assert list(df.columns) == ["time", "u10", "v10", "t2m", "d2m", "blh"]
        assert len(df) == 2
        assert df["t2m"].iloc[0] == pytest.approx(25.0 + 273.15)
        assert df["t2m"].iloc[1] == pytest.approx(20.0 + 273.15)
        assert df["d2m"].iloc[0] == pytest.approx(18.0 + 273.15)
        assert df["d2m"].iloc[1] == pytest.approx(15.0 + 273.15)

    def test_wind_from_north(self, requests_mock):
        """Wind FROM north (0 deg) at 2 m/s -> u~0, v~-2 (blowing southward)."""
        payload = _payload(
            times=["2026-05-20T00:00"],
            temperature=[25.0],
            dew_point=[18.0],
            wind_speed=[2.0],
            wind_direction=[0.0],
            blh=[500.0],
        )
        requests_mock.get(openmeteo.OPENMETEO_URL, json=payload)

        df = openmeteo.fetch_forecast(lat=18.787, lon=98.993)

        assert df["u10"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert df["v10"].iloc[0] == pytest.approx(-2.0)

    def test_wind_from_east(self, requests_mock):
        """Wind FROM east (90 deg) at 2 m/s -> u~-2, v~0 (blowing westward)."""
        payload = _payload(
            times=["2026-05-20T00:00"],
            temperature=[25.0],
            dew_point=[18.0],
            wind_speed=[2.0],
            wind_direction=[90.0],
            blh=[500.0],
        )
        requests_mock.get(openmeteo.OPENMETEO_URL, json=payload)

        df = openmeteo.fetch_forecast(lat=18.787, lon=98.993)

        assert df["u10"].iloc[0] == pytest.approx(-2.0)
        assert df["v10"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_time_is_utc_tz_aware(self, requests_mock):
        """time column should be a UTC tz-aware pandas datetime."""
        payload = _payload(
            times=["2026-05-20T00:00", "2026-05-20T01:00"],
            temperature=[25.0, 20.0],
            dew_point=[18.0, 15.0],
            wind_speed=[2.0, 2.0],
            wind_direction=[0.0, 90.0],
            blh=[500.0, 600.0],
        )
        requests_mock.get(openmeteo.OPENMETEO_URL, json=payload)

        df = openmeteo.fetch_forecast(lat=18.787, lon=98.993)

        assert isinstance(df["time"].dtype, pd.DatetimeTZDtype)
        assert str(df["time"].dt.tz) == "UTC"

    def test_null_blh_becomes_nan(self, requests_mock):
        """A null boundary_layer_height entry should become NaN, not filled."""
        payload = _payload(
            times=["2026-05-20T00:00", "2026-05-20T01:00"],
            temperature=[25.0, 20.0],
            dew_point=[18.0, 15.0],
            wind_speed=[2.0, 2.0],
            wind_direction=[0.0, 90.0],
            blh=[500.0, None],
        )
        requests_mock.get(openmeteo.OPENMETEO_URL, json=payload)

        df = openmeteo.fetch_forecast(lat=18.787, lon=98.993)

        assert df["blh"].iloc[0] == pytest.approx(500.0)
        assert pd.isna(df["blh"].iloc[1])

    def test_http_500_raises(self, requests_mock):
        """A 500 status should raise requests.HTTPError."""
        requests_mock.get(openmeteo.OPENMETEO_URL, status_code=500)

        with pytest.raises(requests.HTTPError):
            openmeteo.fetch_forecast(lat=18.787, lon=98.993)


class TestFetchForecastStations:
    """Tests for fetch_forecast_stations()."""

    def test_concatenates_stations_and_calls_twice(self, requests_mock):
        """Two stations -> concatenated frame with station_id, endpoint called twice."""
        payload = _payload(
            times=["2026-05-20T00:00"],
            temperature=[25.0],
            dew_point=[18.0],
            wind_speed=[2.0],
            wind_direction=[0.0],
            blh=[500.0],
        )
        requests_mock.get(openmeteo.OPENMETEO_URL, json=payload)

        stations = pd.DataFrame(
            {
                "station_id": [1, 2],
                "lat": [18.787, 19.9],
                "lon": [98.993, 99.8],
            }
        )

        df = openmeteo.fetch_forecast_stations(stations)

        assert list(df.columns) == ["station_id", "time", "u10", "v10", "t2m", "d2m", "blh"]
        assert set(df["station_id"]) == {1, 2}
        assert len(df) == 2
        assert requests_mock.call_count == 2
