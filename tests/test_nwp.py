"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

Tests for the dashboard live-forecast helpers (app/lib/nwp.py, app/lib/air4thai.py).
All tests are offline: HTTP is mocked with requests-mock and scalers are faked.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.lib import air4thai
from app.lib import nwp as nwp_lib
from src.data.scrapers import openmeteo


def _nwp_payload(times: list[str]) -> dict:
    n = len(times)
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [25.0] * n,
            "dew_point_2m": [18.0] * n,
            "wind_speed_10m": [2.0] * n,
            "wind_direction_10m": [0.0] * n,
            "boundary_layer_height": [500.0] * n,
        }
    }


class TestFetchForecastWindow:
    """fetch_forecast_window() adds past_days and reuses the openmeteo parser."""

    def test_passes_past_days_and_parses(self, requests_mock):
        requests_mock.get(
            openmeteo.OPENMETEO_URL,
            json=_nwp_payload(["2026-07-10T10:00", "2026-07-10T11:00"]),
        )
        df = nwp_lib.fetch_forecast_window(lat=18.79, lon=98.99, past_days=3, forecast_days=3)
        assert list(df.columns) == ["time", "u10", "v10", "t2m", "d2m", "blh"]
        assert len(df) == 2
        assert df["t2m"].iloc[0] == pytest.approx(25.0 + 273.15)
        qs = requests_mock.last_request.qs
        assert qs["past_days"] == ["3"]
        assert qs["forecast_days"] == ["3"]


class TestBuildLiveWindow:
    """build_live_window() shaping, per-station scaling, coverage, exclusion."""

    def _inputs(self):
        anchor = pd.Timestamp("2026-07-10T12:00", tz="UTC")
        slots = pd.date_range(end=anchor, periods=3, freq="1h")  # 10:00, 11:00, 12:00 UTC
        station_ids = [1, 2]
        scalers = {
            1: {"center_": 15.0, "scale_": 22.0},
            2: {"center_": 10.0, "scale_": 5.0},
        }
        # Station 1: full window (raw 15, 37, 15 -> scaled 0, 1, 0).
        # Station 2: only 1 of 3 hours -> coverage 1/3 < 0.7 -> excluded.
        pm25_hist = pd.DataFrame(
            {
                "station_id": [1, 1, 1, 2],
                "time": [slots[0], slots[1], slots[2], slots[2]],
                "pm25": [15.0, 37.0, 15.0, 12.0],
            }
        )
        nwp_rows = []
        for sid in station_ids:
            for s in slots:
                nwp_rows.append(
                    {
                        "station_id": sid,
                        "time": s,
                        "u10": 1.5,
                        "v10": -2.0,
                        "t2m": 300.0,
                        "d2m": 290.0,
                        "blh": 500.0,
                    }
                )
        nwp = pd.DataFrame(nwp_rows)
        return pm25_hist, nwp, scalers, station_ids, anchor

    def test_shape_and_feature_order(self):
        pm25_hist, nwp, scalers, station_ids, anchor = self._inputs()
        lw = nwp_lib.build_live_window(pm25_hist, nwp, scalers, station_ids, anchor, window_in=3)
        assert lw.x.shape == (2, 3, len(nwp_lib.FEATURE_ORDER))
        assert lw.x.dtype == np.float32

    def test_pm25_scaling(self):
        pm25_hist, nwp, scalers, station_ids, anchor = self._inputs()
        lw = nwp_lib.build_live_window(pm25_hist, nwp, scalers, station_ids, anchor, window_in=3)
        # Feature 0 is pm25_scaled; station 1 -> (raw-15)/22 == [0, 1, 0].
        np.testing.assert_allclose(lw.x[0, :, 0], [0.0, 1.0, 0.0], atol=1e-6)

    def test_low_coverage_station_excluded_and_center_filled(self):
        pm25_hist, nwp, scalers, station_ids, anchor = self._inputs()
        lw = nwp_lib.build_live_window(pm25_hist, nwp, scalers, station_ids, anchor, window_in=3)
        assert lw.excluded == [2]
        assert lw.coverage[1] == pytest.approx(1.0)
        assert lw.coverage[2] == pytest.approx(1 / 3)
        # Excluded station's pm25_scaled column is all zero (== center in raw space).
        np.testing.assert_allclose(lw.x[1, :, 0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_cyclic_and_nwp_features(self):
        pm25_hist, nwp, scalers, station_ids, anchor = self._inputs()
        lw = nwp_lib.build_live_window(pm25_hist, nwp, scalers, station_ids, anchor, window_in=3)
        # hour at last slot = 12 UTC -> hour_sin = sin(2*pi*12/24) ~ 0 (feature idx 1).
        assert lw.x[0, 2, 1] == pytest.approx(0.0, abs=1e-6)
        # NWP features enter raw (idx 5..9): u10=1.5, t2m=300 at every slot.
        assert lw.x[0, 0, 5] == pytest.approx(1.5)
        assert lw.x[0, 0, 7] == pytest.approx(300.0)

    def test_anchor_wind_is_last_slot(self):
        pm25_hist, nwp, scalers, station_ids, anchor = self._inputs()
        lw = nwp_lib.build_live_window(pm25_hist, nwp, scalers, station_ids, anchor, window_in=3)
        np.testing.assert_allclose(lw.anchor_u, [1.5, 1.5])
        np.testing.assert_allclose(lw.anchor_v, [-2.0, -2.0])

    def test_observed_is_last_finite(self):
        pm25_hist, nwp, scalers, station_ids, anchor = self._inputs()
        lw = nwp_lib.build_live_window(pm25_hist, nwp, scalers, station_ids, anchor, window_in=3)
        assert lw.observed[0] == pytest.approx(15.0)
        assert lw.observed[1] == pytest.approx(12.0)


class TestAir4ThaiHistory:
    """fetch_history() parses local time to UTC and screens sentinel values."""

    def test_local_to_utc_and_sentinel(self, requests_mock):
        payload = {
            "result": "OK",
            "error": "",
            "stations": [
                {
                    "stationID": "35t",
                    "params": ["PM25"],
                    "data": [
                        {"DATETIMEDATA": "2026-07-10 12:00:00", "PM25": 30.0},
                        {"DATETIMEDATA": "2026-07-10 13:00:00", "PM25": -1},
                    ],
                }
            ],
        }
        requests_mock.get(air4thai.HISTORY_URL, json=payload)
        df = air4thai.fetch_history(
            "35t", pd.Timestamp("2026-07-10").date(), pd.Timestamp("2026-07-10").date()
        )
        assert list(df.columns) == ["time", "pm25"]
        # 12:00 Asia/Bangkok (UTC+7) -> 05:00 UTC.
        assert df["time"].iloc[0] == pd.Timestamp("2026-07-10T05:00", tz="UTC")
        assert str(df["time"].dt.tz) == "UTC"
        # -1 sentinel becomes NaN.
        assert pd.isna(df["pm25"].iloc[1])
        assert df["pm25"].iloc[0] == pytest.approx(30.0)

    def test_empty_stations_returns_empty(self, requests_mock):
        requests_mock.get(air4thai.HISTORY_URL, json={"result": "OK", "stations": []})
        df = air4thai.fetch_history(
            "35t", pd.Timestamp("2026-07-10").date(), pd.Timestamp("2026-07-10").date()
        )
        assert df.empty


class TestStationCodeMap:
    """load_station_code_map() reads the audit JSON down to {id: code}."""

    def test_reads_air4thai_id(self, tmp_path):
        p = tmp_path / "codes.json"
        p.write_text(
            json.dumps(
                {"225669": {"air4thai_id": "35t", "distance_m": 0.0, "needs_verification": False}}
            ),
            encoding="utf-8",
        )
        m = air4thai.load_station_code_map(p)
        assert m == {225669: "35t"}

    def test_missing_file_returns_empty(self, tmp_path):
        assert air4thai.load_station_code_map(tmp_path / "nope.json") == {}
