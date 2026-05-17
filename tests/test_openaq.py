"""
[TODO: NSC Disclaimer — see booklet page 44]

Tests for src/data/scrapers/openaq.py.
All tests are offline and use requests-mock for HTTP mocking.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from src.data.scrapers import openaq


@pytest.fixture
def mock_session(requests_mock):
    """Create a mock session for testing."""
    return openaq._make_session()


class TestDiscoverLocations:
    """Tests for discover_locations() function."""

    def test_discover_locations_pagination(self, requests_mock, monkeypatch):
        """Mock two pages: page 1 returns 1 location, page 2 returns empty."""
        monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
        # Reload the module to pick up new env var
        import importlib

        importlib.reload(openaq)

        # Mock page 1 with one location
        page1_response = {
            "results": [
                {
                    "id": 225579,
                    "name": "Yupparaj Wittayalai School",
                    "coordinates": {"latitude": 18.787, "longitude": 98.993},
                    "providers": [{"name": "Air4Thai"}],
                    "datetimes": {
                        "first": "2020-01-01T00:00:00Z",
                        "last": "2026-03-01T00:00:00Z",
                    },
                    "sensors": [
                        {
                            "id": 1304368,
                            "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"},
                        }
                    ],
                }
            ]
        }

        # Mock page 2 with empty results
        page2_response = {"results": []}

        requests_mock.register_uri(
            "GET",
            "https://api.openaq.org/v3/locations",
            [
                {"json": page1_response, "status_code": 200},
                {"json": page2_response, "status_code": 200},
            ],
        )

        df = openaq.discover_locations()

        assert len(df) == 1
        assert df.iloc[0]["location_id"] == 225579
        assert df.iloc[0]["name"] == "Yupparaj Wittayalai School"
        assert df.iloc[0]["lat"] == 18.787
        assert df.iloc[0]["lon"] == 98.993
        assert df.iloc[0]["provider"] == "Air4Thai"
        assert df.iloc[0]["sensor_id_pm25"] == 1304368
        assert list(df.columns) == [
            "location_id",
            "name",
            "lat",
            "lon",
            "provider",
            "datetime_first",
            "datetime_last",
            "sensor_id_pm25",
        ]

    def test_discover_locations_skips_no_pm25_sensor(self, requests_mock, monkeypatch):
        """Location with no PM2.5 sensor should be skipped."""
        monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
        import importlib

        importlib.reload(openaq)

        # Location with parameter.id = 5 (not PM2.5)
        page1_response = {
            "results": [
                {
                    "id": 123456,
                    "name": "Some Station",
                    "coordinates": {"latitude": 18.0, "longitude": 99.0},
                    "providers": [{"name": "Other"}],
                    "datetimes": {"first": "2020-01-01T00:00:00Z", "last": "2026-01-01T00:00:00Z"},
                    "sensors": [
                        {
                            "id": 999,
                            "parameter": {"id": 5, "name": "co2", "units": "ppm"},
                        }
                    ],
                }
            ]
        }

        page2_response = {"results": []}

        requests_mock.register_uri(
            "GET",
            "https://api.openaq.org/v3/locations",
            [
                {"json": page1_response, "status_code": 200},
                {"json": page2_response, "status_code": 200},
            ],
        )

        df = openaq.discover_locations()

        assert len(df) == 0

    def test_discover_locations_bom_safe(self, requests_mock, monkeypatch):
        """Response with UTF-8 BOM should be decoded correctly."""
        monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
        import importlib

        importlib.reload(openaq)

        page1_response = {
            "results": [
                {
                    "id": 111,
                    "name": "Test Station",
                    "coordinates": {"latitude": 18.0, "longitude": 99.0},
                    "providers": [{"name": "Air4Thai"}],
                    "datetimes": {"first": "2020-01-01T00:00:00Z", "last": "2026-01-01T00:00:00Z"},
                    "sensors": [
                        {
                            "id": 222,
                            "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"},
                        }
                    ],
                }
            ]
        }

        page2_response = {"results": []}

        # Create BOM-prefixed JSON responses
        page1_json = json.dumps(page1_response).encode("utf-8")
        page1_bom = b"\xef\xbb\xbf" + page1_json

        page2_json = json.dumps(page2_response).encode("utf-8")
        page2_bom = b"\xef\xbb\xbf" + page2_json

        requests_mock.register_uri(
            "GET",
            "https://api.openaq.org/v3/locations",
            [
                {"content": page1_bom, "status_code": 200},
                {"content": page2_bom, "status_code": 200},
            ],
        )

        # Should not raise
        df = openaq.discover_locations()
        assert len(df) == 1


class TestCurateTrainingStations:
    """Tests for curate_training_stations() function."""

    def test_curate_training_stations_filters_provider(self):
        """Only Air4Thai stations should be kept."""
        df = pd.DataFrame(
            {
                "location_id": [1, 2],
                "name": ["Station A", "Station B"],
                "lat": [18.0, 19.0],
                "lon": [99.0, 100.0],
                "provider": ["Air4Thai", "Other"],
                "datetime_first": ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
                "datetime_last": ["2026-12-31T00:00:00Z", "2026-12-31T00:00:00Z"],
                "sensor_id_pm25": [123, 456],
            }
        )

        result = openaq.curate_training_stations(df)

        assert len(result) == 1
        assert result.iloc[0]["provider"] == "Air4Thai"
        assert result.iloc[0]["location_id"] == 1

    def test_curate_training_stations_filters_dates(self):
        """Station must have data before first_before and after last_after."""
        df = pd.DataFrame(
            {
                "location_id": [1, 2],
                "name": ["Station A", "Station B"],
                "lat": [18.0, 19.0],
                "lon": [99.0, 100.0],
                "provider": ["Air4Thai", "Air4Thai"],
                "datetime_first": ["2020-01-01T00:00:00Z", "2022-06-01T00:00:00Z"],
                "datetime_last": ["2026-12-31T00:00:00Z", "2025-12-31T00:00:00Z"],
                "sensor_id_pm25": [123, 456],
            }
        )

        result = openaq.curate_training_stations(df)

        # Only station 1 should pass (first_before=2022-01-01 and last_after=2026-01-01)
        assert len(result) == 1
        assert result.iloc[0]["location_id"] == 1


class TestFetchMeasurements:
    """Tests for fetch_measurements() function."""

    def test_fetch_measurements_iso_datetime_format(self, requests_mock, monkeypatch):
        """URL should contain ISO 8601 datetime with Z suffix."""
        monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
        import importlib

        importlib.reload(openaq)

        page1_response = {
            "results": [
                {
                    "period": {"datetimeFrom": {"utc": "2024-03-01T00:00:00Z"}},
                    "value": 42.5,
                    "parameter": {"name": "pm25", "units": "µg/m³"},
                }
            ]
        }

        page2_response = {"results": []}

        requests_mock.register_uri(
            "GET",
            "https://api.openaq.org/v3/sensors/1234/measurements",
            [
                {"json": page1_response, "status_code": 200},
                {"json": page2_response, "status_code": 200},
            ],
        )

        df = openaq.fetch_measurements(
            sensor_id=1234,
            datetime_from=datetime(2024, 3, 1, 0, 0, 0),
            datetime_to=datetime(2024, 3, 31, 23, 59, 59),
        )

        # Check that the request was made with correct datetime format
        history = requests_mock.request_history
        assert len(history) >= 2
        first_request = history[0]

        # Check URL contains ISO 8601 format with Z
        assert "datetime_from=2024-03-01T00%3A00%3A00Z" in first_request.url
        assert "datetime_to=2024-03-31T23%3A59%3A59Z" in first_request.url

        assert len(df) == 1
        assert df.iloc[0]["value"] == 42.5

    def test_fetch_measurements_retry_on_429(self, requests_mock, monkeypatch, mocker):
        """Should retry on 429 (rate limit) and eventually succeed."""
        monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
        import importlib

        importlib.reload(openaq)

        # Patch time.sleep to avoid actual delays
        mocker.patch("time.sleep")

        page_success = {
            "results": [
                {
                    "period": {"datetimeFrom": {"utc": "2024-03-01T00:00:00Z"}},
                    "value": 42.5,
                    "parameter": {"name": "pm25", "units": "µg/m³"},
                }
            ]
        }

        page_empty = {"results": []}

        requests_mock.register_uri(
            "GET",
            "https://api.openaq.org/v3/sensors/5678/measurements",
            [
                {"status_code": 429},
                {"status_code": 429},
                {"json": page_success, "status_code": 200},
                {"json": page_empty, "status_code": 200},
            ],
        )

        df = openaq.fetch_measurements(
            sensor_id=5678,
            datetime_from=datetime(2024, 3, 1),
            datetime_to=datetime(2024, 3, 31),
        )

        assert len(df) == 1
        assert df.iloc[0]["value"] == 42.5

    def test_fetch_measurements_pagination(self, requests_mock, monkeypatch):
        """Should paginate through all results."""
        monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
        import importlib

        importlib.reload(openaq)

        page1 = {
            "results": [
                {
                    "period": {"datetimeFrom": {"utc": "2024-03-01T00:00:00Z"}},
                    "value": 10.0,
                    "parameter": {"name": "pm25", "units": "µg/m³"},
                },
                {
                    "period": {"datetimeFrom": {"utc": "2024-03-02T00:00:00Z"}},
                    "value": 20.0,
                    "parameter": {"name": "pm25", "units": "µg/m³"},
                },
            ]
        }

        page2 = {
            "results": [
                {
                    "period": {"datetimeFrom": {"utc": "2024-03-03T00:00:00Z"}},
                    "value": 30.0,
                    "parameter": {"name": "pm25", "units": "µg/m³"},
                }
            ]
        }

        page3 = {"results": []}

        requests_mock.register_uri(
            "GET",
            "https://api.openaq.org/v3/sensors/9999/measurements",
            [
                {"json": page1, "status_code": 200},
                {"json": page2, "status_code": 200},
                {"json": page3, "status_code": 200},
            ],
        )

        df = openaq.fetch_measurements(
            sensor_id=9999,
            datetime_from=datetime(2024, 3, 1),
            datetime_to=datetime(2024, 3, 31),
        )

        assert len(df) == 3
        assert df.iloc[0]["value"] == 10.0
        assert df.iloc[1]["value"] == 20.0
        assert df.iloc[2]["value"] == 30.0


class TestFetchLocationsPage:
    """Tests for _fetch_locations_page() function."""

    def test_fetch_locations_page_sets_encoding(self, requests_mock, monkeypatch):
        """_fetch_locations_page should set response.encoding to utf-8-sig."""
        monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
        import importlib

        importlib.reload(openaq)

        response_data = {"results": []}
        requests_mock.register_uri(
            "GET",
            "https://api.openaq.org/v3/locations",
            json=response_data,
        )

        session = openaq._make_session()
        results = openaq._fetch_locations_page(session, (97.0, 16.0, 101.5, 21.0), 2, 1)

        assert results == []
        assert requests_mock.last_request is not None


class TestFetchMeasurementsPage:
    """Tests for _fetch_measurements_page() function."""

    def test_fetch_measurements_page_datetime_format(self, requests_mock, monkeypatch):
        """_fetch_measurements_page should format datetime params correctly."""
        monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
        import importlib

        importlib.reload(openaq)

        response_data = {"results": []}
        requests_mock.register_uri(
            "GET",
            "https://api.openaq.org/v3/sensors/123/measurements",
            json=response_data,
        )

        session = openaq._make_session()
        results = openaq._fetch_measurements_page(
            session,
            sensor_id=123,
            datetime_from=datetime(2024, 3, 1, 12, 30, 45),
            datetime_to=datetime(2024, 3, 2, 18, 45, 30),
            page_size=500,
            page=1,
        )

        assert results == []
        # Verify the URL has the correct datetime format
        assert "datetime_from=2024-03-01T12%3A30%3A45Z" in requests_mock.last_request.url
        assert "datetime_to=2024-03-02T18%3A45%3A30Z" in requests_mock.last_request.url
