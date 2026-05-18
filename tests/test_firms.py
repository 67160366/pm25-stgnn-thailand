"""
[TODO: NSC Disclaimer — see booklet page 44]

Tests for src/data/scrapers/firms.py.
All tests are offline and use requests-mock for HTTP mocking.
"""

from datetime import date

import pytest

from src.data.scrapers import firms


@pytest.fixture
def mock_cache_dir(tmp_path, monkeypatch):
    """Redirect cache directory to a temporary location for test isolation."""
    monkeypatch.setattr(firms, "_CACHE_DIR", tmp_path / "firms_cache")
    return tmp_path / "firms_cache"


class TestFetchHotspots:
    """Tests for fetch_hotspots() function."""

    def test_fetch_hotspots_valid(self, requests_mock, mock_cache_dir, monkeypatch):
        """Should fetch and parse CSV response correctly."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = (
            "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite\n"
            "18.5,99.5,320.5,1.5,1.0,2024-03-01,1200,NOAA20\n"
            "18.6,99.6,330.0,1.5,1.0,2024-03-01,1300,NOAA20\n"
        )

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_NRT/97.0,16.0,101.5,21.0/1",
            text=csv_response,
        )

        df = firms.fetch_hotspots()

        assert len(df) == 2
        assert list(df.columns) == [
            "latitude",
            "longitude",
            "brightness",
            "scan",
            "track",
            "acq_date",
            "acq_time",
            "satellite",
        ]
        assert df.iloc[0]["latitude"] == 18.5
        assert df.iloc[1]["longitude"] == 99.6

    def test_fetch_hotspots_bbox_in_url(self, requests_mock, mock_cache_dir, monkeypatch):
        """URL should contain the bbox coordinates."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_NRT/97.0,16.0,101.5,21.0/1",
            text=csv_response,
        )

        firms.fetch_hotspots(bbox=(97.0, 16.0, 101.5, 21.0), day_range=1, source="VIIRS_NOAA20_NRT")

        assert len(requests_mock.request_history) == 1
        request = requests_mock.request_history[0]
        assert "97.0,16.0,101.5,21.0" in request.url

    def test_fetch_hotspots_day_range_zero_raises(self, mock_cache_dir, monkeypatch):
        """day_range=0 should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        with pytest.raises(ValueError, match="day_range must be between 1 and 10"):
            firms.fetch_hotspots(day_range=0)

    def test_fetch_hotspots_day_range_eleven_raises(self, mock_cache_dir, monkeypatch):
        """day_range=11 should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        with pytest.raises(ValueError, match="day_range must be between 1 and 10"):
            firms.fetch_hotspots(day_range=11)

    def test_fetch_hotspots_day_range_negative_raises(self, mock_cache_dir, monkeypatch):
        """Negative day_range should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        with pytest.raises(ValueError, match="day_range must be between 1 and 10"):
            firms.fetch_hotspots(day_range=-5)

    def test_fetch_hotspots_day_range_ten_succeeds(
        self, requests_mock, mock_cache_dir, monkeypatch
    ):
        """day_range=10 should work without raising."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_NRT/97.0,16.0,101.5,21.0/10",
            text=csv_response,
        )

        df = firms.fetch_hotspots(day_range=10)

        assert len(df) == 1

    def test_fetch_hotspots_day_range_one_succeeds(
        self, requests_mock, mock_cache_dir, monkeypatch
    ):
        """day_range=1 should work without raising."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_NRT/97.0,16.0,101.5,21.0/1",
            text=csv_response,
        )

        df = firms.fetch_hotspots(day_range=1)

        assert len(df) == 1

    def test_fetch_hotspots_with_date_includes_date_in_url(
        self, requests_mock, mock_cache_dir, monkeypatch
    ):
        """When date_ is provided, URL should include the date."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_NRT/97.0,16.0,101.5,21.0/1/2024-03-15",
            text=csv_response,
        )

        df = firms.fetch_hotspots(day_range=1, date_=date(2024, 3, 15))

        assert len(df) == 1
        request = requests_mock.request_history[0]
        assert "2024-03-15" in request.url

    def test_fetch_hotspots_without_date_no_date_in_url(
        self, requests_mock, mock_cache_dir, monkeypatch
    ):
        """When date_ is None, URL should not include a date."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_NRT/97.0,16.0,101.5,21.0/1",
            text=csv_response,
        )

        df = firms.fetch_hotspots(day_range=1, date_=None)

        assert len(df) == 1
        request = requests_mock.request_history[0]
        # URL should end with day_range, not include a date
        assert request.url.endswith("/1")

    def test_fetch_hotspots_custom_source(self, requests_mock, mock_cache_dir, monkeypatch):
        """Should support different FIRMS sources."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/MODIS_NRT/97.0,16.0,101.5,21.0/1",
            text=csv_response,
        )

        df = firms.fetch_hotspots(source="MODIS_NRT", day_range=1)

        assert len(df) == 1
        request = requests_mock.request_history[0]
        assert "MODIS_NRT" in request.url

    def test_fetch_hotspots_caches_result(self, requests_mock, mock_cache_dir, monkeypatch):
        """Result should be cached to file."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_NRT/97.0,16.0,101.5,21.0/1",
            text=csv_response,
        )

        # First call should fetch
        df1 = firms.fetch_hotspots(day_range=1)
        assert len(requests_mock.request_history) == 1

        # Second call should use cache
        df2 = firms.fetch_hotspots(day_range=1)
        assert len(requests_mock.request_history) == 1  # No new request
        assert df1.equals(df2)

        # Verify cache file was created
        cache_files = list(mock_cache_dir.glob("*.csv"))
        assert len(cache_files) == 1


class TestBboxConstant:
    """Tests for BBOX_NORTHERN_THAILAND constant."""

    def test_bbox_northern_thailand_value(self):
        """BBOX_NORTHERN_THAILAND should be the correct bounding box."""
        assert firms.BBOX_NORTHERN_THAILAND == (97.0, 16.0, 101.5, 21.0)
