"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

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


class TestSourceValidation:
    """Tests for source name validation across all FIRMS functions."""

    def test_source_name_validation_sp_accepted(self, requests_mock, mock_cache_dir, monkeypatch):
        """VIIRS_NOAA20_SP should be accepted and build the correct URL."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_SP/97.0,16.0,101.5,21.0/1/2024-03-01",
            text=csv_response,
        )

        df = firms.fetch_hotspots(
            source="VIIRS_NOAA20_SP",
            day_range=1,
            date_=date(2024, 3, 1),
        )

        assert len(df) == 1
        request = requests_mock.request_history[0]
        assert "VIIRS_NOAA20_SP" in request.url
        assert "2024-03-01" in request.url

    def test_source_name_validation_nrt_accepted(self, requests_mock, mock_cache_dir, monkeypatch):
        """VIIRS_NOAA20_NRT should be accepted and build the correct URL."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        requests_mock.register_uri(
            "GET",
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key/VIIRS_NOAA20_NRT/97.0,16.0,101.5,21.0/1",
            text=csv_response,
        )

        df = firms.fetch_hotspots(source="VIIRS_NOAA20_NRT", day_range=1)

        assert len(df) == 1
        request = requests_mock.request_history[0]
        assert "VIIRS_NOAA20_NRT" in request.url

    def test_source_name_validation_invalid_raises(self, mock_cache_dir, monkeypatch):
        """An unrecognised source name should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        with pytest.raises(ValueError, match="Unknown FIRMS source"):
            firms.fetch_hotspots(source="INVALID_SOURCE", day_range=1)

    def test_source_name_validation_all_nrt_sources_accepted(
        self, requests_mock, mock_cache_dir, monkeypatch
    ):
        """All declared NRT source keys should be accepted without raising."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"
        base = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key"
        bbox = "97.0,16.0,101.5,21.0"

        for src in sorted(firms._NRT_SOURCES):
            requests_mock.register_uri(
                "GET",
                f"{base}/{src}/{bbox}/1",
                text=csv_response,
            )

        for src in sorted(firms._NRT_SOURCES):
            df = firms.fetch_hotspots(source=src, day_range=1)
            assert len(df) == 1, f"Expected rows for source={src}"

    def test_source_name_validation_all_sp_sources_accepted(
        self, requests_mock, mock_cache_dir, monkeypatch
    ):
        """All declared SP source keys should be accepted without raising."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"
        base = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key"
        bbox = "97.0,16.0,101.5,21.0"

        for src in sorted(firms._SP_SOURCES):
            requests_mock.register_uri(
                "GET",
                f"{base}/{src}/{bbox}/1/2024-03-01",
                text=csv_response,
            )

        for src in sorted(firms._SP_SOURCES):
            df = firms.fetch_hotspots(source=src, day_range=1, date_=date(2024, 3, 1))
            assert len(df) == 1, f"Expected rows for source={src}"

    def test_fetch_hotspots_historical_invalid_source_raises(self, mock_cache_dir, monkeypatch):
        """fetch_hotspots_historical with invalid source should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        with pytest.raises(ValueError, match="Unknown FIRMS source"):
            firms.fetch_hotspots_historical(
                start_date="2024-01-01",
                end_date="2024-01-05",
                source="NOT_A_SOURCE",
                cache_dir=mock_cache_dir,
            )


class TestFetchHotspotsHistorical:
    """Tests for fetch_hotspots_historical() function."""

    def _sp_url(self, date_str: str, day_range: int) -> str:
        return (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"test-api-key/VIIRS_NOAA20_SP/97.0,16.0,101.5,21.0/{day_range}/{date_str}"
        )

    def test_historical_chunks_25_day_range(self, requests_mock, mock_cache_dir, monkeypatch):
        """A 25-day SP range should result in exactly 5 HTTP requests (5+5+5+5+5).

        SP endpoint caps day_range at 5 (vs 10 for NRT); observed 2026-05-19.
        """
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"

        # Chunks: Jan 1-5 (5d), Jan 6-10 (5d), Jan 11-15 (5d), Jan 16-20 (5d), Jan 21-25 (5d)
        requests_mock.register_uri("GET", self._sp_url("2024-01-05", 5), text=csv_response)
        requests_mock.register_uri("GET", self._sp_url("2024-01-10", 5), text=csv_response)
        requests_mock.register_uri("GET", self._sp_url("2024-01-15", 5), text=csv_response)
        requests_mock.register_uri("GET", self._sp_url("2024-01-20", 5), text=csv_response)
        requests_mock.register_uri("GET", self._sp_url("2024-01-25", 5), text=csv_response)

        df = firms.fetch_hotspots_historical(
            start_date="2024-01-01",
            end_date="2024-01-25",
            source="VIIRS_NOAA20_SP",
            cache_dir=mock_cache_dir,
        )

        assert len(requests_mock.request_history) == 5
        assert len(df) == 5  # 1 row per chunk

    def test_historical_returns_combined_df(self, requests_mock, mock_cache_dir, monkeypatch):
        """Rows from all SP chunks (5-day windows) should be concatenated."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_chunk1 = "latitude,longitude\n18.5,99.5\n18.6,99.6\n"
        csv_chunk2 = "latitude,longitude\n18.7,99.7\n"
        csv_chunk3 = "latitude,longitude\n18.8,99.8\n"

        # 15-day range with 5-day SP chunks → 3 requests: Jan 1-5, Jan 6-10, Jan 11-15
        requests_mock.register_uri("GET", self._sp_url("2024-01-05", 5), text=csv_chunk1)
        requests_mock.register_uri("GET", self._sp_url("2024-01-10", 5), text=csv_chunk2)
        requests_mock.register_uri("GET", self._sp_url("2024-01-15", 5), text=csv_chunk3)

        df = firms.fetch_hotspots_historical(
            start_date="2024-01-01",
            end_date="2024-01-15",
            source="VIIRS_NOAA20_SP",
            cache_dir=mock_cache_dir,
        )

        assert len(df) == 4  # 2 + 1 + 1
        assert list(df["latitude"]) == [18.5, 18.6, 18.7, 18.8]

    def test_historical_single_day(self, requests_mock, mock_cache_dir, monkeypatch):
        """A single-day range should produce exactly one HTTP request."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"
        requests_mock.register_uri("GET", self._sp_url("2024-03-01", 1), text=csv_response)

        df = firms.fetch_hotspots_historical(
            start_date="2024-03-01",
            end_date="2024-03-01",
            source="VIIRS_NOAA20_SP",
            cache_dir=mock_cache_dir,
        )

        assert len(requests_mock.request_history) == 1
        assert len(df) == 1

    def test_historical_end_before_start_raises(self, mock_cache_dir, monkeypatch):
        """end_date before start_date should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        with pytest.raises(ValueError, match="end_date"):
            firms.fetch_hotspots_historical(
                start_date="2024-03-10",
                end_date="2024-03-01",
                source="VIIRS_NOAA20_SP",
                cache_dir=mock_cache_dir,
            )

    def test_historical_sp_source_in_url(self, requests_mock, mock_cache_dir, monkeypatch):
        """SP source name should appear in the request URL."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")

        csv_response = "latitude,longitude\n18.5,99.5\n"
        requests_mock.register_uri("GET", self._sp_url("2024-01-05", 5), text=csv_response)

        firms.fetch_hotspots_historical(
            start_date="2024-01-01",
            end_date="2024-01-05",
            source="VIIRS_NOAA20_SP",
            cache_dir=mock_cache_dir,
        )

        assert "VIIRS_NOAA20_SP" in requests_mock.request_history[0].url


class TestFetchHotspotsHybrid:
    """Tests for fetch_hotspots_hybrid()."""

    _BBOX_STR = "97.0,16.0,101.5,21.0"

    @staticmethod
    def _patch_today(monkeypatch, fixed_today: date) -> None:
        """Monkeypatch date.today() in the firms module to return a fixed date."""
        import datetime as _dt

        class _FakeDate(_dt.date):
            @classmethod
            def today(cls) -> date:
                return fixed_today

        monkeypatch.setattr(firms, "date", _FakeDate)

    def test_invalid_sp_source_raises(self, monkeypatch) -> None:
        """Invalid sp_source should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")
        with pytest.raises(ValueError, match="Unknown FIRMS source"):
            firms.fetch_hotspots_hybrid(sp_source="INVALID_SP")

    def test_invalid_nrt_source_raises(self, monkeypatch) -> None:
        """Invalid nrt_source should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")
        with pytest.raises(ValueError, match="Unknown FIRMS source"):
            firms.fetch_hotspots_hybrid(nrt_source="INVALID_NRT")

    def test_end_before_start_raises(self, monkeypatch) -> None:
        """end_date earlier than start_date should raise ValueError."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")
        with pytest.raises(ValueError, match="end_date"):
            firms.fetch_hotspots_hybrid(start_date="2024-03-10", end_date="2024-03-01")

    def test_old_range_uses_sp_only(self, requests_mock, mock_cache_dir, monkeypatch) -> None:
        """Dates well before the SP cutoff should only trigger SP requests, not NRT."""
        import re

        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")
        self._patch_today(monkeypatch, date(2024, 3, 15))
        # fixed today: sp_cutoff=2024-01-15, nrt_start=2024-03-05
        # Range 2022-01-01..2022-01-05 is before both → SP only
        requests_mock.register_uri("GET", re.compile(".*"), text="latitude,longitude\n18.0,99.0\n")

        firms.fetch_hotspots_hybrid(
            start_date="2022-01-01",
            end_date="2022-01-05",
            sp_lag_days=60,
            nrt_window_days=10,
            cache_dir=mock_cache_dir,
        )

        urls = [r.url for r in requests_mock.request_history]
        assert all("VIIRS_NOAA20_SP" in u for u in urls), "Only SP requests expected"
        assert all("VIIRS_NOAA20_NRT" not in u for u in urls), "No NRT requests expected"

    def test_recent_range_uses_nrt_only(self, requests_mock, mock_cache_dir, monkeypatch) -> None:
        """Dates within the SP lag window should only trigger NRT requests."""
        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")
        self._patch_today(monkeypatch, date(2024, 3, 15))
        # sp_cutoff=2024-01-15, nrt_start=2024-03-05
        # start=2024-03-08 > sp_cutoff → SP skipped; end=2024-03-15 → NRT fetched
        # NRT day_range = (2024-03-15 - 2024-03-08).days + 1 = 8
        nrt_url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/test-api-key"
            f"/VIIRS_NOAA20_NRT/{self._BBOX_STR}/8/2024-03-15"
        )
        requests_mock.register_uri("GET", nrt_url, text="latitude,longitude\n18.0,99.0\n")

        df = firms.fetch_hotspots_hybrid(
            start_date="2024-03-08",
            end_date="2024-03-15",
            sp_lag_days=60,
            nrt_window_days=10,
            cache_dir=mock_cache_dir,
        )

        urls = [r.url for r in requests_mock.request_history]
        assert all("VIIRS_NOAA20_NRT" in u for u in urls), "Only NRT request expected"
        assert all("VIIRS_NOAA20_SP" not in u for u in urls), "No SP request expected"
        assert len(df) == 1

    def test_gap_warning_emitted(self, requests_mock, mock_cache_dir, monkeypatch, caplog) -> None:
        """A gap between SP and NRT coverage should emit a WARNING log."""
        import logging
        import re

        monkeypatch.setenv("FIRMS_API_KEY", "test-api-key")
        self._patch_today(monkeypatch, date(2024, 3, 15))
        # sp_cutoff=2024-01-15, nrt_start=2024-03-05 → uncoverable gap: 2024-01-16..2024-03-04
        requests_mock.register_uri("GET", re.compile(".*"), text="latitude,longitude\n18.0,99.0\n")

        with caplog.at_level(logging.WARNING, logger="src.data.scrapers.firms"):
            firms.fetch_hotspots_hybrid(
                start_date="2022-01-01",
                end_date="2024-03-15",
                sp_lag_days=60,
                nrt_window_days=10,
                cache_dir=mock_cache_dir,
            )

        assert any(
            "uncoverable gap" in r.message for r in caplog.records
        ), "Expected a gap warning in the log"


class TestBboxConstant:
    """Tests for BBOX_NORTHERN_THAILAND constant."""

    def test_bbox_northern_thailand_value(self):
        """BBOX_NORTHERN_THAILAND should be the correct bounding box."""
        assert firms.BBOX_NORTHERN_THAILAND == (97.0, 16.0, 101.5, 21.0)
