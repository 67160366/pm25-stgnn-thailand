"""
[TODO: NSC Disclaimer — see booklet page 44]

Tests for src/data/scrapers/era5.py.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.scrapers import era5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stations_df(n: int = 3) -> pd.DataFrame:
    """Return a minimal stations DataFrame matching interpolate_to_stations contract."""
    return pd.DataFrame(
        {
            "station_id": list(range(1, n + 1)),
            "latitude": [18.0 + i * 0.5 for i in range(n)],
            "longitude": [98.0 + i * 0.5 for i in range(n)],
        }
    )


def _make_era5_dataset(
    lats: list[float] | None = None,
    lons: list[float] | None = None,
    times: int = 4,
    coord_style: str = "lat_lon",
) -> xr.Dataset:
    """Build a minimal in-memory ERA5-like Dataset for testing.

    Args:
        lats: Latitude grid values.
        lons: Longitude grid values.
        times: Number of hourly timesteps.
        coord_style: "lat_lon" uses lat/lon; "latitude_longitude" uses the
            long form names that some API versions produce.

    Returns:
        xr.Dataset with u10, v10, t2m, d2m, blh variables.
    """
    if lats is None:
        lats = [17.5, 18.0, 18.5, 19.0]
    if lons is None:
        lons = [97.5, 98.0, 98.5, 99.0]

    time_coord = pd.date_range("2022-01-01", periods=times, freq="h")
    rng = np.random.default_rng(42)
    shape = (times, len(lats), len(lons))

    if coord_style == "latitude_longitude":
        coords = {"time": time_coord, "latitude": lats, "longitude": lons}
        dims = ("time", "latitude", "longitude")
    else:
        coords = {"time": time_coord, "lat": lats, "lon": lons}
        dims = ("time", "lat", "lon")

    def _da(data: np.ndarray) -> xr.DataArray:
        return xr.DataArray(data.astype("float32"), dims=dims, coords=coords)

    ds = xr.Dataset(
        {
            "u10": _da(rng.standard_normal(shape)),
            "v10": _da(rng.standard_normal(shape)),
            "t2m": _da(rng.standard_normal(shape) + 300),
            "d2m": _da(rng.standard_normal(shape) + 295),
            "blh": _da(np.abs(rng.standard_normal(shape) * 500)),
        }
    )
    return ds


# ---------------------------------------------------------------------------
# _normalise_coords
# ---------------------------------------------------------------------------


class TestNormaliseCoords:
    """Tests for _normalise_coords helper."""

    def test_lat_lon_already_present_no_rename(self):
        """Dataset with lat/lon coords should not be modified."""
        ds = _make_era5_dataset(coord_style="lat_lon")
        result = era5._normalise_coords(ds)
        assert "lat" in result.coords
        assert "lon" in result.coords

    def test_latitude_longitude_renamed_to_lat_lon(self):
        """Dataset with latitude/longitude coords should be renamed."""
        ds = _make_era5_dataset(coord_style="latitude_longitude")
        result = era5._normalise_coords(ds)
        assert "lat" in result.coords
        assert "lon" in result.coords
        assert "latitude" not in result.coords
        assert "longitude" not in result.coords


# ---------------------------------------------------------------------------
# _find_var
# ---------------------------------------------------------------------------


class TestFindVar:
    """Tests for _find_var helper."""

    def test_finds_first_alias(self):
        """Returns first alias that exists as a data variable."""
        ds = _make_era5_dataset()
        assert era5._find_var(ds, ["u10", "U10M", "10u"]) == "u10"

    def test_returns_none_when_none_present(self):
        """Returns None when no alias is present in the dataset."""
        ds = _make_era5_dataset()
        assert era5._find_var(ds, ["nonexistent", "also_missing"]) is None

    def test_falls_back_to_later_alias(self):
        """Returns the second alias if the first is absent."""
        ds = _make_era5_dataset()
        assert era5._find_var(ds, ["nonexistent", "v10"]) == "v10"


# ---------------------------------------------------------------------------
# interpolate_to_stations
# ---------------------------------------------------------------------------


class TestInterpolateToStations:
    """Tests for interpolate_to_stations."""

    def test_output_columns(self, tmp_path):
        """Output DataFrame has the expected column set."""
        ds = _make_era5_dataset()
        nc_path = tmp_path / "era5_test.nc"
        ds.to_netcdf(nc_path)
        stations = _make_stations_df(3)

        result = era5.interpolate_to_stations(nc_path, stations)

        assert set(result.columns) == {"station_id", "datetime", "u10", "v10", "t2m", "d2m", "blh"}

    def test_row_count(self, tmp_path):
        """Row count equals timesteps x station count."""
        n_times = 4
        n_stations = 3
        ds = _make_era5_dataset(times=n_times)
        nc_path = tmp_path / "era5_test.nc"
        ds.to_netcdf(nc_path)
        stations = _make_stations_df(n_stations)

        result = era5.interpolate_to_stations(nc_path, stations)

        assert len(result) == n_times * n_stations

    def test_datetime_is_utc_aware(self, tmp_path):
        """datetime column should be UTC-aware."""
        ds = _make_era5_dataset()
        nc_path = tmp_path / "era5_test.nc"
        ds.to_netcdf(nc_path)
        stations = _make_stations_df(2)

        result = era5.interpolate_to_stations(nc_path, stations)

        assert result["datetime"].dt.tz is not None
        assert str(result["datetime"].dt.tz) == "UTC"

    def test_station_ids_present(self, tmp_path):
        """All station IDs from stations_df appear in output."""
        ds = _make_era5_dataset()
        nc_path = tmp_path / "era5_test.nc"
        ds.to_netcdf(nc_path)
        stations = _make_stations_df(3)

        result = era5.interpolate_to_stations(nc_path, stations)

        assert set(result["station_id"].unique()) == {1, 2, 3}

    def test_latitude_longitude_coords_accepted(self, tmp_path):
        """NetCDF with latitude/longitude coordinate names is handled correctly."""
        ds = _make_era5_dataset(coord_style="latitude_longitude")
        nc_path = tmp_path / "era5_long_coords.nc"
        ds.to_netcdf(nc_path)
        stations = _make_stations_df(2)

        result = era5.interpolate_to_stations(nc_path, stations)

        assert len(result) > 0

    def test_station_id_dtype_is_int(self, tmp_path):
        """station_id column should be integer dtype."""
        ds = _make_era5_dataset()
        nc_path = tmp_path / "era5_test.nc"
        ds.to_netcdf(nc_path)
        stations = _make_stations_df(2)

        result = era5.interpolate_to_stations(nc_path, stations)

        assert pd.api.types.is_integer_dtype(result["station_id"])


# ---------------------------------------------------------------------------
# load_wind_field
# ---------------------------------------------------------------------------


class TestLoadWindField:
    """Tests for load_wind_field."""

    def test_raises_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for a non-existent path."""
        from datetime import UTC, datetime

        nc_path = tmp_path / "does_not_exist.nc"
        dt = datetime(2022, 1, 1, 0, 0, tzinfo=UTC)

        with pytest.raises(FileNotFoundError):
            era5.load_wind_field(nc_path, dt)

    def test_output_shape_and_dims(self, tmp_path):
        """Returns DataArray with dims (time, lat, lon, component) and shape (1, ...)."""
        from datetime import UTC, datetime

        ds = _make_era5_dataset(times=4)
        nc_path = tmp_path / "era5_wind.nc"
        ds.to_netcdf(nc_path)

        dt = datetime(2022, 1, 1, 1, 0, tzinfo=UTC)
        result = era5.load_wind_field(nc_path, dt)

        assert result.dims == ("time", "lat", "lon", "component")
        assert result.sizes["time"] == 1
        assert result.sizes["component"] == 2

    def test_component_0_is_u10_component_1_is_v10(self, tmp_path):
        """component=0 should be u10, component=1 should be v10."""
        from datetime import UTC, datetime

        ds = _make_era5_dataset(times=2)
        nc_path = tmp_path / "era5_wind.nc"
        ds.to_netcdf(nc_path)

        dt = datetime(2022, 1, 1, 0, 0, tzinfo=UTC)
        result = era5.load_wind_field(nc_path, dt)

        # Values should be finite floats (not all-NaN).
        assert not np.all(np.isnan(result.isel(time=0, component=0).values))
        assert not np.all(np.isnan(result.isel(time=0, component=1).values))

    def test_raises_key_error_missing_u10(self, tmp_path):
        """Raises KeyError if u10 variable is absent from the file."""
        from datetime import UTC, datetime

        ds = _make_era5_dataset()
        ds_no_u = ds.drop_vars("u10")
        nc_path = tmp_path / "era5_no_u10.nc"
        ds_no_u.to_netcdf(nc_path)

        dt = datetime(2022, 1, 1, 0, 0, tzinfo=UTC)
        with pytest.raises(KeyError):
            era5.load_wind_field(nc_path, dt)

    def test_latitude_longitude_coords_accepted(self, tmp_path):
        """NetCDF with latitude/longitude coordinate names is handled correctly."""
        from datetime import UTC, datetime

        ds = _make_era5_dataset(coord_style="latitude_longitude", times=2)
        nc_path = tmp_path / "era5_longcoords.nc"
        ds.to_netcdf(nc_path)

        dt = datetime(2022, 1, 1, 0, 0, tzinfo=UTC)
        result = era5.load_wind_field(nc_path, dt)

        assert result.sizes["time"] == 1


# ---------------------------------------------------------------------------
# fetch_era5 — unit tests (no network calls, uses monkeypatching)
# ---------------------------------------------------------------------------


class TestFetchEra5:
    """Tests for fetch_era5() function."""

    def test_raises_file_not_found_when_parquet_absent(self, tmp_path, monkeypatch):
        """Raises FileNotFoundError when stations_metadata.parquet does not exist."""
        absent = tmp_path / "nonexistent.parquet"
        monkeypatch.setattr(era5, "_STATIONS_PARQUET", absent)

        with pytest.raises(FileNotFoundError, match=r"stations_metadata\.parquet"):
            era5.fetch_era5(output_dir=tmp_path)

    def test_raises_file_not_found_message_mentions_discover(self, tmp_path, monkeypatch):
        """FileNotFoundError message should suggest running the discover script."""
        absent = tmp_path / "nonexistent.parquet"
        monkeypatch.setattr(era5, "_STATIONS_PARQUET", absent)

        with pytest.raises(FileNotFoundError) as exc_info:
            era5.fetch_era5(output_dir=tmp_path)

        assert "discover" in str(exc_info.value).lower()

    def test_orchestrates_download_and_interpolation(self, tmp_path, monkeypatch):
        """fetch_era5 calls download_era5_year and interpolate_to_stations per year."""
        from datetime import date

        # Write a minimal stations parquet.
        stations = pd.DataFrame(
            {
                "location_id": [1, 2],
                "lat": [18.0, 18.5],
                "lon": [98.0, 98.5],
                "name": ["A", "B"],
            }
        )
        parquet_path = tmp_path / "stations_metadata.parquet"
        stations.to_parquet(parquet_path, index=False)
        monkeypatch.setattr(era5, "_STATIONS_PARQUET", parquet_path)

        # Fake NetCDF file produced by download.
        ds = _make_era5_dataset(times=2)
        nc_path = tmp_path / "era5_2022.nc"
        ds.to_netcdf(nc_path)

        calls: list[int] = []

        def fake_download(year: int, output_dir, bbox) -> "Path":  # noqa: F821
            calls.append(year)
            return nc_path

        monkeypatch.setattr(era5, "download_era5_year", fake_download)

        result = era5.fetch_era5(
            date_from=date(2022, 1, 1),
            date_to=date(2022, 12, 31),
            output_dir=tmp_path,
        )

        assert calls == [2022]
        assert set(result.columns) == {"station_id", "datetime", "u10", "v10", "t2m", "d2m", "blh"}

    def test_parquet_with_lat_lon_columns_accepted(self, tmp_path, monkeypatch):
        """fetch_era5 accepts parquets using lat/lon column names (discover output)."""
        from datetime import date

        stations = pd.DataFrame(
            {
                "location_id": [10],
                "lat": [18.0],
                "lon": [98.0],
            }
        )
        parquet_path = tmp_path / "stations_metadata.parquet"
        stations.to_parquet(parquet_path, index=False)
        monkeypatch.setattr(era5, "_STATIONS_PARQUET", parquet_path)

        ds = _make_era5_dataset(times=2)
        nc_path = tmp_path / "era5_2022.nc"
        ds.to_netcdf(nc_path)

        monkeypatch.setattr(era5, "download_era5_year", lambda year, output_dir, bbox: nc_path)

        result = era5.fetch_era5(
            date_from=date(2022, 1, 1),
            date_to=date(2022, 12, 31),
            output_dir=tmp_path,
        )

        assert "station_id" in result.columns

    def test_result_sorted_by_station_and_datetime(self, tmp_path, monkeypatch):
        """Output is sorted by (station_id, datetime)."""
        from datetime import date

        stations = pd.DataFrame(
            {
                "location_id": [3, 1, 2],
                "lat": [18.0, 18.5, 19.0],
                "lon": [98.0, 98.5, 99.0],
            }
        )
        parquet_path = tmp_path / "stations_metadata.parquet"
        stations.to_parquet(parquet_path, index=False)
        monkeypatch.setattr(era5, "_STATIONS_PARQUET", parquet_path)

        ds = _make_era5_dataset(times=3)
        nc_path = tmp_path / "era5_2022.nc"
        ds.to_netcdf(nc_path)

        monkeypatch.setattr(era5, "download_era5_year", lambda year, output_dir, bbox: nc_path)

        result = era5.fetch_era5(
            date_from=date(2022, 1, 1),
            date_to=date(2022, 12, 31),
            output_dir=tmp_path,
        )

        station_ids = result["station_id"].tolist()
        assert station_ids == sorted(station_ids), "Output must be sorted by station_id"


# ---------------------------------------------------------------------------
# _cds_area
# ---------------------------------------------------------------------------


class TestCdsArea:
    """Tests for _cds_area bbox conversion."""

    def test_order_is_north_west_south_east(self):
        """CDS area format is [north, west, south, east]."""
        result = era5._cds_area((97.0, 16.0, 101.5, 21.0))
        north, west, south, east = result
        assert north == 21.0
        assert west == 97.0
        assert south == 16.0
        assert east == 101.5
