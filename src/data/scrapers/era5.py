# [TODO: NSC Disclaimer — see booklet page 44]

"""ERA5 reanalysis scraper for PM2.5 STGNN Northern Thailand.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Downloads ERA5 hourly wind and weather fields via cdsapi, then bilinearly
interpolates them to the 18 monitoring-station locations.  The resulting
tidy DataFrame feeds directly into preprocessing.py and graph_builder.py.

CDS API setup: place ``~/.cdsapirc`` with valid UID and API key.
See https://cds.climate.copernicus.eu/how-to-api for setup instructions.
"""

import logging
import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _open_nc(path: Path) -> xr.Dataset:
    """Open a NetCDF4 file safely on paths with non-ASCII characters.

    netCDF4's C library (and h5py's) cannot open files whose path contains
    non-ASCII characters on Windows (e.g. Thai directory names). Copying to a
    temporary file in the system temp dir (always ASCII on Windows) and using
    xr.load_dataset (reads all data into memory, then closes the file handle)
    avoids the C-level path encoding issue entirely.
    """
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copy2(path, tmp_path)
        return xr.load_dataset(tmp_path)
    finally:
        os.unlink(tmp_path)


def _write_nc(ds: xr.Dataset, path: Path) -> None:
    """Write a Dataset to a NetCDF4 file safely on non-ASCII paths."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ds.to_netcdf(tmp_path)
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ERA5 variables requested in every download.
_ERA5_VARIABLES: list[str] = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "boundary_layer_height",
]

# CDS dataset identifier for ERA5 single-level reanalysis.
_CDS_DATASET: str = "reanalysis-era5-single-levels"

# Mapping from CDS short name to tidy-DataFrame column name.
_VAR_RENAME: dict[str, str] = {
    "u10": "u10",
    "v10": "v10",
    "t2m": "t2m",
    "d2m": "d2m",
    "blh": "blh",
}

# Path to station metadata (relative to project root, also accepted as absolute).
_STATIONS_PARQUET: Path = Path("data/processed/stations_metadata.parquet")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cds_area(bbox: tuple[float, float, float, float]) -> list[float]:
    """Convert (west, south, east, north) bbox to CDS [north, west, south, east].

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.

    Returns:
        List [north, west, south, east] as expected by the CDS API.
    """
    west, south, east, north = bbox
    return [north, west, south, east]


def _all_hours() -> list[str]:
    """Return list of all 24 hour strings expected by the CDS API.

    Returns:
        List of strings in "HH:00" format from "00:00" to "23:00".
    """
    return [f"{h:02d}:00" for h in range(24)]


def _all_months() -> list[str]:
    """Return list of all 12 month strings expected by the CDS API.

    Returns:
        List of strings in "MM" format from "01" to "12".
    """
    return [f"{m:02d}" for m in range(1, 13)]


def _all_days() -> list[str]:
    """Return list of all 31 day strings expected by the CDS API.

    The CDS API silently drops days that do not exist for a given month.

    Returns:
        List of strings in "DD" format from "01" to "31".
    """
    return [f"{d:02d}" for d in range(1, 32)]


def _load_stations() -> pd.DataFrame:
    """Load station metadata from the project parquet and normalise column names.

    The parquet produced by the discover step uses ``location_id``/``lat``/``lon``;
    ``interpolate_to_stations`` expects ``station_id``/``latitude``/``longitude``.

    Returns:
        DataFrame with columns ``station_id``, ``latitude``, ``longitude``
        (plus any other columns present in the parquet).

    Raises:
        FileNotFoundError: If ``data/processed/stations_metadata.parquet`` does not exist.
        ValueError: If required coordinate columns cannot be found.
    """
    stations_path = _STATIONS_PARQUET
    if not stations_path.exists():
        raise FileNotFoundError(
            f"Station metadata not found at {stations_path.resolve()}. "
            "Run `uv run python scripts/01_download_all.py discover` first to "
            "generate the stations_metadata.parquet file."
        )

    df = pd.read_parquet(stations_path)

    col_map: dict[str, str] = {}
    if "location_id" in df.columns and "station_id" not in df.columns:
        col_map["location_id"] = "station_id"
    if "lat" in df.columns and "latitude" not in df.columns:
        col_map["lat"] = "latitude"
    if "lon" in df.columns and "longitude" not in df.columns:
        col_map["lon"] = "longitude"
    if col_map:
        df = df.rename(columns=col_map)

    required = {"station_id", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"stations_metadata.parquet is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def _filter_and_sort(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """Filter a tidy ERA5 DataFrame to [date_from, date_to] and sort.

    Args:
        df: DataFrame with a UTC-aware ``datetime`` column.
        date_from: Inclusive start date.
        date_to: Inclusive end date.

    Returns:
        Filtered, sorted copy with reset index.
    """
    dt_from = pd.Timestamp(date_from).tz_localize("UTC")
    dt_to = (
        pd.Timestamp(date_to).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )
    mask = (df["datetime"] >= dt_from) & (df["datetime"] <= dt_to)
    result = df.loc[mask].copy()
    result.sort_values(["station_id", "datetime"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# Known ERA5 short-name aliases for each output column (API version varies).
_ERA5_ALIASES: dict[str, list[str]] = {
    "u10": ["u10", "U10M", "10u"],
    "v10": ["v10", "V10M", "10v"],
    "t2m": ["t2m", "T2M", "2t"],
    "d2m": ["d2m", "D2M", "2d"],
    "blh": ["blh", "BLH", "boundary_layer_height"],
}


def _normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinate names to the expected canonical forms.

    ERA5 downloads from the new CDS API v2 use ``valid_time`` instead of
    ``time``, and ``latitude``/``longitude`` instead of ``lat``/``lon``.

    ERA5 NetCDF files produced by different API versions may use either
    ``latitude``/``longitude`` or ``lat``/``lon`` as coordinate names.

    Args:
        ds: Raw xarray Dataset as loaded from NetCDF.

    Returns:
        Dataset with coordinates renamed to ``lat``/``lon`` if necessary.
    """
    rename_map: dict[str, str] = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename_map["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename_map["valid_time"] = "time"
    return ds.rename(rename_map) if rename_map else ds


def _find_var(ds: xr.Dataset, aliases: list[str]) -> str | None:
    """Return the first alias present in ds.data_vars, or None.

    Args:
        ds: xarray Dataset to search.
        aliases: Candidate variable name strings to check in order.

    Returns:
        First matching variable name, or None if none found.
    """
    for name in aliases:
        if name in ds.data_vars:
            return name
    return None


def _extract_var_arrays(
    ds_interp: xr.Dataset,
    n_times: int,
    n_stations: int,
    nc_path: Path,
) -> dict[str, np.ndarray]:
    """Resolve and extract each ERA5 output column as a (time, station) array.

    Args:
        ds_interp: Station-interpolated Dataset.
        n_times: Number of timesteps.
        n_stations: Number of station points.
        nc_path: Source path used only for log messages.

    Returns:
        Dict mapping column name to float32 ndarray of shape (n_times, n_stations).
    """
    resolved = {col: _find_var(ds_interp, aliases) for col, aliases in _ERA5_ALIASES.items()}
    missing = [col for col, name in resolved.items() if name is None]
    if missing:
        logger.warning("ERA5 file %s missing variables for columns: %s", nc_path, missing)

    arrays: dict[str, np.ndarray] = {}
    for col, var_name in resolved.items():
        if var_name is not None:
            arrays[col] = ds_interp[var_name].values.astype(np.float32)
        else:
            arrays[col] = np.full((n_times, n_stations), np.nan, dtype=np.float32)
    return arrays


def _build_tidy_frames(
    arrays: dict[str, np.ndarray],
    times: pd.DatetimeIndex,
    station_ids: np.ndarray,
) -> list[pd.DataFrame]:
    """Build per-station tidy DataFrames from pre-extracted arrays.

    Args:
        arrays: Dict of column -> (n_times, n_stations) float32 ndarray.
        times: UTC-aware DatetimeIndex with n_times entries.
        station_ids: 1-D array of station IDs, length n_stations.

    Returns:
        List of DataFrames, one per station, each with columns
        ``station_id``, ``datetime``, ``u10``, ``v10``, ``t2m``, ``d2m``, ``blh``.
    """
    frames: list[pd.DataFrame] = []
    for s_idx, sid in enumerate(station_ids):
        frames.append(
            pd.DataFrame(
                {
                    "station_id": sid,
                    "datetime": times,
                    "u10": arrays["u10"][:, s_idx],
                    "v10": arrays["v10"][:, s_idx],
                    "t2m": arrays["t2m"][:, s_idx],
                    "d2m": arrays["d2m"][:, s_idx],
                    "blh": arrays["blh"][:, s_idx],
                }
            )
        )
    return frames


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def _download_era5_month(
    year: int,
    month: int,
    output_dir: Path,
    bbox: tuple[float, float, float, float],
    client: object,
) -> Path:
    """Download one calendar month of ERA5 data.

    Args:
        year: Calendar year (e.g. 2022).
        month: Calendar month 1-12.
        output_dir: Directory for the monthly NetCDF cache file.
        bbox: (west, south, east, north) in WGS84 degrees.
        client: Authenticated cdsapi.Client instance.

    Returns:
        Path to the monthly NetCDF (``era5_{year}_{month:02d}.nc``).
    """
    nc_path = output_dir / f"era5_{year}_{month:02d}.nc"
    if nc_path.exists():
        logger.info("ERA5 %d-%02d already cached: %s", year, month, nc_path)
        return nc_path

    logger.info("Downloading ERA5 %d-%02d ...", year, month)
    request: dict = {
        "product_type": "reanalysis",
        "variable": _ERA5_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": _all_days(),
        "time": _all_hours(),
        "area": _cds_area(bbox),
        "grid": "0.25/0.25",
        "format": "netcdf",
    }
    client.retrieve(_CDS_DATASET, request, str(nc_path))
    logger.info("ERA5 %d-%02d saved to %s", year, month, nc_path)
    return nc_path


def download_era5_year(
    year: int,
    output_dir: Path,
    bbox: tuple[float, float, float, float] = (97.0, 16.0, 101.5, 21.0),
) -> Path:
    """Download one year of ERA5 hourly data to a single NetCDF file.

    The CDS API v2 rejects full-year requests as too large.  This function
    works around the limit by downloading one calendar month at a time
    (12 sequential requests) and merging the results into a single yearly
    file with xarray.  Monthly cache files are kept in ``output_dir`` so
    individual months are not re-downloaded on retry.

    If the merged yearly file already exists the download is skipped entirely.

    Args:
        year: Calendar year to download (e.g. 2022).
        output_dir: Directory where NetCDF files are written.
                    Created if it does not exist.
        bbox: (west, south, east, north) in WGS84 degrees.
              Defaults to the Northern Thailand study area.

    Returns:
        Path to the merged yearly NetCDF (``era5_{year}.nc``).

    Raises:
        RuntimeError: If cdsapi is not installed or authentication fails.
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("cdsapi is not installed. Run `uv add cdsapi` or `uv sync`.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_path = output_dir / f"era5_{year}.nc"

    if nc_path.exists():
        logger.info("ERA5 year=%d already downloaded: %s", year, nc_path)
        return nc_path

    logger.info(
        "Downloading ERA5 year=%d in 12 monthly chunks -> %s", year, nc_path
    )

    try:
        client = cdsapi.Client()
        monthly_paths: list[Path] = []
        for month in range(1, 13):
            mp = _download_era5_month(
                year=year, month=month, output_dir=output_dir, bbox=bbox, client=client
            )
            monthly_paths.append(mp)
    except Exception as exc:
        raise RuntimeError(
            f"ERA5 download failed for year={year}. "
            "Ensure ~/.cdsapirc is present and contains a valid CDS API key. "
            "See https://cds.climate.copernicus.eu/how-to-api for setup. "
            f"Original error: {exc}"
        ) from exc

    # Merge 12 monthly NetCDFs into one yearly file.
    # Use _open_nc/_write_nc to bypass C-library unicode path issues on Windows.
    logger.info("Merging 12 monthly files into %s ...", nc_path)
    datasets = [_open_nc(mp) for mp in monthly_paths]
    ds_merged = xr.concat(datasets, dim="time")
    _write_nc(ds_merged, nc_path)
    for ds in datasets:
        ds.close()

    logger.info("ERA5 year=%d merged and saved to %s", year, nc_path)
    return nc_path


def interpolate_to_stations(nc_path: Path, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Bilinearly interpolate ERA5 grid data to station lat/lon coordinates.

    Loads the NetCDF at ``nc_path`` and uses ``xr.DataArray.interp`` to
    perform bilinear spatial interpolation for each station.  Temperature
    and dewpoint are kept in Kelvin (ERA5 native units).

    Args:
        nc_path: Path to an ERA5 NetCDF file produced by
            :func:`download_era5_year`.
        stations_df: DataFrame with at least three columns:
            ``station_id`` (int), ``latitude`` (float), ``longitude`` (float).

    Returns:
        Tidy DataFrame with columns:
            ``station_id``, ``datetime`` (UTC tz-aware), ``u10``, ``v10``,
            ``t2m``, ``d2m``, ``blh``.
        Row count = (number of timesteps) x (number of stations).
    """
    ds = _open_nc(nc_path)
    ds = _normalise_coords(ds)

    lats = xr.DataArray(stations_df["latitude"].values, dims="station")
    lons = xr.DataArray(stations_df["longitude"].values, dims="station")

    # Bilinear interpolation to all station points simultaneously.
    ds_interp = ds.interp(lat=lats, lon=lons, method="linear")

    n_stations = len(stations_df)
    station_ids = stations_df["station_id"].values

    # Convert time coordinate to UTC-aware pandas timestamps.
    times = pd.to_datetime(ds_interp["time"].values).tz_localize("UTC")

    logger.info(
        "interpolate_to_stations: %d timesteps x %d stations from %s",
        len(times),
        n_stations,
        nc_path,
    )

    arrays = _extract_var_arrays(ds_interp, len(times), n_stations, nc_path)
    frames = _build_tidy_frames(arrays, times, station_ids)

    ds.close()

    result = pd.concat(frames, ignore_index=True)
    result["station_id"] = result["station_id"].astype(int)
    return result


def fetch_era5(
    bbox: tuple[float, float, float, float] = (97.0, 16.0, 101.5, 21.0),
    date_from: date | None = None,
    date_to: date | None = None,
    output_dir: Path = Path("data/raw/era5"),
) -> pd.DataFrame:
    """Fetch ERA5 reanalysis data for the Northern Thailand study area.

    Orchestrates year-by-year download via :func:`download_era5_year` then
    spatial interpolation via :func:`interpolate_to_stations`.  Station
    coordinates are loaded from ``data/processed/stations_metadata.parquet``.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
              Defaults to the 9-province Northern Thailand study area.
        date_from: Inclusive start date.  Defaults to 2022-01-01.
        date_to: Inclusive end date.  Defaults to 2025-12-31.
        output_dir: Directory where NetCDF files are downloaded.
                    Created if it does not exist.

    Returns:
        Tidy DataFrame with columns:
            ``station_id``, ``datetime`` (UTC tz-aware), ``u10``, ``v10``,
            ``t2m``, ``d2m``, ``blh``.
        Filtered to the [date_from, date_to] window and sorted by
        (station_id, datetime).

    Raises:
        FileNotFoundError: If ``data/processed/stations_metadata.parquet``
            does not exist.  Run the station-discovery script first.
        RuntimeError: If cdsapi authentication fails.
    """
    if date_from is None:
        date_from = date(2022, 1, 1)
    if date_to is None:
        date_to = date(2025, 12, 31)

    stations_df = _load_stations()
    output_dir = Path(output_dir)
    start_year = date_from.year
    end_year = date_to.year

    logger.info(
        "fetch_era5: bbox=%s years=%d-%d stations=%d",
        bbox,
        start_year,
        end_year,
        len(stations_df),
    )

    yearly_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        nc_path = download_era5_year(year=year, output_dir=output_dir, bbox=bbox)
        df_year = interpolate_to_stations(nc_path=nc_path, stations_df=stations_df)
        yearly_frames.append(df_year)
        logger.info("fetch_era5: year=%d interpolated %d rows", year, len(df_year))

    if not yearly_frames:
        logger.warning("fetch_era5: no data returned for range %s-%s", date_from, date_to)
        return pd.DataFrame(columns=["station_id", "datetime", "u10", "v10", "t2m", "d2m", "blh"])

    combined = _filter_and_sort(pd.concat(yearly_frames, ignore_index=True), date_from, date_to)

    logger.info(
        "fetch_era5: returning %d rows for %d stations, %s-%s",
        len(combined),
        combined["station_id"].nunique(),
        date_from,
        date_to,
    )
    return combined


def load_wind_field(nc_path: Path, dt: datetime) -> xr.DataArray:
    """Load u10 and v10 wind components at a single datetime as a DataArray.

    Retrieves u10 and v10 from the ERA5 NetCDF, selects the nearest
    timestep to ``dt``, and stacks the two components into a (lat, lon, 2)
    array.  A leading time dimension of size 1 is added so the result is
    compatible with ``graph_builder.build_graph`` which expects
    ``(time, lat, lon, 2)``.

    Args:
        nc_path: Path to an ERA5 NetCDF produced by :func:`download_era5_year`.
        dt: Target datetime (should be UTC-aware; naive datetimes are assumed UTC).

    Returns:
        ``xr.DataArray`` with dims ``(time, lat, lon, component)`` and shape
        ``(1, n_lat, n_lon, 2)``.  component 0 = u10 (eastward), component 1 = v10
        (northward).  Coordinate ``time`` contains ``dt`` rounded to the nearest
        ERA5 timestep.

    Raises:
        FileNotFoundError: If ``nc_path`` does not exist.
        KeyError: If u10 or v10 variables are not present in the file.
    """
    if not nc_path.exists():
        raise FileNotFoundError(f"ERA5 NetCDF not found: {nc_path}")

    ds = _open_nc(nc_path)
    ds = _normalise_coords(ds)

    # Resolve u10 / v10 variable names using module-level aliases.
    u_name = _find_var(ds, _ERA5_ALIASES["u10"])
    v_name = _find_var(ds, _ERA5_ALIASES["v10"])

    if u_name is None:
        raise KeyError(
            f"Cannot find u10 variable in {nc_path}. "
            f"Looked for: {_ERA5_ALIASES['u10']}. Available: {list(ds.data_vars)}"
        )
    if v_name is None:
        raise KeyError(
            f"Cannot find v10 variable in {nc_path}. "
            f"Looked for: {_ERA5_ALIASES['v10']}. Available: {list(ds.data_vars)}"
        )

    if dt.tzinfo is None:
        logger.warning("load_wind_field: dt=%s has no timezone; assuming UTC.", dt.isoformat())

    # ERA5 NetCDF stores time as timezone-naive datetime64[ns] (implicitly UTC).
    # Strip tz from the query datetime to avoid dtype-comparison errors in xarray.
    dt_naive = dt.replace(tzinfo=None)

    u_slice = ds[u_name].sel(time=dt_naive, method="nearest")  # (lat, lon)
    v_slice = ds[v_name].sel(time=dt_naive, method="nearest")  # (lat, lon)

    # Resolved nearest time coordinate (scalar DataArray).
    actual_time = u_slice["time"]

    # Stack into (lat, lon, 2), then add a leading time dim of size 1.
    wind = xr.concat([u_slice, v_slice], dim="component")  # (component, lat, lon)
    wind = wind.transpose("lat", "lon", "component")  # (lat, lon, component)
    wind = wind.expand_dims(dim={"time": [actual_time.values]})
    wind = wind.transpose("time", "lat", "lon", "component")

    ds.close()

    logger.debug("load_wind_field: dt=%s shape=%s", dt.isoformat(), wind.shape)
    return wind
