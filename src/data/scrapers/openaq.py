"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openaq.org/v3"
_TIMEOUT = 30
_API_KEY = os.getenv("OPENAQ_API_KEY", "")
_HEADERS = {
    "X-API-Key": _API_KEY,
    "User-Agent": "NSC2026-PM25STGNN (academic-research)",
}


def _make_session() -> requests.Session:
    """Create a requests.Session with project-standard headers.

    Returns:
        Configured session for OpenAQ v3 requests.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


def _is_retryable(exc: BaseException) -> bool:
    """Return True when exc is an HTTPError with a retryable status code."""
    return (
        isinstance(exc, requests.HTTPError)
        and exc.response is not None
        and exc.response.status_code in (408, 429, 500, 502, 503, 504)
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
)
def _fetch_locations_page(
    session: requests.Session,
    bbox: tuple[float, float, float, float],
    parameters_id: int,
    page: int,
) -> list[dict]:
    """Fetch a single page of location results from the OpenAQ locations endpoint.

    Args:
        session: Authenticated requests session.
        bbox: (west, south, east, north) bounding box in WGS84 degrees.
        parameters_id: OpenAQ parameter ID to filter by (2 = PM2.5).
        page: 1-based page number.

    Returns:
        List of location dicts from the API `results` field, or empty list.

    Raises:
        requests.HTTPError: On non-retryable HTTP errors.
    """
    west, south, east, north = bbox
    params = {
        "iso": "TH",
        "parameters_id": parameters_id,
        "bbox": f"{west},{south},{east},{north}",
        "limit": 100,
        "page": page,
    }
    logger.debug("Fetching locations page %d", page)
    response = session.get(f"{_BASE_URL}/locations", params=params, timeout=_TIMEOUT)
    response.encoding = "utf-8-sig"
    response.raise_for_status()
    return response.json().get("results", [])


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
)
def _fetch_measurements_page(
    session: requests.Session,
    sensor_id: int,
    datetime_from: datetime,
    datetime_to: datetime,
    page_size: int,
    page: int,
) -> list[dict]:
    """Fetch a single page of measurements for a sensor.

    Args:
        session: Authenticated requests session.
        sensor_id: OpenAQ sensor ID.
        datetime_from: Start of the measurement window (UTC).
        datetime_to: End of the measurement window (UTC).
        page_size: Number of results per page.
        page: 1-based page number.

    Returns:
        List of measurement dicts from the API `results` field, or empty list.

    Raises:
        requests.HTTPError: On non-retryable HTTP errors.
    """
    params = {
        "datetime_from": datetime_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datetime_to": datetime_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": page_size,
        "page": page,
    }
    logger.debug("Fetching measurements sensor=%d page=%d", sensor_id, page)
    response = session.get(
        f"{_BASE_URL}/sensors/{sensor_id}/measurements",
        params=params,
        timeout=_TIMEOUT,
    )
    response.encoding = "utf-8-sig"
    response.raise_for_status()
    return response.json().get("results", [])


def discover_locations(
    bbox: tuple[float, float, float, float] = (97.0, 16.0, 101.5, 21.0),
    parameters_id: int = 2,
) -> pd.DataFrame:
    """Discover all OpenAQ monitoring locations within a bounding box.

    Paginates through `GET /v3/locations` until no results remain, then
    extracts PM2.5 sensor IDs and location metadata.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
              Defaults to Northern Thailand study area.
        parameters_id: OpenAQ parameter ID to filter by (2 = PM2.5).

    Returns:
        DataFrame with columns:
            location_id, name, lat, lon, provider,
            datetime_first, datetime_last, sensor_id_pm25.
        Locations with no PM2.5 sensor are excluded.
    """
    session = _make_session()
    records: list[dict] = []
    page = 1

    while True:
        results = _fetch_locations_page(session, bbox, parameters_id, page)
        if not results:
            logger.info("Locations discovery complete: %d total pages fetched", page - 1)
            break
        logger.info("Locations page %d: %d results", page, len(results))

        for loc in results:
            # Resolve PM2.5 sensor ID
            sensor_id_pm25: int | None = None
            for sensor in loc.get("sensors", []):
                if sensor.get("parameter", {}).get("id") == parameters_id:
                    sensor_id_pm25 = sensor["id"]
                    break
            if sensor_id_pm25 is None:
                continue  # Skip locations with no PM2.5 sensor

            provider_name = loc.get("provider", {}).get("name", "")

            coords = loc.get("coordinates", {})

            records.append(
                {
                    "location_id": loc["id"],
                    "name": loc.get("name", ""),
                    "lat": coords.get("latitude"),
                    "lon": coords.get("longitude"),
                    "provider": provider_name,
                    "datetime_first": loc.get("datetimeFirst", {}).get("utc"),
                    "datetime_last": loc.get("datetimeLast", {}).get("utc"),
                    "sensor_id_pm25": sensor_id_pm25,
                }
            )

        page += 1

    columns = [
        "location_id",
        "name",
        "lat",
        "lon",
        "provider",
        "datetime_first",
        "datetime_last",
        "sensor_id_pm25",
    ]
    return pd.DataFrame(records, columns=columns)


def curate_training_stations(
    df_locations: pd.DataFrame,
    provider: str = "Air4Thai",
    first_before: str = "2022-01-01",
    last_after: str = "2026-01-01",
) -> pd.DataFrame:
    """Filter discovered locations to stations suitable for ML training.

    Keeps only Air4Thai stations that were active before the training window
    starts and were still reporting after the training window ends.

    Args:
        df_locations: DataFrame returned by :func:`discover_locations`.
        provider: Provider name to filter on (case-sensitive).
        first_before: Station must have data before this date (ISO 8601).
        last_after: Station must still be active after this date (ISO 8601).

    Returns:
        Filtered DataFrame with reset index.
    """
    first_before_ts = pd.to_datetime(first_before, utc=True)
    last_after_ts = pd.to_datetime(last_after, utc=True)
    mask = (
        (df_locations["provider"] == provider)
        & (pd.to_datetime(df_locations["datetime_first"], utc=True) < first_before_ts)
        & (pd.to_datetime(df_locations["datetime_last"], utc=True) > last_after_ts)
    )
    result = df_locations[mask].reset_index(drop=True)
    logger.info(
        "curate_training_stations: %d/%d stations kept (provider=%s)",
        len(result),
        len(df_locations),
        provider,
    )
    return result


def fetch_measurements(
    sensor_id: int,
    datetime_from: datetime,
    datetime_to: datetime,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Fetch all PM2.5 measurements for a sensor within a time window.

    Paginates through `GET /v3/sensors/{sensor_id}/measurements` until
    the API returns an empty results page.

    Args:
        sensor_id: OpenAQ sensor ID for the PM2.5 instrument.
        datetime_from: Inclusive start of the query window (UTC).
        datetime_to: Inclusive end of the query window (UTC).
        page_size: Results per page (max 1000 per OpenAQ v3 docs).

    Returns:
        DataFrame with columns:
            timestamp_utc, sensor_id, value, parameter_name, parameter_units.
        Empty DataFrame if no data is returned.
    """
    session = _make_session()
    records: list[dict] = []
    page = 1

    while True:
        results = _fetch_measurements_page(
            session, sensor_id, datetime_from, datetime_to, page_size, page
        )
        if not results:
            logger.info(
                "fetch_measurements sensor=%d: complete, %d total pages", sensor_id, page - 1
            )
            break
        logger.info(
            "fetch_measurements sensor=%d page=%d: %d results", sensor_id, page, len(results)
        )

        for item in results:
            period = item.get("period", {})
            datetime_from_utc = period.get("datetimeFrom", {}).get("utc")
            parameter = item.get("parameter", {})
            records.append(
                {
                    "timestamp_utc": datetime_from_utc,
                    "sensor_id": sensor_id,
                    "value": item.get("value"),
                    "parameter_name": parameter.get("name"),
                    "parameter_units": parameter.get("units"),
                }
            )

        page += 1

    columns = ["timestamp_utc", "sensor_id", "value", "parameter_name", "parameter_units"]
    return pd.DataFrame(records, columns=columns)


def backfill_station(
    sensor_id: int,
    start_year: int,
    end_year: int,
    output_dir: Path,
) -> list[Path]:
    """Download and persist annual measurement files for a single station.

    Iterates over each year in [start_year, end_year] (inclusive), skips
    years where the parquet file already exists, and calls
    :func:`fetch_measurements` per month (OpenAQ times out on full-year ranges).

    Args:
        sensor_id: OpenAQ sensor ID for the PM2.5 instrument.
        start_year: First year to backfill (inclusive).
        end_year: Last year to backfill (inclusive).
        output_dir: Directory under which parquet files are written.
                    Created if it does not exist.

    Returns:
        List of Path objects for all output files (existing + newly written).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    for year in range(start_year, end_year + 1):
        path = output_dir / f"sensor_{sensor_id}_{year}.parquet"
        paths.append(path)

        if path.exists():
            logger.info("Skipping %s (already exists)", path)
            continue

        # Fetch in 7-day windows — monthly ranges trigger 408 from OpenAQ
        chunk_frames: list[pd.DataFrame] = []
        chunk_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31, 23, 59, 59)
        while chunk_start <= year_end:
            chunk_end = min(
                datetime(
                    chunk_start.year,
                    chunk_start.month,
                    chunk_start.day,
                    23,
                    59,
                    59,
                )
                + timedelta(days=6),
                year_end,
            )
            df_chunk = fetch_measurements(sensor_id, chunk_start, chunk_end)
            chunk_frames.append(df_chunk)
            logger.info(
                "sensor=%d %s-%s: %d rows",
                sensor_id,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                len(df_chunk),
            )
            time.sleep(1)  # polite pause to avoid 408 timeouts
            chunk_start = chunk_end + timedelta(seconds=1)

        df = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
        df.to_parquet(path, index=False)
        logger.info("Saved %s (%d rows)", path, len(df))

    return paths
