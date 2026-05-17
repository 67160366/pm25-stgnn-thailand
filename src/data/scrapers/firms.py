"""
[TODO: NSC Disclaimer — see booklet page 44]

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

import io
import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
_TIMEOUT = 30
_CACHE_DIR = Path("data/raw/firms")

BBOX_NORTHERN_THAILAND = (97.0, 16.0, 101.5, 21.0)


def fetch_hotspots(
    bbox: tuple[float, float, float, float] = BBOX_NORTHERN_THAILAND,
    day_range: int = 1,
    source: str = "VIIRS_NOAA20_NRT",
    date_: date | None = None,
) -> pd.DataFrame:
    """Fetch fire hotspots from the NASA FIRMS area CSV API.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        day_range: Number of days to fetch (1-10 inclusive).
        source: FIRMS data source key (VIIRS_NOAA20_NRT, VIIRS_SNPP_NRT, MODIS_NRT).
        date_: Reference date for the query (None = today in API default).

    Returns:
        DataFrame with the raw FIRMS CSV columns.

    Raises:
        ValueError: If day_range is not in [1, 10].
    """
    if day_range < 1 or day_range > 10:
        raise ValueError(f"day_range must be between 1 and 10, got {day_range}")

    api_key = os.getenv("FIRMS_API_KEY", "")
    lon_w, lat_s, lon_e, lat_n = bbox
    bbox_str = f"{lon_w},{lat_s},{lon_e},{lat_n}"
    date_str = date_.isoformat() if date_ is not None else "latest"

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_filename = f"firms_{source}_{lon_w}_{lat_s}_{lon_e}_{lat_n}_{day_range}_{date_str}.csv"
    cache_path = _CACHE_DIR / cache_filename

    if cache_path.exists():
        logger.info("Loading FIRMS hotspots from cache: %s", cache_path)
        return pd.read_csv(cache_path)

    if date_ is not None:
        url = f"{_BASE_URL}/{api_key}/{source}/{bbox_str}/{day_range}/{date_.isoformat()}"
    else:
        url = f"{_BASE_URL}/{api_key}/{source}/{bbox_str}/{day_range}"

    logger.info(
        "Fetching FIRMS hotspots source=%s bbox=%s day_range=%d date=%s",
        source,
        bbox_str,
        day_range,
        date_str,
    )
    response = requests.get(url, timeout=_TIMEOUT)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    df.to_csv(cache_path, index=False)
    logger.info("Cached FIRMS hotspots to %s (%d rows)", cache_path, len(df))

    return df
