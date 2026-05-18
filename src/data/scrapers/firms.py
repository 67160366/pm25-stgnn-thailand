"""
[TODO: NSC Disclaimer — see booklet page 44]

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

import io
import logging
import os
from datetime import date, timedelta
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

# Valid FIRMS data source keys.
# NRT (Near Real-Time): available for the last ~10 days from today.
# SP (Standard Product): scientifically validated; available from satellite launch
# through approximately today - _SP_LAG_DAYS.
#
# Observed lag as of 2026-05-18:
#   VIIRS_NOAA20_SP last confirmed data date: 2026-03-31 (48+ days behind today).
#   Conservative cutoff: today - 60 days to avoid fetching empty windows.
#   Known limitation: ~38-day gap between SP end and NRT start cannot be
#   backfilled from the FIRMS area CSV API; this is a NASA pipeline constraint.
_NRT_SOURCES: frozenset[str] = frozenset({"VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"})
_SP_SOURCES: frozenset[str] = frozenset({"VIIRS_NOAA20_SP", "VIIRS_SNPP_SP"})
_VALID_SOURCES: frozenset[str] = _NRT_SOURCES | _SP_SOURCES

# Conservative SP processing lag (days behind today).
_SP_LAG_DAYS: int = 60
# Maximum day_range the FIRMS area CSV API accepts per product type.
# NRT endpoint accepts up to 10; SP endpoint accepts only up to 5 (observed 2026-05-19).
_NRT_WINDOW_DAYS: int = 10
_SP_WINDOW_DAYS: int = 5


def _effective_cache_dir(cache_dir: Path | None) -> Path:
    """Return the active cache directory, falling back to the module default."""
    return cache_dir if cache_dir is not None else _CACHE_DIR


def fetch_hotspots(
    bbox: tuple[float, float, float, float] = BBOX_NORTHERN_THAILAND,
    day_range: int = 1,
    source: str = "VIIRS_NOAA20_NRT",
    date_: date | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch fire hotspots from the NASA FIRMS area CSV API.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        day_range: Number of days to fetch (1-10 inclusive).
        source: FIRMS data source key.  Must be one of ``_VALID_SOURCES``.
            NRT sources (e.g. ``VIIRS_NOAA20_NRT``) cover the last ~10 days.
            SP sources (e.g. ``VIIRS_NOAA20_SP``) cover 2022-01-01 through
            approximately today - 60 days.
        date_: Reference date for the query; the API returns ``day_range`` days
            ending on this date.  Pass ``None`` to use the API's default
            (most-recent available).
        cache_dir: Override the module-level ``_CACHE_DIR``.  Primarily used
            in tests.

    Returns:
        DataFrame with the raw FIRMS CSV columns.

    Raises:
        ValueError: If ``day_range`` is not in [1, 10] or ``source`` is not a
            recognised FIRMS source key.
    """
    if day_range < 1 or day_range > 10:
        raise ValueError(f"day_range must be between 1 and 10, got {day_range}")

    if source not in _VALID_SOURCES:
        raise ValueError(
            f"Unknown FIRMS source {source!r}. " f"Valid sources: {sorted(_VALID_SOURCES)}"
        )

    api_key = os.getenv("FIRMS_API_KEY", "")
    lon_w, lat_s, lon_e, lat_n = bbox
    bbox_str = f"{lon_w},{lat_s},{lon_e},{lat_n}"
    date_str = date_.isoformat() if date_ is not None else "latest"

    active_cache_dir = _effective_cache_dir(cache_dir)
    active_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_filename = f"firms_{source}_{lon_w}_{lat_s}_{lon_e}_{lat_n}_{day_range}_{date_str}.csv"
    cache_path = active_cache_dir / cache_filename

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


def fetch_hotspots_historical(
    bbox: tuple[float, float, float, float] = BBOX_NORTHERN_THAILAND,
    start_date: str = "2022-01-01",
    end_date: str | None = None,
    source: str = "VIIRS_NOAA20_SP",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch FIRMS hotspots for a date range by chunking into 10-day windows.

    The FIRMS area CSV API caps ``day_range`` at 10, so this function splits
    the requested range into sequential 10-day chunks and concatenates the
    results.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        start_date: ISO 8601 date string for the first day to fetch, inclusive.
        end_date: ISO 8601 date string for the last day to fetch, inclusive.
            Defaults to today.
        source: FIRMS data source key.  Defaults to ``VIIRS_NOAA20_SP`` for
            historical backfill.  Must be one of ``_VALID_SOURCES``.
        cache_dir: Override the module-level ``_CACHE_DIR``.  Primarily used
            in tests.

    Returns:
        Concatenated DataFrame of all hotspot rows across the date range.
        Returns an empty DataFrame if no hotspots were found in any chunk.

    Raises:
        ValueError: If ``source`` is not a recognised FIRMS source key, or if
            ``end_date`` is earlier than ``start_date``.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"Unknown FIRMS source {source!r}. " f"Valid sources: {sorted(_VALID_SOURCES)}"
        )

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date is not None else date.today()

    if end < start:
        raise ValueError(f"end_date ({end}) must be >= start_date ({start})")

    frames: list[pd.DataFrame] = []
    chunk_start = start
    # SP endpoint caps day_range at 5; NRT allows up to 10.
    max_chunk = _SP_WINDOW_DAYS if source in _SP_SOURCES else _NRT_WINDOW_DAYS

    while chunk_start <= end:
        days_remaining = (end - chunk_start).days + 1
        days_in_chunk = min(max_chunk, days_remaining)
        # The API date parameter is the END date of the window.
        chunk_end = chunk_start + timedelta(days=days_in_chunk - 1)

        logger.debug(
            "Fetching historical chunk source=%s %s to %s (day_range=%d)",
            source,
            chunk_start.isoformat(),
            chunk_end.isoformat(),
            days_in_chunk,
        )

        df_chunk = fetch_hotspots(
            bbox=bbox,
            day_range=days_in_chunk,
            source=source,
            date_=chunk_end,
            cache_dir=cache_dir,
        )
        if not df_chunk.empty:
            frames.append(df_chunk)

        chunk_start += timedelta(days=days_in_chunk)

    if not frames:
        logger.warning(
            "fetch_hotspots_historical: no hotspot rows returned for bbox=%s source=%s %s→%s",
            bbox,
            source,
            start.isoformat(),
            end.isoformat(),
        )
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "fetch_hotspots_historical: %d total rows, source=%s %s→%s",
        len(result),
        source,
        start.isoformat(),
        end.isoformat(),
    )
    return result


def fetch_hotspots_hybrid(
    bbox: tuple[float, float, float, float] = BBOX_NORTHERN_THAILAND,
    start_date: str = "2022-01-01",
    end_date: str | None = None,
    sp_source: str = "VIIRS_NOAA20_SP",
    nrt_source: str = "VIIRS_NOAA20_NRT",
    sp_lag_days: int = _SP_LAG_DAYS,
    nrt_window_days: int = _NRT_WINDOW_DAYS,  # NRT max window
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch FIRMS hotspots using SP for history and NRT for recent days.

    Covers the widest possible date range by combining two FIRMS products:

    - **SP** (Standard Product): scientifically validated, covers 2022-01-01
      through approximately today - ``sp_lag_days`` (observed: ~60 days).
    - **NRT** (Near Real-Time): covers the last ``nrt_window_days`` days.

    Known limitation: a gap of roughly ``sp_lag_days - nrt_window_days`` days
    (observed ~48 days as of 2026-05-18) between the SP trailing edge and the
    NRT leading edge cannot be filled from the FIRMS area CSV API.  This is a
    NASA pipeline constraint.  The gap is logged at WARNING level.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        start_date: ISO 8601 date string for the first day to fetch, inclusive.
        end_date: ISO 8601 date string for the last day to fetch, inclusive.
            Defaults to today.
        sp_source: Standard Product source key (default ``VIIRS_NOAA20_SP``).
        nrt_source: Near Real-Time source key (default ``VIIRS_NOAA20_NRT``).
        sp_lag_days: Conservative SP processing lag in days.  Dates within this
            window of today are assumed to have no SP data yet.
        nrt_window_days: How many recent days NRT can cover (max 10 per API).
        cache_dir: Override the module-level ``_CACHE_DIR``.

    Returns:
        Concatenated DataFrame from SP and/or NRT depending on the date range.
        Returns an empty DataFrame if neither source returned rows.

    Raises:
        ValueError: If any source key is not recognised, or if ``end_date`` is
            earlier than ``start_date``.
    """
    for src in (sp_source, nrt_source):
        if src not in _VALID_SOURCES:
            raise ValueError(
                f"Unknown FIRMS source {src!r}. " f"Valid sources: {sorted(_VALID_SOURCES)}"
            )

    today = date.today()
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date is not None else today

    if end < start:
        raise ValueError(f"end_date ({end}) must be >= start_date ({start})")

    sp_cutoff = today - timedelta(days=sp_lag_days)
    nrt_start = today - timedelta(days=nrt_window_days)

    frames: list[pd.DataFrame] = []

    # -- SP segment: [start, min(end, sp_cutoff)] --
    if start <= sp_cutoff:
        sp_end = min(end, sp_cutoff)
        logger.info(
            "Hybrid: fetching SP segment %s → %s (source=%s)",
            start.isoformat(),
            sp_end.isoformat(),
            sp_source,
        )
        df_sp = fetch_hotspots_historical(
            bbox=bbox,
            start_date=start.isoformat(),
            end_date=sp_end.isoformat(),
            source=sp_source,
            cache_dir=cache_dir,
        )
        if not df_sp.empty:
            frames.append(df_sp)
    else:
        logger.debug("Hybrid: start_date %s is within SP lag window; skipping SP segment.", start)

    # -- Gap warning --
    if sp_cutoff < nrt_start and start <= sp_cutoff and end >= nrt_start:
        gap_start = sp_cutoff + timedelta(days=1)
        gap_end = nrt_start - timedelta(days=1)
        logger.warning(
            "Hybrid: uncoverable gap %s → %s (%d days). "
            "This is a FIRMS API pipeline constraint — SP data is not yet "
            "processed and NRT data is no longer available for this window.",
            gap_start.isoformat(),
            gap_end.isoformat(),
            (gap_end - gap_start).days + 1,
        )

    # -- NRT segment: [max(start, nrt_start), end] --
    nrt_fetch_start = max(start, nrt_start)
    if nrt_fetch_start <= end:
        nrt_days = min((end - nrt_fetch_start).days + 1, nrt_window_days)
        logger.info(
            "Hybrid: fetching NRT segment %s → %s (source=%s, day_range=%d)",
            nrt_fetch_start.isoformat(),
            end.isoformat(),
            nrt_source,
            nrt_days,
        )
        df_nrt = fetch_hotspots(
            bbox=bbox,
            day_range=nrt_days,
            source=nrt_source,
            date_=end,
            cache_dir=cache_dir,
        )
        if not df_nrt.empty:
            frames.append(df_nrt)
    else:
        logger.debug(
            "Hybrid: end_date %s is before NRT window start %s; skipping NRT segment.",
            end,
            nrt_start,
        )

    if not frames:
        logger.warning(
            "fetch_hotspots_hybrid: no rows returned for bbox=%s %s→%s",
            bbox,
            start.isoformat(),
            end.isoformat(),
        )
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "fetch_hotspots_hybrid: %d total rows, %s→%s",
        len(result),
        start.isoformat(),
        end.isoformat(),
    )
    return result
