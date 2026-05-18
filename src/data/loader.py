"""
[TODO: NSC Disclaimer — see booklet page 44]

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data.scrapers.openaq import curate_training_stations, discover_locations

logger = logging.getLogger(__name__)


def build_station_metadata(
    output_path: Path = Path("data/processed/stations_metadata.parquet"),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Discover + curate + cache the core station set.

    Args:
        output_path: Destination parquet file for the curated station table.
            Defaults to ``data/processed/stations_metadata.parquet``.
        use_cache: When True and ``output_path`` already exists, load and
            return the cached file without hitting the API.

    Returns:
        DataFrame with columns: location_id, name, lat, lon, provider,
        datetime_first, datetime_last, sensor_id_pm25.
    """
    if use_cache and output_path.exists():
        logger.info("Loading station metadata from cache: %s", output_path)
        return pd.read_parquet(output_path)

    logger.info("Discovering locations via OpenAQ v3 …")
    df_locations = discover_locations()

    logger.info("Curating training stations …")
    df = curate_training_stations(df_locations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved station metadata to %s (%d stations)", output_path, len(df))

    return df
