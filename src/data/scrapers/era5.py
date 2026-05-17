"""
[TODO: NSC Disclaimer — see booklet page 44]

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

# ERA5 variables needed when this module is implemented:
#   2m_temperature
#   10m_u_component_of_wind
#   10m_v_component_of_wind
#   boundary_layer_height
#   total_precipitation
#   2m_dewpoint_temperature

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_era5(
    bbox: tuple[float, float, float, float] = (97.0, 16.0, 101.5, 21.0),
    date_from: date | None = None,
    date_to: date | None = None,
    output_dir: Path = Path("data/raw/era5"),
) -> pd.DataFrame:
    """Fetch ERA5 reanalysis data for the Northern Thailand study area.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        date_from: Start date of the ERA5 request.
        date_to: End date of the ERA5 request.
        output_dir: Directory to write downloaded NetCDF files.

    Returns:
        DataFrame with ERA5 variables interpolated to station locations.

    Raises:
        NotImplementedError: ERA5 download is not yet implemented.
            See docs/SESSION1_NOTES.md for the implementation plan.
    """
    raise NotImplementedError(
        "ERA5 download is not yet implemented. "
        "See docs/SESSION1_NOTES.md for the implementation plan and "
        "required CDS API profile setup."
    )
