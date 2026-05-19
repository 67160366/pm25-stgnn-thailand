"""
[TODO: NSC Disclaimer — see booklet page 44]

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

# DBSCAN parameters for fire hotspot clustering.
# eps=25 km converted to radians (haversine metric uses radians).
_DBSCAN_EPS_KM: float = 25.0
_EARTH_RADIUS_KM: float = 6371.0
_DBSCAN_EPS_RAD: float = _DBSCAN_EPS_KM / _EARTH_RADIUS_KM
_DBSCAN_MIN_SAMPLES: int = 2

# Bounding boxes (lon_min, lat_min, lon_max, lat_max) for country attribution.
# Priority order: Thailand > Myanmar > Laos > other.
# Boxes are approximate; border regions may be misclassified.
# See SESSION2_NOTES.md for why shapely was not used.
_COUNTRY_BBOXES: list[tuple[str, float, float, float, float]] = [
    ("Thailand", 97.3, 5.6, 105.7, 20.5),
    ("Myanmar", 92.2, 9.8, 101.2, 28.5),
    ("Laos", 100.1, 13.9, 107.6, 22.5),
]

# Required FIRMS CSV columns (superset; extra columns are kept).
_FIRMS_DATE_COL: str = "acq_date"
_FIRMS_TIME_COL: str = "acq_time"
_FIRMS_LAT_COL: str = "latitude"
_FIRMS_LON_COL: str = "longitude"
_FIRMS_FRP_COL: str = "frp"


def _country_from_centroid(lon: float, lat: float) -> str:
    """Return the country name for a given centroid using simple bbox lookup.

    Priority order: Thailand, Myanmar, Laos, other.
    Border regions may be misclassified; this is documented in SESSION2_NOTES.md.

    Args:
        lon: Centroid longitude (WGS84).
        lat: Centroid latitude (WGS84).

    Returns:
        One of ``"Thailand"``, ``"Myanmar"``, ``"Laos"``, ``"other"``.
    """
    for country, lon_min, lat_min, lon_max, lat_max in _COUNTRY_BBOXES:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return country
    return "other"


def _load_firms_csvs(firms_dir: Path) -> pd.DataFrame:
    """Load and concatenate all FIRMS CSV files from a directory.

    Args:
        firms_dir: Directory containing FIRMS area CSV files.

    Returns:
        Combined DataFrame with at minimum ``acq_date``, ``latitude``,
        ``longitude``, and ``frp`` columns.

    Raises:
        RuntimeError: If no CSV files are found or none contain valid rows.
    """
    csv_files = sorted(firms_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No FIRMS CSV files found in {firms_dir}")
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        if df.empty or _FIRMS_LAT_COL not in df.columns:
            logger.debug("Skipping empty or invalid FIRMS file: %s", path.name)
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError(f"All FIRMS CSVs in {firms_dir} were empty or missing lat column")
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d FIRMS hotspot rows from %d files", len(combined), len(frames))
    return combined


def _cluster_daily(day_df: pd.DataFrame) -> pd.DataFrame:
    """Run DBSCAN on a single day's hotspot points and aggregate clusters.

    Points not assigned to any cluster (label -1) are treated as
    single-point clusters so no fire information is discarded.

    Args:
        day_df: DataFrame for one date with ``latitude``, ``longitude``,
            and ``frp`` columns.

    Returns:
        DataFrame with one row per cluster:
        ``cluster_id``, ``centroid_lat``, ``centroid_lon``,
        ``total_frp``, ``point_count``, ``country``.
        Empty DataFrame if ``day_df`` has no rows.
    """
    if day_df.empty:
        return pd.DataFrame()

    coords = np.radians(day_df[[_FIRMS_LAT_COL, _FIRMS_LON_COL]].values)
    labels = DBSCAN(
        eps=_DBSCAN_EPS_RAD,
        min_samples=_DBSCAN_MIN_SAMPLES,
        metric="haversine",
    ).fit_predict(coords)

    # Treat noise points (label -1) as individual clusters with unique negative IDs
    # to avoid silently discarding isolated fire detections.
    noise_offset = int(labels.max()) + 1 if labels.max() >= 0 else 0
    noise_ids = np.arange(noise_offset, noise_offset + (labels == -1).sum())
    adjusted = labels.copy()
    adjusted[labels == -1] = noise_ids

    rows: list[dict] = []
    for cluster_id in np.unique(adjusted):
        mask = adjusted == cluster_id
        subset = day_df[mask]
        lat = float(subset[_FIRMS_LAT_COL].mean())
        lon = float(subset[_FIRMS_LON_COL].mean())
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "centroid_lat": lat,
                "centroid_lon": lon,
                "total_frp": float(subset[_FIRMS_FRP_COL].sum()),
                "point_count": len(subset),
                "country": _country_from_centroid(lon, lat),
            }
        )
    return pd.DataFrame(rows)


def cluster_hotspots(
    firms_dir: Path | str = Path("data/raw/firms"),
    output_path: Path | str = Path("data/processed/hotspots.parquet"),
    sp_source_prefix: str = "firms_VIIRS_NOAA20_SP",
) -> pd.DataFrame:
    """Cluster daily FIRMS hotspots with DBSCAN and save to parquet.

    Pipeline:
        1. Load all FIRMS CSVs from ``firms_dir`` (SP product preferred;
           NRT included if present).
        2. Parse acquisition date from ``acq_date`` column.
        3. Group by date; run DBSCAN per day (eps=25 km, min_samples=2).
        4. Aggregate clusters: centroid, total FRP, point count, country flag.
        5. Save to ``output_path``.

    Country flag uses a simple bbox geocoder (Thailand > Myanmar > Laos > other).
    Border-region accuracy is approximate; see SESSION2_NOTES.md.

    Args:
        firms_dir: Directory containing FIRMS CSV files.
        output_path: Output parquet path.
        sp_source_prefix: Filename prefix used to identify SP product files
            (loaded first; NRT files are also included).

    Returns:
        DataFrame with columns:
        ``date``, ``cluster_id``, ``centroid_lat``, ``centroid_lon``,
        ``total_frp``, ``point_count``, ``country``.
    """
    firms_dir = Path(firms_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _load_firms_csvs(firms_dir)

    # Parse date column
    raw[_FIRMS_DATE_COL] = pd.to_datetime(raw[_FIRMS_DATE_COL], format="%Y-%m-%d").dt.date

    # Drop rows with missing coordinates or FRP
    before = len(raw)
    raw = raw.dropna(subset=[_FIRMS_LAT_COL, _FIRMS_LON_COL, _FIRMS_FRP_COL])
    if len(raw) < before:
        logger.warning("Dropped %d FIRMS rows with NaN coordinates or FRP", before - len(raw))

    # Cluster per day
    daily_results: list[pd.DataFrame] = []
    dates = sorted(raw[_FIRMS_DATE_COL].unique())
    logger.info("Clustering %d unique dates", len(dates))
    for day in dates:
        day_df = raw[raw[_FIRMS_DATE_COL] == day].reset_index(drop=True)
        clusters = _cluster_daily(day_df)
        if not clusters.empty:
            clusters.insert(0, "date", day)
            daily_results.append(clusters)

    if not daily_results:
        logger.warning("No clusters produced from FIRMS data — returning empty DataFrame")
        return pd.DataFrame()

    result = pd.concat(daily_results, ignore_index=True)

    # Enforce schema types
    result["date"] = pd.to_datetime(result["date"]).dt.date
    result["cluster_id"] = result["cluster_id"].astype("int64")
    result["centroid_lat"] = result["centroid_lat"].astype("float32")
    result["centroid_lon"] = result["centroid_lon"].astype("float32")
    result["total_frp"] = result["total_frp"].astype("float32")
    result["point_count"] = result["point_count"].astype("int32")
    result["country"] = result["country"].astype("str")

    col_order = [
        "date",
        "cluster_id",
        "centroid_lat",
        "centroid_lon",
        "total_frp",
        "point_count",
        "country",
    ]
    result = result[col_order]

    result.to_parquet(output_path, index=False)
    logger.info(
        "Saved %d hotspot clusters (%d dates) to %s",
        len(result),
        result["date"].nunique(),
        output_path,
    )
    return result
