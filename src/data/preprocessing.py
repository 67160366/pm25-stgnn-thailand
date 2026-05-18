"""
[TODO: NSC Disclaimer — see booklet page 44]

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Training window used to fit scalers (never touch val/test split).
_SCALER_FIT_END_YEAR: int = 2023

# Gap-policy thresholds (DESIGN.md §4.3).
_GAP_INTERPOLATE_H: int = 6  # gaps strictly < 6 h → linear interpolate
_GAP_FFILL_H: int = 24  # gaps 6-24 h -> forward-fill; >24 h -> mask in loss
_GAP_EXCLUDE_H: int = 7 * 24  # gaps > 168 h in a month → exclude that month


def _load_raw_parquets(
    raw_dir: Path,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Load all sensor-year parquets and map sensor_id → station_id.

    Args:
        raw_dir: Directory containing ``sensor_{id}_{year}.parquet`` files.
        metadata: 18-row DataFrame with at least ``sensor_id_pm25`` and
            ``location_id`` columns.

    Returns:
        Long DataFrame with columns ``[timestamp_utc, station_id, pm25_raw]``.
        Rows with non-PM2.5 parameters are dropped.
    """
    sensor_to_station: dict[int, int] = dict(
        zip(metadata["sensor_id_pm25"], metadata["location_id"], strict=True)
    )
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("sensor_*.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            logger.warning("Empty parquet skipped: %s", path.name)
            continue
        sid = int(df["sensor_id"].iloc[0])
        if sid not in sensor_to_station:
            logger.warning("sensor_id=%d not in metadata, skipping %s", sid, path.name)
            continue
        df = df[df["parameter_name"] == "pm25"].copy()
        if df.empty:
            logger.warning("No pm25 rows in %s", path.name)
            continue
        df["station_id"] = sensor_to_station[sid]
        df = df[["timestamp_utc", "station_id", "value"]].rename(columns={"value": "pm25_raw"})
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No valid parquets found in {raw_dir}")
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    combined["pm25_raw"] = combined["pm25_raw"].astype("float32")
    logger.info("Loaded %d rows from %d parquet files", len(combined), len(frames))
    return combined


def _resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample sub-hourly raw measurements to a strict hourly UTC grid.

    Groups by station and takes the mean of all readings within each hour.
    Stations with zero rows for a year are skipped with a WARNING.

    Args:
        df: Long DataFrame from ``_load_raw_parquets``.

    Returns:
        DataFrame indexed by ``(station_id, timestamp)`` with ``pm25_raw`` column.
        Full hourly grid 2022-01-01 00:00 → 2025-12-31 23:00 UTC per station.
    """
    full_index = pd.date_range("2022-01-01", "2025-12-31 23:00", freq="1h", tz="UTC")
    frames: list[pd.DataFrame] = []
    for station_id, group in df.groupby("station_id"):
        group = group.set_index("timestamp_utc")["pm25_raw"]
        if group.empty:
            logger.warning("station_id=%d has no data — skipped", station_id)
            continue
        resampled = group.resample("1h").mean()
        # Reindex to full hourly grid; missing hours become NaN
        resampled = resampled.reindex(full_index)
        frame = pd.DataFrame(
            {"station_id": station_id, "pm25_raw": resampled.values.astype("float32")},
            index=full_index,
        )
        frame.index.name = "timestamp"
        frames.append(frame)
    result = pd.concat(frames).reset_index()
    logger.info("Resampled to %d hourly rows across %d stations", len(result), len(frames))
    return result


def _label_gap_runs(series: pd.Series) -> pd.Series:
    """Return a Series of gap run-lengths (0 where data exists, N where gap of N hours).

    Args:
        series: Hourly time series for one station (may contain NaN).

    Returns:
        Series aligned to ``series.index`` with run-length at each NaN position.
    """
    is_null = series.isna()
    run_ids = is_null.ne(is_null.shift()).cumsum()
    # transform('sum') counts the total group size; mask to 0 where data exists
    return is_null.groupby(run_ids).transform("sum").where(is_null, 0)


def _apply_gap_policy(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Apply DESIGN.md §4.3 gap policy to a single-station hourly series.

    Gap policy (applied to ORIGINAL gaps, not post-fill):
        <6 h   -> linear interpolate
        6-24 h -> forward-fill from last known value
        >24 h  -> leave NaN; set mask_in_loss=True
        >168 h (7 d) in any month -> set exclude_from_training=True for that month

    Args:
        series: Hourly PM2.5 series with DatetimeTZDtype(UTC) index.

    Returns:
        Tuple of (filled, mask_in_loss, exclude_from_training), all aligned to
        the input index.
    """
    if not series.isna().any():
        false_series = pd.Series(False, index=series.index)
        return series.copy(), false_series, false_series.copy()

    gap_lengths = _label_gap_runs(series)
    short_mask = (gap_lengths > 0) & (gap_lengths < _GAP_INTERPOLATE_H)
    medium_mask = (gap_lengths >= _GAP_INTERPOLATE_H) & (gap_lengths <= _GAP_FFILL_H)
    long_mask = gap_lengths > _GAP_FFILL_H

    filled = series.copy()

    # Short gaps (< 6h): linear interpolation — only copy interpolated values
    # into short-gap positions so medium/long gaps are never contaminated.
    interp = series.interpolate(method="linear", limit=_GAP_INTERPOLATE_H - 1, limit_area="inside")
    filled[short_mask] = interp[short_mask]

    # Medium gaps (6-24h): forward-fill from original series.
    # Using the original (not already-modified) series prevents interpolated
    # short-gap values from propagating into adjacent medium gaps.
    ffilled = series.ffill(limit=_GAP_FFILL_H)
    filled[medium_mask] = ffilled[medium_mask]

    # Long gaps (>24h) remain NaN — mask_in_loss=True is returned as long_mask.

    # Exclude-from-training: any month where a gap exceeded 7 consecutive days.
    exclude_from_training = pd.Series(False, index=series.index)
    long_gap_idx = series.index[gap_lengths > _GAP_EXCLUDE_H]
    if len(long_gap_idx) > 0:
        months_to_exclude: set[tuple[int, int]] = {(ts.year, ts.month) for ts in long_gap_idx}
        for year, month in months_to_exclude:
            month_mask = (series.index.year == year) & (series.index.month == month)
            exclude_from_training[month_mask] = True
        logger.debug(
            "station: %d station-months excluded from training due to >7-day gaps",
            len(months_to_exclude),
        )

    return filled, long_mask, exclude_from_training


def _fit_scalers(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Fit RobustScaler per station on 2022-2023 training data only.

    Robust scaling (median + IQR) is preferred over z-score because PM2.5
    has heavy outliers during burning season (>500 ug/m3).

    Args:
        df: DataFrame with ``station_id``, ``timestamp``, and ``pm25_raw`` columns.
            ``pm25_raw`` may contain NaN (masked gaps).

    Returns:
        Dict mapping station_id → {center_: float, scale_: float}.
        ``center_`` is the median; ``scale_`` is the IQR (Q75 - Q25).
    """
    train_mask = df["timestamp"].dt.year <= _SCALER_FIT_END_YEAR
    scalers: dict[int, dict[str, float]] = {}
    for station_id, group in df[train_mask].groupby("station_id"):
        vals = group["pm25_raw"].dropna().values
        if len(vals) < 10:
            logger.warning(
                "station_id=%d has <10 training values; using global median/IQR fallback",
                station_id,
            )
            vals = df[df["station_id"] == station_id]["pm25_raw"].dropna().values
        if len(vals) == 0:
            logger.error("station_id=%d has zero valid PM2.5 values — using scale_=1", station_id)
            scalers[station_id] = {"center_": 0.0, "scale_": 1.0}
            continue
        center = float(np.median(vals))
        q75, q25 = float(np.percentile(vals, 75)), float(np.percentile(vals, 25))
        scale = q75 - q25
        if scale == 0:
            logger.warning(
                "station_id=%d IQR=0 (constant signal); using scale_=1 to avoid div-by-zero",
                station_id,
            )
            scale = 1.0
        scalers[station_id] = {"center_": center, "scale_": scale}
    logger.info("Fitted RobustScaler for %d stations", len(scalers))
    return scalers


def _normalize(df: pd.DataFrame, scalers: dict[int, dict[str, float]]) -> pd.DataFrame:
    """Add ``pm25_scaled`` column using per-station RobustScaler params.

    Args:
        df: DataFrame with ``station_id`` and ``pm25_raw`` columns.
        scalers: Output of ``_fit_scalers``.

    Returns:
        Same DataFrame with ``pm25_scaled`` column added (NaN where pm25_raw is NaN).
    """
    df = df.copy()
    scaled = np.full(len(df), float("nan"), dtype="float32")
    for station_id, params in scalers.items():
        mask = df["station_id"] == station_id
        raw = df.loc[mask, "pm25_raw"].values.astype(float)
        scaled[mask.values] = ((raw - params["center_"]) / params["scale_"]).astype("float32")
    df["pm25_scaled"] = scaled
    return df


def _add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour-of-day and day-of-year as sin/cos pairs.

    Cyclic encoding maps periodic features onto a unit circle so the model
    sees continuity across midnight (hour 23 → 0) and year boundaries.

    Args:
        df: DataFrame with a ``timestamp`` column (DatetimeTZDtype UTC).

    Returns:
        Same DataFrame with four new float32 columns:
        ``hour_sin``, ``hour_cos``, ``doy_sin``, ``doy_cos``.
    """
    df = df.copy()
    hour = df["timestamp"].dt.hour.astype("float32")
    doy = df["timestamp"].dt.dayofyear.astype("float32")
    df["hour_sin"] = np.sin(2 * math.pi * hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * math.pi * hour / 24).astype("float32")
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365).astype("float32")
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365).astype("float32")
    return df


def preprocess(
    raw_dir: Path | str = Path("data/raw/openaq"),
    processed_dir: Path | str = Path("data/processed"),
    metadata_path: Path | str | None = None,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline and save outputs.

    Pipeline (DESIGN.md §4.3, §6.1):
        1. Load raw sensor-year parquets → long DataFrame.
        2. Resample to strict hourly UTC grid (2022-01-01 to 2025-12-31).
        3. Apply gap policy per station (interpolate / ffill / mask).
        4. Fit RobustScaler on 2022-2023 training data only.
        5. Normalize all years using fitted params.
        6. Add cyclic temporal features.
        7. Save dataset.parquet and scalers.json.

    Args:
        raw_dir: Directory with OpenAQ sensor-year parquets.
        processed_dir: Output directory for dataset.parquet and scalers.json.
        metadata_path: Path to stations_metadata.parquet.
            Defaults to ``processed_dir / "stations_metadata.parquet"``.

    Returns:
        Final DataFrame matching DESIGN.md §6.1 schema (minus ERA5 weather columns,
        which are added in Session 3).
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path is None:
        metadata_path = processed_dir / "stations_metadata.parquet"
    metadata = pd.read_parquet(metadata_path)

    logger.info("=== Preprocessing pipeline start ===")

    # 1. Load raw parquets
    raw = _load_raw_parquets(raw_dir, metadata)

    # 2. Resample to hourly grid
    hourly = _resample_hourly(raw)

    # 3. Apply gap policy per station
    filled_frames: list[pd.DataFrame] = []
    for station_id, group in hourly.groupby("station_id"):
        group = group.sort_values("timestamp").copy()
        series = group.set_index("timestamp")["pm25_raw"]
        filled, mask_in_loss, exclude = _apply_gap_policy(series)
        result = pd.DataFrame(
            {
                "timestamp": filled.index,
                "station_id": station_id,
                "pm25_raw": filled.values.astype("float32"),
                "mask_in_loss": mask_in_loss.values,
                "exclude_from_training": exclude.values,
            }
        )
        filled_frames.append(result)
        n_mask = int(mask_in_loss.sum())
        n_excl = int(exclude.sum())
        logger.info(
            "station_id=%d: masked=%d h, excluded=%d h (%.1f%% of series)",
            station_id,
            n_mask,
            n_excl,
            100 * n_excl / max(len(series), 1),
        )

    dataset = pd.concat(filled_frames, ignore_index=True)

    # 4. Fit scalers (training years only)
    scalers = _fit_scalers(dataset)
    scalers_path = processed_dir / "scalers.json"
    with open(scalers_path, "w") as fh:
        json.dump({str(k): v for k, v in scalers.items()}, fh, indent=2)
    logger.info("Saved scaler params to %s", scalers_path)

    # 5. Normalize
    dataset = _normalize(dataset, scalers)

    # 6. Cyclic features
    dataset = _add_cyclic_features(dataset)

    # 7. Finalise schema order and types
    col_order = [
        "timestamp",
        "station_id",
        "pm25_raw",
        "pm25_scaled",
        "mask_in_loss",
        "exclude_from_training",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    dataset = dataset[col_order].copy()
    dataset["timestamp"] = dataset["timestamp"].dt.tz_convert("UTC")
    dataset["station_id"] = dataset["station_id"].astype("int64")
    dataset["mask_in_loss"] = dataset["mask_in_loss"].astype(bool)
    dataset["exclude_from_training"] = dataset["exclude_from_training"].astype(bool)

    out_path = processed_dir / "dataset.parquet"
    dataset.to_parquet(out_path, index=False)
    logger.info(
        "=== Preprocessing done — saved %d rows to %s ===",
        len(dataset),
        out_path,
    )
    return dataset
