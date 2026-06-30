# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Merge ERA5 weather features into the processed dataset parquet.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Joins data/processed/era5.parquet (u10, v10, t2m, d2m, blh) into
data/processed/dataset.parquet on station_id + timestamp. Safe to
re-run — existing ERA5 columns are dropped before re-merging.

Usage:
    uv run python scripts/02_preprocess.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_ERA5_COLS: list[str] = ["u10", "v10", "t2m", "d2m", "blh"]


def main() -> None:
    dataset_path = Path("data/processed/dataset.parquet")
    era5_path = Path("data/processed/era5.parquet")

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.parquet not found: {dataset_path}")
    if not era5_path.exists():
        raise FileNotFoundError(
            f"era5.parquet not found: {era5_path}. "
            "Run: uv run python scripts/01_download_all.py era5"
        )

    logger.info("Loading dataset.parquet ...")
    df = pd.read_parquet(dataset_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    logger.info("Loading era5.parquet ...")
    era5 = pd.read_parquet(era5_path)
    # ERA5 uses 'datetime' column; align name for merge key.
    era5 = era5.rename(columns={"datetime": "timestamp"})
    era5["timestamp"] = pd.to_datetime(era5["timestamp"], utc=True)

    # Idempotent: drop existing ERA5 columns before re-merging.
    existing = [c for c in _ERA5_COLS if c in df.columns]
    if existing:
        logger.info("Dropping existing ERA5 columns for re-merge: %s", existing)
        df = df.drop(columns=existing)

    logger.info("Merging ERA5 features on station_id + timestamp ...")
    merged = df.merge(
        era5[["station_id", "timestamp", *_ERA5_COLS]],
        on=["station_id", "timestamp"],
        how="left",
    )

    n_missing = merged[_ERA5_COLS].isna().sum().sum()
    if n_missing > 0:
        logger.warning(
            "ERA5 merge left %d NaN cells across %d total — check ERA5 date coverage.",
            n_missing,
            merged[_ERA5_COLS].size,
        )
    else:
        logger.info("ERA5 merge complete — no NaN cells.")

    logger.info("Merged shape: %s  columns: %s", merged.shape, list(merged.columns))
    merged.to_parquet(dataset_path, index=False)
    logger.info("Saved enriched dataset to %s", dataset_path)


if __name__ == "__main__":
    main()
