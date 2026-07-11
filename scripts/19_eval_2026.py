# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Frozen-model evaluation on the Jan-Apr 2026 burning season (100% out-of-sample).

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Roadmap item 8 (DEMO_ROADMAP.md): the 2026 burning-season data did not exist when
the model was designed or trained, so scoring the FROZEN checkpoints on it is a true
out-of-sample test that rules out every leakage path. Honest evaluation is the
project's thesis: if the model loses to persistence, that IS the reported result.

Design constraints (iron rule until SIMS submit):
  * Never touches the 2022-2025 dataset, scalers, or report JSONs.
  * Reuses the training-era ``scalers.json`` as-is (NO refit) — a frozen-model eval
    must feed inputs consistent with the checkpoint.
  * Reuses ``src.data.preprocessing`` helpers, ``hotspot_clustering.cluster_hotspots``,
    ``src.data.scrapers.era5`` interpolation, and ``src.training.evaluation`` so the
    numbers are directly comparable to ``outputs/evaluation_test2025.json``.
  * All 2026 artifacts are separate files (openaq_2026/, firms_2026/, dataset_2026,
    hotspots_2026, era5_2026, evaluation_2026burn.json).

Subcommands (run in order; downloads are slow, so they are separate):
    backfill-openaq  -- OpenAQ PM2.5 for 18 stations, Jan-Apr 2026 -> openaq_2026/
    backfill-firms   -- FIRMS SP hotspots, Jan-Apr 2026 -> firms_2026/
    backfill-era5    -- ERA5 months 01-04 2026 -> era5_2026.parquet
    build            -- dataset_2026.parquet + hotspots_2026.parquet
    evaluate         -- score frozen checkpoints -> outputs/evaluation_2026burn.json

Usage:
    uv run python scripts/19_eval_2026.py backfill-openaq
    uv run python scripts/19_eval_2026.py backfill-firms
    uv run python scripts/19_eval_2026.py backfill-era5
    uv run python scripts/19_eval_2026.py build
    uv run python scripts/19_eval_2026.py evaluate
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from src.data.preprocessing import (
    _add_cyclic_features,
    _apply_gap_policy,
    _load_raw_parquets,
    _normalize,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_2026")

app = typer.Typer(add_completion=False)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

# --- 2026 evaluation window (Jan 1 -> Apr 30, burning season) ---
_WINDOW_START = date(2026, 1, 1)
_WINDOW_END = date(2026, 4, 30)
_BBOX = (97.0, 16.0, 101.5, 21.0)  # west, south, east, north — matches BBOX_NORTHERN_THAILAND

# Separate raw/processed locations — never collide with 2022-2025 artifacts.
_OPENAQ_2026_DIR = _PROJECT_ROOT / "data" / "raw" / "openaq_2026"
_FIRMS_2026_DIR = _PROJECT_ROOT / "data" / "raw" / "firms_2026"
_ERA5_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "era5"
_ERA5_2026_PARQUET = _PROJECT_ROOT / "data" / "processed" / "era5_2026.parquet"
_DATASET_2026 = _PROJECT_ROOT / "data" / "processed" / "dataset_2026.parquet"
_HOTSPOTS_2026 = _PROJECT_ROOT / "data" / "processed" / "hotspots_2026.parquet"
_METADATA = _PROJECT_ROOT / "data" / "processed" / "stations_metadata.parquet"
_SCALERS = _PROJECT_ROOT / "data" / "processed" / "scalers.json"
_OUTPUT_JSON = _PROJECT_ROOT / "outputs" / "evaluation_2026burn.json"

# Frozen checkpoints scored on 2026 (both mtgnn architecture).
_CHECKPOINTS: list[tuple[str, str]] = [
    ("mtgnn_split2", "checkpoints_split2/mtgnn/best_model.pt"),  # report protocol (primary)
    ("mtgnn_demo", "checkpoints/mtgnn/best_model.pt"),  # app/demo model
]

_ERA5_COLS = ["u10", "v10", "t2m", "d2m", "blh"]


def _full_index_2026() -> pd.DatetimeIndex:
    """Return the hourly UTC grid for the Jan-Apr 2026 evaluation window."""
    return pd.date_range(
        _WINDOW_START.isoformat(),
        f"{_WINDOW_END.isoformat()} 23:00",
        freq="1h",
        tz="UTC",
    )


def _split_bounds_2026() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the split-bounds map placing the whole 2026 window in 'test'."""
    idx = _full_index_2026()
    return {"test": (idx[0], idx[-1])}


# ---------------------------------------------------------------------------
# Pure dataset-frame construction (testable, no network)
# ---------------------------------------------------------------------------


def _resample_hourly_2026(raw_long: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Resample raw sub-hourly PM2.5 to a strict hourly grid over ``full_index``.

    Mirrors ``preprocessing._resample_hourly`` but onto a caller-supplied index so
    the 2026 window is isolated from the 2022-2025 grid. Stations absent from the
    raw frame are left out here; the loader later reindexes them to NaN (masked).

    Args:
        raw_long: Long frame with ``timestamp_utc``, ``station_id``, ``pm25_raw``.
        full_index: Target hourly UTC index.

    Returns:
        Long frame with ``timestamp``, ``station_id``, ``pm25_raw`` on the grid.
    """
    frames: list[pd.DataFrame] = []
    for station_id, group in raw_long.groupby("station_id"):
        series = group.set_index("timestamp_utc")["pm25_raw"]
        if series.empty:
            logger.warning("station_id=%d has no 2026 data — skipped", station_id)
            continue
        resampled = series.resample("1h").mean().reindex(full_index)
        frame = pd.DataFrame(
            {"station_id": station_id, "pm25_raw": resampled.values.astype("float32")},
            index=full_index,
        )
        frame.index.name = "timestamp"
        frames.append(frame)
    return pd.concat(frames).reset_index()


def build_2026_dataset_frame(
    raw_long: pd.DataFrame,
    scalers: dict[int, dict[str, float]],
    full_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the 2026 dataset frame with FROZEN normalization (no scaler refit).

    Replicates ``preprocessing.preprocess`` steps 2-6 (resample, gap policy,
    normalize, cyclic features) but reuses the supplied training-era ``scalers``
    verbatim so inputs stay consistent with the frozen checkpoints. ERA5 columns
    are merged separately by :func:`build`.

    Args:
        raw_long: Long frame (``timestamp_utc``, ``station_id``, ``pm25_raw``).
        scalers: Existing per-station RobustScaler params from scalers.json.
        full_index: Hourly UTC grid for the 2026 window.

    Returns:
        Frame matching the 2022-2025 dataset schema minus ERA5 columns.
    """
    hourly = _resample_hourly_2026(raw_long, full_index)

    filled_frames: list[pd.DataFrame] = []
    for station_id, group in hourly.groupby("station_id"):
        series = group.sort_values("timestamp").set_index("timestamp")["pm25_raw"]
        filled, mask_in_loss, exclude = _apply_gap_policy(series)
        filled_frames.append(
            pd.DataFrame(
                {
                    "timestamp": filled.index,
                    "station_id": station_id,
                    "pm25_raw": filled.values.astype("float32"),
                    "mask_in_loss": mask_in_loss.values,
                    "exclude_from_training": exclude.values,
                }
            )
        )
    dataset = pd.concat(filled_frames, ignore_index=True)

    dataset = _normalize(dataset, scalers)  # frozen: uses supplied center_/scale_
    dataset = _add_cyclic_features(dataset)

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
    return dataset


def _load_scalers_int(path: Path) -> dict[int, dict[str, float]]:
    """Load scalers.json with integer station keys (frozen params, read-only)."""
    with open(path) as fh:
        raw: dict[str, dict[str, float]] = json.load(fh)
    return {int(k): v for k, v in raw.items()}


def _station_coverage(dataset: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, dict]:
    """Per-station fraction of hours with a finite PM2.5 reading in the window."""
    total_hours = dataset["timestamp"].nunique()
    name_by_id = dict(zip(metadata["location_id"], metadata["name"], strict=False))
    coverage: dict[str, dict] = {}
    present = set(dataset["station_id"].unique().tolist())
    for sid in metadata["location_id"].tolist():
        if sid in present:
            sub = dataset[dataset["station_id"] == sid]["pm25_raw"]
            n_valid = int(np.isfinite(sub.values).sum())
        else:
            n_valid = 0
        coverage[str(sid)] = {
            "name": name_by_id.get(sid, ""),
            "valid_hours": n_valid,
            "total_hours": int(total_hours),
            "coverage_pct": round(100.0 * n_valid / max(total_hours, 1), 1),
        }
    return coverage


# ---------------------------------------------------------------------------
# Backfill subcommands (real network — never run in tests)
# ---------------------------------------------------------------------------


@app.command(name="backfill-openaq")
def backfill_openaq() -> None:
    """Fetch OpenAQ PM2.5 for the 18 curated stations, Jan-Apr 2026 -> openaq_2026/."""
    import time

    from src.data.scrapers.openaq import fetch_measurements

    _OPENAQ_2026_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(_METADATA)

    for _, row in meta.iterrows():
        sensor_id = int(row["sensor_id_pm25"])
        out_path = _OPENAQ_2026_DIR / f"sensor_{sensor_id}_2026.parquet"
        if out_path.exists():
            logger.info("Skipping %s (exists)", out_path.name)
            continue

        chunk_frames: list[pd.DataFrame] = []
        chunk_start = datetime(_WINDOW_START.year, _WINDOW_START.month, _WINDOW_START.day)
        window_end = datetime(_WINDOW_END.year, _WINDOW_END.month, _WINDOW_END.day, 23, 59, 59)
        while chunk_start <= window_end:
            chunk_end = min(
                chunk_start + timedelta(days=6, hours=23, minutes=59, seconds=59), window_end
            )
            df_chunk = fetch_measurements(sensor_id, chunk_start, chunk_end)
            chunk_frames.append(df_chunk)
            logger.info(
                "sensor=%d %s->%s: %d rows",
                sensor_id,
                chunk_start.date(),
                chunk_end.date(),
                len(df_chunk),
            )
            time.sleep(1)
            chunk_start = chunk_end + timedelta(seconds=1)

        df = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s (%d rows, %s)", out_path.name, len(df), row["name"])

    typer.echo(f"OpenAQ 2026 backfill complete -> {_OPENAQ_2026_DIR}")


@app.command(name="backfill-firms")
def backfill_firms() -> None:
    """Fetch FIRMS SP hotspots for Jan-Apr 2026 into a separate firms_2026/ cache."""
    from src.data.scrapers.firms import fetch_hotspots_historical

    _FIRMS_2026_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_hotspots_historical(
        bbox=_BBOX,
        start_date=_WINDOW_START.isoformat(),
        end_date=_WINDOW_END.isoformat(),
        source="VIIRS_NOAA20_SP",
        cache_dir=_FIRMS_2026_DIR,
    )
    typer.echo(
        f"FIRMS 2026 SP backfill: {len(df)} rows, {_WINDOW_START}->{_WINDOW_END} "
        f"-> {_FIRMS_2026_DIR}"
    )


@app.command(name="backfill-era5")
def backfill_era5() -> None:
    """Download ERA5 months 01-04 of 2026 and interpolate to stations -> era5_2026.parquet.

    Uses ``era5._download_era5_month`` (months 1-4 only) instead of
    ``download_era5_year`` because the latter forces all 12 months and would fail on
    not-yet-existing future months. era5.py itself is left unmodified.
    """
    import cdsapi

    from src.data.scrapers.era5 import (
        _download_era5_month,
        _filter_and_sort,
        _load_stations,
        interpolate_to_stations,
    )

    _ERA5_RAW_DIR.mkdir(parents=True, exist_ok=True)
    stations_df = _load_stations()
    client = cdsapi.Client()

    frames: list[pd.DataFrame] = []
    for month in range(_WINDOW_START.month, _WINDOW_END.month + 1):
        nc_path = _download_era5_month(
            year=2026, month=month, output_dir=_ERA5_RAW_DIR, bbox=_BBOX, client=client
        )
        frames.append(interpolate_to_stations(nc_path=nc_path, stations_df=stations_df))
        logger.info("ERA5 2026-%02d interpolated", month)

    combined = _filter_and_sort(pd.concat(frames, ignore_index=True), _WINDOW_START, _WINDOW_END)
    _ERA5_2026_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_ERA5_2026_PARQUET, index=False)
    typer.echo(
        f"ERA5 2026 done: {len(combined):,} rows, {combined['station_id'].nunique()} stations "
        f"-> {_ERA5_2026_PARQUET}"
    )


# ---------------------------------------------------------------------------
# Build subcommand
# ---------------------------------------------------------------------------


@app.command()
def build() -> None:
    """Build dataset_2026.parquet (frozen scalers + ERA5) and hotspots_2026.parquet."""
    from src.data.hotspot_clustering import cluster_hotspots

    metadata = pd.read_parquet(_METADATA)
    scalers = _load_scalers_int(_SCALERS)
    full_index = _full_index_2026()

    # --- PM2.5 dataset frame (frozen normalization) ---
    raw_long = _load_raw_parquets(_OPENAQ_2026_DIR, metadata)
    raw_long = raw_long[
        (raw_long["timestamp_utc"] >= full_index[0]) & (raw_long["timestamp_utc"] <= full_index[-1])
    ].reset_index(drop=True)
    dataset = build_2026_dataset_frame(raw_long, scalers, full_index)

    # --- Merge ERA5 (raw, as the frozen model was trained) ---
    if _ERA5_2026_PARQUET.exists():
        era5 = pd.read_parquet(_ERA5_2026_PARQUET).rename(columns={"datetime": "timestamp"})
        era5["timestamp"] = pd.to_datetime(era5["timestamp"], utc=True)
        dataset = dataset.merge(
            era5[["station_id", "timestamp", *_ERA5_COLS]],
            on=["station_id", "timestamp"],
            how="left",
        )
        n_missing = int(dataset[_ERA5_COLS].isna().sum().sum())
        logger.info("ERA5 merged; %d NaN cells across ERA5 columns", n_missing)
    else:
        logger.warning(
            "era5_2026.parquet missing — dataset_2026 has NO weather features; run "
            "backfill-era5 first for a faithful frozen eval."
        )

    _DATASET_2026.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(_DATASET_2026, index=False)
    logger.info(
        "Saved %s (%d rows, %d stations)",
        _DATASET_2026,
        len(dataset),
        dataset["station_id"].nunique(),
    )

    # --- Hotspot clusters (2026-only FIRMS cache) ---
    if _FIRMS_2026_DIR.exists() and any(_FIRMS_2026_DIR.glob("*.csv")):
        cluster_hotspots(firms_dir=_FIRMS_2026_DIR, output_path=_HOTSPOTS_2026)
        logger.info("Saved %s", _HOTSPOTS_2026)
    else:
        logger.warning(
            "No FIRMS 2026 CSVs found in %s — run backfill-firms first.", _FIRMS_2026_DIR
        )

    cov = _station_coverage(dataset, metadata)
    typer.echo("Per-station 2026 PM2.5 coverage:")
    for sid, info in cov.items():
        typer.echo(f"  {sid:>8} {info['coverage_pct']:>5.1f}%  {info['name']}")


# ---------------------------------------------------------------------------
# Evaluate subcommand
# ---------------------------------------------------------------------------


@app.command()
def evaluate(device: str = typer.Option("cpu", help="Torch device (cpu/cuda).")) -> None:
    """Score the frozen checkpoints on the 2026 window -> outputs/evaluation_2026burn.json."""
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from torch_geometric.loader import DataLoader

    from src.data.loader import PM25GraphDataset
    from src.training.evaluation import (
        build_ground_truth,
        denorm_pred,
        predict,
        rmse_block,
        station_scalers,
    )

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — using CPU.")
        device = "cpu"

    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config")
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")

    ds = PM25GraphDataset(
        dataset_path=_DATASET_2026,
        hotspots_path=_HOTSPOTS_2026,
        metadata_path=_METADATA,
        scalers_path=_SCALERS,
        split="test",
        window_in=cfg.data.window_in,
        horizons=horizons,
        exclude_stations=list(cfg.data.exclude_stations),
        graph_config={"wind_mode": wind_mode},
        full_index=_full_index_2026(),
        split_bounds=_split_bounds_2026(),
    )
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    logger.info(
        "2026 eval: %d samples, %d stations, wind_mode=%s", len(ds), ds.n_stations, wind_mode
    )

    centers, scales = station_scalers(ds)
    y_ug, mask, pers_ug = build_ground_truth(ds, horizons)
    persistence_rmse = rmse_block(pers_ug, y_ug, mask, horizons)
    logger.info("Persistence RMSE (ug/m3): %s", persistence_rmse)

    per_ckpt: dict[str, dict] = {}
    for name, ckpt_rel in _CHECKPOINTS:
        ckpt_path = _PROJECT_ROOT / ckpt_rel
        if not ckpt_path.exists():
            logger.warning("Checkpoint missing: %s — skipping", ckpt_path)
            continue
        with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
            mcfg = compose(config_name="config", overrides=["model=mtgnn"])
        model = instantiate(mcfg.model, n_stations=ds.n_stations, horizons=horizons)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)

        pred_ug = denorm_pred(predict(model, loader, device), centers, scales, ds.n_stations)
        rmse = rmse_block(pred_ug, y_ug, mask, horizons)
        beats = {f"{h}h": bool(rmse[f"{h}h"] < persistence_rmse[f"{h}h"]) for h in horizons}
        improvement = {
            f"{h}h": round(
                (persistence_rmse[f"{h}h"] - rmse[f"{h}h"]) / persistence_rmse[f"{h}h"] * 100.0, 1
            )
            for h in horizons
        }
        per_ckpt[name] = {
            "checkpoint": ckpt_rel,
            "rmse_ug_m3": rmse,
            "beats_persistence": beats,
            "improvement_vs_persistence_pct": improvement,
        }
        logger.info("%s RMSE=%s beats=%s", name, rmse, beats)

    metadata = pd.read_parquet(_METADATA)
    coverage = _station_coverage(pd.read_parquet(_DATASET_2026), metadata)

    results = {
        "split_description": "frozen eval, Jan-Apr 2026 (100% out-of-sample burning season)",
        "window": {"start": _WINDOW_START.isoformat(), "end": _WINDOW_END.isoformat()},
        "n_samples": len(ds),
        "n_stations": ds.n_stations,
        "horizons_h": horizons,
        "wind_mode": wind_mode,
        "persistence_rmse_ug_m3": persistence_rmse,
        "checkpoints": per_ckpt,
        "per_station_coverage": coverage,
        "note": (
            "Frozen-model out-of-sample evaluation. No retrain and no scaler refit: the "
            "training-era per-station RobustScaler (scalers.json) is reused verbatim, and "
            "ERA5 weather features enter raw exactly as during training. RMSE in ug/m3, "
            "targets/persistence from raw PM2.5, model predictions denormalized via "
            "per-station IQR; NaN targets and low-coverage stations masked. Protocol matches "
            "scripts/11_ablation_eval.py so numbers are comparable to "
            "outputs/evaluation_test2025.json. FIRMS provenance: VIIRS_NOAA20_SP (Standard "
            "Product; the Jan-Apr 2026 window is > the ~60-day SP lag, so no NRT was needed)."
        ),
    }

    _OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"{'method':<20}" + "".join(f"{f'{h}h':>10}" for h in horizons))
    print("-" * 64)
    print(f"{'persistence':<20}" + "".join(f"{persistence_rmse[f'{h}h']:>10.2f}" for h in horizons))
    for name, block in per_ckpt.items():
        r = block["rmse_ug_m3"]
        print(f"{name:<20}" + "".join(f"{r[f'{h}h']:>10.2f}" for h in horizons))
    print("=" * 64 + "  (RMSE ug/m3, lower is better)\n")
    logger.info("Saved %s", _OUTPUT_JSON)


if __name__ == "__main__":
    app()
