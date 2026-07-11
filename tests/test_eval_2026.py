"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

Unit tests for the frozen 2026 out-of-sample evaluation (scripts/19_eval_2026.py)
and the opt-in ``full_index``/``split_bounds`` parameters added to
``src.data.loader.PM25GraphDataset``.

All tests use synthetic in-memory / tmp_path data — no real files, no network.
The key property under test is FROZEN normalization: the 2026 dataset frame must
reuse the supplied training-era scalers verbatim (no refit), otherwise the
out-of-sample claim breaks.
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import PM25GraphDataset

_ROOT = Path(__file__).resolve().parent.parent


def _load_eval_module():
    """Import the digit-prefixed scripts/19_eval_2026.py by file path."""
    spec = importlib.util.spec_from_file_location(
        "eval_2026_mod", _ROOT / "scripts" / "19_eval_2026.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_2026 = _load_eval_module()


# ---------------------------------------------------------------------------
# Frozen-normalization behaviour
# ---------------------------------------------------------------------------


def _synthetic_raw_long(start: str, end: str, station_ids: list[int]) -> pd.DataFrame:
    """Build a raw-long frame (timestamp_utc, station_id, pm25_raw) with no gaps."""
    ts = pd.date_range(start, end, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for sid in station_ids:
        vals = 30.0 + 10.0 * rng.standard_normal(len(ts))
        for t, v in zip(ts, vals, strict=True):
            rows.append({"timestamp_utc": t, "station_id": sid, "pm25_raw": float(v)})
    return pd.DataFrame(rows)


def test_frozen_scaling_uses_supplied_params_not_data_median():
    """pm25_scaled must equal (raw - center_)/scale_ from the SUPPLIED scalers."""
    full_index = pd.date_range("2026-01-01", "2026-01-05 23:00", freq="1h", tz="UTC")
    raw = _synthetic_raw_long("2026-01-01", "2026-01-05 23:00", [100, 101])
    # Deliberately off-center scalers so a refit would give different numbers.
    scalers = {100: {"center_": 10.0, "scale_": 2.0}, 101: {"center_": 50.0, "scale_": 8.0}}
    scalers_snapshot = copy.deepcopy(scalers)

    frame = eval_2026.build_2026_dataset_frame(raw, scalers, full_index)

    for sid, params in scalers.items():
        sub = frame[frame["station_id"] == sid]
        expected = (sub["pm25_raw"].to_numpy() - params["center_"]) / params["scale_"]
        np.testing.assert_allclose(sub["pm25_scaled"].to_numpy(), expected, rtol=1e-5, atol=1e-4)

    # Scalers must not be mutated (frozen contract).
    assert scalers == scalers_snapshot


def test_build_frame_schema_and_index():
    """Frame has the exact 2022-2025 schema (minus ERA5) on the 2026 grid."""
    full_index = pd.date_range("2026-01-01", "2026-01-03 23:00", freq="1h", tz="UTC")
    raw = _synthetic_raw_long("2026-01-01", "2026-01-03 23:00", [100])
    scalers = {100: {"center_": 20.0, "scale_": 5.0}}

    frame = eval_2026.build_2026_dataset_frame(raw, scalers, full_index)

    assert list(frame.columns) == [
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
    assert frame["mask_in_loss"].dtype == bool
    assert frame["timestamp"].min() == full_index[0]
    assert frame["timestamp"].max() == full_index[-1]


def test_station_coverage_reports_missing_station_as_zero():
    """A station present in metadata but absent from the frame reports 0% coverage."""
    full_index = pd.date_range("2026-01-01", "2026-01-02 23:00", freq="1h", tz="UTC")
    raw = _synthetic_raw_long("2026-01-01", "2026-01-02 23:00", [100])
    scalers = {100: {"center_": 20.0, "scale_": 5.0}}
    frame = eval_2026.build_2026_dataset_frame(raw, scalers, full_index)

    metadata = pd.DataFrame(
        {
            "location_id": [100, 999],
            "name": ["Present", "Dead2026"],
            "lat": [18.0, 19.0],
            "lon": [99.0, 100.0],
        }
    )
    cov = eval_2026._station_coverage(frame, metadata)

    assert cov["100"]["coverage_pct"] == pytest.approx(100.0, abs=0.5)
    assert cov["999"]["valid_hours"] == 0
    assert cov["999"]["coverage_pct"] == 0.0


# ---------------------------------------------------------------------------
# Loader opt-in full_index / split_bounds
# ---------------------------------------------------------------------------


def _write_2026_dataset(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create synthetic 2026 dataset/hotspots/metadata/scalers files."""
    full_index = pd.date_range("2026-01-01", "2026-01-07 23:00", freq="1h", tz="UTC")
    raw = _synthetic_raw_long("2026-01-01", "2026-01-07 23:00", [100, 101])
    scalers = {100: {"center_": 20.0, "scale_": 5.0}, 101: {"center_": 22.0, "scale_": 6.0}}
    frame = eval_2026.build_2026_dataset_frame(raw, scalers, full_index)
    # Add zero ERA5 columns so the loader has weather features present.
    for col in ["u10", "v10", "t2m", "d2m", "blh"]:
        frame[col] = 0.0
    dataset_path = tmp_path / "dataset_2026.parquet"
    frame.to_parquet(dataset_path, index=False)

    meta = pd.DataFrame(
        {
            "location_id": [100, 101],
            "name": ["S100", "S101"],
            "lat": [18.5, 19.0],
            "lon": [99.0, 99.5],
            "provider": ["Air4Thai", "Air4Thai"],
        }
    )
    metadata_path = tmp_path / "stations_metadata.parquet"
    meta.to_parquet(metadata_path, index=False)

    scalers_path = tmp_path / "scalers.json"
    scalers_path.write_text(json.dumps({str(k): v for k, v in scalers.items()}))

    hotspots = pd.DataFrame(
        {
            "date": ["2026-01-03"],
            "cluster_id": [1],
            "centroid_lat": [18.5],
            "centroid_lon": [99.0],
            "total_frp": [100.0],
            "country": ["Thailand"],
        }
    )
    hotspots_path = tmp_path / "hotspots_2026.parquet"
    hotspots.to_parquet(hotspots_path, index=False)
    return dataset_path, hotspots_path, metadata_path, scalers_path


def test_loader_custom_2026_window_produces_samples(tmp_path: Path):
    """PM25GraphDataset with a 2026 full_index/split_bounds yields anchors in-window."""
    dataset_path, hotspots_path, metadata_path, scalers_path = _write_2026_dataset(tmp_path)
    full_index = pd.date_range("2026-01-01", "2026-01-07 23:00", freq="1h", tz="UTC")
    split_bounds = {"test": (full_index[0], full_index[-1])}

    ds = PM25GraphDataset(
        dataset_path=dataset_path,
        hotspots_path=hotspots_path,
        metadata_path=metadata_path,
        scalers_path=scalers_path,
        split="test",
        window_in=6,
        horizons=[1, 2, 3, 6],
        graph_config={"wind_mode": "constant_ne"},
        full_index=full_index,
        split_bounds=split_bounds,
    )
    assert len(ds) > 0
    assert ds.n_stations == 2
    # Every anchor timestamp must fall inside the 2026 window.
    anchor_ts = full_index[ds._anchor_indices]
    assert anchor_ts.min() >= full_index[0]
    assert anchor_ts.max() <= full_index[-1]


def test_loader_defaults_unchanged_reject_2026_data(tmp_path: Path):
    """Without the opt-in params the loader uses the 2022-2025 grid (2026 data drops out)."""
    dataset_path, hotspots_path, metadata_path, scalers_path = _write_2026_dataset(tmp_path)
    # Default full_index is 2022-2025; 2026 rows land outside → no valid test anchors.
    with pytest.raises(RuntimeError, match="No valid anchor indices"):
        PM25GraphDataset(
            dataset_path=dataset_path,
            hotspots_path=hotspots_path,
            metadata_path=metadata_path,
            scalers_path=scalers_path,
            split="test",
            window_in=6,
            horizons=[1, 2, 3, 6],
            graph_config={"wind_mode": "constant_ne"},
        )
