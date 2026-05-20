# SESSION 3 NOTES — Training Stack Debug & ERA5 Integration

Date: 2026-05-20
Branch: `feat/session-3-era5-models`

---

## What Was Built

| Deliverable | File | Status |
|---|---|---|
| Training target scale fix | `src/data/loader.py` | Fixed — `pm25_scaled` not `pm25_raw` |
| NaN guard in trainer | `src/training/trainer.py` | Fixed — raises RuntimeError on NaN loss |
| `n_stations` dataset property | `src/data/loader.py` | Added — derives from actual station list |
| Training script n_stations fix | `scripts/03_train.py` | Fixed — uses dataset property, not Hydra config |
| ERA5 scraper: monthly chunking | `src/data/scrapers/era5.py` | Fixed — 12 req/year bypasses CDS v2 size limit |
| ERA5 scraper: Windows path fix | `src/data/scrapers/era5.py` | Fixed — temp-file roundtrip bypasses C library |
| ERA5 scraper: coord normalise | `src/data/scrapers/era5.py` | Fixed — `valid_time`->`time` before concat |
| ERA5 download: 2022-2025 | `data/processed/era5.parquet` | Done — 631,152 rows, 0 NaN |
| ERA5 preprocessing merge | `scripts/02_preprocess.py` | Done — idempotent merge into dataset.parquet |
| ERA5 features in loader | `src/data/loader.py` | Done — 10 features (was 5) |
| Model config update | `configs/model/a3tgcn.yaml`, `mtgnn.yaml` | Done — `n_features: 10` |
| Stale docstring cleanup | `src/models/base.py`, `src/training/losses.py` | Fixed — normalized scale, not raw ug/m3 |
| Training tests | `tests/test_training.py` | 31 tests — losses/metrics 100%, trainer 88% |
| ERA5 download tests | `tests/test_era5.py` | Existing tests |
| Total test suite | `tests/` | 163/163 green |

---

## Bugs Found and Fixed

### Bug 1: Training crash at epoch ~9 (NaN weights)

**Symptom**: Training appeared to freeze after epoch 9 — no output, no error.

**Root cause**: `batch["station"].y` was built from `pm25_raw` (range 15-500 ug/m3).
The model outputs normalized values (~0-5). MSE computed `(~2 - ~80)^2 ~ 6000` per
step instead of ~1. After ~9 epochs of explosive gradient updates, all weights went
NaN. The early-stopping comparison `NaN < inf = False` silently incremented the
patience counter, and the training loop kept running but produced no visible log
output (log format only printed when loss improved).

**Fix**:
- `loader.py __getitem__`: changed target to `y_col = self._pm25_scaled[t_h, :]`
- `trainer.py`: added NaN guard — raises `RuntimeError` immediately on first NaN loss
- `losses.py`, `base.py`: updated docstrings to say "normalized scale" not "raw ug/m3"

### Bug 2: Windows cp874 UnicodeEncodeError on special characters

**Symptom**: `--- Logging error ---` in stderr; log messages silently dropped.

**Root cause**: Windows Thai terminal (cp874 encoding) cannot encode `µ` (U+00B5) or
`³` (U+00B3). Affected the final summary line in `scripts/03_train.py`.

**Fix**: Use ASCII equivalents (`ug/m3`, `(normalized scale)`) in all log strings.

### Bug 3: n_stations mismatch when `exclude_stations` is non-empty

**Symptom**: `ValueError: Total nodes X is not divisible by n_stations=18` when
`exclude_stations=[<id>]` removes a station, leaving 17 nodes but the model
expects 18.

**Root cause**: `scripts/03_train.py` instantiated the model with `n_stations=18`
hardcoded from the Hydra config, not from the actual post-filter dataset size.

**Fix**:
- Added `n_stations` property to `PM25GraphDataset` (`len(self._station_ids)`)
- Training script now uses `train_ds.n_stations`

### Bug 4: ERA5 CDS API v2 — full-year request rejected (HTTP 403)

**Root cause**: New Copernicus platform (2024+) has per-request cost limits.
A full year x 5 variables exceeds the threshold.

**Fix**: Download one month at a time (12 requests/year, ~3 MB each), merge with
`xr.concat`. Monthly files cached as `era5_{year}_{month:02d}.nc` — safe to retry.

### Bug 5: ERA5 CDS API v2 — licence not accepted (HTTP 403)

**Root cause**: CDS API v2 requires explicit one-time licence acceptance per dataset
via the web UI. The `.cdsapirc` key alone is not enough.

**Fix**: Manual one-time step — accept at:
`https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download`
("Manage licences" tab).

### Bug 6: netCDF4 C library cannot open Thai-character paths on Windows

**Symptom**: `FileNotFoundError` when opening monthly `.nc` files that exist on disk.

**Root cause**: The `netCDF4` C library uses `fopen()` with the raw path bytes. On
Windows with a Thai-character working directory (`D:\งาน\...`), the C layer cannot
decode the path. The `scipy` backend was tried as a workaround but only handles
NetCDF3 — ERA5 files are NetCDF4 (HDF5-based).

**Fix**: `_open_nc(path)` and `_write_nc(ds, path)` helpers copy files to/from a
temp file in the system temp directory (always ASCII on Windows), bypassing the C
library's path handling entirely. `xr.load_dataset` (loads all data into memory
before closing the file handle) used for reads.

### Bug 7: `xr.concat` dimension mismatch — duplicate `time` dims

**Symptom**: `ValueError: broadcasting cannot handle duplicate dimensions on a
variable: ['time', 'time', 'lat', 'lon']` during bilinear interpolation.

**Root cause**: ERA5 monthly files from CDS API v2 use `valid_time` as the time
dimension name (not `time`). The merge step called `xr.concat(datasets, dim="time")`
without first normalising coordinates. This created a new outer `time` dim (size=12)
stacked over the existing `valid_time` dim, producing shape `(12, 744, lat, lon)`.
When `_normalise_coords` later renamed `valid_time`->`time`, the result had two
dimensions both named `time`.

**Fix**: Call `_normalise_coords(_open_nc(mp))` for each monthly dataset before
`xr.concat`. All datasets then have `time` as their 1D dim; concat aligns correctly.
Verified: 2022 merge produces exactly 8760 timesteps with clean coords.

---

## Decisions Made

### ERA5 feature fill strategy: zero for missing values

ERA5 is a complete reanalysis product — NaN values should not occur in practice.
The loader uses `fillna(0.0)` for ERA5 columns (same as cyclic features), with a
warning emitted if any ERA5 column is absent from `dataset.parquet`. This makes
the loader forward-compatible: if ERA5 is not yet merged, the model gets zeros
and continues to train (degraded, but not crashed).

### `wind_mode="from_field"` deferred to Session 4

The graph builder already has a `from_field` mode wired, which reads `u10`/`v10`
from the ERA5 DataArray to build dynamic wind-aware Type B edges. Session 3 added
ERA5 data and features but did not activate `from_field` mode in the loader or
graph builder — this is Session 4 work (requires deciding per-sample vs. per-batch
wind field lookup and the performance trade-off of dynamic edge recomputation).

---

## Checkpoints

### Checkpoint: Training smoke test (3 epochs, post-fix)

Model: MTGNNModel, n_features=10, n_stations=18, device=CUDA

| Epoch | train_loss | val_rmse_24h |
|---|---|---|
| 1 | 3.164 | 1.54 |
| 2 | 1.802 | 1.54 |
| 3 | 1.784 | 1.54 |

Loss declining cleanly. Val RMSE higher than pre-ERA5 smoke test (~0.62) because
the model now has 10 features and wider weight matrices — needs more epochs to
converge. No NaN, no crash. CUDA confirmed.

Pre-ERA5 3-epoch smoke test (Session 3 early, 5 features):
- Loss: 0.646 -> 0.376 -> 0.354; val_rmse_24h: 0.64 -> 0.62 -> 0.62

### Checkpoint: ERA5 data completeness

| Year | Timesteps | Stations | Rows |
|---|---|---|---|
| 2022 | 8760 | 18 | 157,680 |
| 2023 | 8760 | 18 | 157,680 |
| 2024 | 8784 (leap) | 18 | 158,112 |
| 2025 | 8760 | 18 | 157,680 |
| **Total** | **35,064** | **18** | **631,152** |

NaN cells after merge: 0. Columns: `u10, v10, t2m, d2m, blh`.

### Checkpoint: dataset.parquet final schema

| Column | Type | Notes |
|---|---|---|
| `timestamp` | datetime64[us, UTC] | Hourly |
| `station_id` | int64 | 18 stations |
| `pm25_raw` | float32 | ug/m3, NaN where missing |
| `pm25_scaled` | float32 | RobustScaler normalized |
| `mask_in_loss` | bool | True = exclude from loss |
| `exclude_from_training` | bool | True = exclude anchor |
| `hour_sin`, `hour_cos` | float32 | Cyclic hour encoding |
| `doy_sin`, `doy_cos` | float32 | Cyclic day-of-year encoding |
| `u10`, `v10` | float32 | ERA5 wind components (m/s) |
| `t2m`, `d2m` | float32 | ERA5 temperature, dewpoint (K) |
| `blh` | float32 | ERA5 boundary layer height (m) |

---

## Deviations from Plan

| Planned | Actual | Rationale |
|---|---|---|
| ERA5 as single yearly request | Monthly chunking (12 req/year) | CDS API v2 rejects large requests |
| `xr.open_dataset(path)` for merge | Temp-file roundtrip | Thai-character path crashes netCDF4 C library |
| `wind_mode="from_field"` in Session 3 | Deferred to Session 4 | Path/performance trade-offs need design |
| Full 100-epoch production training | Smoke-tested only (3 epochs) | Full training is Session 4 after wind wiring |

---

## Open Questions

All closed for Session 3.

---

## Recommendations for Session 4

1. **`wind_mode="from_field"`**: Wire ERA5 `u10`/`v10` into `build_type_b_edges`.
   Decision needed: compute wind field once per timestep from `_u10`/`_v10` arrays
   already in the loader, or re-read from ERA5 parquet. The former is cheaper.

2. **Full production training**: Run `scripts/03_train.py model=mtgnn` for 100 epochs.
   Expected ~2-3 h on GPU. Log to W&B. Save best checkpoint for explainability.

3. **`src/explain/`**: Implement GB-IG (Graph-based Integrated Gradients).
   - `gnn_explainer.py`: wrapper around PyG's GNNExplainer for edge attribution
   - `gb_ig.py`: GB-IG path integral along graph feature space
   - `attribution.py`: aggregate attributions to source-country level
   Reference: GB-IG paper (2025) https://arxiv.org/abs/2509.07648

4. **`src/viz/`**: `maps.py` (station map + attribution overlay) and
   `timeseries.py` (forecast vs. actual per station).

5. **`app/streamlit_app.py`**: Live forecast dashboard + source attribution view.
   NSC Disclaimer footer must appear (booklet page 44).

6. **Station 1304386 exclusion decision** (carried from Session 2):
   20.9% coverage in 2023. Re-evaluate once full training results are available —
   poor coverage may degrade model if not excluded via `exclude_stations`.
