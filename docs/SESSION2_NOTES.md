# SESSION 2 NOTES — Preprocessing & Graph Construction

Date: 2026-05-19
Branch: `feat/session-2-preprocessing-graph`

---

## What Was Built

| Deliverable | File | Status |
|---|---|---|
| DESIGN.md graph spec | `docs/DESIGN.md` (v2.1) | Done — §5.2, §5.3, §6 added |
| FIRMS SP backfill | `src/data/scrapers/firms.py` | Done — SP+NRT hybrid |
| FIRMS SP fix (5-day cap) | `src/data/scrapers/firms.py` | Fixed — SP capped at 5, not 10 |
| Preprocessing pipeline | `src/data/preprocessing.py` | Done |
| Hotspot clustering | `src/data/hotspot_clustering.py` | Done |
| Graph builder | `src/data/graph_builder.py` | Done |
| Preprocessing tests | `tests/test_preprocessing.py` | 18 tests, all green |
| Graph builder tests | `tests/test_graph_builder.py` | 16 tests (+ 2 integration), all green |
| FIRMS tests updated | `tests/test_firms.py` | 23 tests, all green |
| Total test suite | `tests/` | 70/70 green |

---

## API Findings and Surprises

### FIRMS SP Product: day_range Capped at 5 (not 10)

The implementer assumed `VIIRS_NOAA20_SP` (Standard Product) accepts the same
`day_range=[1..10]` as NRT products. Observed: SP returns HTTP 400 for `day_range>5`.

- **NRT endpoint**: `day_range` [1..10] per request.
- **SP endpoint**: `day_range` [1..5] per request.

Fixed in `firms.py` via `_SP_WINDOW_DAYS=5` constant. The 4-year backfill required
~292 separate HTTP requests (4 years / 5 days per chunk).

### FIRMS SP Processing Lag: ~48–60 Days

Probed 2026-05-18:
- Last confirmed SP data: 2026-03-31 (48 days behind today)
- Conservative cutoff: `today - 60 days`
- **Known gap**: 2026-04-01 to 2026-05-08 (~38 days) has no coverage from either
  SP or NRT. This is a NASA pipeline constraint logged at WARNING level.

For ML training (2022-2025), pure SP covers the entire window.

### FIRMS SP vs NRT Scope

`fetch_hotspots_hybrid()` was implemented:
- SP segment: `[start_date, today - 60 days]`
- NRT segment: last 10 days
- Gap warning emitted if window falls between the two

### 2023 OpenAQ Coverage Anomaly

All 18 stations show ~40-43% hourly coverage in 2023 (vs 80-92% in 2022/2024).
Root cause: extended sensor outages at Air4Thai stations (verified by interval
analysis — gaps up to 745 h = 31 days). This is a real data quality issue, not
a resolution change. The preprocessing gap policy handles it:

- >24h gaps: `mask_in_loss=True` (17.6% of dataset rows)
- Months with >7-day gaps: `exclude_from_training=True` (25.2% of rows, mostly 2023)

Station 1304386 (Debaratana, Mae Hong Son) is consistently the worst:
20.9% in 2023, below 50% in all four years. Flag for potential exclusion during
model training.

### CLI Arrow Character Encoding Bug

The `scripts/01_download_all.py` FIRMS completion message contained `→` (U+2192),
which cannot be encoded in Windows cp874 console encoding. Fixed by replacing
with `->` (ASCII hyphen-arrow).

---

## Decisions Made

### Normalization: RobustScaler (median + IQR)

Chosen over z-score because PM2.5 has heavy outliers during burning season (>500 ug/m3).
RobustScaler is less sensitive to these spikes and produces better-scaled features
for the rainy season baseline.

Scaler fitted on 2022-2023 training data only. Parameters saved to
`data/processed/scalers.json` for inference use.

### Country Geocoder: Simple Bbox Lookup

Chose simple bbox geocoder (Thailand > Myanmar > Laos > other) over shapely + country
polygons for two reasons:
1. `shapely` is not in `pyproject.toml` (adding it requires user approval per CLAUDE.md)
2. For source attribution at the model level, exact border classification is less
   critical than the aggregate FRP signal

Bboxes verified against Natural Earth admin boundaries (corners checked on reference map).
Border regions (e.g., Golden Triangle area, Mae Hong Son – Myanmar border) may be
misclassified. Country distribution from 294-file FIRMS dataset: Thailand 86.8%,
Myanmar 12.7%, Laos 0.5% — directionally correct.

### Synthetic Wind for Session 2

`wind_mode="constant_ne"` used as default: u=3.5, v=3.5 m/s (NE direction, ~5 m/s).
Three modes implemented:
- `constant_ne`: constant synthetic NE wind
- `random`: uniform random u,v in [-5, 5] m/s per location
- `from_field`: real ERA5 DataArray (Session 3)

### Equirectangular Bearing Approximation

Used `bearing = atan2(dlon * cos(src_lat), dlat)` (equirectangular, not great-circle).
At bbox extremes (~500 km, lat 16-21 deg), deviation < 1.5 deg = < 0.02 alignment error.
Verified adequate for our domain; true great-circle bearing deferred to Session 3 review.

---

## Checkpoint A — OpenAQ Coverage Summary

72 parquet files (18 stations × 4 years). Key findings:

- **2023 systemic gap**: All stations at 38-43% coverage. Root cause: Air4Thai sensor outages.
- **Station 1304386 (Debaratana, Mae Hong Son)**: Below 50% in ALL years (worst: 20.9% in 2023).
  Recommend flagging for exclusion threshold review in Session 3.

---

## Checkpoint B — Preprocessing Output

File: `data/processed/dataset.parquet`

| Metric | Value |
|---|---|
| Rows | 631,152 |
| Stations | 18 |
| Date range | 2022-01-01 00:00 UTC → 2025-12-31 23:00 UTC |
| `mask_in_loss=True` | 17.6% |
| `exclude_from_training=True` | 25.2% |

Schema matches DESIGN.md §6.1 (ERA5 weather columns deferred to Session 3).

---

## Checkpoint D — Graph Statistics (2022-03-15, burning season)

| Metric | Value |
|---|---|
| Station nodes | 18 |
| Hotspot nodes | 39 |
| Type A edges | 74 (avg degree 4.1) |
| Type B edges | 71 (avg degree 3.9) |
| Type C edges | 303 (avg degree 16.8) |
| Type A weight range | 0.14 – 0.89 |
| Type B weight range | 0.26 – 4.00 |
| Inter-station distances | 5.9 – 465.5 km (median 177 km) |

---

## Deviations from Plan

| Planned | Actual | Rationale |
|---|---|---|
| FIRMS SP day_range=10 | day_range=5 | API returns 400 for SP with day_range>5 |
| Architect agent for preprocessing design | Implemented directly | Architect hit usage limit; design was clear from plan |
| EDA notebooks | Completed in Session 2 (commits c34facc, 77c13ff) | Initially deferred; completed after user flagged as DoD requirement |

---

## Open Questions

All closed for Session 2.

---

## Recommendations for Session 3

1. **ERA5 wind integration**: Plug `wind_mode="from_field"` with real CDS API u10/v10.
   The `_get_wind_uv()` function is already wired for this.

2. **Station 1304386 exclusion decision**: Review 20.9% 2023 coverage. If excluded,
   update `data/processed/stations_metadata.parquet` and reduce graph to 17 stations.

3. **EDA notebooks**: Build `notebooks/01_eda_dataset.ipynb` and
   `notebooks/02_graph_construction_viz.ipynb` now that all processed data exists.

4. **Dataset loader** (`src/data/loader.py`): Implement the temporal sliding window
   dataset that stacks `build_graph()` calls per timestep.

5. **Baseline models**: A3TGCN → MTGNN, using the `HeteroData` objects from graph_builder.

6. **Country bbox accuracy**: Consider adding shapely + Natural Earth polygons to replace
   the bbox geocoder if source attribution results look geographically implausible.

7. **FIRMS gap (2026-04-01 to 2026-05-08)**: ~38 days with no SP or NRT coverage.
   For real-time demo, this window will have no hotspot signal. Log at WARNING.
