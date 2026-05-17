# SESSION1 NOTES — Data Pipeline Foundation

Date: 2026-05-18
Branch: `feat/session-1-data-pipeline`

---

## What Was Built

| Deliverable | File | Status |
|---|---|---|
| Repo scaffolding | `src/**/__init__.py`, `data/**/.gitkeep` | Done |
| OpenAQ v3 scraper | `src/data/scrapers/openaq.py` | Working |
| FIRMS hotspot scraper | `src/data/scrapers/firms.py` | Working |
| ERA5 downloader | `src/data/scrapers/era5.py` | Stub |
| Loader orchestration | `src/data/loader.py` | Working |
| Offline unit tests | `tests/test_openaq.py`, `tests/test_firms.py`, `tests/test_era5.py` | 25/25 green |
| CLI download script | `scripts/01_download_all.py` | Working |
| README scaffolding | `README.md` | Done (bilingual, TODO disclaimer) |
| NSC disclaimer headers | All `src/**/*.py` | Placeholder added |

---

## API Findings (Surprises During Real Runs)

### OpenAQ v3 — Field Name Mismatches vs. Spec

The spec described two fields that turned out to be wrong:

1. **`providers` (plural array)** — The actual response uses `provider` (singular object):
   - Wrong: `loc.get("providers", [])[0]["name"]`
   - Correct: `loc.get("provider", {}).get("name", "")`

2. **`datetimes.first` / `datetimes.last`** — The actual response uses camelCase nested objects:
   - Wrong: `loc.get("datetimes", {}).get("first")`
   - Correct: `loc.get("datetimeFirst", {}).get("utc")`

Both mismatches were caught by the live `discover` run returning 0 Air4Thai stations.

### OpenAQ 408 Timeouts on Monthly/Annual Ranges

The `/measurements` endpoint consistently returns `408 Request Timeout` for date ranges
larger than ~2 weeks. The spec suggested downloading a full year at once, which does not work.

**Fix:** `backfill_station` now fetches in 7-day chunks with a 1-second polite pause between
requests. This is consistent with the verified 8-day test range in the prompt.

Retry logic updated to include 408 alongside 429 and 5xx. Wait: 2s/4s/8s exponential.

### OpenAQ Station Count

- Total PM2.5 locations in bbox: **173**
- After `curate_training_stations` (Air4Thai, first < 2022-01-01, last > 2026-01-01): **18 stations**

This is above the "15 core + up to 10 extended" target in CLAUDE.md.

### Backfill — Data Gaps

Station 225579 (Yupparaj Wittayalai School, Chiang Mai), 2025:
- Total rows: **5,619** across 52 weekly chunks
- Notable gaps: several weeks in July–November 2025 returned 0 rows (sensor offline or data not yet ingested by OpenAQ)
- This is expected for historical data; gaps should be handled during preprocessing (interpolation or masking)

### FIRMS

FIRMS API works reliably. For 2026-04-01 to 2026-04-05 (5 days, Northern Thailand bbox):
- **19,798 hotspot rows** returned from `VIIRS_NOAA20_NRT` source

### Python Encoding Issue (Windows Thai Path)

The repository path `D:\งาน\NSC\pm25-stgnn-thailand` contains Thai characters.
Python's `site.py` reads the editable install `.pth` file using the system locale (cp874),
but the file is stored as UTF-8. This causes:

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x87
```

**Temporary fix applied:** Rewrote `.venv/Lib/site-packages/_editable_impl_pm25_stgnn_thailand.pth`
in cp874 encoding using `uvx python`.

**Important:** This fix is lost whenever `uv sync` is re-run, because uv regenerates the file
in UTF-8. After every `uv sync`, re-run the encoding fix:

```bash
PYTHONUTF8=1 uvx python -c "
pth = r'D:\งาน\NSC\pm25-stgnn-thailand\.venv\Lib\site-packages\_editable_impl_pm25_stgnn_thailand.pth'
content = r'D:\งาน\NSC\pm25-stgnn-thailand'
with open(pth, 'wb') as f:
    f.write(content.encode('cp874') + b'\n')
"
```

A permanent fix (adding a `sitecustomize.py` or moving to a non-Thai path) is recommended
before Session 2.

---

## Deviations from Plan

| Planned | Actual | Rationale |
|---|---|---|
| Monthly chunking in `backfill_station` | 7-day chunking | OpenAQ 408s on monthly ranges |
| Single checkpoint B shape report | Combined with checkpoint C (CLI demo) | Encoding issue prevented early Python runs |
| Annual `fetch_measurements` calls | 7-day window loop | API constraint, not API docs constraint |

---

## State of Each Scraper

| Scraper | State | Notes |
|---|---|---|
| `openaq.py` | Working | Verified on 18 stations; 5,619 rows for station 225579 (2025) |
| `firms.py` | Working | 19,798 rows for 5-day window, Northern Thailand |
| `era5.py` | Stub | `raise NotImplementedError` — needs CDS profile setup |
| `loader.py` | Working | 18 stations cached to `data/processed/stations_metadata.parquet` |

---

## Open Questions for User

1. **ERA5 CDS credentials** — Do you have a CDS API key? ERA5 implementation is blocked until
   `CDS_API_URL` and `CDS_API_KEY` are set in `.env`. If not, we can defer ERA5 fully to Session 3
   and use only OpenAQ + FIRMS features for initial model experiments.

2. **Data gaps** — Station 225579 has multi-week gaps in mid-2025. Should preprocessing fill gaps
   by linear interpolation, forward-fill, or mask them as missing? This affects the training
   window design.

3. **Python path encoding** — The `.pth` file fix needs to be re-applied after every `uv sync`.
   Would it be acceptable to clone the repo to a path without Thai characters (e.g., `C:\projects\pm25-stgnn`)
   for development? Or should we add a `sitecustomize.py` permanent fix?

4. **All 18 stations vs. 15 core** — CLAUDE.md says "15 core + up to 10 extended". Should we
   drop 3 stations to hit exactly 15, or use all 18 as-is?

---

## Recommendations for Session 2

Session 2 should build the preprocessing pipeline (`src/data/preprocessing.py`):

1. **Load and merge** parquets from `data/raw/openaq/` for all 18 stations
2. **Resample to hourly** — raw data is sub-hourly at some stations; aggregate to 1-hour mean
3. **Handle missing data** — choose a strategy (linear interpolation / forward-fill / mask)
4. **Align FIRMS hotspots** to station grid — aggregate hotspot counts/FRP within bbox per time step
5. **Build the station graph** — draft `src/data/graph_builder.py` with static adjacency (distance-based);
   wind-aware dynamic adjacency can follow in Session 3
6. **Write interim parquets** to `data/interim/` — one file per station, hourly, cleaned
7. **Add `test_preprocessing.py`** with offline unit tests

Before Session 2:
- Complete the backfill for all 18 stations (run `uv run python scripts/01_download_all.py backfill --start-year 2022 --end-year 2025` — this will take several hours)
- Resolve the Python encoding issue (see Open Questions #3)
- Decide on ERA5 approach (Open Questions #1)
