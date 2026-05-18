# DESIGN.md for PM2.5 STGNN Thailand

This document is the current architecture and design reference for the project.

## 1. Goals

- Build a spatial-temporal graph neural network for PM2.5 forecasting.
- Support dataset ingestion from Air4Thai and OpenAQ.
- Keep model and preprocessing code modular and testable.
- Separate architecture decisions from implementation details.

## 2. Repository Structure

- `src/` contains the implementation code.
- `docs/` contains design, session notes, and architecture rationale.
- `.claude/` contains agent definitions and Claude settings.

## 3. Station Selection

18 Air4Thai stations were selected from the OpenAQ v3 `/locations` endpoint
using the following criteria:

- `provider.name == "Air4Thai"`
- Bounding box: lon 97.0–101.5, lat 16.0–21.0 (9 Northern Thai provinces)
- `datetimeFirst.utc < 2022-01-01` — data predates training window start
- `datetimeLast.utc > 2026-01-01` — data extends past end of test window
- Parameter: PM2.5

Result: **18 stations** (15 designated core + 3 extended), covering all 9 target provinces.
Discovery run verified 2026-05-17 via `scripts/01_download_all.py discover`.

## 4. Data Sources

### 4.1 OpenAQ v3

Primary historical source. Use `/v3/sensors/{id}/measurements` (not `/hours`).
Datetime params must be full ISO 8601 with timezone. Chunked in 7-day windows
to avoid 408 timeouts. See `docs/api_quirks.md` for full field-name notes.

### 4.2 NASA FIRMS

Fire hotspot source. `day_range` capped at 10 per request. Bbox in lon,lat order.
Source: `VIIRS_NOAA20_NRT`.

### 4.3 Missing Data Policy

Applied during preprocessing (`src/data/preprocessing.py`):

| Gap length | Treatment |
|---|---|
| < 6 hours | Linear interpolation |
| 6 – 24 hours | Forward-fill, capped at gap length |
| > 24 hours | Mask in loss function (do not impute) |
| > 7 consecutive days | Exclude entire month from training |

### 4.4 ERA5

Deferred to Session 3. See section 11.

## 5. Decision Notes

- Prefer explicit agent roles: architect, implementer, tester, reviewer.
- Store project memory in `CLAUDE.md`.
- Use a modern Python packaging config via `pyproject.toml`.
- Keep environment examples out of source control via `.env.example`.

## 11. Deferred Items

| Item | Deferred to | Reason |
|---|---|---|
| ERA5 reanalysis data (wind, temperature, humidity) | Session 3 | Requires CDS API credentials; not blocking Sessions 1–2. ERA5 features are additive — the model can train without them. Implementation stub exists at `src/data/scrapers/era5.py`. |
