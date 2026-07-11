# API Quirks

Hard-won integration notes for the external data sources. Referenced from
CLAUDE.md ("Data sources") and README.md. Keep this terse and factual — one
entry per gotcha, with the fix and where it lives in code.

## Native PyG dependencies (install order)

`torch-scatter`, `torch-sparse`, and `torch-geometric-temporal` have native
C++/CUDA code that links against an already-installed torch. uv resolves all
deps before installing any, so these cannot live in `pyproject.toml`. Run
`uv sync` first, then `./install_native_deps.ps1` (Windows) or the manual
commands in `docs/INSTALL.md`. Wheels come from https://data.pyg.org/whl/.

## OpenAQ v3 (primary historical source)

- Use `/v3/sensors/{id}/measurements` for historical data, **not** `/hours`.
- Datetime params must be full ISO 8601 **with timezone**
  (e.g. `2024-03-01T00:00:00Z`), not bare dates.
- Responses may arrive UTF-16 with a BOM. Decode with
  `response.encoding = "utf-8-sig"` before parsing.

## NASA FIRMS

- `day_range` is capped at 10 per request; page longer spans in ≤10-day chunks.
- Bounding box is given in **lon,lat** order.

## air4thai

Two public endpoints, no API key. Canonical URLs (per CLAUDE.md) are plain
`http://`:

- Realtime: `http://air4thai.pcd.go.th/services/getNewAQI_JSON.php`
- History: `http://air4thai.com/forweb/getHistoryData.php`

Quirks observed (2026-07-10, building the live dashboard mode):

- **Header:** both endpoints need `User-Agent: Mozilla/5.0` or they reject the
  request.
- **HTTPS redirect + broken TLS chain:** the `http://` URLs 301-redirect to
  `https://`, and the served certificate chain often fails local verification
  (`SSLCertVerificationError: unable to get local issuer certificate`) on
  Windows. Policy: try `verify=True` first and fall back to `verify=False`
  **only** on `requests.exceptions.SSLError`, logging a warning. These are
  read-only public endpoints and no credentials are sent, so the fallback is no
  worse than the documented plain-`http://` baseline. Implemented in
  `app/lib/air4thai._get_json` (and in the one-time builder
  `scripts/14_air4thai_station_map.py`, which also exposes an `--insecure` flag).
- **History window:** ~90-day rolling window only. Fine for filling a live 24h
  input window; **not** usable for ML training (use OpenAQ for that).
- **History timezone:** `DATETIMEDATA` is Asia/Bangkok local time. Convert to
  tz-aware UTC to match the model's hourly grid. Sentinel/negative PM2.5 values
  (e.g. `-1`) mean "no reading" — coerce to NaN.
- **Response schema:** history returns `{result, error, stations:[{stationID,
  params, data:[{DATETIMEDATA, PM25}], summary}]}`; realtime returns
  `{stations:[{stationID, nameEN, areaEN, lat, long, AQILast{PM25{value}}, ...}]}`.
- **Station-code mapping:** our stations are keyed by OpenAQ `location_id`;
  air4thai keys on its own `stationID` (e.g. `36t`). No column links them, so
  `scripts/14_air4thai_station_map.py` recovers the map by nearest-coordinate +
  name cross-check into `configs/air4thai_station_codes.json`.
- **Mae Chaem missing:** our `location_id` 225693 ("Debaratana Vejjanukul
  Hospital, Mae Chaem") is **absent from the air4thai realtime/history feed** as
  of 2026-07-10 (nearest live station is ~70 km away). It has no code entry and
  is center-filled (graph input only) and hidden from the live-mode selector.

## Open-Meteo (live NWP)

- Free, no API key, non-commercial. `boundary_layer_height` is available on the
  default forecast endpoint (confirmed 2026-07-10) alongside `temperature_2m`,
  `dew_point_2m`, `wind_speed_10m`, `wind_direction_10m`.
- The public scraper (`src/data/scrapers/openmeteo.py`) cannot pass `past_days`;
  the dashboard's `app/lib/nwp.fetch_forecast_window` adds it (max 92) while
  reusing `openmeteo._parse_hourly`, so one call covers the input window + horizon.
- Wind is reported as speed + meteorological direction (from-north); convert to
  ERA5-style `u10`/`v10` eastward/northward components.

## ERA5 via cdsapi

See `feedback_era5_quirks.md` (auto-memory) and `docs/INSTALL.md` for CDS API v2
setup. ERA5 reanalysis lags several days, which is why live mode uses Open-Meteo
NWP instead (with the honesty caveat surfaced in the dashboard).
