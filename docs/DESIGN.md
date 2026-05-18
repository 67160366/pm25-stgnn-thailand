<!-- v2.1 — 2026-05-18: Add §5.2 Graph Edge Types, §5.3 build_graph API, §6 Feature Schema -->

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

## 5. Graph Construction

### 5.1 Decision Notes

- Prefer explicit agent roles: architect, implementer, tester, reviewer.
- Store project memory in `CLAUDE.md`.
- Use a modern Python packaging config via `pyproject.toml`.
- Keep environment examples out of source control via `.env.example`.

### 5.2 Graph Edge Types

The graph is heterogeneous with two node types (station, hotspot) and three
directed/undirected edge types per timestep:

| Type | Nodes | Distance cap | Condition | Weight formula |
|---|---|---|---|---|
| A — static spatial | station ↔ station | ≤ 100 km | — | exp(−d / 50) |
| B — wind-aware dynamic | station → station | ≤ 200 km | alignment > 0.3 | alignment × wind_speed × exp(−d / 100) |
| C — hotspot influence | hotspot → station | ≤ 500 km | alignment > 0.4 | alignment × FRP × exp(−d / 200) |

**Wind alignment** = cos(θ), where θ is the angle between the wind vector (u, v) and
the bearing from source node to target node. A value of 1 = perfectly downwind; 0 = perpendicular; −1 = upwind.

**Distance** d is great-circle distance in km (haversine).

Type A edges are symmetric (undirected). Types B and C are directed.

### 5.3 `build_graph` — Locked API

```python
def build_graph(
    df_stations: pd.DataFrame,   # 18-row metadata (lat, lon, station_id)
    df_hotspots: pd.DataFrame,   # daily hotspot clusters (centroid_lat, centroid_lon, total_frp, date)
    wind_field: xr.DataArray,    # shape (time, lat, lon, 2), components u and v
    config: dict,                # thresholds, wind_mode, etc.
) -> torch_geometric.data.HeteroData:
    ...
```

`wind_field` is an `xr.DataArray` shaped `(time, lat, lon, 2)` (u and v components).
For Session 2 (no ERA5 yet), pass a synthetic DataArray with constant or random wind.
Session 3 plugs in real ERA5 u10/v10.

The returned `HeteroData` contains:
- `data['station'].x` — node feature matrix (pm25_scaled, hour_sin, hour_cos, doy_sin, doy_cos)
- `data['hotspot'].x` — node feature matrix (total_frp, centroid_lat, centroid_lon)
- `data['station', 'type_a', 'station'].edge_index` and `.edge_attr`
- `data['station', 'type_b', 'station'].edge_index` and `.edge_attr`
- `data['hotspot', 'type_c', 'station'].edge_index` and `.edge_attr`

## 6. Feature Schema

### 6.1 Station Node Features (per timestep)

| Column | Type | Description |
|---|---|---|
| pm25_scaled | float32 | RobustScaler-normalised PM2.5 |
| hour_sin | float32 | sin(2π × hour / 24) |
| hour_cos | float32 | cos(2π × hour / 24) |
| doy_sin | float32 | sin(2π × dayofyear / 365) |
| doy_cos | float32 | cos(2π × dayofyear / 365) |

Weather features (u10, v10, t2m, rh) are NULL until ERA5 arrives in Session 3.

### 6.2 Hotspot Node Features (per timestep)

| Column | Type | Description |
|---|---|---|
| total_frp | float32 | Sum of FRP for the cluster (MW) |
| centroid_lat | float32 | Cluster centroid latitude |
| centroid_lon | float32 | Cluster centroid longitude |

### 6.3 Edge Attributes

| Edge type | Attribute | Description |
|---|---|---|
| Type A | weight | exp(−d / 50) |
| Type B | weight, alignment, wind_speed | scalar weight + components |
| Type C | weight, alignment, frp | scalar weight + components |

## 11. Deferred Items

| Item | Deferred to | Reason |
|---|---|---|
| ERA5 reanalysis data (wind, temperature, humidity) | Session 3 | Requires CDS API credentials; not blocking Sessions 1–2. ERA5 features are additive — the model can train without them. Implementation stub exists at `src/data/scrapers/era5.py`. |
