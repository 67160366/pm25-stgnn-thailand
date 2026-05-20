# Session 4 Notes — Explainability, Viz, and Streamlit App

**Date:** 2026-05-20
**Branch:** `feat/session-4-explain-app`
**Goal:** Build the explanation and presentation layer (GB-IG, Streamlit dashboard)

---

## What was built

### Explain module (`src/explain/`)

| File | Description |
|---|---|
| `gb_ig.py` | `integrated_gradients()` — IG path integral over station features (completeness axiom). `occlusion_country_attribution()` — masks hotspot FRP by country, returns normalized [0,1] scores. |
| `gnn_explainer.py` | `gradient_x_input()` — fast single-pass Gradient × Input saliency (no completeness). |
| `attribution.py` | `load_hotspot_countries()`, `station_source_report()` (runs IG + occlusion, returns combined report), `batch_attribution()` (loops over sample list). |
| `__init__.py` | Public exports for all five attribution functions. |

### Viz module (`src/viz/`)

| File | Description |
|---|---|
| `maps.py` | `station_map()` — Plotly Scattermapbox with Thai PCD AQI colour scale. `attribution_bar_map()` — country scores as % bars. `feature_importance_bar()` — IG feature ranking. |
| `timeseries.py` | `forecast_vs_actual()` — line chart with AQI threshold bands. `multi_station_heatmap()` — all-station PM2.5 heatmap. `error_metrics_table()` — RMSE/MAE/R2 Plotly table. |

### Streamlit app (`app/streamlit_app.py`)

4 tabs:
1. **Live Forecast** — predicted PM2.5 at selected horizon, map + table
2. **Haze History** — multi-station heatmap + per-station time series
3. **Source Attribution** — IG feature importance + occlusion country attribution
4. **About** — project description + NSC Disclaimer placeholder

Usage: `uv run streamlit run app/streamlit_app.py`

### Training (`scripts/03_train.py`)

Production training launched in background:
- Model: MTGNN, 252K parameters
- Device: CPU (no CUDA on this machine)
- Epochs: 100, patience: 10
- Train: 15,110 samples, Val: 8,713 samples
- Wind mode: `from_field` (ERA5 pre-interpolated per station)
- Epoch 1 val RMSE: 1.54 (normalized scale) — ~3 min/epoch, ~5h total

---

## Key decisions

### IG vs exact GB-IG
Chose "practical IG-for-graphs" rather than exact 2025 GB-IG paper implementation.
- Standard IG path integral over station node features (`gb_ig.py`)
- Occlusion for country attribution (model-agnostic, interpretable for NSC)
- Fast Gradient×Input baseline for real-time Streamlit use

### n_ig_steps in Streamlit
Set to 30 in the dashboard for interactive speed. Paper-quality requires 100.

### Streamlit model loading
Uses `st.cache_resource` + Hydra compose for clean config loading.
Graceful degradation if `best_model.pt` not found (shows "train first" warning).

---

## Bugs found/fixed

- `gb_ig.py` line 93: originally had broken tensor-to-scalar comparison in debug log.
  Fixed by tester agent during test writing.
- `app/streamlit_app.py`: removed unused `OmegaConf` and `DataLoader` imports.

---

## Tests

- `tests/test_explain.py`: 44 tests covering all attribution functions.
  All use `_StubModel` (minimal PM25ModelBase) + synthetic HeteroData.
  Runtime: ~6s. Coverage: gb_ig 96%, gnn_explainer 92%, attribution 100%.
- Full suite: 207 tests passing.

---

## Commits

- `b0df19a` feat: implement GB-IG explain module + wire ERA5 wind-mode passthrough
- `076e790` feat: add viz module and Streamlit dashboard

---

## Next session (Session 5): Evaluation & polish

1. Wait for training to complete (~5h from session start)
2. Load best checkpoint, evaluate on test split
3. Generate attribution report for Chiang Mai station during peak haze event
4. Write SESSION5_NOTES + update DESIGN.md with results
5. Prepare NSC submission: fill in Disclaimer text, finalize README
