# SESSION 2 — Handoff Prompt (Claude Code plans)

Copy everything below the line into a NEW Claude Code session (`claude`
in the repo root). This prompt asks Claude Code to plan Session 2 itself
based on context already in the repo, rather than executing a pre-written
detailed plan.

---

We're starting Session 2 of the PM2.5 STGNN Thailand project (NSC 2026).
Session 1 is merged to main (tagged `session-1-complete`). The data
pipeline foundation is in place.

## Your task

PLAN Session 2 yourself. Read the repo, understand state, propose the
plan, wait for my approval. I'll review and adjust, then we execute.

## Setup before planning

1. Read these files in full, in order:
   - `CLAUDE.md` (project memory; conventions; updated scope: 18 stations)
   - `docs/DESIGN.md` (architectural decisions, especially sections 4.3,
     5, 6, 11)
   - `docs/SESSION1_NOTES.md` (what was built, API findings, deviations)
   - `pyproject.toml` (locked deps)
   - `install_native_deps.ps1` (native deps install workflow)
   - `README.md` (quickstart)

2. Inspect repo state:
   - `git log --oneline main -20` — see Session 1 commits
   - `git tag` — confirm `session-1-complete` exists
   - `ls src/data/scrapers/` — confirm openaq.py, firms.py, era5.py stub
   - `ls data/processed/` — confirm `stations_metadata.parquet` (18 rows)
   - `ls data/raw/` — see what's already downloaded
   - `uv run pytest tests/ -v` — confirm 25/25 still green

3. Verify the subagents are loaded (`/agents`): architect, implementer,
   tester, reviewer. If missing, STOP and tell me.

4. Create + switch to feature branch:
   `git checkout -b feat/session-2-preprocessing-graph`

## Session 2 scope

Build the graph construction foundation. Three deliverables:

1. **Preprocessing pipeline** (`src/data/preprocessing.py`)
   - Apply the missing-data policy from DESIGN.md section 4.3 exactly:
     <6h interpolate, 6-24h forward-fill capped, >24h mask in loss,
     >7 days exclude that month from training
   - Resample to strict hourly grid (UTC)
   - Per-station normalization (z-score or robust scaler, your call —
     document the choice)
   - Cyclic encoding (hour, dayofyear) per DESIGN.md schema
   - Output: `data/processed/dataset.parquet` matching DESIGN.md 4.2 schema
     (minus weather fields — those come in Session 3 when ERA5 arrives)

2. **Hotspot clustering** (`src/data/hotspot_clustering.py`)
   - Read FIRMS CSVs from `data/raw/firms/`
   - DBSCAN cluster within 25 km, daily aggregation
   - Country flag per cluster (Thailand / Myanmar / Laos / other) — use
     a simple bbox-based geocoder or shapely with country polygons.
     Document your choice.
   - Output: `data/processed/hotspots.parquet` keyed by
     (cluster_id, date) with centroid, total FRP, country

3. **Graph builder** (`src/data/graph_builder.py`) — THE CRITICAL FILE
   - Implements the three edge types from DESIGN.md section 5.2:
     - Type A: static spatial (distance ≤100km, weight = exp(-d/50))
     - Type B: wind-aware dynamic (distance ≤200km, alignment >0.3,
       weight = alignment × wind_speed × exp(-d/100))
     - Type C: hotspot influence (distance ≤500km, alignment >0.4,
       weight = alignment × FRP × exp(-d/200))
   - Signature locked per DESIGN.md section 5.3:
     ```python
     def build_graph(df_stations, df_hotspots, wind_field, config)
         -> torch_geometric.data.HeteroData
     ```
   - **Wind data note**: ERA5 is deferred to Session 3. Design the API
     to accept `wind_field: xr.DataArray` as a parameter — for Session 2,
     test using SYNTHETIC wind data (constant wind from NE, or random
     per-timestep). Session 3 plugs in real ERA5.
   - Document this synthetic-wind-for-now decision in
     `docs/SESSION2_NOTES.md`.

## Pre-flight (run AS PART of Session 2, not before)

Before any new code, run full backfill:

```
uv run python scripts/01_download_all.py backfill --start-year 2022 --end-year 2025
uv run python scripts/01_download_all.py firms --start-date 2022-01-01 --end-date 2025-12-31
```

The backfill will take 30-90 minutes (18 stations × 4 years). Run it
early in the session, in parallel with planning the preprocessing module.

⚠️ FIRMS NRT (Near-Real-Time) endpoints likely don't cover 2022-2024.
For historical hotspots you may need the FIRMS Archive endpoint instead.
If you find this is the case:
- STOP and tell me before proceeding
- Don't silently switch endpoints without flagging
- Document the finding in SESSION2_NOTES.md

## Subagent routing (suggested, you decide)

| Work | Agent | Rationale |
|---|---|---|
| Pre-flight backfill | (no agent — run script) | mechanical |
| Preprocessing module design | architect | gap policy + edge cases |
| Preprocessing implementation | implementer | from architect spec |
| Hotspot clustering | implementer | DBSCAN + geo logic |
| Graph builder design | architect | CRITICAL — wind math + edge logic |
| Graph builder implementation | implementer (or architect) | depends on complexity |
| All unit tests | tester | mechanical |
| EDA notebooks | implementer | exploration only |
| Pre-commit review | reviewer | DESIGN.md adherence + security |

## EDA notebooks (required)

Two notebooks in `notebooks/`:

1. `01_eda_dataset.ipynb`
   - Station coverage map (Folium): 18 stations on Northern Thailand map
   - PM2.5 distribution per station (histograms + boxplots)
   - Time series for 4 burning seasons overlaid (one figure per station)
   - Missing-data heatmap (stations × time, colored by gap length)
   - Province-level summary table
   - Language: English titles/code; Thai narrative in markdown cells.
     Goal: figures must be copy-paste-able into proposal/final report.

2. `02_graph_construction_viz.ipynb`
   - Visualize a single timestep's graph (one figure per edge type +
     one combined figure)
   - Demo wind-aware adjacency under different wind directions
     (test with synthetic wind from N, S, E, W)
   - Hotspot density heatmap for one burning season
   - Sanity check: edge weights distribution, node degree distribution

## Checkpoints — STOP and report

- **Checkpoint A** — after pre-flight backfill: report rows-per-station
  coverage table (stations × years × % complete). Flag any station with
  <50% coverage.
- **Checkpoint B** — after preprocessing: paste the schema and row count
  of `data/processed/dataset.parquet`. Confirm gap policy applied
  correctly.
- **Checkpoint C** — after graph builder design (architect plan, before
  implementation): paste the proposed module interface, internal
  functions, and 3-5 critical unit tests. I will review the math before
  implementation.
- **Checkpoint D** — after graph builder implementation + tests: paste
  graph statistics (avg nodes, avg edges per type, avg degree) and
  visualization screenshots from notebook 02.

## Definition of Done

- [ ] Branch `feat/session-2-preprocessing-graph`, never on main
- [ ] Pre-flight: 4 years of backfill data in `data/raw/openaq/`
- [ ] Pre-flight: FIRMS data in `data/raw/firms/` (history decision flagged)
- [ ] `src/data/preprocessing.py` — schema matches DESIGN.md 4.2
- [ ] `src/data/hotspot_clustering.py` — output keyed correctly
- [ ] `src/data/graph_builder.py` — three edge types, type-checked
- [ ] `tests/test_graph_builder.py` — at least 8 tests covering:
    - Static edge symmetry
    - Wind alignment math (known angles)
    - Hotspot edge under reversed wind
    - Empty hotspot day
    - Distance threshold boundaries (just inside / just outside)
- [ ] `tests/test_preprocessing.py` — gap policy correctness
- [ ] `notebooks/01_eda_dataset.ipynb` runs end-to-end
- [ ] `notebooks/02_graph_construction_viz.ipynb` runs end-to-end
- [ ] `uv run pytest tests/ -v` — all green
- [ ] `uv run ruff check src/` clean
- [ ] `docs/SESSION2_NOTES.md` written (same structure as session 1)
- [ ] Reviewer agent has approved the branch

## Hard constraints

- DO NOT push to main; stay on feature branch
- DO NOT write any model code, training, or XAI in this session
- DO NOT fetch ERA5 (deferred to Session 3)
- DO NOT add dependencies beyond pyproject.toml without asking
- DO NOT add torch-scatter, torch-sparse, or torch-geometric-temporal
  to pyproject.toml (breaks uv sync)
- DO NOT make real network calls in tests
- DO NOT commit anything under data/
- DO NOT use print() in src/
- Type hints (PEP 604) + Google docstrings everywhere
- Gap policy must match DESIGN.md section 4.3 exactly
- Graph builder must match DESIGN.md section 5 specifications
- Wind data is synthetic for now — real ERA5 in Session 3

## Workflow

1. PLAN MODE first.
2. Read context files (listed above).
3. Inspect repo state.
4. Start pre-flight backfill in a separate terminal (or as a background
   process) so it runs while you plan.
5. Propose a step-by-step plan with files, subagent assignment, and
   acceptance criteria per step.
6. WAIT for my approval before exiting plan mode.

After approval:

7. Execute step by step. Commit on feature branch after each logical
   unit. Conventional Commits.
8. Stop at Checkpoints A, B, C, D and report.
9. At the end, invoke reviewer agent.
10. Write `docs/SESSION2_NOTES.md`.
11. Report: what works, what's blocked, recommendations for Session 3
    (which will be ERA5 + baseline models).
12. Do NOT merge to main — I will review and merge.

Begin now. Start by reading the context files and reporting your
understanding of where Session 1 left off, then propose the plan.
