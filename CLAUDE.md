# PM2.5 STGNN Thailand — Project Memory

## Context

NSC 2026 entry (28th National Software Contest, Thailand).
Category 14 — โปรแกรมเพื่องานการพัฒนาด้านวิทยาศาสตร์และเทคโนโลยี
University level (ระดับนิสิต นักศึกษา).

Theme: "Sustainability Innovation"
Repo: https://github.com/67160366/pm25-stgnn-thailand

Deadlines:
- Proposal: 29 May 2026
- Final report: 17 July 2026
- Demo (final round): 21 Aug 2026

## What we're building

Explainable Spatio-Temporal Graph Neural Network for PM2.5 forecasting AND
source attribution in 9 Northern Thai provinces.

Read `docs/DESIGN.md` for full architectural decisions and rationale.

Three novelties:
1. Wind-aware dynamic adjacency (transboundary haze from Myanmar/Laos)
2. FIRMS hotspot clusters as first-class graph nodes
3. Source attribution via Graph-based Integrated Gradients (GB-IG)

## Scope (locked — do not expand)

- 9 Northern provinces: Chiang Mai, Chiang Rai, Lampang, Lamphun,
  Mae Hong Son, Nan, Phayao, Phrae, Tak
- Bounding box: lon 97.0–101.5, lat 16.0–21.0
- 15 core + up to 10 extended stations, all from `provider.name == "Air4Thai"`
- Time range: 2022-01-01 → 2025-12-31 (3 train seasons + val + test)
- Forecast horizons: 6h, 12h, 24h, 48h
- Hourly resolution

## Tech stack (locked)

- Python 3.11 (via uv)
- PyTorch 2.2+, PyG 2.5+, PyG Temporal
- Hydra (config), Weights & Biases (experiment tracking)
- Streamlit (demo)
- pytest + ruff + black (quality)

## Subagents available

- `architect` (Opus) — design, complex algos, cross-file debugging
- `implementer` (Sonnet) — code from clear spec
- `tester` (Haiku) — tests, lint, fixtures
- `reviewer` (Opus) — review before commits, security check

Default to PLAN MODE for any task touching >1 file. Wait for approval
before exiting plan mode.

## Coding conventions

- Type hints required, PEP 604 style (`list[int]`, `int | None`)
- Docstrings: Google style, terse, focus on intent
- No `print()` in `src/` — use `logging.getLogger(__name__)`
- Black line length: 100
- Ruff for linting; resolve all warnings before commit
- Specific exceptions, no bare `except:` or `except Exception:` without
  re-raise/log

## Repo structure (locked — do not refactor without approval)

```
pm25-stgnn-thailand/
├── .claude/
│   ├── settings.json
│   └── agents/
├── CLAUDE.md, README.md, pyproject.toml, .env.example, .gitignore
├── docs/
│   ├── DESIGN.md
│   ├── SESSION{N}_NOTES.md
│   └── api_quirks.md
├── data/
│   ├── raw/         # gitignored
│   ├── interim/     # gitignored
│   └── processed/   # gitignored
├── configs/                    # Hydra YAML
├── notebooks/                  # EDA only
├── src/
│   ├── data/
│   │   ├── scrapers/{openaq,air4thai,firms,era5}.py
│   │   ├── preprocessing.py
│   │   ├── graph_builder.py    # critical, has tests
│   │   └── loader.py
│   ├── models/{base,a3tgcn,pm25gnn,mtgnn}.py
│   ├── training/{trainer,losses,metrics}.py
│   ├── explain/{gnn_explainer,gb_ig,attribution}.py
│   └── viz/{maps,timeseries}.py
├── app/streamlit_app.py
├── scripts/01_*.py to 05_*.py
└── tests/
```

## Commands

- Install (two steps): `uv sync` then `./install_native_deps.ps1` (Windows)
  or `./install_native_deps.sh` (Linux). The native step installs
  torch-scatter, torch-sparse, and torch-geometric-temporal which cannot
  go in pyproject.toml because they require torch to exist before they
  build.
- Add dep: `uv add <pkg>` (NEVER edit pyproject.toml deps manually)
- Realtime snapshot: `uv run python scripts/01_download_all.py realtime`
- Discover stations: `uv run python scripts/01_download_all.py discover`
- Backfill: `uv run python scripts/01_download_all.py backfill --start-year 2022 --end-year 2025`
- Train: `uv run python scripts/03_train.py model=mtgnn`
- Test: `uv run pytest tests/ -v`
- Lint: `uv run ruff check src/ && uv run black --check src/`
- Format: `uv run black src/ tests/ && uv run ruff check --fix src/`
- Dashboard: `uv run streamlit run app/streamlit_app.py`

## Native dependencies (read before `uv sync`)

torch-scatter, torch-sparse, and torch-geometric-temporal are PyG
extensions with native C++/CUDA code that need to link against an
already-installed torch. Because uv resolves all dependencies before
installing any, these cannot be in pyproject.toml.

After cloning or pulling fresh:
  1. `uv sync` — installs torch + all pure-Python deps
  2. `./install_native_deps.ps1` (or `./install_native_deps.sh`) — installs the three native
     packages from PyG's wheel index at https://data.pyg.org/whl/

This is also documented in README.md Quickstart and docs/api_quirks.md.

## Data sources (verified 2026-05-17)

Full details in `docs/DESIGN.md` section 4.1. Critical points:

1. **OpenAQ v3 is primary.** Use `/v3/sensors/{id}/measurements` (NOT `/hours`)
   for historical. Datetime params MUST be full ISO 8601 with timezone
   (e.g. `2024-03-01T00:00:00Z`).
2. **OpenAQ responses may be UTF-16 with BOM.** Decode with
   `response.encoding = "utf-8-sig"`.
3. **air4thai realtime** works with `User-Agent: Mozilla/5.0` header.
4. **air4thai history** has only ~90-day rolling window; NOT usable for ML
   training, only for live cross-validation.
5. **NASA FIRMS** day_range capped at 10. Bbox in lon,lat order.

## Hard restrictions

- NEVER commit anything inside `data/` (gitignored)
- NEVER commit `.env` or API keys
- NEVER push to `main` — feature branch + PR
- NEVER use `requirements.txt` — use `pyproject.toml` + `uv.lock`
- NEVER add a dependency without explaining why in chat first
- NEVER use `print()` in `src/` — use `logging`
- NEVER write `except Exception:` without re-raising or specific handling
- NEVER make real network calls in tests — always mock with requests-mock

## Working style

- Default to PLAN MODE for tasks touching >1 file
- After each meaningful change: tests, lint, then commit
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- Commits small + descriptive
- If a requirement is ambiguous, ASK — don't guess
- If wanting to refactor structure, STOP and ask
- Reference paper/docs in docstrings when implementing published algorithms

## NSC-specific

- Final report follows NSC template (TH Sarabun New 16pt; sections per
  booklet pages 29-30)
- The Disclaimer text from booklet page 44 (Thai + English) must appear in:
  - README.md
  - `app/streamlit_app.py` footer
  - Header comment of every Python file in `src/` (use placeholder
    `[TODO: NSC Disclaimer]` until final wording locked)
- All source must be original. Flag if proposing to copy non-trivial code
  from elsewhere (allowed for clearly-credited reference implementations
  like PM2.5-GNN, but must cite).

## When uncertain

1. Read CLAUDE.md and DESIGN.md again
2. Read the relevant `SESSION{N}_NOTES.md` from previous sessions
3. Ask in chat. Do not silently make assumptions.

## Reference materials

1. PM2.5-GNN paper (Wang et al. 2020): https://arxiv.org/abs/2002.12898
2. MTGNN (Wu et al. 2020, KDD)
3. PyG Temporal docs: https://pytorch-geometric-temporal.readthedocs.io/
4. GB-IG paper (2025): https://arxiv.org/abs/2509.07648
5. OpenAQ API v3: https://docs.openaq.org/
6. air4thai realtime: http://air4thai.pcd.go.th/services/getNewAQI_JSON.php
7. air4thai history: http://air4thai.com/forweb/getHistoryData.php
8. NASA FIRMS API: https://firms.modaps.eosdis.nasa.gov/api/
9. ERA5 via cdsapi: https://cds.climate.copernicus.eu/how-to-api
