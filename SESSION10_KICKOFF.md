# SESSION 10 — KICKOFF, HANDOFF & FULL PROJECT ONBOARDING
# PM2.5 STGNN Thailand · NSC 2026 Category 14 · finish the final report (deadline 17 Jul 2026)

> Paste/keep this as the first context in the new session. Project rule: **reply in Thai for
> explanations, English for code/identifiers.** This single file is meant to fully onboard a fresh
> session — read it top to bottom once, then act.

---

## 0. FIRST ACTIONS (in order)
1. Read this file fully.
2. Read `CLAUDE.md` (hard rules, conventions, native deps) and the auto-memory
   `MEMORY.md` + `memory/project_state.md` (live status, the gotchas).
3. Skim `docs/REPORT_DRAFT.md` (the report we just wrote — it is the single best summary of the
   whole project) and `docs/SESSION8_NOTES.md` (the authoritative honest results).
4. Confirm the Session-10 mission in §9, then work task-by-task with the discipline in §10.
5. **Do NOT re-run experiments to "improve" the headline.** Honest framing is the deliverable.

---

## 1. WHAT THIS PROJECT IS (the one-paragraph teach)
Northern Thailand (9 provinces) has an annual dry-season PM2.5 haze crisis from biomass burning +
transboundary transport (Myanmar/Laos). We built an **Explainable Spatio-Temporal Graph Neural
Network**: it (a) **forecasts** PM2.5 at 18 Air4Thai stations for 6/12/24/48 h, and (b) **attributes
the source** of the pollution (which country's fires) using Graph-based Integrated Gradients + an
occlusion method. It integrates ERA5 weather and NASA FIRMS satellite fire hotspots as nodes in a
heterogeneous graph. It is an NSC 2026 (National Software Contest) entry, Category 14 (science &
technology development), university level. Rubric weights (Cat 14): Report 25 · Technique 20 ·
Creativity 20 · Social/Economic 15 · Look&Feel 15 · Presentation 5.

## 2. THE HONEST THESIS (★ the most important thing — lead everything with this)
After rigorous evaluation (held-out test, bootstrap CIs, multi-seed ablation):
- The **forecast edge over a persistence baseline is marginal** — present only at 48 h (~2%) and
  **NOT statistically significant** in a single year (95% CI spans 0). At 24 h it ties / slightly loses.
- **Multi-seed ablation shows the graph machinery does NOT robustly improve accuracy** beyond
  training-seed noise. A non-graph gradient-boosted baseline (GBM) nearly matches MTGNN.
- The **demonstrated, defensible contribution is explainable source attribution**: on held-out 2025
  data the model produces proximity-consistent transboundary (Myanmar) attribution that persistence
  and a non-graph model simply cannot. That, plus an honest 48 h early-warning + an operational
  hybrid, is the story.
- **Methodological honesty IS the selling point** (the "AI governance / ธรรมาภิบาล" the rubric
  rewards): we use held-out test, report CIs/std, ran multi-seed, and caught our own denorm bug.
Never walk this back into "we beat persistence / the graph helps." If a number tempts you to
overclaim, re-read this section.

## 3. ARCHITECTURE DEEP-DIVE (data → graph → model → XAI)

### 3.1 Data pipeline (`src/data/`, `scripts/01`–`02`)
- **PM2.5**: Air4Thai (PCD), 18 stations, hourly. Target = `pm25_scaled` (per-station RobustScaler);
  `pm25_raw` is the µg/m³ ground truth for metrics. `data/processed/dataset.parquet`
  (631,152 rows = 18 stations × 35,064 hours, 2022-01-01 → 2025-12-31 UTC; long format).
- **Weather**: ERA5 (Copernicus) — `u10, v10, t2m, d2m, blh`, hourly, interpolated to stations.
- **Fire**: NASA FIRMS VIIRS/MODIS, daily; spatially clustered → `data/processed/hotspots.parquet`
  (cols: `date` [⚠️ **datetime.date objects, not strings**], `cluster_id`, `centroid_lat`,
  `centroid_lon`, `total_frp`, `point_count`, `country`).
- **10 model features per station** (`loader._FEATURE_COLS`): `pm25_scaled, hour_sin, hour_cos,
  doy_sin, doy_cos, u10, v10, t2m, d2m, blh`.
- `data/processed/scalers.json` = per-station RobustScaler `{center_, scale_}` keyed by station_id
  (string). `stations_metadata.parquet` = 18 stations (`location_id`→station_id, `name`, `lat`,
  `lon`; **no province column** — parse province from `name` after the comma).

### 3.2 The heterogeneous graph (`src/data/graph_builder.py` — critical, has tests)
Node types `station` (18) + `hotspot` (M clusters). **Three edge types** (the "novelties"):
- **type_a (geographic)**: static, undirected; weight `exp(-d/50)`, distance ≤ 100 km.
- **type_b (wind-aware, dynamic)**: directed, updated hourly from ERA5 wind:
  `A_wind[i,j] = max(0, cos(θ_wind, θ_ij)) · exp(-d/λ)`; ≤ 200 km, alignment > 0.3.
- **type_c (fire → station, bipartite)**: hotspot influences a downwind station; ≤ 500 km,
  alignment > 0.4. `data["hotspot"].x = [total_frp, centroid_lat, centroid_lon]`,
  `data["hotspot"].country` = list of country labels (drives attribution).
- `graph_config={"wind_mode": "from_field"}` is what the trained checkpoints use — **always pass it**
  (the loader's bare default is `constant_ne`, which is NOT what was trained).

### 3.3 Model (`src/models/mtgnn.py`)
`MTGNNModel` (Wu et al., KDD 2020): graph-learning layer (adaptive adjacency) + temporal conv (TCN)
+ graph conv (GCN). `forward(HeteroData) -> (B*N, H)` **normalized** predictions (H=4) — you MUST
denormalize before showing µg/m³. Config (`configs/model/mtgnn.yaml`): n_features=10, hidden_dim=64,
n_layers=3, kernel_size=7, dropout=0.3, n_stations=18. **Ablation gate flags** `use_type_a/b/c`,
`use_adaptive` (float gates in `forward`; disabling zeroes a channel's contribution AND gradient).
Baseline: `src/models/a3tgcn.py` (small Attention-Temporal-GCN).

### 3.4 Explainability (`src/explain/`)
- `gb_ig.py`: `integrated_gradients(...)` (per-feature IG, completeness axiom) +
  `occlusion_country_attribution(...)` (zero each country's hotspots, measure prediction drop →
  per-country scores in [0,1] summing to 1). `gnn_explainer.py` = grad×input saliency (note: despite
  the filename it is NOT a PyG edge-mask explainer).
- `attribution.py`: `station_source_report(model, data, station_idx, horizon_idx, ...)` = the one-call
  entry point → `{ig_feature_importance, country_attribution, ...}`.

### 3.5 ⚠️ TWO MODELS — the single most confusing thing (internalize this)
There are two trained MTGNN setups, both evaluated on **2025 data**:
| | "main / pitch" model | "report / split2" model |
|---|---|---|
| trained on | 2022–2024 (old 2-way split) | 2022–2023 (new 3-way split) |
| 2025 role | **validation** (model-selection) | **held-out test** (never seen) |
| checkpoint | `checkpoints/mtgnn/best_model.pt` (in tree) | `checkpoints_split2/mtgnn` (**NOT in tree**) |
| result JSONs | `evaluation_val2025.json`, `significance_val2025.json`, `nwp_sensitivity.json`, `attribution_march2024.json`, `transboundary_attr_test.json` | `evaluation_test2025.json`, `significance_test2025.json`, `ablation_multiseed.json`, `baseline_ml_test.json`, `transboundary_attr_test_split2.json` |
| used by | the dashboard + the pitch deck (live inference uses this; split2 ckpts aren't on disk) | the **report's primary numbers** |
- **The report leads with the held-out test (report model); the deck/dashboard use the main model.**
  Both are 2025. **Never label the val numbers "2024"** even though the current loader split has
  val=2024 — those val JSONs predate the re-split and physically contain 2025 data (this trap bit us
  twice: dashboard + report; see §6).

## 4. REPO STRUCTURE (annotated; do not refactor locked dirs without asking)
```
src/data/{scrapers,preprocessing,graph_builder*,loader}   *has tests
src/models/{base,mtgnn,a3tgcn}
src/explain/{gb_ig,attribution,gnn_explainer}
src/training/{trainer,losses,metrics,evaluation}   evaluation.py = SHARED ug/m3 math (reuse always)
src/viz/{maps,timeseries}                          Plotly figures (used by the dashboard)
app/streamlit_app.py                               entry: st.navigation (6 pages)
app/lib/{aqi,data_access,inference,ui}.py          dashboard shared helpers (cached)
app/views/{overview,forecast,attribution,transboundary,performance,about}.py
scripts/01..05  pipeline (download/preprocess/.../04_evaluate/05_attribution)
scripts/06..12  rigor (06 nwp, 07 significance, 08/11/12 ablation, 09 ml-baseline, 10 transboundary)
scripts/13_report_figures.py                       NEW: report PNGs from JSONs (matplotlib)
scripts/generate_proposal.py  generate_pitch_deck.py  (python-docx / python-pptx generators)
docs/ DESIGN.md, SESSION{1..8}_NOTES.md, CRITICAL_REVIEW.md, PROPOSAL_MEMO.md, REPORT_DRAFT.md
outputs/ *.json (results), pitch/ (deck+scripts), figures/report/*.png
configs/ (Hydra)  ·  tests/ (213 pass)  ·  data/ (gitignored)  ·  checkpoints/ (gitignored)
20260218_NSC2026_Booklet.pdf  (the NSC rules; report structure idx 29-30, disclaimer idx 44)
```

## 5. AUTHORITATIVE NUMBERS (the ONLY values allowed; cite the JSON; verify before use)
- **Held-out test 2025 (report model, primary)** `evaluation_test2025.json` + `baseline_ml_test.json`:
  persistence 2.96/5.19/8.70/12.99 · MTGNN 4.52/5.93/8.76/**12.66** · A3TGCN 6.93/7.86/9.92/13.34 ·
  GBM 3.78/5.58/9.55/14.22 (RMSE µg/m³, 6/12/24/48h).
- **Main model on 2025 (deck/dashboard)** `evaluation_val2025.json`: MTGNN 4.31/5.80/**8.67**/**12.68**;
  A3TGCN 6.56/7.48/9.63/13.18; hybrid 2.96/5.19/8.67/12.68; MTGNN 252,588 params (epoch 15, norm
  0.4576), A3TGCN 27,164 (epoch 97, norm 0.4928).
- **Significance (48h, both not significant)**: test +2.54% CI [-1.44, 6.28]; val +2.37% CI [-3.31, 8.10].
- **Multi-seed ablation** `ablation_multiseed.json`: full mean 5.01/6.19/8.93/12.70; temporal_only
  delta +0.04/+0.13/+0.18/+0.14; **every robust_beyond_noise = false**.
- **Transboundary (held-out, Mae Hong Son, 2025-02-16)** `transboundary_attr_test_split2.json`:
  37.3% connected-foreign → **36.6% Myanmar** (lead with this); main model 33.2% (agrees). Magnitude
  is model-dependent (2025-03-25: main 100% / split2 ~4%).
- **Chiang Mai Mar-2024** `attribution_march2024.json`: Thailand ~100%, FRP 128× (109,564 vs 854),
  hotspot +3.49 µg/m³, peak 141–144; IG d2m 0.0345 / t2m 0.0344.
- **NWP** `nwp_sensitivity.json`: crossover 24h 0.25×, 48h 0.5× (robust BELOW 0.5×, lost AT 0.5×).
- **FORBIDDEN (denorm-bug, never reappear): 10.21 · 14.32 · +5.1% · +10.6% · hotspot 4.46.**

## 6. GOTCHAS & DATA PROVENANCE (the traps that already bit us — read before coding)
1. **`.pth` Thai-path crash** (`memory/feedback_pth_encoding_fix.md`): after `uv sync`/`uv add`/any
   `pyproject.toml` change, uv rewrites `.venv/Lib/site-packages/_editable_impl_pm25_stgnn_thailand.pth`
   to the Thai absolute path → Windows cp874 site init crash. Fix: `printf '../../..\n' >` that file,
   then run everything as `UV_NO_SYNC=1 uv run ...` so uv doesn't re-clobber it. (Ruff is Rust = it
   survives a broken `.pth`; black/python crash — don't be fooled.) In **plan mode** you cannot patch
   the `.pth` (no edits) → use `.venv/Scripts/python.exe` directly for read-only checks.
2. **Denorm bug (Bug-7)**: every µg/m³ number must flow through `src/training/evaluation.py`
   (`station_scalers`, `build_ground_truth`, `predict`, `denorm_pred`, `rmse_block`). Ad-hoc denorm =
   the bug that overstated the proposal. Sanity check: normalized RMSE@24h == checkpoint (≈0.4576/0.4469).
3. **val2025 = 2025 data, NOT 2024** (§3.5). Label by what the file physically is.
4. **`hotspots.parquet["date"]` = `datetime.date` objects, not "YYYY-MM-DD" strings.** Filter via
   `.astype(str) == "YYYY-MM-DD"` or compare to `datetime.date(...)`. (This made the dashboard fire
   overlay silently empty — fixed in `dc8056c`.)
5. **ASCII only in logs/commit messages** (Windows cp874 cannot encode µ/³ → write "ug/m3").
6. **Two transboundary JSONs' internal `checkpoint` field is stale** (`scripts/10:227` hardcodes the
   pitch path). Cite transboundary results by **filename**, not the field. (Worth fixing in 10 later.)
7. Subagent reports can be wrong: an Explore agent mis-reported the hotspot `date` dtype (#4). Always
   verify a load-bearing claim against the real file before relying on it.

## 7. COMMANDS
- Install: `uv sync` then `./install_native_deps.ps1` (torch-scatter/sparse/PyG-temporal) +
  the `.pth` fix (#6.1). Add dep: `uv add <pkg>` (never edit pyproject deps by hand).
- Run anything: `UV_NO_SYNC=1 uv run python <script>` (after `.pth` fix). Tests: `uv run pytest tests/ -q`.
  Lint: `uv run ruff check <path> && uv run black --check <path>`.
- Dashboard: `UV_NO_SYNC=1 uv run streamlit run app/streamlit_app.py` (opens localhost:8501).
- Report figures: `uv run python scripts/13_report_figures.py`.
- Read-only Python in plan mode (no uv re-sync): `.venv/Scripts/python.exe -c "..."`.

## 8. SESSION 9 STATE — DONE (7 commits on `feat/session-4-explain-app`, **NOT pushed**)
`531e3f7` SS4b pitch updates (Q3/new-Q4/Q10) + new pitch deck · `eddf0a9` full 6-page dashboard ·
`beac8a9` NWP wording fix · `159be2e` NSC report draft (all 13 sections) · `3b28391` report polish ·
`dc8056c` dashboard hotspot date-filter fix · `b7578f3` report result figures + wiring.
- **Pitch deck** `outputs/pitch/pitch_deck_aicamp.pptx` (9 slides) + `outputs/pitch/{presenter_script_th,
  qa_defense}.md` — reviewer-approved, honest, AI-Camp-ready.
- **Dashboard** — full multipage, public-friendly, reviewer-approved (a streamlit server may still be
  running on localhost:8501 from this session; restart to pick up the hotspot fix).
- **Report** `docs/REPORT_DRAFT.md` — Markdown, all 13 NSC sections, 8 figures embedded,
  Disclaimer (TH+EN) + author/advisor filled, reviewer-approved (APPROVE).
- Identity (from `proposal_text.txt`): lead **นายรณชัย ขาวสะอาด** (67160366@go.buu.ac.th); advisor
  **ดร.วัชรพงศ์ อยู่ขวัญ**; **ม.บูรพา คณะวิทยาการสารสนเทศ สาขาปัญญาประดิษฐ์ประยุกต์ฯ**. NSC **project
  code is still a placeholder**.

## 9. SESSION 10 MISSION — finish the report into a submittable artifact (deadline 17 Jul 2026)
Do these in order; commit each with the discipline in §10. **Suggested work split (tools/mode/agent)
is given per task.**

- **P10.1 — Word generation (biggest; produces the submittable .docx).** New
  `scripts/generate_report.py` that extends the python-docx pattern in `scripts/generate_proposal.py`
  (helpers `fmt`/`_set_cs_font` set **TH Sarabun New 16pt**, A4, 1" margins, page numbers, cover per
  booklet idx 43; `add_picture` for the 8 figures). Source content = `docs/REPORT_DRAFT.md` verbatim
  (numbers already verified — do NOT change them). Output e.g. `outputs/NSC2026_Final_Report.docx`.
  → **PLAN MODE first** (new multi-file-ish task). **Explore agent**: study `generate_proposal.py`
  helpers + confirm TH Sarabun New is available/embeddable. Build in main thread; **verify** by
  re-opening the .docx with python-docx (assert headings/tables/8 images present) + a `reviewer`
  subagent audit (numbers match REPORT_DRAFT, all NSC sections, font/size). Note: a Markdown→docx
  converter (e.g. pandoc) is an alternative but pandoc isn't installed and won't give TH-Sarabun
  cover/formatting control — prefer python-docx.
- **P10.2 — NSC Disclaimer rollout (quick win).** Replace `[TODO: NSC Disclaimer ...]` placeholders in
  every `src/**`, `app/**`, `scripts/**` header, `README.md`, and the dashboard About/footer with the
  real booklet text (now in `docs/REPORT_DRAFT.md` §12, TH+EN). → no plan mode needed; a **`tester`/
  `implementer` subagent or a careful `grep + sed`-style pass** fits; then `grep -rn "TODO: NSC
  Disclaimer"` must return nothing in those trees; run `pytest`/`ruff` after.
- **P10.3 — fill the NSC project code** (ask the user for it) and confirm author romanization in the
  English disclaimer (`Mr. Ronnachai Khaosa-ard` is a guess — flagged in REPORT_DRAFT §12).
- **P10.4 — INSTALL + USER GUIDE** (booklet requires คู่มือการติดตั้ง + คู่มือการใช้งาน in the
  appendix). Write `docs/INSTALL.md` + `docs/USER_GUIDE.md` (uv sync + native deps + .pth fix; how to
  run dashboard + read each page). → straightforward; `implementer` subagent from a clear spec.
- **P10.5 (optional rigor, only if asked)** in `docs/SESSION8_NOTES.md` §9: more seeds, other border
  stations (Mae Sot 225626, Mae Sai 225567), residual-target training. **None change the honest
  conclusion** — don't chase a better headline.
- **Push / PR**: only when the user asks. Branch is `feat/session-4-explain-app`; never push to main.

## 10. HOW TO WORK HERE (tools / mode / agent playbook — this is "use the right tool for the job")
- **Plan mode** (`EnterPlanMode`): default for any task touching >1 file or with design choices
  (CLAUDE.md rule). Flow: Phase-1 **Explore** subagents → design → `ExitPlanMode` for approval.
  Use `AskUserQuestion` only for genuine forks (don't re-litigate settled decisions).
- **Explore subagent** (read-only fan-out): use it to (a) map unfamiliar code before planning, and
  (b) **independently re-extract numbers from JSONs** to cross-check your own (this caught/avoided
  several errors this session). Give it exact files + ask for file:line citations.
- **reviewer subagent (Opus)**: run BEFORE committing anything number- or honesty-sensitive (it
  caught the val-label honesty bug in both the dashboard and the report this session). Feed it the
  verified numbers + the honesty rules; fix findings, then `SendMessage` it to confirm the fix.
- **architect** (design/algorithms/cross-file debugging), **implementer** (code from a clear spec),
  **tester** (tests/lint/fixtures) — per `CLAUDE.md`. Don't spawn agents for trivial work; the user
  asked for *systematic* subagent use, not gratuitous spawning.
- **Verify after every step (anti-hallucination)**: grep each emitted number against its source JSON;
  for figures, **Read the PNG** (the Read tool renders images); for the dashboard, use
  `streamlit.testing.v1.AppTest` headless (`AppTest.from_file(...).run(); assert not at.exception`);
  for the .docx, reopen with python-docx and assert structure.
- **Commits**: feature branch only, **commit only when the user asks**, Conventional Commits, ASCII
  body, stage only the files for that change (leave the untracked booklet PDF, `ex/`, `SESSION*` files
  alone), end every message with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. If a
  `git add <dir>` silently skips files, check `.gitignore` (the generic `lib/` rule hides `app/lib/`
  → we added `!app/lib/`; use `git add -f` if needed).
- **Memory**: update `memory/project_state.md` at the end of meaningful work; record non-obvious
  gotchas as their own memory file + a one-line pointer in `MEMORY.md`.

## 11. WHERE TO FIND THINGS
| What | Where |
|---|---|
| The honest results + synthesis | `docs/SESSION8_NOTES.md` (§ near lines 195–203) |
| The whole project, readable | `docs/REPORT_DRAFT.md` |
| Architecture/design decisions | `docs/DESIGN.md` |
| Shared µg/m³ math (reuse!) | `src/training/evaluation.py` |
| Result numbers | `outputs/*.json` (see §5 for which holds what) |
| Report figures + generator | `outputs/figures/report/*.png`, `scripts/13_report_figures.py` |
| Pitch deck + scripts | `outputs/pitch/` |
| NSC rules / report template / disclaimer | `20260218_NSC2026_Booklet.pdf` (idx 29-30, 43-44) |
| Identity (names/contact) | `proposal_text.txt` |
| Gotchas / live status | `memory/` (`project_state.md`, `feedback_*.md`) |

## 12. DEADLINES
- AI Camp pitch: **2–3 Jul 2026** (deck + script ready).
- NSC final report + install/user manuals via SIMS: **17 Jul 2026, 17:00** (the Session-10 goal).
- Demo / final round: **21 Aug 2026**.

---
**TL;DR for Session 10:** the pitch, dashboard, and an honest, reviewer-approved Markdown report (with
figures) are DONE and committed (not pushed). Turn `docs/REPORT_DRAFT.md` into the submittable
**TH-Sarabun-16pt Word document** (P10.1), roll out the real **Disclaimer** (P10.2), get the **project
code** + INSTALL/USER guides (P10.3–4). Keep the honesty (§2), reuse `evaluation.py` for every number,
mind the gotchas (§6), and verify + reviewer-audit before each commit.
