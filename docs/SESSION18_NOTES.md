# SESSION 18 NOTES — 2026-07-12
Status: PHASE_COMPLETE

## What was accomplished

Roadmap items 2 + 4 complete (orchestration: architect spec → implementer
A/B/C sequential, Fable verified every gate + oracle independently before
each commit). Iron rule respected — zero report-dependency files modified;
all outputs are NEW files.

- **`ebddde3` — item 2: border-station attribution matrix**
  - `src/explain/transboundary_events.py`: shared helper; reuses script-10
    logic via importlib by path (script 10 untouched — it produced the
    frozen report JSON). Border criterion: ≤50 km haversine to nearest
    MMR/LAO vertex in the committed borders geojson → 5 stations
    (225567 Maesai 2.1km, 225626 Mae Sot 6.5km, 225648 MHS 13.5km,
    2328 Chiang Rai NREO 40.7km, 225674 Nan 45.5km).
  - `scripts/20_transboundary_matrix.py` → `outputs/transboundary_matrix.json`
    (5 stations × 2 ckpts {demo, report} × top-5 events, split=test).
  - ORACLE PASSED: MHS report-ckpt events byte-identical to frozen
    `transboundary_attr_test_split2.json` (2025-03-18 = 0.627).
  - Headline: true border stations (Maesai, Mae Sot) show high foreign
    attribution under BOTH checkpoints; MHS flagship is 0.627 (report) vs
    0.0 (demo) — model-dependence made visible, not hidden.

- **`71fd620` — item 4: multi-seed attribution uncertainty**
  - `scripts/21_transboundary_uncertainty.py` → `outputs/transboundary_uncertainty.json`:
    same 5 report-protocol events (selection is model-independent → identical
    across seeds) × 3 full-variant seeds (report_orig + full_s0 + full_s1).
  - ORACLE PASSED: report_orig reproduces frozen split2 values; event
    dates/order identical.
  - Headline (measured, honest): flagship 2025-03-18 foreign attribution
    mean 0.542, min–max 0.0–1.0, std 0.505; mean per-event spread 0.397.
    full_s0 gives ZERO total attribution mass on 2/5 events — verified NOT
    a load bug (distinct md5; mass=1.0 on other 3 events); consistent with
    ablation_multiseed.json (type_c within training noise). Direction holds
    2/3 seeds on flagship; magnitude swings full range.

- **`bc46e4e` — items 2/4 UI + qa_defense**
  - `app/views/transboundary.py`: two additive sections below the untouched
    flagship narrative (heatmap + ckpt radio; mean bars with asymmetric
    min–max error bars + per-seed table). Graceful st.info if JSON absent.
    Caveats rendered from the JSONs' caveats[] (3+3 strings, spec 4.3).
  - `app/lib/data_access.py`: 2 additive cached loaders.
  - `outputs/pitch/qa_defense.md` Q10: measured seed-spread paragraph
    appended (54.2% mean / 0–100% range / 39.7 pp mean spread; full_s0
    zero-mass disclosed). Existing wording untouched.

Gates before each commit (run by Fable, not trusted from agents): pytest
full suite (361→376 as tests landed), ruff (src/ app/ tests/ scripts/),
black (src/ tests/). Package C additionally verified: streamlit headless
boot HTTP 200 + `streamlit.testing.v1.AppTest` exercising `render()` with
the real JSONs (0 exceptions, both sections + 6 caveats rendered).

Also this session: pushed `4534e17..85e444a` (user-approved), CI green on
GitHub Actions for all three.

## Current state (branch, last commit, open work)

- Branch `feat/demo-features` @ `bc46e4e` (+ this handoff commit),
  **NOT pushed** — per-push ask rule; user approval covered only the
  earlier `..85e444a` push.
- Working tree: only pre-existing untracked report/proposal artifacts.
- SIMS final report: NOT submitted (deadline 17 Jul 17:00) → **iron rule
  still in force** (no edits to pre-existing outputs/*.json,
  outputs/figures/report/, checkpoints*/, scripts/generate_report.py,
  docs/REPORT_DRAFT.md, data/processed 2022-25 files).
- Telegram bot: still not created (user action, needed before demo 21 Aug).
- Architect spec (full detail incl. JSON schemas) archived at scratchpad
  `SPEC_items_2_4.md` — session-local; key decisions replicated in the
  module docstrings, so losing the scratchpad is acceptable.

## What was NOT done (and why)

- Push of `ebddde3..` — needs fresh user approval (per-push rule).
- Roadmap item 11 (HYSPLIT cross-check) — next in queue per SESSION17;
  items 1/3/6 (Open-Meteo live mode, interactive counterfactual, Telegram)
  also remain from the original 7.
- DEMO_ROADMAP.md not edited to mark 2/4 done — progress tracked in session
  notes per existing convention.
- No update to presenter script yet with matrix/uncertainty talking points
  (listed in DEMO_ROADMAP "งานเก็บตก" — qa_defense Q10 updated, presenter
  script not).

## Next session must start with (exact first actions)

1. Ask: SIMS submitted? (deadline 17 Jul — if past and submitted, iron
   rule lifts.)
2. Ask push approval for `85e444a..HEAD` (3 feat + 1 docs commits), then
   verify CI via GitHub API (gh CLI not installed; git-credential-manager
   token works — see session transcript pattern).
3. Continue roadmap: item 11 (HYSPLIT) or items 1/3/6 — ask user priority.
   Item 3 (counterfactual UI) is cheap now: occlusion already in gb_ig.py
   and the new matrix demonstrates per-country occlusion end-to-end.

## Critical context to preserve (gotchas, fragile files, decisions)

- **full_s0 zero-mass finding**: occluding fires barely changes full_s0's
  prediction on 2/5 flagship events. On stage, phrase as "ทิศทางคงอยู่
  2/3 seeds, ขนาดแกว่ง 0–100% จึงรายงานช่วงเสมอ" — never claim the 0.627
  number alone.
- `src/explain/transboundary_events.py` importlib-loads script 10 by path
  (deliberate layering inversion; do NOT "fix" by editing script 10 or
  copy-pasting — docstring explains).
- `n_ig_steps` CLI key on scripts 20/21 is metadata-only (script-10's
  _summarize_event hardcodes 50); logger.warning fires if ≠50.
- Scripts 20/21 checkpoints/seeds are FIXED constants (not CLI) so the
  artifacts stay canonical.
- Event selection is model-independent (FIRMS FRP + peak PM2.5 + type_c
  graph) — load-bearing for cross-seed comparability; guarded by tests.
- New JSONs (`transboundary_matrix.json`, `transboundary_uncertainty.json`)
  are DEMO artifacts, not report protocol — do not cite in REPORT_DRAFT
  unless SIMS resubmission window is used deliberately.
- app idiom: `st.plotly_chart(fig, width="stretch")` (not
  use_container_width).
- AppTest smoke pattern (works, reusable):
  `AppTest.from_string("...transboundary.render()")` with sys.path insert.
- `uv run` only; never bare `uv sync` without re-applying the .pth Thai-path
  fix (memory feedback_pth_encoding_fix).

## Open questions

- Push timing for `85e444a..HEAD` (user decision).
- Next roadmap priority: 11 (HYSPLIT independent validation) vs 1
  (Open-Meteo live forecast) vs 3 (counterfactual UI) vs 6 (Telegram bot,
  blocked on user creating the bot).
- Whether matrix/uncertainty findings go into a SIMS resubmission before
  17 Jul (booklet allows resubmission; would require lifting iron rule
  deliberately for a re-generated report — user decision).
