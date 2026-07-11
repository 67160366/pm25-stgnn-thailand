# SESSION 17 NOTES — 2026-07-12
Status: PHASE_COMPLETE

## What was accomplished

Roadmap items 8 + 10 done in parallel (user-approved pre-SIMS, iron rule
respected — zero report-dependency files modified; loader.py change is
strictly opt-in, see below). Both committed on `feat/demo-features`:

- **`c48bcbc` — item 10: exceedance re-scoring of test-2025 forecasts**
  - New `src/training/exceedance.py` (pure numpy: binary_event strict `>`,
    contingency POD/FAR/CSI/accuracy, Brier, BSS vs no-leakage climatology),
    `scripts/18_exceedance_eval.py`, `tests/test_exceedance.py` (27 cases),
    `outputs/exceedance_test2025.json`.
  - Climatology base rate from train+val 2022–2024 ONLY (37.5→0.1951,
    75→0.0441); mask semantics identical to loader.py validity
    (`~mask_in_loss & ~exclude & isfinite`), n=3637 matches
    evaluation_test2025.json.
  - Result (honest thesis): model beats persistence only at 48h
    (BSS 0.190 vs 0.170 @75 µg/m³); loses 6–12h on false alarms.
    CAVEAT to always disclose: 2025 test base rates (34% @37.5, 5.5% @75)
    exceed 2022–24 climatology → BSS inflated for BOTH model and persistence.

- **`67c12b4` — item 8: frozen-model eval on Jan–Apr 2026 burning season**
  - `src/data/loader.py` gains opt-in `full_index` / `split_bounds` params
    (None → old module globals; all existing callers byte-identical).
  - New `scripts/19_eval_2026.py` (typer: backfill-openaq/firms/era5, build,
    evaluate; separate paths `data/raw/{openaq,firms}_2026/`,
    `data/processed/*_2026.parquet` — 2022-25 files untouched),
    `tests/test_eval_2026.py` (5 tests incl. frozen-scaler no-refit assert),
    `outputs/evaluation_2026burn.json`.
  - Frozen protocol: training-era `scalers.json` reused verbatim, no refit.
  - Result: n=2809, 18 stations. RMSE persistence 4.08/7.22/11.99/17.52 vs
    mtgnn_split2 5.47/7.67/11.46/15.92 — model beats persistence at 24h
    (+4.4%) and 48h (+9.1%) on true out-of-sample 2026 data (stronger than
    2025's +2.5% @48h). CAVEAT: burn-season-only RMSE not comparable to
    full-year test-2025 magnitudes. FIRMS 2026 = VIIRS_NOAA20_SP.

Gates before each commit: pytest 342 passed, ruff (src/ app/ tests/
scripts/) clean, black (src/ tests/) clean.

## Current state (branch, last commit, open work)

- Branch `feat/demo-features` @ `67c12b4`, **NOT pushed** (user said
  "ยังไม่ push" at session end; origin is at `a7f2d31`, CI green there).
- Working tree: only pre-existing untracked report/proposal artifacts
  (booklet PDF, docx, pitch, proposal_text.txt, review_prompt.md) — leave.
- SIMS final report: NOT submitted yet (deadline 17 Jul 2026 17:00) →
  **iron rule still in force** (no edits to outputs/*.json existing files,
  outputs/figures/report/, checkpoints*/, scripts/generate_report.py,
  docs/REPORT_DRAFT.md, existing data/processed 2022-25 files).
- Telegram bot: still not created (user action, needed before demo 21 Aug).

## What was NOT done (and why)

- Push of `c48bcbc` + `67c12b4` — user deferred; per-push ask rule.
- Roadmap items 2/4 (multi-station border attribution matrix + multi-seed
  attribution uncertainty) and 11 (HYSPLIT cross-check) — user ended
  session ("พอแค่นี้ก่อน"); these are next in queue, all new-file work.

## Next session must start with (exact first actions)

1. Ask: SIMS submitted yet? (if yes → iron rule lifts; if past 17 Jul and
   unclear, ask before touching anything report-adjacent)
2. Ask push approval for `4534e17..67c12b4`, then verify CI via GitHub API
   (gh CLI not installed; use git-credential-manager token; repo private).
3. If continuing roadmap: items 2/4 next, delegate per orchestration
   pattern (architect/implementer/tester, disjoint file scopes, Fable runs
   combined gates + selective staging).

## Critical context to preserve (gotchas, fragile files, decisions)

- Exceedance event convention is strict `>` (verification stats, Wilks);
  app/lib/telegram.py alert uses `>=` — deliberate, do not "unify".
- Climatology reference MUST come from 2022–2024 train+val only; never
  recompute from test/2026 (leakage).
- `checkpoints_split2/mtgnn` = report protocol (primary);
  `checkpoints/mtgnn/best_model.pt` = demo model — 2025 was its val split,
  so it is NOT fully held-out on 2025; 2026 eval is honest for both.
  [NEEDS VERIFICATION] demo checkpoint provenance if ever cited as a
  distinct model in reports (architect flag, unresolved).
- loader.py `full_index`/`split_bounds` are opt-in; default path must stay
  byte-identical — any future edit there re-verify old callers.
- 2026 artifacts live in separate files/dirs by design; do not merge into
  2022-25 datasets.
- Concurrent-agent work: stage selectively per work item (this session a
  WIP architect edit briefly failed the other item's gate run — excluded
  via selective `git add`, resolved before its own commit).
- `uv run` only, never bare `uv sync` without re-applying the .pth
  Thai-path fix (see memory feedback_pth_encoding_fix).

## Open questions

- Push timing for the two commits (user decision pending).
- Whether items 2/4 results go in the final report (only if SIMS not yet
  submitted at completion time — otherwise demo-only material).
