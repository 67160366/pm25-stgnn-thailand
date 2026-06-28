# SESSION 8 — KICKOFF & HANDOFF
# Rigor & Validation: turn "advanced technique, unproven value" into "rigorously characterized"

> Paste/keep this as the first context in the new Claude Code session.
> Reply in Thai for explanations, English for code/identifiers (project rule).

---

## 0. Read first (in order)
1. `CLAUDE.md` — conventions, hard restrictions, native deps
2. `docs/DESIGN.md` — architecture (graph edge types §5.2, feature schema §6)
3. `docs/CRITICAL_REVIEW.md` — **why this session exists** (the honest weaknesses)
4. `docs/SESSION7_NOTES.md` — denorm-bug fix, reproducible eval, NWP sensitivity
5. Verified result files: `outputs/evaluation_val2025.json`, `outputs/attribution_march2024.json`,
   `outputs/nwp_sensitivity.json`
6. Shared eval helpers you will reuse everywhere: `src/training/evaluation.py`

---

## 1. Where things stand (Session 7 — DONE, committed on `feat/session-4-explain-app`)
- **Found + fixed a denormalization reporting bug.** Old µg/m³ numbers were overstated
  (used a flat ~22 scale instead of per-station RobustScaler `scale_`). Verified 3 ways
  (normalized RMSE == checkpoint; denorm(scaled)==raw; raw-only persistence). **No retraining
  was needed** — the model/checkpoint are correct; only reporting was wrong.
- New/changed: `scripts/04_evaluate.py` (reproducible RMSE + hybrid), `src/training/evaluation.py`
  (shared helpers), `scripts/05_attribution.py` (reproducible attribution µg, C901 fixed),
  `scripts/06_nwp_sensitivity.py` (NWP sensitivity), corrected `outputs/*.json`, corrected
  `docs/PROPOSAL_MEMO.md`, `docs/SESSION7_NOTES.md`, pitch materials in `outputs/pitch/`.
- 6 commits landed (chore gitignore/hydra, feat eval, fix attribution, feat NWP, docs, chore deps).
- Reviewer (opus) APPROVED the diff; 207 tests pass; our files ruff/black clean.
- **Then a critical review (`docs/CRITICAL_REVIEW.md`) found the project's real value is
  weaker than the proposal claimed.** Session 8 addresses those holes.

## 2. Verified numbers — USE THESE, never the old proposal numbers
> The **submitted** proposal (29 May, immutable) still has the OLD wrong numbers
> (10.21 / +5.1% / +10.6% / 4.46). Pitch & report must use the corrected ones below.

val 2025, 18 stations, 3,637 samples, identical mask, RMSE µg/m³:
| Method | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| Persistence | 2.96 | 5.19 | 8.70 | 12.99 |
| MTGNN | 4.31 | 5.80 | **8.67** | **12.68** |
| A3TGCN | 6.56 | 7.48 | 9.63 | 13.18 |
| Climatology | 17.89 | — | 17.90 | 17.92 |

- MTGNN beats persistence ONLY at 24h (+0.3%, noise-level) and 48h (+2.4%).
- Attribution (March 2024, Chiang Mai, TRAIN data): Thailand≈100%, FRP ratio 128×,
  hotspot impact mean 3.49 µg/m³ (peak samples only). IG: d2m 0.0345 > t2m 0.0344 >
  pm25_scaled 0.0140 ≫ everything else (wind ≈ 0.00005).
- NWP sensitivity: MTGNN 24h edge gone at 0.25× feature-noise, 48h edge gone at 0.5×.
- Checkpoint: `checkpoints/mtgnn/best_model.pt` (252,588 params, best epoch 15,
  val_rmse_norm_24h 0.4576). A3TGCN at `checkpoints/a3tgcn/best_model.pt`.

## 3. Mission for Session 8
**Goal: credibility, not a bigger headline number.** Each experiment yields valuable
knowledge whether the result is positive or negative. Do NOT p-hack toward "winning".
The deliverable is an honest, rigorous characterization of what the model and its
novelties actually contribute — for the **final report (deadline 17 Jul 2026)**.

---

## 4. PLAN — sequenced by deadline

### PHASE 1 — pre-pitch, CHEAP, NO GNN retrain (do first; arms the pitch Q&A honestly)
Goal: tighten the honesty of existing claims without retraining. ~1 session.

**P1.1 — Significance / CI on existing val results** [review finding #2, #9]
- Goal: is MTGNN's +2.4%@48h real? is +0.3%@24h just noise?
- Method: paired **block bootstrap** (resample by DAY, not hour, to respect temporal
  autocorrelation) on per-position squared errors, MTGNN vs persistence, per horizon.
  Report 95% CI of the RMSE difference and % improvement.
- Files: `scripts/07_significance.py` (reuse `src/training/evaluation.py`), `outputs/significance_val2025.json`
- Effort: S · Agent: implementer
- Caveat: i.i.d. bootstrap over-claims significance under autocorrelation — block-bootstrap by day.

**P1.2 — Inference-time channel ablation (quick diagnostic)** [finding #4]
- Goal: quick signal on whether type_b / type_c / adaptive matter, using the EXISTING model.
- Method: at inference, separately (a) empty type_b edge_index, (b) zero hotspot.x + empty
  type_c, (c) zero the adaptive contribution; measure val RMSE delta per horizon.
- Files: `scripts/08_inference_ablation.py`, `outputs/ablation_inference.json`
- Effort: M · Agent: implementer
- Caveat (state loudly in output): this measures the *trained model's sensitivity* to removing
  a channel at inference, NOT the value of training with it. A real ablation requires retrain
  (Phase 2). This is a lower-bound diagnostic only.

**P1.3 — Non-graph ML baseline** [finding #7]
- Goal: does a simple non-graph model match MTGNN? (justifies/undermines the graph)
- Method: per-station features = flattened 24h window × 10 features; train sklearn
  `GradientBoostingRegressor` (or `HistGradientBoostingRegressor`) per horizon, OR a small MLP.
  Train on train split, evaluate on val with the SAME mask via `evaluation.build_ground_truth`.
- Files: `scripts/09_ml_baseline.py`, `outputs/baseline_ml.json`
- Effort: M · Agent: implementer
- Dep check: sklearn is already used (RobustScaler) — confirm before assuming; NO new deps without asking.

### PHASE 2 — for the report, REQUIRES RETRAIN (the decisive credibility work)
> ⚠️ Changing the split invalidates ALL current numbers (eval, attribution, NWP, pitch).
> Keep current val-2025 numbers for the **pitch**; switch to the proper split for the **report**.
> Decide with the user before starting (see §8).

**P2.1 — Proper 3-way temporal split + retrain** [finding #1, #8]
- Method: edit `src/data/loader.py` `_SPLIT_BOUNDS` → train 2022–2023, val 2024 (model
  selection), test 2025 (report only, touched once). Retrain MTGNN + A3TGCN. Re-run
  `04_evaluate.py` on **test**. Bonus: March 2024 attribution event moves to VAL (no longer
  train) → addresses finding #8.
- Files: `configs`/`loader.py` split, re-run training + eval; new `outputs/evaluation_test2025.json`
- Effort: L · Agent: architect (split/decision) + implementer (runs)
- Caveats: less train data (2 yr) may lower accuracy — document honestly. Re-confirm the
  denorm path still holds (normalized RMSE == new checkpoint).

**P2.2 — Proper ablation (retrain variants)** [finding #4 — HIGHEST VALUE]
- Method: add boolean config flags to `MTGNNModel` (`use_type_b`, `use_type_c`,
  `use_adaptive`) gating the corresponding terms in `forward` (the spatial sum a+b+g+c).
  Retrain variants on the Phase-2.1 split: {full, −type_b, −type_c, −adaptive, −all_graph
  (temporal only), optionally −ERA5_features}. Evaluate each on test. Tabulate each novelty's
  RMSE contribution per horizon.
- Files: `src/models/mtgnn.py` (flags), `configs/model/mtgnn_ablation_*.yaml`, `scripts/03_train.py`
  (already config-driven), `outputs/ablation_retrained.json`
- Effort: L (model change + N retrains + analysis) · Agent: architect (model flags) + implementer
- Caveat: this is the experiment that proves/disproves the project's core contribution.
  Be ready to report a NEGATIVE result honestly (e.g., "type_b adds <0.1 µg/m³").

### PHASE 3 — attribution rigor [findings #5, #6, #8]
**P3.1 — Transboundary event hunt:** search `data/processed/hotspots.parquet` for dates with
high Myanmar/Laos FRP near border stations (Mae Hong Son / Tak / Chiang Rai); run attribution;
report whether the model attributes cross-border at all. Files: `scripts/10_transboundary_attr.py`.
**P3.2 — Attribution tautology/sensitivity test:** artificially scale a country's FRP
(×0.5/×2/×5) and check the attribution responds sensibly (not a trivial linear echo of input).
**P3.3 — Attribution on held-out data:** rerun the March-2024 case after Phase 2.1 (it is now
in val), and on a test-2025 event.
- Effort: M each · Agent: implementer (architect if interpretation is ambiguous)

### PHASE 4 — OPTIONAL model improvement (only if time; higher risk) [finding #2,#3]
**P4.1 — Residual-target training:** target = `pm25 − persistence`; may genuinely improve
short horizons. Clearly scoped retrain experiment; compare fairly on the Phase-2 test set.
Files: target plumbing in `loader.py`/`trainer.py`, `outputs/residual_eval.json`.

---

## 5. Plan re-review — will this actually help? (honest self-check)
**Yes, it directly closes every HIGH finding:**
- #1 → P2.1 (test set) · #2/#9 → P1.1 (significance) · #4 → P1.2 + **P2.2 (ablation)** ·
  #5/#6/#8 → P3 · #7 → P1.3 · #3 already quantified (NWP).
**What it does NOT do (be honest):** it will not make a marginal model "win". If the model is
fundamentally ~persistence, rigor documents that honestly — it does not change it. A better
headline would need P4 (residual) or architecture work, which is riskier and must not become
p-hacking. **The win here is credibility/defensibility, which is exactly what the review said
is missing.**
**Biggest risk:** Phase 2/4 retraining feasibility (see §6 device note) and the split change
rippling through all numbers. Mitigation: Phase 1 needs no retrain — do it first; gate Phase 2
on a user decision and a GPU check.
**Recommended scope if time-limited:** do Phase 1 fully + **P2.2 ablation** + **P3.1
transboundary** well, rather than all of P1–P4 half-done.

---

## 6. Constraints & gotchas (READ before coding)
- **Device/time:** `trainer.device=cuda` with CPU fallback. Training DID happen (checkpoints
  exist) but speed on this machine is unknown — **check GPU availability first**; if CPU-only,
  budget time for the ~5 retrains in Phase 2 (or reduce epochs/variants). `num_workers=0` on Windows.
- **Native deps:** after any fresh `uv sync`, run `./install_native_deps.ps1` (torch-scatter/
  sparse/geometric-temporal). See CLAUDE.md.
- **.pth encoding fix** (memory): after `uv sync`, replace the Thai abs-path in the `.pth` with
  `../../..` or Python crashes.
- **Conventions:** type hints PEP604; Google docstrings; **no `print()` in `src/`** (scripts may
  print); no bare `except`; ruff (incl. C90 mccabe ≤10, ANN, S) + black (line 100) must be clean;
  `scripts/**` ignores only S101.
- **Denorm lesson (do not repeat Bug 7):** any µg/m³ number must use per-station RobustScaler
  `scale_` from `data/processed/scalers.json`. ALWAYS sanity-check normalized RMSE == checkpoint
  `val_rmse_24h`. Reuse `src/training/evaluation.py` — do not re-implement denorm.
- **No new dependency** without explaining + asking the user first.
- **Commits:** feature branch only, never main; commit only when the user asks; Conventional
  Commits; end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage
  ONLY session files (leave untracked `generate_*.py`, PDFs, `ex/` alone).
- **ASCII in log/commit messages** (Windows cp874 cannot encode µ/³ — Bug 2).
- **Hydra run dirs** now go to `outputs/hydra/` (gitignored). `04`/new scripts use
  `initialize_config_dir` (no run-dir); `03_train.py`/`05_attribution.py` use `@hydra.main`.

## 7. File map
| What | Where |
|---|---|
| Shared eval helpers (reuse!) | `src/training/evaluation.py` |
| Eval / attribution / NWP scripts | `scripts/04_evaluate.py`, `scripts/05_attribution.py`, `scripts/06_nwp_sensitivity.py` |
| Model | `src/models/mtgnn.py` (forward: spatial sum a+b+g+c at §8), `a3tgcn.py`, `base.py` |
| Data loader / split | `src/data/loader.py` (`_SPLIT_BOUNDS`, `_build_anchor_index`) |
| Graph builder | `src/data/graph_builder.py` |
| Attribution internals | `src/explain/gb_ig.py`, `src/explain/attribution.py` |
| Results | `outputs/*.json` |
| Pitch materials | `outputs/pitch/presenter_script_th.md`, `qa_defense.md` |
| Review (why session 8 exists) | `docs/CRITICAL_REVIEW.md` |

## 8. Decisions needed from the user before Phase 2
1. **Re-split & retrain?** Phase 2 changes `_SPLIT_BOUNDS` and invalidates all current numbers.
   Confirm: keep val-2025 numbers for the **pitch**, adopt train2022-23/val2024/test2025 for the
   **report**? (Recommended.)
2. **Compute budget:** is a GPU available for ~5 retrains? If CPU-only, reduce epochs/variants?
3. **Scope:** full P1–P4, or the recommended minimal (P1 + P2.2 ablation + P3.1)?
4. **Pitch panel:** still want the optional dashboard "Model-vs-Persistence + NWP" panel before
   AI Camp?

## 9. Suggested agent division
- **architect (Opus/Max):** P2.1 split decision, P2.2 model flags, P4 design, ambiguous interpretation.
- **implementer (Sonnet, opus for denorm-touching):** P1.1–P1.3, P3 scripts, training runs from spec.
- **tester (Haiku):** lint, tests, fixtures for new scripts.
- **reviewer (Opus/Max):** review before each commit (denorm correctness, no leakage, conventions).
- Keep correctness-critical denorm/eval logic with the main thread or opus agents; verify every
  µg number against the normalized checkpoint metric.

---
**First action in Session 8:** read §0 files, confirm the §8 decisions with the user, then start
Phase 1 (P1.1 → P1.3) which needs no retrain. Do not start Phase 2 before the user confirms the
re-split and compute budget.
