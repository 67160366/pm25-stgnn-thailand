# Session 8 Notes - Rigor & Validation (significance, ablation, non-graph baseline, re-split)

**Date:** 2026-06-28
**Branch:** `feat/session-4-explain-app`
**Goal:** Turn "advanced technique, unproven value" into a rigorously characterized,
*honest* account of what the model and its three novelties actually contribute. Credibility,
not a bigger headline number (see `docs/CRITICAL_REVIEW.md`). Targets the final report
(17 Jul 2026); arms the AI Camp pitch Q&A (2-3 Jul).

---

## 1. Phase 1 - no retrain (pitch-ready). Run on the pre-re-split val=2025 + `checkpoints/mtgnn`.

All three new scripts reuse `src/training/evaluation.py` for denorm/mask/persistence, so every
ug/m3 number is identical to `scripts/04_evaluate.py` (no denorm drift; Bug-7 guard).

### P1.1 Significance / CI - `scripts/07_significance.py` -> `outputs/significance_val2025.json`
Paired **block bootstrap by forecast-origin day** (B=2000, seed 42), MTGNN vs persistence.

| Horizon | MTGNN | Persist | %impr (point) | 95% CI (% impr) | significant? |
|---|---|---|---|---|---|
| 6h  | 4.30 | 2.96 | -45.3% | [-58.3, -33.9] | worse (sig) |
| 12h | 5.79 | 5.19 | -11.6% | [-18.8,  -5.6] | worse (sig) |
| 24h | 8.67 | 8.70 | +0.33% | [ -5.2,  +5.5] | **no** |
| 48h | 12.69| 12.99| +2.37% | [ -3.3,  +8.1] | **no** |

**Reading:** Neither the 24h nor the 48h advantage over persistence is statistically
distinguishable from zero (both CIs straddle 0). The headline "+2.4% @48h" is within noise on
this single val year. Do NOT claim a significant forecast win.

### P1.2 Inference channel ablation - `scripts/08_inference_ablation.py` -> `outputs/ablation_inference.json`
Knock out one channel at a time on the EXISTING checkpoint (no retrain). RMSE delta vs full:

| Variant | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| -type_b (wind)   | +0.01 | +0.01 | +0.01 | +0.00 |
| -hotspot (type_c)| +0.53 | +0.38 | +0.34 | +0.40 |
| -adaptive        | +0.01 | +0.01 | +0.00 | -0.01 |
| -all_novelties   | +0.39 | +0.28 | +0.27 | +0.34 |

**Reading:** Of the three novelties, only the **FIRMS hotspot channel** measurably affects the
trained model (~0.3-0.5 ug/m3). Wind-aware `type_b` and adaptive adjacency are inference-inert
(consistent with IG: station-level wind ~ 0). **Lower-bound diagnostic only** - the network can
route around a disabled channel; the decisive test is the retrain ablation (Phase 2.2).

### P1.3 Non-graph ML baseline - `scripts/09_ml_baseline.py` -> `outputs/baseline_ml.json`
`HistGradientBoostingRegressor` per horizon on the SAME flattened 24h x 10-feature window /
targets / mask (no graph). sklearn already a dep (no new deps).

| Method | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| persistence       | 2.96 | 5.19 | 8.70 | 12.99 |
| HistGBR (no graph)| 3.83 | 5.35 | 8.73 | 12.86 |
| MTGNN (graph)     | 4.30 | 5.79 | 8.67 | 12.69 |

**Reading:** A plain gradient-boosted tree nearly matches the STGNN - tied at 24h (8.73 vs
8.67), MTGNN ahead by only 0.17 ug/m3 (1.3%) at 48h; the tree beats MTGNN at 6h/12h. The graph
machinery buys little over a non-graph baseline on these features (finding #7).

### Combined honest message (for pitch Q&A + report)
The forecasting advantage over simple/non-graph baselines is marginal and **not statistically
significant**, and of the three novelties only the fire/hotspot channel shows (lower-bound)
usage. Lead the narrative with **source attribution (XAI)**, **48h early-warning**, and the
**hybrid operational guarantee** - not RMSE deltas. This is engineering honesty (AI governance
rubric), exactly what the critical review asked for.

### Note: ug/m3 numbers vs the committed JSON (float/rounding wobble)
A FRESH `04_evaluate` run in this environment gives MTGNN 4.30/5.79/8.67/12.69 (07/08/09 all
agree). The committed `evaluation_val2025.json` shows 4.31/5.80/8.67/12.68 - a <=0.01 wobble
from the Session-7 GPU run (the 24h primary and persistence baseline are stable; normalized
RMSE@24h still matches the checkpoint). Not a bug. For the report, regenerate evals consistently.

---

## 2. Phase 2 setup - re-split + ablation flags (report). Retrains running in background.

### Re-split decision (data-informed; user delegated the call)
Per-year usable PM2.5 coverage drove the choice: 2022 90.7%/JFMA 94.9%; 2023 61.9%/JFMA 38.5%
(weak, known PCD gap); 2024 94.3%/JFMA 96.3% (best); 2025 39.0%/JFMA 74.6% (sparse).
**Chosen split:** train **2022-23** / val **2024** (model selection) / test **2025** (held-out,
report only). 2024 anchors selection (best coverage); 2025 is most recent and matches the pitch
domain. `src/data/loader.py` `_SPLIT_BOUNDS` updated.

New sample counts: **train 15,110** (was 23,894; -37%, the documented less-data caveat),
**val 8,713**, **test 3,637** (identical to old val-2025 -> test domain == pitch domain).

**Report caveats:** 2-yr train ~ 1.3 strong burning seasons (2022 full + 2023 partial);
test-2025 is small/sparse -> wider CIs / limited power; less train data may lower accuracy.

### MTGNN ablation flags - `src/models/mtgnn.py` + `configs/model/mtgnn.yaml`
Added `use_type_a` / `use_type_b` / `use_type_c` / `use_adaptive` (default True). In `forward`,
`spatial = w_a*a + w_b*b + w_adaptive*g + w_c*c` with float gates (no new branches -> ruff C90
still <=10; no new params -> existing checkpoint loads identically). Disabling a channel zeroes
its contribution AND its gradient, so a retrained variant measures its true contribution.
Smoke-tested: every gate changes the forward output; all variants finite. 207 tests still pass.

### Retrain queue (background) -> `checkpoints_split2/` (gitignored; NEVER overwrites pitch ckpts)
Protocol = original (lr 1e-3, wd 1e-4, cosine, max_epochs 100, patience 15) so the only change
vs the pitch model is the split. ~130s/epoch on the RTX 3060 Ti (Windows num_workers=0 + larger
val). Order: `mtgnn` (full) -> `mtgnn_no_type_b` -> `mtgnn_no_type_c` -> `mtgnn_no_adaptive` ->
`mtgnn_temporal_only` -> `a3tgcn`. Launched via scratchpad `run_retrains.sh`; progress in
`checkpoints_split2/queue.log` and per-variant `train.log`. ETA: MTGNN set ~5h, +A3TGCN ~3h.

---

## 3. Status & next steps
- **DONE:** P1.1-P1.3 (committed-ready), reviewed (main-thread max-effort; the reviewer subagent
  hit a session limit). Phase 2 split + flags done, validated, tests green. Retrains launched.
- **RUNNING:** background retrain queue (6 runs) -> `checkpoints_split2/`.
- **PENDING (after retrains):** `04_evaluate.py split=test output=outputs/evaluation_test2025.json`
  (verify normalized RMSE == new checkpoint); `outputs/ablation_retrained.json` (each novelty's
  RMSE contribution per horizon); re-run P1.1-P1.3 on `split=test` for the report.
- **PENDING:** P3.1 transboundary attribution (`scripts/10_transboundary_attr.py`); P3.3 re-run
  March-2024 attribution (now in val).
- **NOT committed yet** (per hard rule - awaiting user ask). Suggested commits: (1) Phase 1
  scripts + result JSONs; (2) Phase 2 split + ablation flags + configs + .gitignore; (3) docs.

## 4. Files changed this session
- New: `scripts/07_significance.py`, `scripts/08_inference_ablation.py`, `scripts/09_ml_baseline.py`,
  `outputs/significance_val2025.json`, `outputs/ablation_inference.json`, `outputs/baseline_ml.json`,
  `docs/SESSION8_NOTES.md`.
- Changed: `src/data/loader.py` (`_SPLIT_BOUNDS`), `src/models/mtgnn.py` (ablation flags),
  `configs/model/mtgnn.yaml` (flag keys), `.gitignore` (`checkpoints_split2/`).
