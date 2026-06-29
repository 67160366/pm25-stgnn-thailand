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

## 2b. Phase 3.1 - transboundary attribution (no retrain; pitch checkpoint) -> `scripts/10_transboundary_attr.py`
Directly tests finding #6. At Mae Hong Son (Myanmar border), rank split dates by foreign
(Myanmar+Laos) FRP, take the peak-PM2.5 anchor, read hotspots CONNECTED via type_c edges, then
occlusion country attribution + IG. Outputs `outputs/transboundary_attr_{test,train}.json`.

**Held-out 2025 (test) - the model DOES attribute cross-border:**

| date | peak PM2.5 | conn. foreign % | foreign attr | attribution |
|---|---|---|---|---|
| 2025-03-25 | 126.7 | 26.8% | **1.00** | Myanmar 100% |
| 2025-02-16 | 38.7 | 37.3% | 0.33 | Myanmar 33% |
| 2025-02-13 | 27.6 | 6.5% | 1.00 | Myanmar 100% |
| 2025-03-05 | 52.4 | 6.8% | 0.01 | Thailand 99% |
| 2025-03-18 | 69.1 | 0.2% | 0.00 | Thailand 100% |

**Train 2022-23:** foreign fires near Mae Hong Son were minimal (connected-foreign <=3.8%; even
the all-time strongest Myanmar day 2022-04-14 [25k FRP] had ~0% connected to THIS station), so
the model attributes ~Thailand.

**Reading (refutes #6, also held-out so addresses #8):** foreign attribution TRACKS
connected-foreign proximity - the model attributes to Myanmar exactly when foreign fires are
near the border station, including the severe 2025-03-25 episode (126.7 ug/m3 -> 100% Myanmar).
It is NOT a linear echo of the FRP fraction (100% attr vs 26.8% input; sometimes amplifies),
so it is not the trivial tautology the review feared - but report the magnitude with that
caveat. Note: total foreign FRP != foreign influence at a given station; proximity (type_c
edges) governs it, which is why the interior Chiang Mai case was Thailand-100%.

---

## 2c. Phase 2 analysis - held-out TEST (2025) -> `scripts/11_ablation_eval.py`, re-run 07/09/10
Denorm sanity PASS (full MTGNN val norm RMSE@24h = 0.4469 == checkpoint). New checkpoints in
`checkpoints_split2/`; scripts 07/09/10 gained a `ckpt=` override.

### Held-out test RMSE (ug/m3) - `outputs/evaluation_test2025.json` + `baseline_ml_test.json`
| method | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| persistence | 2.96 | 5.19 | 8.70 | 12.99 |
| MTGNN (full) | 4.52 | 5.93 | 8.76 | **12.66** |
| A3TGCN | 6.93 | 7.86 | 9.92 | 13.34 |
| non-graph HistGBR | 3.78 | 5.58 | 9.55 | 14.22 |
| hybrid (pers<24h, MTGNN>=24h) | 2.96 | 5.19 | 8.76 | 12.66 |

- **vs persistence:** MTGNN beats ONLY at 48h (+2.5%); loses 6/12/24h (24h 8.76 vs 8.70).
  Significance (`significance_test2025.json`): 48h +2.54%, CI [-1.4, +6.3] -> **NOT significant**.
- **vs non-graph baseline (HELD-OUT):** MTGNN better at 24h/48h than the GBM (which generalizes
  worse to 2025 at long horizons). BUT the multi-seed ablation (below) shows this is the TEMPORAL
  architecture, not the graph (temporal_only also beats the GBM) - so it does NOT vindicate the graph.
- **Hybrid footgun:** switching at 24h makes 24h = MTGNN (8.76) > persistence (8.70). On held-out
  test only 48h beats persistence, so the report should use a **48h-cutoff** hybrid.

### Retrain ablation (decisive, finding #4) - `outputs/ablation_retrained.json`
delta vs full on test (+ = removing the channel hurts):
| variant | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| -type_b (wind) | +0.39 | +0.67 | +0.23 | +0.33 |
| -type_c (fire) | +0.04 | +0.17 | +0.28 | +0.13 |
| -adaptive | +0.43 | +0.22 | +0.07 | +0.00 |
| temporal_only (no graph) | +1.16 | +0.91 | +0.52 | +0.63 |

- **SUPERSEDED by the multi-seed result below.** The single seed suggested temporal_only was
  clearly worst (graph helps), but that was a SEED ARTIFACT - a lucky `full` run plus an unlucky
  `temporal_only` run. With 3 seeds the effect vanishes into noise (see next subsection).

### Multi-seed ablation (AUTHORITATIVE, finding #4) - `outputs/ablation_multiseed.json` (scripts/12)
3 seeds/variant (orig + s0 + s1), evaluated on test. RMSE mean +/- std (ug/m3):
| variant | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| full | 5.01+-0.46 | 6.19+-0.23 | 8.93+-0.15 | 12.70+-0.07 |
| no_type_b | 4.95+-0.04 | 6.38+-0.19 | 9.02+-0.10 | 12.76+-0.20 |
| no_type_c | 4.67+-0.58 | 6.05+-0.26 | 8.84+-0.22 | 12.62+-0.15 |
| no_adaptive | 5.06+-0.10 | 6.28+-0.12 | 8.93+-0.10 | 12.76+-0.10 |
| temporal_only | 5.05+-0.64 | 6.32+-0.51 | 9.11+-0.19 | 12.84+-0.39 |

**KEY HONEST FINDING:** NO variant's mean delta vs full exceeds the combined seed std at ANY
horizon. temporal_only (no graph) is only +0.04/+0.13/+0.18/+0.14 vs full - WITHIN noise. So **the
graph machinery does NOT robustly improve forecast accuracy beyond training-seed variance** - the
honest (negative) answer to finding #4. The single-seed "graph helps" was noise.
- Consequence: the MTGNN's edge over the non-graph GBM at long horizons is the TEMPORAL conv
  architecture, NOT the graph (temporal_only also beats the GBM at 24h/48h). Corrects the #7 bullet.
- full (multi-seed): beats persistence only at 48h (12.70 vs 12.99; ~2.2%, robust to seed but not
  significant over test days); 24h is a slight loss on average (8.93 vs 8.70).

### Transboundary on the new model - `outputs/transboundary_attr_test_split2.json`
The split2 model also attributes cross-border, but magnitude differs from the pitch model:
2025-02-16 (37.3% connected-foreign) -> **36.6% Myanmar** (well calibrated; pitch model 33%);
2025-03-25 (26.8%) -> 3.6% (pitch 100%); 0.2% foreign -> 0%. Capability holds across BOTH models
(non-zero, proximity-tracking); MAGNITUDE is model-dependent -> lead the report with the
well-calibrated 2025-02-16 case, not the pitch model's 100%.

### One-paragraph synthesis (for the report) - REVISED after multi-seed
Against a strong persistence baseline the forecast edge is marginal (48h only, ~2%, not significant
over test days). Multi-seed ablation shows the graph machinery does NOT robustly improve accuracy
beyond training noise, so do NOT claim the novelties boost forecasting. The defensible contribution
is the **source-attribution capability the graph ENABLES**: validated, proximity-consistent
transboundary (Myanmar) attribution on held-out data, which persistence and a GBM cannot produce at
all. Lead with XAI / source attribution (+ honest 48h early-warning); present the forecast and
ablation results transparently as marginal/null. That honesty is itself the AI-governance strength
the NSC rubric rewards.

---

## 3. Status & next steps
- **DONE:** P1.1-P1.3 (committed-ready), reviewed (main-thread max-effort; the reviewer subagent
  hit a session limit). Phase 2 split + flags done, validated, tests green. Retrains launched.
- **DONE:** all 6 retrains complete + Phase 2 analysis (section 2c) - held-out test eval, retrain
  ablation, and significance/baseline/transboundary re-run on the new split2 model.
- **DONE (Phase 3.1):** `scripts/10_transboundary_attr.py` refutes finding #6 - cross-border
  attribution demonstrated on held-out 2025 (sections 2b/2c).
- **PENDING:** P3.3 (re-run March-2024 attribution, now in val - `05_attribution.py` needs a split
  override); multi-seed retrains to firm up per-channel ablation; final report write-up.
- **Committed:** Phase 1 + Phase 2 setup + P3.1 + docs (5 commits). Phase 2 ANALYSIS (script 11 +
  test/ablation JSONs + 07/09/10 ckpt override) pending commit this turn.

## 4. Files changed this session
- New: `scripts/07_significance.py`, `scripts/08_inference_ablation.py`, `scripts/09_ml_baseline.py`,
  `scripts/10_transboundary_attr.py`, `scripts/11_ablation_eval.py`, and `outputs/`:
  `significance_val2025.json`, `significance_test2025.json`, `ablation_inference.json`,
  `baseline_ml.json`, `baseline_ml_test.json`, `ablation_retrained.json`, `evaluation_test2025.json`,
  `transboundary_attr_test.json`, `transboundary_attr_train.json`, `transboundary_attr_test_split2.json`,
  plus `docs/SESSION8_NOTES.md`.
- Changed: `src/data/loader.py` (`_SPLIT_BOUNDS`), `src/models/mtgnn.py` (ablation flags),
  `configs/model/mtgnn.yaml` (flag keys), `.gitignore` (`checkpoints_split2/`),
  `scripts/07,09,10` (`ckpt=` override for the new split2 checkpoints).
