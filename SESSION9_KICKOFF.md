# SESSION 9 - KICKOFF & HANDOFF
# Write the NSC 2026 final report - HONEST and rigorous. Deadline: 17 Jul 2026.

> Paste/keep this as the first context in the new Claude Code session.
> Reply in Thai for explanations, English for code/identifiers (project rule).

---

## 0. Read first (in order)
1. `CLAUDE.md` - conventions, hard restrictions, native deps, NSC report template rules
2. `docs/SESSION8_NOTES.md` - **all Session 8 results + the honest synthesis** (most important)
3. `docs/CRITICAL_REVIEW.md` - the weaknesses Session 8 set out to test
4. `docs/DESIGN.md` - architecture (graph edge types, feature schema)
5. Result JSONs (held-out TEST 2025, 3-way split): `outputs/evaluation_test2025.json`,
   `ablation_multiseed.json` (AUTHORITATIVE ablation), `significance_test2025.json`,
   `baseline_ml_test.json`, `transboundary_attr_test_split2.json`, `attribution_march2024_val.json`.
   Pitch-era (val-2025): `outputs/evaluation_val2025.json`, `significance_val2025.json`, etc.

---

## 1. Where things stand - Session 8 DONE (11 commits on `feat/session-4-explain-app`, NOT pushed)
Session 8 turned "advanced technique, unproven value" into a rigorously characterized, honest
account. Every HIGH finding in `CRITICAL_REVIEW.md` was tested - several came back NEGATIVE, and
we report them honestly (that honesty is the point; do NOT walk it back).

Built (all reuse `src/training/evaluation.py` for denorm/mask; all lint-clean; 207 tests pass):
- `07_significance.py` (paired block-bootstrap CI), `08_inference_ablation.py` (lower-bound diag),
  `09_ml_baseline.py` (non-graph HistGBR), `10_transboundary_attr.py` (cross-border attribution),
  `11_ablation_eval.py` (single-seed retrain ablation - SUPERSEDED), `12_ablation_multiseed.py`
  (AUTHORITATIVE multi-seed ablation).
- 3-way split in `loader.py` (`_SPLIT_BOUNDS`: train 2022-23 / val 2024 / test 2025); MTGNN gate
  flags `use_type_a/b/c` + `use_adaptive` in `mtgnn.py`+`mtgnn.yaml`; `trainer.seed` in `03_train.py`.
- Retrained on the new split -> `checkpoints_split2/` (5 MTGNN variants + a3tgcn) and
  `checkpoints_split2_seeds/` (seeds 0,1 for 5 variants). All gitignored; pitch `checkpoints/` untouched.

## 2. THE HONEST RESULTS - use THESE in the report

**Held-out TEST 2025 (3-way split), RMSE ug/m3.** Multi-seed full MTGNN (mean +- std over 3 seeds)
is the fair headline; the single selected checkpoint (evaluation_test2025.json) is in parentheses.

| method | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| persistence | 2.96 | 5.19 | 8.70 | 12.99 |
| MTGNN (multi-seed mean) | 5.01+-0.46 | 6.19+-0.23 | 8.93+-0.15 | **12.70+-0.07** |
| MTGNN (selected ckpt) | (4.52) | (5.93) | (8.76) | (12.66) |
| non-graph HistGBR | 3.78 | 5.58 | 9.55 | 14.22 |
| A3TGCN | 6.93 | 7.86 | 9.92 | 13.34 |

- vs persistence: MTGNN beats ONLY at 48h (~2%, robust to seed, NOT significant over test days,
  CI [-1.4, +6.3]); at 24h it LOSES on the multi-seed mean (8.93 vs 8.70).
- Multi-seed ablation (`ablation_multiseed.json`): NO channel's delta vs full exceeds combined seed
  std at any horizon; temporal_only (no graph) is +0.04/+0.13/+0.18/+0.14 vs full = WITHIN noise.
  **=> the graph does NOT robustly improve accuracy.** (The single-seed `ablation_retrained.json`
  said otherwise - it was a seed artifact; cite the multi-seed result.)
- The MTGNN's edge over the GBM at long horizons is the TEMPORAL conv architecture, not the graph.

**Transboundary attribution (held-out 2025, Mae Hong Son) - the project's defensible win:**
2025-02-16: 37.3% connected-foreign FRP -> 36.6% Myanmar attribution (well calibrated). Capability
holds across two models (proximity-tracking, non-zero); magnitude is model-dependent (lead with
2025-02-16, NOT the pitch model's "100%"). March-2024 Chiang Mai (now held-out val) stays
Thailand-100% with meteorology-dominated IG -> not a memorization artifact (interior station; few
foreign fires connect). Persistence/GBM cannot produce attribution at all.

> The submitted proposal (29 May, immutable) has OLD overstated numbers. The pitch used val-2025.
> The REPORT uses the held-out 3-way split (test 2025) numbers above.

## 3. The honest one-paragraph story (LEAD THE REPORT WITH THIS)
The forecast edge over persistence is marginal (48h only, ~2%, not significant) and the graph
machinery does not robustly improve accuracy beyond training-seed noise. The project's demonstrated,
validated contribution is **explainable source attribution**: the heterogeneous fire/wind graph
ENABLES proximity-consistent transboundary (Myanmar) attribution on held-out data - something
persistence and a non-graph baseline cannot do. Lead with XAI + honest 48h early-warning; present
RMSE/ablation transparently as marginal/null. Methodological honesty (held-out test, CIs, multi-seed,
a denorm-bug catch) is itself the "AI governance / thammaphiban" strength the NSC rubric rewards.

## 4. Mission for Session 9 - write the NSC final report (deadline 17 Jul 2026)
Follow the NSC template (TH Sarabun New 16pt; sections per booklet pages 29-30; see CLAUDE.md). Use
the Section 2/3 numbers + story above. Do NOT re-run experiments to "improve" the headline (no
p-hacking). Honest framing is the deliverable.

## 5. PLAN
- **P9.1** Draft report sections (abstract, intro, related work, data/methods, experiments,
  results, honest discussion + limitations, conclusion). Lead results with attribution; RMSE/ablation
  reported with CIs/std and the negative findings stated plainly.
- **P9.2** Finalize the NSC Disclaimer (booklet page 44, Thai+English) and replace the
  `[TODO: NSC Disclaimer ...]` placeholders in README.md, `app/streamlit_app.py` footer, and every
  `src/**` + `scripts/**` file header.
- **P9.3** Figures: per-horizon RMSE bar (persistence/MTGNN/GBM), multi-seed ablation (mean+-std),
  transboundary attribution map/table, the March-2024 vs Mae-Hong-Son contrast. (Add a small viz
  script under `scripts/` or `src/viz/`; reuse `src/viz/`.)
- **P9.4** (optional) Dashboard "About" tab still has `[TODO: NSC Disclaimer]`.

## 6. Constraints & gotchas (READ before coding)
- **Denorm (Bug-7):** every ug/m3 number via `src/training/evaluation.py` per-station RobustScaler
  `scale_`; ALWAYS sanity-check normalized RMSE == checkpoint (`scripts/12` shows the pattern, PASS=0.4469).
- **Checkpoints:** report numbers come from `checkpoints_split2{,_seeds}/` (3-way split). NEVER
  overwrite the pitch `checkpoints/mtgnn/best_model.pt`. To eval a split2 model with scripts
  07/09/10, pass `ckpt=checkpoints_split2/mtgnn/best_model.pt`; ablation variants must be loaded with
  their gate flags (see `scripts/11`/`12`).
- **Single-seed vs multi-seed:** cite `ablation_multiseed.json` (authoritative), not the superseded
  `ablation_retrained.json`. The graph-helps claim was a seed artifact - do not reintroduce it.
- **Hybrid:** use a 48h cutoff (persistence < 48h, MTGNN at 48h); the 24h-cutoff loses on the test set.
- **ASCII** in logs/commit messages (Windows cp874 cannot encode u/3 -> use "ug/m3"). PEP604 hints;
  Google docstrings; no `print()` in `src/`; ruff (C90<=10, ANN, S; scripts ignore S101) + black(100).
- **Commits:** feature branch only (never main/push without asking); Conventional Commits; commit
  only when the user asks; stage ONLY session files (leave untracked `generate_*.py`, PDFs, `ex/`,
  `SESSION*_KICKOFF/HANDOFF`); end messages `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Native deps:** after a fresh `uv sync`, run `./install_native_deps.ps1` + the `.pth` Thai-path fix.

## 7. File map
| What | Where |
|---|---|
| Honest results + synthesis | `docs/SESSION8_NOTES.md` |
| Shared eval (reuse!) | `src/training/evaluation.py` |
| Rigor scripts | `scripts/07_significance.py` ... `scripts/12_ablation_multiseed.py` |
| Attribution | `scripts/05_attribution.py` (split= override), `scripts/10_transboundary_attr.py`, `src/explain/*` |
| Split / model flags | `src/data/loader.py` (`_SPLIT_BOUNDS`), `src/models/mtgnn.py` (gate flags), `configs/model/mtgnn.yaml` |
| Report checkpoints | `checkpoints_split2/`, `checkpoints_split2_seeds/` (gitignored) |
| Report numbers | `outputs/*_test*.json`, `outputs/ablation_multiseed.json`, `outputs/transboundary_attr_test_split2.json` |
| NSC template / disclaimer | `CLAUDE.md` (NSC-specific section), booklet pages 29-30 + 44 |

## 8. Decisions needed from the user (Session 9)
1. Report language: Thai body with English technical terms (NSC default), or bilingual?
2. Headline framing: confirm leading with XAI/attribution + honesty (not RMSE) per Section 3.
3. Headline RMSE: report multi-seed mean+-std (recommended, honest) or the single selected checkpoint?
4. Figures: which of P9.3 to generate, and a target format (PNG for the booklet)?
5. Push branch / open PR now, or keep local until the report is drafted?

## 9. Optional further rigor (only if asked - not required)
More seeds (>3) for tighter ablation CIs; other border stations (Mae Sot 225626, Mae Sai 225567)
for transboundary; residual-target training (P4, riskier). None change the honest conclusion.

---
**First action in Session 9:** read SS0 files, confirm SS8 decisions with the user, then start P9.1
(draft report) using the Section 2/3 honest numbers. Do not re-run experiments to chase a better
headline.
