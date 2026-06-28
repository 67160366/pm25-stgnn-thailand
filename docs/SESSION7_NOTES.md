# Session 7 Notes — Denorm Bug Fix, Reproducible Evaluation, AI Camp Prep

**Date:** 2026-06-28
**Branch:** `feat/session-4-explain-app`
**Goal:** Prepare for AI Camp pitch (2–3 Jul 2026); harden the technical
"cons" raised by reviewers (per-horizon RMSE, persistence comparison, ERA5
latency, attribution rigor).

---

## 1. MAJOR: denormalization reporting bug found and fixed

### What was wrong

The committed `outputs/evaluation_val2025.json` and the µg/m³ figures copied into
`docs/PROPOSAL_MEMO.md` (and the already-submitted proposal) were **overstated**.
They were produced by an ad-hoc script (not in the repo) that denormalized
normalized-scale RMSE using a wrong, near-constant scale (~22) instead of the
correct **per-station RobustScaler `scale_` (IQR)** from `data/processed/scalers.json`.

The bug is in the **reporting/denorm step only**. Training is correct — targets
were normalized (`pm25_scaled`) and the checkpoint's stored normalized metric
(`val_rmse_24h = 0.4576`) is right. **No retraining was needed.**

### Three independent verifications

1. Recomputed **normalized** RMSE@24h for MTGNN = **0.4575** ≡ checkpoint's stored
   `val_rmse_24h` = 0.4576 → model, alignment, and masking are correct.
2. `denorm(pm25_scaled) == pm25_raw` for all 519,935 rows (max diff 1.9e-5) →
   `scalers.json` is consistent with the raw data.
3. Persistence RMSE computed **directly from raw PM2.5** (no denorm path) matches
   the new script: 2.96 / 5.19 / 8.70 / 13.00. The committed file's persistence
   (3.79 / 6.51 / 10.76 / 16.02) cannot come from correct per-station denorm
   (max IQR = 29, mean 19.92; the committed numbers imply ~22 flat).

### Verified results (val 2025, 18 stations, 3,637 samples)

RMSE in µg/m³, all methods scored on identical valid positions; persistence =
last-observed value carried forward.

| Horizon | Persistence | MTGNN | A3TGCN | Hybrid (oper.) | MTGNN vs persist |
|---|---|---|---|---|---|
| 6h  | 2.96 | 4.31 | 6.56 | 2.96 | −45.6% |
| 12h | 5.19 | 5.80 | 7.48 | 5.19 | −11.8% |
| **24h** | 8.70 | **8.67** | 9.63 | 8.67 | **+0.3%** |
| **48h** | 12.99 | **12.68** | 13.18 | 12.68 | **+2.4%** |

- MTGNN still beats persistence at 24h and 48h, but the margin is **much smaller**
  than previously claimed (+5.1% / +10.6% → +0.3% / +2.4%).
- A3TGCN does **not** beat persistence at any horizon (earlier draft claimed 48h).
- Normalized ranking is unchanged: MTGNN (0.4576) > A3TGCN (0.4928).

### Attribution µg figure corrected

`outputs/attribution_march2024.json` `hotspot_impact_ug_m3_24h` used the same wrong
scale. Recomputed with Chiang Mai's correct `scale_` = 17.2 (station_id 225669):

- mean **3.49** µg/m³ (range 2.50–5.64) — was 4.46 (3.2–7.2).
- `country_attribution` (Thailand 1.0 / Myanmar 0.0), FRP ratio (128×), and IG
  feature ranking are **denorm-independent and unchanged**.

### NWP sensitivity (Con#1: ERA5 latency)

`scripts/06_nwp_sensitivity.py` corrupts all five ERA5 inputs (u10, v10, t2m, d2m,
blh) with zero-mean Gaussian noise (σ = fraction × per-feature std over val),
flowing into both node features and the wind-aware graph edges, averaged over 3
seeds. Baseline (noise 0) reproduces `04_evaluate.py` MTGNN exactly (8.67/12.68).

| Noise (×std) | 24h | 48h | beats persist? |
|---|---|---|---|
| 0.00 (clean) | 8.67 | 12.68 | 24h ✓, 48h ✓ |
| 0.10 | 8.69 | 12.69 | 24h ✓, 48h ✓ |
| 0.25 | 8.79 | 12.76 | 24h ✗, 48h ✓ |
| 0.50 | 9.22 | 13.03 | both ✗ |
| 1.00 | 11.21 | 14.45 | both ✗ |

Persistence (ERA5-independent): 24h = 8.70, 48h = 12.99.

**Honest reading:** the 24h edge (+0.03 µg/m³) is razor-thin and disappears at
0.25× weather-feature noise; the 48h edge (+0.31) survives to 0.50×. **Operational
implication / Con#1 rebuttal:** under imperfect NWP, MTGNN's robust value is at
**48h** (the early-warning horizon that matters). At ≤24h the system should fall
back to persistence — which the **hybrid policy already does for 6/12h and a
conservative variant could extend to 24h**. ERA5 is used only at *training* time
(no latency problem there); inference needs NWP, and we have quantified exactly how
much accuracy that costs.

---

## 2. New / changed code (reproducibility)

| File | Change |
|---|---|
| `scripts/04_evaluate.py` | **New.** Reproducible per-horizon RMSE in µg/m³ for MTGNN + A3TGCN vs persistence, + hybrid operational ensemble. Regenerates `evaluation_val2025.json`. Run: `uv run python scripts/04_evaluate.py` |
| `src/training/evaluation.py` | **New.** Shared eval helpers (`station_scalers`, `build_ground_truth`, `predict`, `denorm_pred`, `rmse_block`) used by `04_evaluate.py` and `06_nwp_sensitivity.py` so denorm/persistence/masking are identical everywhere. |
| `scripts/05_attribution.py` | Extended to compute `hotspot_frp_summary` and `hotspot_impact_ug_m3_24h` reproducibly with the correct per-station scale (were ad-hoc before); lint cleanup. |
| `scripts/06_nwp_sensitivity.py` | **New (P4).** ERA5-perturbation sensitivity (Con#1). |
| `outputs/evaluation_val2025.json` | Regenerated with correct numbers. |
| `outputs/attribution_march2024.json` | `hotspot_impact` corrected (3.49). |
| `docs/PROPOSAL_MEMO.md` | RMSE table, abstract, A3TGCN table, framing, attribution µg updated to verified numbers. |
| `docs/SESSION6_NOTES.md` | Correction banner added. |
| `configs/config.yaml` | Hydra `run.dir` → `outputs/hydra/${now}` (isolate run logs). |
| `.gitignore` | Ignore `outputs/hydra/` + `outputs/20*/` (hydra junk); result JSON stays tracked. |

### Hybrid operational ensemble (D5)

`hybrid = persistence for horizons <24h, MTGNN for >=24h`. By construction it is
**at least as accurate as persistence at every horizon** and equals MTGNN at the
long horizons. This is the recommended deployment policy and a clean pitch point.

---

## 3. Impact on the AI Camp pitch

- The submitted proposal (29 May) contains the **old, overstated** numbers and
  cannot be changed. The pitch deck/report will use the **verified** numbers.
- If asked about the discrepancy: we *found and fixed* a denormalization reporting
  bug after submission and made evaluation fully reproducible
  (`scripts/04_evaluate.py`). Frame as engineering rigor / AI governance — a plus
  for the "ธรรมาภิบาล AI" rubric dimension (weight 10).
- Do **not** lead the pitch with marginal RMSE gains. Lead with: source
  attribution (XAI, persistence cannot do this), 48h early-warning, joint
  multi-station spatial forecasting, and the hybrid operational guarantee.

---

## 4. Status of Session 7 deliverables

**Done:**
- **Phase A cleanup** — extracted shared `src/training/evaluation.py` (04 reproduces
  identical numbers); relocated Hydra run dir → `outputs/hydra/` + gitignore junk;
  fixed `05_attribution.py` C901 + nits.
- **P3 Dashboard** — Streamlit verified end-to-end with the real MTGNN checkpoint
  (model load, forward, attribution, all 5 viz fns); runs at localhost:8501,
  demo-ready. (Tab About still has `[TODO: NSC Disclaimer]`.)
- **P4 NWP sensitivity** — `scripts/06_nwp_sensitivity.py` + `outputs/nwp_sensitivity.json`;
  see §1 "NWP sensitivity".

**Pending:**
- **P5** presenter script (5 min, solo, Thai) + Q&A defense + AI Governance — in progress.
- Optional dashboard pitch panel (Model-vs-Persistence + NWP).
- Reviewer pass + commit (stage only our files).

### Lint status (whole repo)
- Our files (04, 05, 06, `evaluation.py`): ruff + black clean; **207 tests pass**.
- Pre-existing tracked debt: `01_download_all.py` (B008 ×2 — typer idiom
  false-positive, candidate for `# noqa: B008`), `03_train.py` (E501 ×1); both also
  need `black`.
- Untracked user scripts `generate_proposal.py` (101) + `generate_architecture_diagram.py`
  (47) — out of scope; do not commit, or fix later.
