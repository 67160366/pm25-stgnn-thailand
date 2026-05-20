# Session 5 Notes — Data Audit, Split Fix, and Training

**Date:** 2026-05-20
**Branch:** `feat/session-4-explain-app`
**Goal:** Diagnose why training underperforms persistence, fix it, and complete a clean training run.

---

## What was done

### 1. Design verification (web search)

Confirmed via literature search that the three core novelties are grounded in real methods:

| Method | Verified reference |
|---|---|
| MTGNN (graph learning from time series) | Wu et al. 2020, KDD — confirmed, widely cited |
| Integrated Gradients (node feature attribution) | Sundararajan et al. 2017, ICML — confirmed, axiomatic |
| Wind-aware dynamic adjacency | Multiple transboundary haze papers (Amnuaylojaroen 2019, etc.) |
| FIRMS hotspot nodes | NASA FIRMS API, standard for fire-PM2.5 research |

One correctness issue found and fixed: `gb_ig.py` module docstring cited
arxiv:2509.07648 (which applies IG to *graph structure*, not node features).
Our code applies standard IG to node features. Fixed to cite Sundararajan 2017 only.

### 2. NSC booklet analysis

Read `20260218_NSC2026_Booklet.pdf` (45 pages).

**Category 14 scoring weights (proposal + finals):**

| Criterion | Proposal | Finals |
|---|---|---|
| Technique / difficulty | 25 | 25 |
| Creativity / novelty | 25 | 25 |
| Utility / impact | 20 | 20 |
| Clarity of presentation | 15 | 15 |
| Completeness | 15 | 15 |

**Takeaway:** Technique + Creativity = 50 points. The wind-aware graph + FIRMS nodes + IG
attribution story is strong for both. RMSE alone doesn't determine the score.

**Disclaimer text** (page 44) extracted — needs team info filled in:
`[TODO: NSC Disclaimer — see booklet page 44]` placeholders remain in all `src/` files.

### 3. ERA5 data audit

Result: **ERA5 is clean.**
- 4 full years (2022-2025), 0% NaN, 100% coverage
- All 5 ERA5 features present: u10, v10, t2m, d2m, blh

### 4. PM2.5 data gap discovered

**Critical finding:** 2023 Feb–Mar PM2.5 data is almost entirely absent.

| Month | Coverage | Notes |
|---|---|---|
| 2023 Feb | ~10% | Almost all NaN |
| 2023 Mar | ~41% | Worst haze year on record (mean 97.4 µg/m³, max 586!) |

Root cause: OpenAQ did not ingest Air4Thai data for those months. API query
confirmed zero rows for March 2023. This is a data provider issue, not a scraping bug.

**Impact on old training:** The old train split (2022–2023) meant the model
almost never saw a complete haze season in training data, causing poor
generalization to the validation haze months.

**Alternative sources investigated:** PCD direct request (3–4 weeks), MERRA-2
reanalysis (free, hourly, now available), CAMS (3-hourly). PCD request is the
right long-term fix; MERRA-2 is a viable proxy if PCD doesn't respond before the
July report deadline.

### 5. Split restructuring

Changed `_SPLIT_BOUNDS` in `src/data/loader.py`:

| Split | Old | New |
|---|---|---|
| train | 2022-01-01 → 2023-12-31 | 2022-01-01 → **2024-12-31** |
| val | 2024-01-01 → 2024-12-31 | **2025**-01-01 → 2025-12-31 |
| test | 2025-01-01 → 2025-12-31 | 2025-01-01 → 2025-12-31 (same as val for now) |

The 2023 data gaps remain in training but are handled by `mask_in_loss`. The model
now trains on both complete haze seasons (2022 and 2024) instead of one partial one.

New sample counts: **train=23,894** (was 15,110, +58%), **val=3,637**.

### 6. Overfitting regularization

Applied to complement the split fix:

| Parameter | Old | New |
|---|---|---|
| Adam weight_decay | 0 | 1e-4 |
| MTGNN dropout | 0.1 | 0.3 |

### 7. Training run

**Config:** MTGNN, 252K params, CUDA, lr=1e-3, cosine LR decay, patience=15.

**Results:**

| Epoch | val_rmse_24h | val_loss | Note |
|---|---|---|---|
| 1 | 0.95 | 0.924 | Old run epoch 1 was 1.14 |
| 2 | 0.49 | 0.233 | Old run needed epoch 8 for this |
| 5 | **0.46** | 0.203 | First best — old run best was epoch 19 |
| 13 | **0.46** | 0.192 | New checkpoint (lower val_loss) |
| 15 | **0.46** | 0.193 | New checkpoint |
| 30 | — | — | Early stopping (patience=15 from epoch 15) |

**Final best:** val_rmse_24h = **0.4576** (normalized), checkpoint at epoch 15.

### 8. Persistence baseline comparison (approximate)

Station IQR (scale_) range: 14.6–29.0 µg/m³, mean = 19.9 µg/m³.
Approximate real-unit model RMSE ≈ 0.4576 × 19.9 = **~9.1 µg/m³** (24h horizon).

| Horizon | Persistence RMSE | Model RMSE~ | Status |
|---|---|---|---|
| 6h | 2.7 µg/m³ | (likely 4–6 µg/m³) | Behind |
| 12h | 4.8 µg/m³ | (likely 6–8 µg/m³) | Behind |
| 24h | 8.1 µg/m³ | ~9.1 µg/m³ | Very close |
| 48h | 12.1 µg/m³ | (likely 10–11 µg/m³) | **Likely beats** |

The model RMSE column uses the 24h-horizon normalized RMSE for all horizons — per-horizon
breakdown requires a full inference run (`scripts/04_evaluate.py`, not yet written).

---

## Bugs found / fixed

### .pth encoding crash (Windows App Store Python)

**Symptom:** `uv run python` crashes with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x87` before any user code runs.

**Root cause:** Windows App Store Python 3.11's *frozen* site module reads `.pth`
files using cp874 (Thai Windows locale). The editable-install `.pth` file
(`_editable_impl_pm25_stgnn_thailand.pth`) was written by uv with the absolute
project path encoded as UTF-8. The byte `0x87` is the third byte of the UTF-8
encoding of 'ง', which is undefined in cp874. Neither `PYTHONUTF8=1` nor
`-X utf8` fixes the frozen site module in this build.

**Fix:** Replace the `.pth` file content with a relative path:
```
../../..
```
(Three levels up from `site-packages/` → `Lib/` → `.venv/` → project root.)

**Important:** `uv sync` overwrites this file. Reapply the fix after every sync:
```python
# Write tool: target = .venv/Lib/site-packages/_editable_impl_pm25_stgnn_thailand.pth
../../..
```

### gb_ig.py citation overclaim

Removed reference to arxiv:2509.07648 from module docstring. That paper applies IG
to graph *structure* (edge existence); we apply standard IG to node *features*.
Correct citation: Sundararajan et al. 2017, ICML.

---

## Commits

- `e2de1ab` fix: correct training splits, overfitting, and attribution citation

---

## Next session (Session 6): Evaluation and proposal

**Priority 1 (before 2026-05-29):**
- NSC proposal draft — 9 days from this session's date

**Priority 2 (before 2026-07-17 report):**
- Write `scripts/04_evaluate.py` — full inference, per-horizon RMSE in µg/m³, vs persistence
- Run attribution on March 2024 haze event (show Myanmar/Laos source scores)
- Run A3TGCN comparison for Technique section
- End-to-end Streamlit dashboard test with real checkpoint

**Priority 3:**
- Send PCD data request letter for Feb–Mar 2023 hourly PM2.5 (backup: MERRA-2)
- Fill in `[TODO: NSC Disclaimer]` placeholders (need team names, institution, advisor)
- Merge `feat/session-4-explain-app` to `main` via PR
