# Critical Review — PM2.5 STGNN (Session 7, 2026-06-28)

A deliberately unflattering, evidence-based review of the whole project's results, for
internal use and to drive the Session 8 work plan. Written from a skeptical senior-reviewer
stance: separate "good NSC competition entry" from "demonstrably solves the problem".

## Bottom line

- As an **NSC university-level entry** (advanced technique on a real, important problem;
  real multi-source pipeline; STGNN + XAI; reproducible eval; honest about a denorm bug):
  **solid and defensible.**
- As a claim of **"a system that improves PM2.5 forecasting / provides validated,
  deployable source attribution for policy":** **not supported by current evidence.**
  Forecast gains over a strong-but-simple baseline are marginal and vanish under realistic
  (NWP) inputs; the attribution flagship is unvalidated and possibly near-tautological; the
  three novelties have no ablation proving they help.

## Evidence gathered (verified this session)

### Verified val-2025 RMSE (µg/m³), all on identical mask
| Method | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| Persistence | 2.96 | 5.19 | 8.70 | 12.99 |
| Same-hour-yesterday (valid h≤24 only) | 8.70 | 8.70 | 8.70 | n/a* |
| Climatology (train station×month×hour mean) | 17.89 | 17.89 | 17.90 | 17.92 |
| **MTGNN** | 4.31 | 5.80 | **8.67** | **12.68** |
| A3TGCN | 6.56 | 7.48 | 9.63 | 13.18 |

\* The same-hour-yesterday value at 48h (8.71) computed during review used look-ahead
(anchor+24, which is future relative to the forecast origin) — **invalid, discarded**. A
valid diurnal baseline at 48h collapses to persistence.

**Reading:** Climatology is far worse (17.9), so persistence is a *fair, strong* baseline
(not a strawman) and there is genuine signal beyond seasonal means. But MTGNN beats the best
simple baseline by only **+0.3% @24h (noise-level)** and **+2.4% @48h**, and **loses** at
6h/12h. IG feature importance: d2m 0.0345, t2m 0.0344, pm25_scaled 0.0140; all wind/time
features ≤0.0005 (station-level wind ≈ 0.00005).

## Findings by severity

### HIGH
1. **No held-out test set.** `src/data/loader.py` `_SPLIT_BOUNDS`: `val` and `test` are
   the identical 2025 window. Model selection (best epoch 15) and reported numbers are on
   the same data → optimistic bias.
2. **Forecast benefit is marginal.** Beats the strongest simple baseline by +0.3%/+2.4%
   (24h/48h), loses at 6h/12h. No significance/CI reported; +0.3% is almost certainly noise.
3. **NWP fragility.** `outputs/nwp_sensitivity.json`: the 24h edge disappears at 0.25×
   feature-noise, 48h at 0.5×. Production uses NWP (not ERA5 reanalysis), so the edge likely
   vanishes in deployment.
4. **No ablation of the 3 novelties.** No experiment shows type_b (wind), type_c (hotspot),
   or adaptive adjacency improves accuracy. IG says station-level wind features ≈ 0. The
   novelties — the project's claimed contribution — are unproven.
5. **Attribution flagship is fragile / possibly tautological.** (a) IG shows predictions are
   driven by temperature/humidity + autocorrelation, not the fire/wind graph; (b) occluding
   Myanmar (1 small-FRP node) → ≈0 is expected from a tiny input, not a learned insight, and
   risks merely echoing the 128× FRP input ratio; (c) hotspot impact (3.49 µg/m³) is measured
   only on 10 peak samples, not val-wide.
6. **Transboundary capability never demonstrated.** The motivating use case (Myanmar/Laos
   haze) — the only case study is Thailand-100%. No event shows cross-border attribution.
7. **No non-graph ML baseline.** Only persistence is compared. Nothing (AR/GBM/MLP/LSTM on
   the same features) justifies the *graph* specifically.

### MEDIUM
8. **Attribution computed on train data** (March 2024 ∈ train) → explains memorized data.
9. **Single-year val, no CV, no significance.**
10. **Hybrid ensemble is near-vacuous** — "use persistence when the model is worse"; mostly
    defaults to persistence and demonstrates little model value.

## Fair context (steelman)
- NSC criteria reward technique, social relevance, depth, feasibility — not SOTA deployment.
- Beating persistence on short-horizon PM2.5 is genuinely hard; marginal gains are normal in
  the literature.
- Climatology being far worse confirms real signal is captured.
- The denorm-bug catch + reproducible eval is engineering maturity above typical student level.

## The one question the project must answer
> Strip the framing: the model beats a simple baseline by ~2% at one horizon (gone under real
> inputs), and its own XAI says it leans on temperature + autocorrelation, not the fire/wind
> graph that is the novelty. **What is the demonstrated, validated contribution — can you
> prove the graph/fire/wind machinery earns its complexity (ablation) and that the attribution
> reflects reality (validation)?**

## What would make the work credible (→ Session 8 plan, see SESSION8_KICKOFF.md)
1. A true held-out test set (3-way temporal split) + re-report.
2. Ablation of each novelty (retrain variants) — the single highest-value experiment.
3. A non-graph ML baseline to justify the graph.
4. Attribution validation: a transboundary event; a tautology/sensitivity test on FRP;
   significance/CI on the 48h gain; attribution on held-out data.
