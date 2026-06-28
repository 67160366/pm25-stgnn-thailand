# Proposal Writing Memo — Session 7

**For:** New session tasked with writing the NSC 2026 Category 14 proposal  
**Deadline:** 2026-05-29 (submit via SIMS: https://www.nstda.or.th/sims)  
**Branch:** `feat/session-3-era5-models`

---

## What you must produce

A PDF proposal document following NSC 2026 Category 14 format:
- Language: Thai primary, English permitted for technical terms
- Font: TH Sarabun New, 16pt
- Submit PDF via SIMS system (Category 14 = โปรแกรมเพื่องานการพยาบาล/วิทยาศาสตร์/เทคโนโลยี)

**Scoring:** Technique=25, Creativity=25, Utility=20, Clarity=15, Completeness=15 (total 100)

---

## Example proposals to read first

All 3 examples are in `ex/` folder. Open them to understand format and Thai language tone.

| File | Pages | Notes |
|---|---|---|
| `ex/LumoPack.pdf` | 9 | Best reference — AI packaging platform, Category 14, clean structure |
| `ex/Pharmasoft.pdf` | 16 | Longer version with more detail — good for tech spec sections |
| `ex/Smart_Pipeline_Proposal 1.pdf` | 10 | Another engineering project example |

**LumoPack structure (use as template):**
1. Cover page — Thai+English title, Category 14, 3 team members (name/DOB/address/phone/email/student ID), advisor name/institution
2. Abstract (Thai) + Keywords (Thai and English)
3. Background and problem statement (ที่มาและความสำคัญ)
4. Objectives (วัตถุประสงค์) — numbered list
5. Solution novelties + system architecture overview
6. Storyboard / user flow scenarios
7. Algorithm description
8. Tech stack (Frontend/Backend/DB/Libraries), Input/Output spec, Functional spec, Scope + Limitations
9. Bibliography

---

## What is still missing — ASK THE USER

Before writing, collect these from the user:

```
[ ] Team member 1: full name (Thai+English), DOB, address, phone, email, student ID, institution
[ ] Team member 2: full name (Thai+English), DOB, address, phone, email, student ID, institution  
[ ] Team member 3: full name (Thai+English), DOB, address, phone, email, student ID, institution
[ ] Advisor: full name (Thai+English), title, institution, phone, email
[ ] Institution full name (Thai+English)
[ ] Faculty/Department name
```

---

## All technical content — ready to use

### Thai title (draft)
> ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วยโครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลา สำหรับภาคเหนือของประเทศไทย

### English title (draft)
> Explainable Spatio-Temporal Graph Neural Network for PM2.5 Forecasting and Source Attribution in Northern Thailand

---

### Abstract (draft — translate/adapt to Thai)

PM2.5 haze in Northern Thailand (9 provinces) is a recurring public health crisis driven by biomass burning and transboundary transport. Existing forecasting systems rely on persistence or NWP models that cannot attribute pollution sources. We present an Explainable Spatio-Temporal Graph Neural Network (STGNN) that jointly forecasts PM2.5 at 18 Air4Thai monitoring stations (6h, 12h, 24h, 48h horizons) and attributes predictions to fire hotspot sources via Graph-based Integrated Gradients (GB-IG). Our MTGNN model achieves RMSE = 8.67 µg/m³ at 24h and 12.68 µg/m³ at 48h, beating a strong persistence baseline at the operationally critical 24h and 48h horizons (by 0.3% and 2.4% respectively). Attribution reveals that the March 2024 Chiang Mai haze peak (141–144 µg/m³) was driven by local Thai biomass burning (128× higher FRP than Myanmar sources), with fire hotspots contributing ~3.5 µg/m³ to 24h predictions. The system provides early warning capability for haze event management.

**Keywords:** PM2.5, Spatio-Temporal GNN, Source Attribution, Integrated Gradients, Northern Thailand, Haze Forecasting, FIRMS, ERA5

---

### Background / ที่มาและความสำคัญ (key facts to include)

- Northern Thailand suffers annual haze season (Jan–April), with PM2.5 frequently exceeding WHO 24h guideline (15 µg/m³) and Thailand standard (50 µg/m³)
- 9 provinces in scope: Chiang Mai, Chiang Rai, Lampang, Lamphun, Mae Hong Son, Nan, Phayao, Phrae, Tak
- March 2024 peak: Chiang Mai city measured 141–144 µg/m³ hourly PM2.5
- Primary sources: biomass burning (agricultural, forest), transboundary transport from Myanmar and Laos
- Existing tools (Thailand Air Quality Index website, PCD alerts) are reactive — no short-term forecast or fire source attribution
- GNN-based spatial forecasting outperforms persistence at 24h+ horizons (cite MTGNN paper, PM2.5-GNN paper)

---

### Objectives / วัตถุประสงค์

1. พัฒนาระบบพยากรณ์ PM2.5 รายชั่วโมงล่วงหน้า 6, 12, 24, 48 ชั่วโมง สำหรับ 18 สถานีตรวจวัดในภาคเหนือ
2. บูรณาการข้อมูลจุดความร้อน (FIRMS) เป็น node พิเศษในกราฟเพื่อจำลองการลำเลียงฝุ่นจากแหล่งเผาไหม้
3. พัฒนาโมดูล XAI (Graph-based Integrated Gradients) เพื่อระบุสัดส่วนผลกระทบจากแหล่งกำเนิดไฟ (ไทย/เมียนมา/ลาว)
4. ตรวจสอบความถูกต้องของระบบบน validation split ปี 2025 และเปรียบเทียบกับ persistence baseline
5. จัดทำ dashboard แสดงผลพยากรณ์และ attribution แบบ real-time สำหรับหน่วยงานที่เกี่ยวข้อง

---

### Three Novelties (ความแปลกใหม่ / จุดเด่น)

1. **Wind-aware dynamic graph adjacency** — Edge weights between monitoring stations updated hourly using ERA5 wind vectors (u10, v10) to capture transboundary transport direction. Three edge types: (a) geographic proximity, (b) wind-aligned corridor, (c) hotspot-to-station bipartite
2. **FIRMS fire hotspot clusters as first-class graph nodes** — NASA FIRMS daily active fire data (VIIRS/MODIS) is spatially clustered and added as auxiliary nodes. Each hotspot node encodes total FRP, centroid lat/lon. The bipartite graph channel (type_c) connects hotspot nodes to downwind station nodes.
3. **Graph-based Integrated Gradients (GB-IG) for source attribution** — Post-hoc XAI method that computes feature-level attribution across graph structure. Enables quantifying the contribution of each fire source country to each station's prediction.

---

### Architecture (for diagram and description)

```
Input: 18 station nodes × 10 features × T=24 hours history
       + M hotspot cluster nodes × 3 features (FRP, lat, lon)

Three edge channels:
  type_a: k-NN geographic proximity (k=5, fixed)
  type_b: wind-aligned corridor (cosine wind similarity, hourly update)
  type_c: hotspot bipartite (hotspot → station if downwind within 300km)

Model: MTGNN (Wu et al. KDD 2020)
  - Adaptive adjacency learning (learnable node embeddings)
  - 3 layers × TCN (temporal) + GCN (spatial)
  - hidden_dim=64, dropout=0.3, weight_decay=1e-4
  - Output: 18 stations × 4 horizons (6h/12h/24h/48h)

XAI: Graph-based Integrated Gradients
  - Baseline: zero input (no PM2.5, no ERA5, no hotspots)
  - 50 interpolation steps
  - Attribution aggregated by: country (Thailand/Myanmar/Laos), feature type (weather/fire/lag)
```

---

### Results (use these exact numbers)

**Model: MTGNN — 252,588 parameters, best epoch 15**

| Horizon | Persistence RMSE | MTGNN RMSE | Improvement |
|---|---|---|---|
| 6h | 2.96 µg/m³ | 4.31 µg/m³ | -45.6% (behind) |
| 12h | 5.19 µg/m³ | 5.80 µg/m³ | -11.8% (behind) |
| **24h** | **8.70 µg/m³** | **8.67 µg/m³** | **+0.3% (beats)** |
| **48h** | **12.99 µg/m³** | **12.68 µg/m³** | **+2.4% (beats)** |

Validation split: 2025-01-01 to 2025-12-31, 18 stations, 3,637 samples.  
Source: `outputs/evaluation_val2025.json` (reproducible via `scripts/04_evaluate.py`).  
Note: persistence uses last-observed carry-forward; all methods scored on identical valid positions. Earlier drafts overstated these µg/m³ figures due to a denormalization bug (now fixed); the normalized metrics and model ranking are unchanged.

**Framing:** Short-horizon (6h/12h) performance behind persistence is expected in PM2.5 forecasting because autocorrelation dominates short-range — consistent with published PM2.5-GNN literature (Wang et al. 2020). The 48h improvement (+2.4%) is the most actionable for public health early warning of haze events. The primary value of the model over persistence is not marginal point-RMSE but (a) source attribution, which persistence cannot provide, (b) joint multi-station spatial forecasting, and (c) the operational hybrid policy that is at least as accurate as persistence at every horizon.

**Baseline comparison: A3TGCN (27,164 parameters, best epoch 97)**

| Horizon | A3TGCN RMSE |
|---|---|
| 6h | 6.56 µg/m³ |
| 12h | 7.48 µg/m³ |
| 24h | 9.63 µg/m³ |
| 48h | 13.18 µg/m³ |

MTGNN is 7.1% more accurate at 24h (normalized RMSE 0.4576 vs 0.4928; ~10% in µg/m³) with 9.3× more parameters and converges 6.5× faster. A3TGCN does not beat persistence at any horizon in µg/m³.

---

### Attribution Finding (Section: XAI Results)

**Event:** March 2024 PM2.5 peak at Chiang Mai (Top-10 hourly peaks: 141–144 µg/m³)

**Country attribution:** Thailand = 100%, Myanmar = 0%

**Why this is correct (not a bug):**
- Each peak sample has 23 hotspot clusters: **22 Thai, 1 Myanmar**
- Thai total FRP = 109,564 vs Myanmar total FRP = 854 (**128× ratio**)
- Myanmar delta (zeroing all Myanmar hotspot nodes) ≈ 0.000 normalized units for all 10 samples

**Hotspot impact on prediction:** Fire hotspot channel adds **mean 3.49 µg/m³** (range 2.5–5.6 µg/m³) above no-fire baseline to 24h predictions.

**Feature importance (Integrated Gradients, mean |attribution|):**

| Feature | Mean |attr| |
|---|---|
| d2m (dewpoint temperature) | 0.0345 |
| t2m (air temperature) | 0.0344 |
| pm25_scaled (lagged PM2.5) | 0.0140 |
| blh, u10, v10, hour_sin/cos, doy_sin/cos | <<0.001 |

Interpretation: Meteorological conditions (humidity/temperature) dominate feature attribution. Fire hotspots contribute via the graph structure (type_c edges) rather than node features.

Source: `outputs/attribution_march2024.json`

---

### NSC Proposal Narrative (XAI section)

> "ระบบ XAI ของเราสามารถระบุได้ว่าเหตุการณ์หมอกควันสูงสุดในเดือนมีนาคม 2024 ที่เชียงใหม่ (141–144 µg/m³) เกิดจากการเผาไหม้ในประเทศไทยเป็นหลัก ไม่ใช่แหล่งกำเนิดข้ามพรมแดน โดยพบว่าค่า FRP ของไฟในไทยสูงกว่าเมียนมา 128 เท่า ผลลัพธ์นี้สอดคล้องกับข้อมูลดาวเทียม FIRMS และให้ข้อมูลที่มีประโยชน์สำหรับการตัดสินใจเชิงนโยบาย"

---

### Tech Stack (for technical spec section)

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| GNN Framework | PyTorch 2.2, PyG 2.5, PyG Temporal |
| Model | MTGNN (Wu et al. KDD 2020) |
| XAI | Graph-based Integrated Gradients |
| Config | Hydra |
| Experiment tracking | Weights & Biases |
| Dashboard | Streamlit |
| PM2.5 data | Air4Thai (18 stations) |
| Fire data | NASA FIRMS VIIRS/MODIS |
| Weather | ERA5 via CDS API (u10, v10, t2m, d2m, blh) |
| Normalization | RobustScaler per station |

---

### Input / Output Specification

**Input:**
- 24 hours of PM2.5 observations from 18 stations (pm25_scaled)
- ERA5 meteorological fields: u10, v10, t2m, d2m, blh (hourly, ERA5 grid interpolated to stations)
- NASA FIRMS fire hotspot clusters: total FRP, centroid lat/lon (daily)
- Time encodings: hour_sin/cos, doy_sin/cos

**Output:**
- PM2.5 forecast for each of 18 stations at 4 horizons: 6h, 12h, 24h, 48h
- Source attribution map: per-country contribution (%) to each station's 24h prediction
- Feature importance ranking (IG attribution by feature type)

---

### Scope and Limitations

**Scope:**
- 9 Northern Thai provinces, 18 Air4Thai stations
- Bounding box: lon 97.0–101.5, lat 16.0–21.0
- Forecast horizon: 6–48 hours, hourly resolution
- Historical validation: 2022–2024 training, 2025 validation

**Limitations:**
- PM2.5 data from Air4Thai only (no PCD stations in 2022–2024)
- ERA5 is reanalysis (5–7 day latency); an operational system must use NWP forecast
  input at inference — quantified below
- Model does not account for dust transport from China/India
- Attribution at country level only (not sub-national regions)

---

### ERA5 Latency & NWP Sensitivity (addressing reviewer Con#1)

ERA5 reanalysis has a 5–7 day latency, so a production system must substitute an
imperfect NWP forecast for the weather inputs at inference. We **quantified** this
cost: `scripts/06_nwp_sensitivity.py` corrupts all five ERA5 inputs (u10, v10, t2m,
d2m, blh) with zero-mean Gaussian noise (σ = fraction × per-feature std over the
2025 validation period), propagating into both node features and the wind-aware
graph edges, averaged over 3 seeds.

| Noise (× natural std) | MTGNN 24h | MTGNN 48h |
|---|---|---|
| 0.00 (clean ERA5) | 8.67 | 12.68 |
| 0.25 | 8.79 | 12.76 |
| 0.50 | 9.22 | 13.03 |
| 1.00 | 11.21 | 14.45 |

Persistence (weather-independent): 24h = 8.70, 48h = 12.99.

**Finding:** the 48h advantage is the more robust one (survives weather-input error
up to 0.5× natural variability), while the 24h margin is marginal and erased by
modest NWP error. **Mitigation / design response:** (1) the operational system
relies on the model at the 48h early-warning horizon and falls back to persistence
at short horizons (the hybrid policy); (2) deployment should use the best available
NWP (e.g. ECMWF HRES / GFS), keeping weather-feature error well below natural
variability; (3) ERA5 latency affects *inference only* — training on reanalysis is
unaffected. Source: `outputs/nwp_sensitivity.json`.

---

### Bibliography (for references section)

1. Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2020). Connecting the dots: Multivariate time series forecasting with graph neural networks. *KDD 2020*. (MTGNN)
2. Wang, Y., Li, Y., Song, Y., & Rong, X. (2020). PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting. *SIGSPATIAL 2020*. arXiv:2002.12898
3. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *ICLR 2018*.
4. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017*. (Integrated Gradients)
5. NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
6. Copernicus ERA5 Reanalysis: https://doi.org/10.24381/cds.adbb2d47

---

## Key source files to reference

| What | Where |
|---|---|
| Evaluation results | `outputs/evaluation_val2025.json` |
| Attribution results | `outputs/attribution_march2024.json` |
| Model architecture | `src/models/mtgnn.py`, `src/models/a3tgcn.py` |
| Graph builder | `src/data/graph_builder.py` |
| XAI implementation | `src/explain/gb_ig.py` |
| Data pipeline | `src/data/loader.py`, `src/data/preprocessing.py` |
| Design rationale | `docs/DESIGN.md` |
| Session 6 findings | `docs/SESSION6_NOTES.md` |
| Example proposals | `ex/LumoPack.pdf`, `ex/Pharmasoft.pdf`, `ex/Smart_Pipeline_Proposal 1.pdf` |

---

## Checklist before submitting

- [ ] Collect team member info (names, IDs, DOB, contact) from user
- [ ] Fill cover page
- [ ] Write abstract in Thai (use draft above as base)
- [ ] Translate objectives to Thai
- [ ] Include architecture diagram (can sketch or use ASCII → convert)
- [ ] Include results table (copy from above)
- [ ] Add attribution finding narrative
- [ ] Fill bibliography
- [ ] Add NSC Disclaimer text (Thai+English) — get final wording from NSC booklet page 44
- [ ] Export to PDF, check font is TH Sarabun New 16pt
- [ ] Upload to SIMS before 2026-05-29
