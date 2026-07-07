# PART E — OUTPUT & RESULTS

> อ้างอิงโค้ด/ผลจริง: `src/models/mtgnn.py`, `src/training/metrics.py`,
> `outputs/*.json`, `docs/CRITICAL_REVIEW.md`, `docs/SESSION{7,8}_NOTES.md`

## E1. Model Output

**model predict อะไร:** ค่า PM2.5 (สเกล normalized) ล่วงหน้าทุก horizon พร้อมกัน ต่อทุกสถานี

**output shape** (`mtgnn.py:304`):
```
forward(data) → predictions (B*N, H)   # B=batch, N=18 สถานี, H=4 horizon (6,12,24,48h)
```
- input: `data['station'].x` = `(B*N, T_in, F)` = `(B·18, 24, 10)`
- head เป็น 2-layer MLP → `n_horizons=4` outputs; **clip ≥ 0** (ฝุ่นติดลบไม่ได้)
- ค่า output อยู่บนสเกล `pm25_scaled` → ตอนรายงานต้อง **denormalize กลับด้วย per-station IQR**
  (จุดที่เคยพลาด — ดู E2 / Bug denorm)

**metric ที่ใช้** (`metrics.py` `compute_metrics`): per-horizon **RMSE, MAE, MAPE** คำนวณเฉพาะ
ตำแหน่ง valid (ใช้ mask):

| metric | ความหมาย | หมายเหตุในโค้ด |
|---|---|---|
| RMSE | รากที่สองของ error กำลังสองเฉลี่ย — ลงโทษ error ใหญ่หนัก | metric หลักที่รายงาน |
| MAE | ค่า error สัมบูรณ์เฉลี่ย — ตีความเป็น µg/m³ ตรงๆ | |
| MAPE | error เป็น % ของค่าจริง | **ข้ามจุดที่ \|target\| < 1 µg/m³** กันหารใกล้ศูนย์ |

**ผลลัพธ์ตอนนี้** (held-out test 2025, `evaluation_test2025.json`, 3,637 samples, retrain 3-way split):

| RMSE (µg/m³) | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| **Persistence** (ค่าชั่วโมงล่าสุด) | **2.96** | **5.19** | 8.70 | 12.99 |
| **MTGNN** | 4.52 | 5.93 | 8.76 | **12.66** |
| A3TGCN | 6.93 | 7.86 | 9.92 | 13.34 |
| Hybrid (persist <24h, MTGNN ≥24h) | 2.96 | 5.19 | 8.76 | **12.66** |

**อ่านผลอย่างตรงไปตรงมา:**
- MTGNN **แพ้ persistence ชัดเจนที่ 6h/12h** (แย่กว่า ~53%/14%)
- ที่ **24h แทบเท่า/แย่กว่านิดเดียว** (8.76 vs 8.70) — **ดีกว่าจริงเฉพาะ 48h** (12.66 vs 12.99)
- A3TGCN แย่กว่า MTGNN ทุก horizon (baseline ที่เบากว่า ใช้แค่ Type A — PART D)
- ทางออกที่ซื่อสัตย์ = **Hybrid**: ใช้ persistence สั้น + MTGNN ยาว → เอาส่วนดีของแต่ละตัว

**นัยสำคัญเชิงสถิติ** (`significance_test2025.json`, block bootstrap by day, B=2000):
- 6h: `prob_model_better = 0.0`, `significant_at_95 = false` → MTGNN แย่กว่าอย่างมีนัยสำคัญ
- horizon ยาวที่ "ดีกว่า" ก็ **ไม่ผ่านนัยสำคัญ 95%** เช่นกัน → gain ไม่ robust

**ผลลม/ไฟช่วยไหม?** `ablation_multiseed.json` (retrain 3 seeds): variant `full` vs `no_type_b` /
`no_type_c` ต่างกันภายในช่วง std ของกันและกัน → **ตัด edge ลม/ไฟออกแทบไม่เปลี่ยนความแม่นยำ**
และ `nwp_sensitivity.json` แสดงว่าใส่ noise ลง ERA5 ก็แทบไม่ขยับ RMSE → กราฟไม่ได้ช่วย accuracy จริง

## E2. Honest Thesis

**thesis จริงของโปรเจค** (จาก `CRITICAL_REVIEW.md` §Bottom line):
> เป็นผลงาน **advanced technique + real problem + reproducible eval + ซื่อสัตย์เรื่อง bug**
> → *แข็งแรงในฐานะงาน NSC*; แต่ **"gain การพยากรณ์เหนือ baseline ง่ายๆ = marginal และหายไป
> ภายใต้เงื่อนไขจริง"** → *อ่อนในฐานะ "ระบบพยากรณ์ที่ดีกว่าเดิม"*

**"forecast edge marginal" แปลว่า:** ส่วนที่โมเดลชนะ baseline (persistence) **บางมาก** — แค่
เศษเสี้ยว µg/m³ ที่ 48h และ **ไม่ผ่านนัยสำคัญทางสถิติ** (CI คร่อม 0) ไม่ใช่การพยากรณ์ที่ดีขึ้นจริง

**"graph does not robustly help accuracy" แปลว่า:** โครงสร้างกราฟ (Type B ลม + Type C ไฟ) ซึ่งเป็น
**จุดขายทางเทคนิค กลับพิสูจน์ไม่ได้ว่าช่วยความแม่นยำ** — ablation ตัดออกแล้ว RMSE แทบไม่เปลี่ยน,
เติม noise ลม/ไฟก็ไม่กระทบ → "robustly" = ไม่มั่นคง/ไม่ซ้ำได้ในเชิงประโยชน์ต่อ accuracy

**เหตุการณ์สำคัญ — denorm bug (Session 7):** ตัวเลข µg/m³ ในข้อเสนอเดิม (และ SESSION6) **สูงเกินจริง**
เพราะสคริปต์ ad-hoc denormalize ด้วยสเกลคงที่ผิด (~22) แทน per-station IQR จริงจาก `scalers.json`
→ Session 7 แก้ให้ eval ทุกตัวใช้ `src/training/evaluation.py` ตัวเดียว (denorm/mask/persistence
เหมือนกันหมด) เลข µg/m³ จึงตรงกันทุกไฟล์ — **ตัวโมเดลไม่ผิด ผิดที่ขั้นรายงาน**

**แล้ว value จริงอยู่ที่ไหน?**
= **Explainable transboundary source attribution** (ไม่ใช่ตัวเลขพยากรณ์)
โปรเจคทำ pipeline ที่ตอบได้ว่า "ฝุ่นที่สถานีชายแดนมาจากไฟประเทศไหน" ด้วย GB-IG + occlusion
เช่น `transboundary_attr_test.json` เหตุการณ์ 2025-03-25 ที่แม่ฮ่องสอน:
```
peak PM2.5 = 126.7 µg/m³
connected FRP: Thailand 1305.9, Myanmar 478.2 (foreign fraction 0.268)
model country_attribution: Myanmar 1.0   → ระบุฝุ่นข้ามแดนจากพม่าได้
```
+ ผลลบที่ยืนยันได้ + eval ทำซ้ำได้ + pipeline หลายแหล่ง = สิ่งที่ป้องกันได้จริงในเวที NSC

## E3. Output Files

โครงสร้าง `outputs/`:

| ไฟล์ | คืออะไร | สร้างโดย |
|---|---|---|
| `evaluation_{val,test}2025.json` | RMSE per-horizon vs persistence | `scripts/04_evaluate.py` |
| `significance_{val,test}2025.json` | block-bootstrap CI/นัยสำคัญ | `scripts/07_significance.py` |
| `ablation_{multiseed,retrained,inference}.json` | ตัด Type B/C ดูผล | ablation scripts (S8) |
| `nwp_sensitivity.json` | ใส่ noise ERA5 ดูความไว | S8 |
| `baseline_ml{,_test}.json` | baseline non-graph (ML ธรรมดา) | S8 |
| `attribution_march2024*.json` | source attribution เคส Chiang Mai | `scripts/` explain |
| `transboundary_attr_*.json` | attribution ชายแดน (พม่า/ลาว) | explain |
| `best_model.pt` | checkpoint โมเดล | trainer |
| `figures/`, `pitch/` | รูปผล + pitch deck | generate scripts |
| `hydra/` | config snapshot ต่อ run | Hydra (อัตโนมัติ) |
| โฟลเดอร์ timestamp `2026-05-*` | run logs รายครั้ง | Hydra output dir |

**`NSC2026_Final_Report.docx` สร้างยังไง / ทำไมไม่ track git:**
- สร้างโดย `scripts/generate_report.py` (Session 10) — python-docx, เนื้อหา transcribe verbatim
  จาก `docs/REPORT_DRAFT.md`, ฟอนต์ TH Sarabun New 16pt, มี TOC/page-number field, disclaimer จริง
- **เป็น artifact ที่ generate ได้ใหม่เสมอจากซอร์ส** (draft + figures) → ไม่ track ใน git ตามหลัก
  "ไม่ commit ของ generated"; แหล่งความจริงคือ `REPORT_DRAFT.md` + สคริปต์ ไม่ใช่ไฟล์ .docx

> ⚠️ **ยึด `outputs/*.json` + `CRITICAL_REVIEW.md` + `SESSION{7,8}_NOTES.md` เป็นความจริงของผล**
> `DESIGN.md` และ SESSION6 มีตัวเลขก่อนแก้ denorm bug — ล้าสมัย

---

## สรุป PART E

- output = `(B·18, 4)` ค่า PM2.5 normalized (clip ≥0) → denorm ด้วย per-station IQR; metric = RMSE/MAE/MAPE
- ผลจริง: MTGNN **แพ้ persistence** ที่ 6h/12h, 24h แทบเท่า, **ดีกว่าเฉพาะ 48h** และ **ไม่ผ่านนัยสำคัญ**; A3TGCN แย่สุด
- กราฟ (ลม/ไฟ) **ไม่ช่วย accuracy อย่างมั่นคง** (ablation + noise ยืนยัน)
- honest thesis: value อยู่ที่ **explainable transboundary attribution** + ความซื่อสัตย์ + reproducibility
  ไม่ใช่ headline พยากรณ์
- `NSC2026_Final_Report.docx` = generated จาก `REPORT_DRAFT.md` จึงไม่ track git

> **ต้องการลงลึก PART ไหน หรือไปต่อ PART F (Project Configuration)?**
