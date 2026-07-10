# Model Card — PM2.5 STGNN Thailand

NSC 2026 หมวด 14 · Explainable Spatio-Temporal GNN สำหรับพยากรณ์ PM2.5 และระบุแหล่งกำเนิด
ในภาคเหนือของประเทศไทย (ที่มา: `docs/DESIGN.md`, `docs/REPORT_DRAFT.md`)

---

## 1. รายละเอียดโมเดล (Model details)

| หัวข้อ | รายละเอียด | ที่มา |
|---|---|---|
| สถาปัตยกรรมหลัก | **MTGNN** (Wu et al., 2020, KDD) — graph learning layer (adaptive adjacency) + temporal convolution (TCN) + graph convolution (GCN); `hidden_dim=64`, 3 layers, `kernel_size=7`, `dropout=0.3` | `docs/REPORT_DRAFT.md` §4.2(3) |
| โมเดลเปรียบเทียบ | **A3TGCN** (Attention Temporal GCN — GCN + GRU + temporal attention), baseline ที่ไม่ใช้ external graph edges 3 ชนิด | `docs/REPORT_DRAFT.md` §4.2, §4.3 |
| Baseline ที่ไม่ใช้กราฟ | scikit-learn `HistGradientBoostingRegressor` ต่อขอบฟ้า (flattened 24h × 10-feature window) | `outputs/baseline_ml_test.json` |
| Baseline อ้างอิง | Persistence (carry-forward ค่าล่าสุดที่สังเกต) | `outputs/evaluation_test2025.json` |
| Framework | PyTorch 2.6.0, PyTorch Geometric ≥2.6, PyTorch Geometric Temporal ≥0.54 | `pyproject.toml`, `install_native_deps.ps1` |
| จำนวนพารามิเตอร์ | MTGNN 252,588 (best epoch 15) · A3TGCN 27,164 (best epoch 97) | `outputs/evaluation_val2025.json` |
| Input features (10 ต่อสถานีต่อ timestep) | `pm25_scaled` (RobustScaler ต่อสถานี), `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`, `u10`, `v10`, `t2m`, `d2m`, `blh` (ERA5) | `docs/DESIGN.md` §6.1, `docs/REPORT_DRAFT.md` §4.4 |
| Input window / Output | ย้อนหลัง 24 ชั่วโมง → พยากรณ์พร้อมกันที่ 18 สถานี, ขอบฟ้า 6/12/24/48 ชม. | `docs/REPORT_DRAFT.md` §4.2(3), §4.4 |
| กราฟ | Heterogeneous: 18 station nodes + M hotspot nodes, 3 ชนิดเส้นเชื่อม (type_a static-spatial, type_b wind-aware, type_c fire-bipartite) + adaptive adjacency ที่เรียนรู้ได้ | `docs/DESIGN.md` §5.2 |
| โมดูล XAI | Graph-based Integrated Gradients (GB-IG, 50 interpolation steps, zero baseline) + occlusion แยกตามประเทศไฟ (ไทย/เมียนมา/ลาว) | `docs/REPORT_DRAFT.md` §4.2(4) |
| เวอร์ชัน | v1 (checkpoint หลัก: `checkpoints/mtgnn/best_model.pt`; checkpoint held-out test: `checkpoints_split2/mtgnn`) | `outputs/evaluation_val2025.json`, `outputs/evaluation_test2025.json` |

## 2. วัตถุประสงค์การใช้งานและผู้ใช้ (Intended use + users)

- **วัตถุประสงค์:** พยากรณ์ PM2.5 ล่วงหน้า 6/12/24/48 ชั่วโมงที่ 18 สถานี Air4Thai ใน 9 จังหวัดภาคเหนือ พร้อมระบุสัดส่วนแหล่งกำเนิดฝุ่นเชิงปริมาณ (ไทย/เมียนมา/ลาว) เพื่อสนับสนุนการเตือนภัยล่วงหน้าและนโยบายควบคุมมลพิษ
- **ผู้ใช้ที่ตั้งใจ:** หน่วยงานสาธารณสุข/สิ่งแวดล้อม (เช่น กรมควบคุมมลพิษ), หน่วยงานปกครองท้องถิ่น, นักวิจัย/สถาบันการศึกษา, ประชาชนทั่วไปผ่าน dashboard (ที่มา: `docs/REPORT_DRAFT.md` §5)
- **ไม่ใช่วัตถุประสงค์:** ไม่ใช่ระบบเตือนภัยฉุกเฉินอย่างเป็นทางการ ไม่ใช้แทนประกาศของกรมควบคุมมลพิษ (PCD) และไม่ควรใช้กล่าวโทษเชิงการทูตระหว่างประเทศจากผลการระบุแหล่งกำเนิด (ที่มา: `docs/REPORT_DRAFT.md` §3.2 ข้อ 4)

## 3. ข้อมูลการฝึก (Training data)

3-way held-out split ตาม `docs/DESIGN.md` §1 / `docs/REPORT_DRAFT.md` §3.1, §6:

| Split | ช่วงเวลา | บทบาท |
|---|---|---|
| Train | 2565–2566 (2022–2023) | ฝึกโมเดล |
| Validation | 2567 (2024) | เลือกโมเดล (main/dashboard checkpoint) |
| Test (held-out) | 2568 (2025) | ประเมินผลสุดท้าย โมเดลไม่เคยเห็นข้อมูลนี้ตอนฝึก |

18 สถานี Air4Thai (15 core + 3 extended), bbox lon 97.0–101.5, lat 16.0–21.0
(ที่มา: `docs/DESIGN.md` §3, `CLAUDE.md`)

## 4. ผลการประเมิน (Metrics) — held-out test 2568

RMSE (µg/m³, ยิ่งต่ำยิ่งดี) บนชุด held-out test 2568 (3,637 ตัวอย่าง, 18 สถานี, โมเดลเทรนใหม่บน
2565–2566 เท่านั้น) (ที่มา: `outputs/evaluation_test2025.json`):

| วิธี | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| Persistence | 2.96 | 5.19 | 8.70 | 12.99 |
| **MTGNN** | 4.52 | 5.93 | 8.76 | **12.66** |
| A3TGCN | 6.93 | 7.86 | 9.92 | 13.34 |
| GBM (ไม่ใช้กราฟ) | 3.78 | 5.58 | 9.55 | 14.22 |

นัยสำคัญทางสถิติ (paired block-bootstrap, B=2000, จับคู่ตามวันต้นทางการพยากรณ์) ของ MTGNN เทียบ
persistence ที่ 48h บนชุด held-out test: **+2.54%, 95% CI [-1.44%, +6.28%]** — คร่อม 0 คือ
**ยังไม่มีนัยสำคัญทางสถิติ** (ที่มา: `outputs/significance_test2025.json`)

## 5. ข้อจำกัด (Limitations) — วิทยานิพนธ์หลักที่ต้องอ่านก่อนใช้งาน

โครงงานนี้ยึดหลักการประเมินผลอย่างเข้มงวดและซื่อสัตย์ (ที่มา: บทคัดย่อ/Abstract ของ
`docs/REPORT_DRAFT.md`):

1. **ความได้เปรียบเชิงพยากรณ์เหนือ persistence มีจำกัดและเล็กน้อย** — เด่นเฉพาะที่ 48h (RMSE 12.66
   เทียบ 12.99 µg/m³ บน held-out test 2568 ≈ 2%) ส่วน 6h/12h persistence เหนือกว่าโมเดลอย่างชัดเจน
   เนื่องจาก PM2.5 มี autocorrelation สูง และ 24h ใกล้เคียงกัน
   (ที่มา: `outputs/evaluation_test2025.json`, `docs/REPORT_DRAFT.md` §6.1)
2. **ความได้เปรียบที่ 48h ยังไม่มีนัยสำคัญทางสถิติในข้อมูลปีเดียว** — 95% CI ของ % improvement คร่อม
   0 ทั้งบนชุด held-out test (`[-1.44%, +6.28%]`) และโมเดลหลักบนปี 2568
   (`[-3.31%, +8.10%]`) (ที่มา: `outputs/significance_test2025.json`, `outputs/significance_val2025.json`)
3. **กลไกกราฟไม่ได้ยกระดับความแม่นยำเหนือความผันผวนจากการสุ่ม seed อย่างมีนัย** — multi-seed
   ablation (3 seeds ต่อ variant) พบว่าถอดกราฟทั้งหมด (temporal-only) ต่างจากโมเดลเต็มเพียง
   +0.04/+0.13/+0.18/+0.14 µg/m³ (6/12/24/48h) และทุก variant (ถอด type_b ลม / type_c ไฟ /
   adaptive adjacency / กราฟทั้งหมด) มี `robust_beyond_noise = false` ทุกขอบฟ้า — ผลต่างอยู่ในช่วง
   noise จากการสุ่ม seed ไม่ใช่ผลจริงของกลไกกราฟ (ที่มา: `outputs/ablation_multiseed.json`)
   ทั้งนี้ข้อสรุป null นี้จำกัดอยู่ที่กราฟขนาด 18 โหนด — กราฟเล็กระดับนี้อาจเล็กเกินกว่าที่กลไก GNN
   จะแสดงผลได้ จึงยังสรุปไม่ได้ว่ากลไกกราฟไร้ประโยชน์ในเครือข่ายสถานีที่ใหญ่กว่า
4. **คุณค่าที่พิสูจน์ได้ของระบบคือการระบุแหล่งกำเนิดที่อธิบายได้ (source attribution) ไม่ใช่ RMSE** —
   บนข้อมูล held-out 2568 ที่สถานีชายแดนแม่ฮ่องสอน (18 มี.ค. 2568) โมเดลระบุสัดส่วนต่างชาติ 62.7%
   ใกล้เคียงสัดส่วนไฟจริงที่เชื่อมถึงสถานี (72.1%) และไล่ตามสัดส่วนไฟจริงทั้ง 5 เหตุการณ์ทดสอบ
   ขณะที่เหตุการณ์เชียงใหม่ มี.ค. 2567 ระบุเป็นไฟไทยเป็นหลัก (~99.7%)
   (ที่มา: `outputs/transboundary_attr_test_split2.json`, `outputs/attribution_march2024.json`)
5. **ขนาดของสัดส่วน attribution ขึ้นกับโมเดล (model-dependent)** — โมเดลหลัก (checkpoint ที่ใช้ใน
   dashboard/pitch) ระบุสัดส่วนต่างชาติเพียง ~0–6% ในชุดเหตุการณ์เดียวกันที่โมเดลรายงาน (held-out
   split) ระบุ 62.7% — ทิศทางเชิงคุณภาพ (ชายแดน→ต่างชาติสูง, เมืองใน→ไทยสูง) สอดคล้องกันแต่ขนาด
   ตัวเลขไม่คงที่ข้ามโมเดล/checkpoint จึงต้องรายงานคู่กับ caveat นี้เสมอ
   (ที่มา: `outputs/transboundary_attr_test.json` เทียบ `outputs/transboundary_attr_test_split2.json`)
6. **ERA5 เป็น reanalysis ที่มีความล่าช้า** ใช้ฝึกได้แต่ inference จริงต้องใช้ NWP forecast แทน —
   ความได้เปรียบที่ 24h หายไปเมื่อ noise ถึง 0.25× ของส่วนเบี่ยงเบนธรรมชาติ ส่วน 48h ทนได้ถึงต่ำกว่า
   0.5× (ที่มา: `outputs/nwp_sensitivity.json`)
7. **Attribution เป็นการประมาณจากโมเดล ตรวจสอบเทียบ FIRMS ไม่ใช่การวัดโดยตรงในอากาศ** และอยู่ใน
   ระดับประเทศ ไม่ใช่ระดับพื้นที่ย่อย (ที่มา: `docs/REPORT_DRAFT.md` §3.2 ข้อ 4)

**สรุปคำแนะนำการใช้งาน:** ควรใช้ระบบในบทบาทการเตือนภัยล่วงหน้า 48 ชั่วโมง**ร่วมกับ**การระบุแหล่ง
กำเนิดที่อธิบายได้ ไม่ใช่มุ่งเอาชนะ persistence ด้าน RMSE เป็นหลัก (ที่มา: `docs/REPORT_DRAFT.md` §9)

## 6. ข้อพิจารณาทางจริยธรรม (Ethical considerations)

- ซอฟต์แวร์นี้เป็นผลงานทางวิชาการ (NSC 2026) **ไม่ใช่เครื่องมือเตือนภัยทางการ** และไม่ทดแทน
  ประกาศ/คำเตือนอย่างเป็นทางการของกรมควบคุมมลพิษ (PCD) หรือหน่วยงานสาธารณสุข
- ผลการระบุแหล่งกำเนิด (source attribution) เป็นการประมาณระดับประเทศจากแบบจำลอง **ไม่ควรใช้เป็น
  หลักฐานกล่าวโทษเชิงการทูตหรือกฎหมาย** ระหว่างประเทศ (ที่มา: `docs/REPORT_DRAFT.md` §3.2 ข้อ 4)
- ข้อกำหนดการใช้งานฉบับเต็ม (Disclaimer ภาษาไทย/อังกฤษ ตามแบบฟอร์ม NSC หน้า 44) อยู่ที่
  `README.md` หัวข้อ "ข้อกำหนดการใช้งาน / Disclaimer" และท้าย `app/streamlit_app.py`
- ดู `docs/REPORT_DRAFT.md` §6.6 สำหรับกรณีที่ทีมพัฒนาตรวจพบและแก้ไขข้อผิดพลาดการ
  denormalization ในการรายงานผลด้วยตนเอง (ธรรมาภิบาล AI)

---

*เอกสารนี้เป็น one-page model card ตามมาตรฐานความโปร่งใสของโมเดล ML สร้างขึ้นสำหรับ NSC 2026
ทุกตัวเลขคัดลอกจากไฟล์ `outputs/*.json` หรือ `docs/*.md` ที่อ้างอิงไว้ในวงเล็บ ณ วันที่ตรวจสอบ
2026-07-10*
