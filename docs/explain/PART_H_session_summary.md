# PART H — SESSION SUMMARY (ปิดท้าย)

> สรุปจาก `docs/SESSION{1..10}_NOTES.md` (ไม่มี SESSION9 — ข้ามหมายเลข)
> รูปแบบต่อ session: เป้าหมาย → สิ่งที่สร้าง → สิ่งที่เรียนรู้/แก้ → deviation → ส่งต่อ

## Session 1 — Data Pipeline Foundation (2026-05-18)
- **เป้าหมาย:** วาง scaffold repo + scraper แหล่งข้อมูล
- **สร้าง:** OpenAQ v3 scraper, FIRMS scraper, ERA5 (stub), loader, `scripts/01_download_all.py`,
  test 25/25, README bilingual, disclaimer placeholder ทุกไฟล์ `src/`
- **เรียนรู้/แก้:** OpenAQ ต้องใช้ `/measurements` ไม่ใช่ `/hours`; response UTF-16+BOM → `utf-8-sig`;
  air4thai realtime ต้องมี User-Agent
- **deviation:** ERA5 ยังเป็น stub (เลื่อนไป S3)
- **ส่งต่อ:** preprocessing + graph construction

## Session 2 — Preprocessing & Graph Construction (2026-05-19)
- **เป้าหมาย:** แปลง raw → dataset + สร้างกราฟ (หัวใจ)
- **สร้าง:** `DESIGN.md v2.1` (§5.2 graph spec), `preprocessing.py`, `hotspot_clustering.py`,
  `graph_builder.py`, FIRMS SP+NRT hybrid, test รวม **70/70 green**
- **เรียนรู้/แก้:** FIRMS **SP cap = 5 วัน** (ไม่ใช่ 10); country attribution ทำด้วย bbox (ไม่ใช้ shapely
  → ชายแดนอาจผิด); DBSCAN noise point ไม่ทิ้ง แปลงเป็นคลัสเตอร์เดี่ยว
- **deviation:** country geocode แบบ bbox แทน polygon จริง (ยอมรับข้อจำกัด)
- **ส่งต่อ:** training stack + ERA5 จริง

## Session 3 — Training Stack Debug & ERA5 Integration (2026-05-20)
- **เป้าหมาย:** ทำ training ให้รันได้ + เอา ERA5 เข้าจริง
- **สร้าง:** target ใช้ `pm25_scaled`, NaN guard ใน trainer, ERA5 monthly chunking + Windows path fix +
  coord normalise, `era5.parquet` (631,152 rows, 0 NaN), **feature 5→10**, `n_features:10` ใน config
- **เรียนรู้/แก้:** เทรนด้วย raw µg/m³ → loss ระเบิด NaN ~epoch 9 → ต้องใช้ scaled; CDS v2 ปฏิเสธ
  request ทั้งปี → ดึงทีละเดือน; netCDF เปิด path ไทยไม่ได้ → temp ASCII roundtrip
- **deviation:** ERA5 เป็น u10/v10/t2m/d2m/blh (5 ตัว) แทน rh ตาม DESIGN เดิม
- **ส่งต่อ:** explainability + dashboard

## Session 4 — Explainability, Viz, Streamlit (2026-05-20)
- **เป้าหมาย:** สร้างชั้นอธิบาย + นำเสนอ (GB-IG, dashboard)
- **สร้าง:** `explain/{gb_ig,gnn_explainer,attribution}.py` (IG completeness, occlusion country,
  gradient×input), `viz/`, Streamlit app หลายหน้า
- **เรียนรู้/แก้:** วาง IG ตาม completeness axiom; occlusion mask FRP รายประเทศ
- **ส่งต่อ:** ตรวจว่าทำไมโมเดลแพ้ persistence

## Session 5 — Data Audit, Split Fix, Training (2026-05-20)
- **เป้าหมาย:** วินิจฉัยว่าทำไมเทรนแพ้ persistence แล้วแก้ + เทรนรอบสะอาด
- **สร้าง:** verify 3 novelties ด้วย literature; แก้ citation bug ใน `gb_ig.py`; รอบเทรนใหม่
- **เรียนรู้/แก้:** ยืนยัน MTGNN/IG/wind-adjacency/FIRMS-node มีรากงานจริง; พบ attribution
  Thailand=100%/Myanmar=0% (ตั้งข้อสงสัย)
- **ส่งต่อ:** debug attribution + A3TGCN

## Session 6 — Attribution Debug & A3TGCN (2026-05-21)
- **เป้าหมาย:** สืบเคส Thailand=100% + fix occlusion + เทรน A3TGCN
- **สร้าง:** วิเคราะห์ occlusion, เทรน A3TGCN
- **⚠️ สำคัญ:** ตัวเลข µg/m³ ใน session นี้ **ผิดจาก denorm bug** (overstated) — ถูกแก้/แทนที่ใน S7
  (ranking โมเดล + metric normalized ไม่เปลี่ยน)
- **ส่งต่อ:** แก้ denorm bug + eval ที่ทำซ้ำได้

## Session 7 — Denorm Bug Fix, Reproducible Eval, AI Camp Prep (2026-06-28)
- **เป้าหมาย:** เตรียม pitch AI Camp (2–3 ก.ค.) + แก้ข้ออ่อนที่ reviewer จี้ (per-horizon RMSE,
  persistence, ERA5 latency, attribution rigor)
- **สร้าง:** **แก้ denorm bug** — รวม denorm/mask/persistence ไว้ที่ `src/training/evaluation.py`
  ตัวเดียว ทุก µg/m³ ตรงกัน; `evaluation_val2025.json` เลขที่ถูกต้อง
- **เรียนรู้/แก้:** ข้อเสนอเดิม denorm ด้วยสเกลคงที่ ~22 (ผิด) แทน per-station IQR → เลขสูงเกินจริง;
  **training ถูกต้องอยู่แล้ว** ผิดแค่ขั้นรายงาน
- **ส่งต่อ:** เพิ่ม rigor (significance, ablation, baseline)

## Session 8 — Rigor & Validation (2026-06-28)
- **เป้าหมาย:** เปลี่ยน "เทคนิคสูง แต่ค่ายังไม่พิสูจน์" → บัญชีที่ *ซื่อสัตย์และวัดได้* ของสิ่งที่
  3 novelties ให้จริง (เน้น credibility ไม่ใช่เลขสวย)
- **สร้าง:** `07_significance.py` (block bootstrap CI), ablation (multiseed/retrained/inference),
  `nwp_sensitivity.json`, baseline non-graph ML, re-split 3-way
- **เรียนรู้/แก้:** MTGNN แพ้ persistence สั้นอย่างมีนัยสำคัญ; gain ยาว **ไม่ผ่าน 95% CI**;
  ablation ตัด Type B/C แทบไม่เปลี่ยน RMSE; noise ERA5 ก็ไม่ขยับ → **graph ไม่ช่วย accuracy มั่นคง**
- **ส่งต่อ:** เขียน final report ที่ซื่อสัตย์

## Session 10 — Finish NSC Final Report (2026-07-01, ปัจจุบัน)
- **เป้าหมาย:** เปลี่ยน `REPORT_DRAFT.md` (ผ่าน reviewer) → Word ส่งได้จริง (TH Sarabun New 16pt);
  roll out disclaimer จริง; เติมรหัสโครงการ; เขียนคู่มือ install/user — **ไม่มีวิจัยใหม่ format only**
- **สร้าง:** `scripts/generate_report.py` → `outputs/NSC2026_Final_Report.docx` (TOC + page-number
  field, disclaimer จริง), install/user guide
- **เรียนรู้/แก้:** ใช้ python-docx `_set_cs_font` complex-script trick; ไม่นำเลข denorm-bug เดิมมาใช้ซ้ำ
- **honest thesis คงเดิม:** forecast edge marginal/ไม่ significant; graph ไม่ช่วย accuracy มั่นคง;
  **value ที่พิสูจน์ได้ = explainable transboundary source attribution**
- **ส่งต่อ (สถานะปัจจุบัน):** 3 commits ยังไม่ push; เหลือ finalize + ส่ง (deadline report 17 ก.ค. 2026)

---

## เส้นเวลาโดยรวม (arc ของโปรเจค)

```
S1 scaffold+scraper → S2 preprocess+graph(หัวใจ) → S3 training+ERA5(5→10 feat)
→ S4 explain+dashboard → S5 audit(ทำไมแพ้ persistence?) → S6 attribution debug[เลขผิด]
→ S7 แก้ DENORM BUG + eval ทำซ้ำได้ → S8 RIGOR(significance/ablation → graph ไม่ช่วย)
→ S10 finalize report ซื่อสัตย์
```

**บทเรียนใหญ่ที่สุดของโปรเจค:** จุดพลิกอยู่ที่ S7–S8 — จาก "อ้าง gain" มาเป็น "วัดอย่างเข้มงวด
แล้วยอมรับผลลบ" ความซื่อสัตย์เชิงวิทยาศาสตร์นี้เองที่กลายเป็นจุดแข็งของงาน NSC มากกว่าตัวเลข RMSE

---

## สรุปทั้งเอกสาร (PART A–H)

| PART | สาระ 1 บรรทัด |
|---|---|
| A | STGNN อธิบายได้ พยากรณ์ + ระบุแหล่งฝุ่น 9 จังหวัดเหนือ; thesis ซื่อสัตย์กับผลลบ |
| B | pipeline: OpenAQ/FIRMS/ERA5 → preprocess (gap policy, RobustScaler) → dataset.parquet |
| C | heterograph: node station/hotspot + edge Type A(สเปเชียล)/B(ลม)/C(ไฟ); wind align = cos(θ) |
| D | station feat 10 มิติ, hotspot 3; model ใช้แค่ weight ของ edge_attr; ERA5 คือ feat ที่เคยขาด |
| E | MTGNN แพ้ persistence สั้น/ไม่ significant; graph ไม่ช่วย accuracy; value = attribution |
| F | 2-step install; `.env`(OPENAQ/FIRMS) + `.cdsapirc`; Windows Thai path + `.pth` fix |
| G | 10 test files (mock network), 70/70 green; black+ruff line-100, ignore ไฟล์เนื้อหาไทย |
| H | S1→S10 arc; จุดพลิก = S7 denorm fix + S8 rigor → ยอมรับผลลบอย่างซื่อสัตย์ |

> จบครบทั้ง PART A–H — เอกสารชุดนี้อยู่ใน `docs/explain/PART_{A..H}_*.md`
