# SESSION 12 NOTES — 2026-07-07
Status: COMPLETE

งาน A จากแผนแก้ปัญหา: แก้ป้ายประเทศ bounding box → point-in-polygon ทั้ง pipeline,
รัน attribution ซ้ำ, อัปเดตรายงานฉบับส่ง SIMS ให้ตัวเลขตรงกันทั้งระบบ **ไม่มีการเทรนใหม่**
(ป้ายประเทศเป็น metadata ตอนจัดกลุ่ม attribution — ยืนยันจากโค้ดว่าไม่ใช่ฟีเจอร์เข้าโมเดล)

## What was accomplished

- **`src/data/geocode.py` (ใหม่):** ray-casting point-in-polygon เวกเตอร์ด้วย numpy + bbox
  pre-filter อ่าน `app/assets/borders_th_mm_la.geojson` — cross-validate กับ implementation เดิม
  ใน `app/lib/geo.py` ตรงกัน **3000/3000 จุดสุ่ม**; `hotspot_clustering._country_from_centroid`
  delegate มาที่นี่ (ลบ `_COUNTRY_BBOXES`); `app/lib/geo.py` เหลือ re-export + `border_geojson()`
- **Regenerate `hotspots.parquet`:** 26,839 clusters เท่าเดิม, geometry/FRP identical ทุกแถว,
  **ป้ายประเทศเปลี่ยน 6,758 กลุ่ม (25.2%)** — Thailand 23,293→17,070, Myanmar 3,406→7,036
  (+4,165 จากไทย), Laos 140→2,673 (+2,010 จากไทย), other 0→60 (bbox เดิม priority ไทยกลืน
  โซนทับซ้อน 97.3–101.2°E ทั้งหมด)
- **รัน attribution ซ้ำทั้ง 5 ไฟล์** (05×2 + 10×3, ทั้ง pitch และ split2 checkpoints):
  - **ชุดเหตุการณ์ top-5 เปลี่ยนทั้งชุด** (จัดอันดับด้วย foreign FRP ซึ่งตอนนี้ถูกต้อง):
    เคสเรือธงใหม่ = **2025-03-18** (peak 69.1 µg/m³, ไฟต่างชาติที่เชื่อมถึง 72.1% →
    โมเดลรายงานระบุต่างชาติ **62.7%** [MM 51.6/LA 11.1]) + เคสลาว 2025-03-13 (52.8%→40.3%)
  - **Monotonic tracking ทั้ง 5 เหตุการณ์:** 72.1→62.7, 52.8→40.3, 11.5→0.5, 3.7→0, 1.4→0
    — เรื่องเล่าแข็งแรงกว่าเคสเดิม (02-16: 37.3→36.6 ซึ่งหลุดจาก top-5 ไปแล้ว)
  - **Claim "เกิดในทั้งสองโมเดล" ตายแล้ว:** pitch model ให้ต่างชาติ ~0–6% บนเหตุการณ์ชุดใหม่
    → เขียน caveat ใหม่: ความสามารถ tracking เป็นของโมเดลรายงาน (split2), ขนาดขึ้นกับโมเดล
  - เชียงใหม่ มี.ค. 2024: Thailand ~100% คงเดิม (99.7% pitch / 99.6% split2); FRP เทียบใหม่
    ไทย 108,569 vs ต่างชาติรวม 1,849 = **~59 เท่า** (เดิม "128 เท่า" vs เมียนมาอย่างเดียว)
  - แก้บั๊ก metadata `10_transboundary_attr.py`: field `checkpoint` เคยเขียนค่า default เสมอ
- **รายงาน + เอกสาร sync ตัวเลขใหม่ทั้งหมด:** `REPORT_DRAFT.md` (บทคัดย่อ TH/EN, §6.5 (ก)(ข) +
  หมายเหตุการแก้ป้าย, คำบรรยายรูป 6/7), `generate_report.py` (transcription เดียวกัน),
  `USER_GUIDE.md`, `docs/explain/PART_B + PART_H`, หน้า dashboard `transboundary.py`
  (เคสเด่น, default date, กล่องความซื่อสัตย์) — **ลบ display-override ใน `data_access.py`**
  (แผนที่กับ JSON มาจากป้ายชุดเดียวกันแล้ว, desync จบ)
- `13_report_figures.py`: แผนที่เลือกวัน flagship อัตโนมัติจาก JSON (ไม่ hardcode วันที่)
- **`outputs/NSC2026_Final_Report.docx` regenerate + verify PASS:** เลขเก่า 12 ตัว ABSENT
  (128 เท่า, 109,564, 16 ก.พ., 37.3/36.6/33.2%, denorm-bug ต้องห้าม 5 ตัว), เลขใหม่ 10 ตัว
  PRESENT, รูป 8 รูปครบ
- **Tests: 225 passed** (ใหม่ `tests/test_geocode.py` 12 ตัว), ruff + black เขียวทั้ง repo
- **`docs/DEMO_ROADMAP.md` (ใหม่):** ผู้ใช้อนุมัติทำฟีเจอร์ demo ทั้ง 7 ข้อ (NWP สด, attribution
  matrix, counterfactual โต้ตอบ, multi-seed uncertainty, ERA5-scaling retrain, Telegram แจ้งเตือน,
  CI/model card) — เริ่มหลังส่งรายงาน 17 ก.ค. เท่านั้น

## Current state
- Branch `feat/session-4-explain-app`; commit และ push ในเทิร์นเดียวกับ notes นี้
- Gates เขียว: pytest 225, ruff/black clean, docx verified

## What was NOT done (and why)
- `outputs/pitch/qa_defense.md` + presenter script ยังอ้างตัวเลขเก่า — งาน demo prep
  (อยู่ใน DEMO_ROADMAP "งานเก็บตก"); เอกสาร AICAMP = historical ปล่อยตามเดิม
- ERA5-scaling retrain (roadmap ข้อ 5) — ห้ามก่อนส่งรายงาน

## Next session must start with
1. อ่าน `docs/DEMO_ROADMAP.md` — ก่อน 17 ก.ค.: **ไม่ทำฟีเจอร์ใหม่** เหลือแค่ user เปิด docx
   ใน Word กด F9 (TOC) แล้วอัปโหลด SIMS พร้อม INSTALL.md + USER_GUIDE.md
2. หลังส่งรายงาน: เริ่ม roadmap ข้อ 1 (Open-Meteo NWP) — verify ตัวแปร blh ใน API ก่อน

## Critical context to preserve
- **ป้ายประเทศ = metadata เท่านั้น** (จัดกลุ่มใน `occlusion_country_attribution` +
  แสดงผล) ไม่ใช่ input โมเดล — แก้ป้าย/geojson ไม่ต้องเทรนใหม่
- เลขที่ห้ามกลับมาใช้เพิ่ม: 37.3/36.6/33.2%, "128 เท่า", เคส 16 ก.พ. 2568 เป็นเรือธง,
  "เกิดในทั้งสองโมเดล" (นอกเหนือจาก denorm-bug 5 ตัวเดิม)
- Hydra ของ `05_attribution.py` ต้องใช้ `+checkpoint=` / `+output=` (มี + — struct mode);
  usage ใน docstring ไม่มี + จะ crash (`ConfigKeyError`)
- ERA5 เข้าโมเดลดิบ ไม่ scale + fillna(0) (`loader.py:213-214`) — ยืนยันแล้ว เป็นคันโยก
  roadmap ข้อ 5

## Open questions
- ไม่มีที่ block การส่งรายงาน
