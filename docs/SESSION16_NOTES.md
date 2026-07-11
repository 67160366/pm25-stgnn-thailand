# SESSION 16 NOTES — 2026-07-11
Status: COMPLETE

ปิดคิวค้างจาก S15 + งาน license compliance ที่โผล่จากการปิด open question S13
ทำแบบ orchestrator: Fable ตรวจ+gates+commit ทุกจุด; implementer ×2 ขนาน (held-out
conformal / province fix), general-purpose ทำ research→apply (geoBoundaries),
agent เดิมปลุกต่อด้วย SendMessage สำหรับงาน follow-up (MODEL_CARD, attribution)

## What was accomplished

- **สถานะจากผู้ใช้:** ยังไม่ submit SIMS (เดดไลน์ 17 ก.ค. 17:00) → กติกาเหล็กคุมตลอด
  session ยกเว้นข้อยกเว้นที่ผู้ใช้อนุมัติชัดเจน (ดูข้อสุดท้าย); ยังไม่สร้าง Telegram bot
- **`a586dea` แก้ 🟡 `_province_from_name`:** คืน `""` เมื่อชื่อไม่มี comma (เดิม echo ชื่อเต็ม)
  + ลบ workaround ที่ call site script 16; ตรวจแล้ว consumer ทุกจุดรองรับ ("" → ไม่โชว์ซ้ำ)
- **`22eb6ea` Held-out conformal coverage บน test 2025** (ข้อ NOT-done จาก S15):
  `scripts/17_conformal_holdout.py` → `outputs/conformal/holdout_coverage_test2025.json`
  (**commit ไฟล์ JSON ด้วย** เหมือน intervals) — pooled **6h 0.885 / 12h 0.882 / 24h 0.882 /
  48h 0.870** vs nominal 0.90; Fable verify เองว่า logic การใช้ halfwidth ตรงกับ
  `app/lib/inference.conformal_halfwidths` + clip ล่างที่ 0 ของ forecast.py เป๊ะ
  **สถานี 225585 undercover หนักสุดและแย่ลงตาม horizon (0.818→0.698)**
- **`409edfe`** caption แถบ 90% หน้า forecast อ้างเลข held-out 0.87–0.88 แล้ว
- **`de2658a`** MODEL_CARD.md ได้ §5 uncertainty ใหม่ (ตารางเทียบ in-sample 0.900 ที่ห้ามอ้าง
  vs held-out ที่ควรอ้าง + ข้อจำกัด 225585/live-NWP; §5→§6, §6→§7 — ไม่มี cross-ref พัง)
- **`a7f2d31` ปิด open question S13 (geoBoundaries):** verify เชิงประจักษ์ —
  `borders_th_mm_la.geojson` = geoBoundaries **gbOpen ADM0 simplified** build 2023-12-12
  (`wmgeolab/geoBoundaries@9469f09`) geometry ตรงทุกพิกัด แค่ properties ถูก rename;
  license **per-boundary**: THA/LAO = ODbL 1.0, MMR = CC BY-SA 2.0 (ทั้งหมด OSM-derived →
  ต้องเครดิต © OpenStreetMap contributors; share-alike ครอบแค่ตัว geojson ไม่ลามโค้ด)
  → เติม attribution ใน About page + README refs + DATA_CARD §4 (ปลด NEEDS VERIFICATION 2 จุด)
- **`0ed94cb` เครดิตในรายงาน (ข้อยกเว้นกติกาเหล็ก — ผู้ใช้อนุมัติผ่าน AskUserQuestion):**
  reference ข้อ 8 (Runfola 2020 + OSM/ODbL/CC BY-SA) ใน REPORT_DRAFT.md + generate_report.py
  sync กัน; docx regen + verify (เครดิตมา, 8 รูป, เลขต้องห้ามหายครบ, TH Sarabun New คงเดิม)
- **Push แรก (ผู้ใช้อนุมัติ):** `604a7be..a7f2d31` — **CI SUCCESS ทั้งสอง run**
- Gates ทุก commit: pytest **310 passed**, ruff 4 dirs + black clean

## Current state (branch, last commit, open work)

- Branch `feat/demo-features` @ `0ed94cb` — commit สุดท้าย (report credit) **ยังไม่ push**
  (per-push ask; push ก่อนหน้าถึง `a7f2d31` แล้ว CI เขียว)
- docx ล่าสุด regen แล้วมีเครดิต — **ผู้ใช้ต้องเปิด Word กด F9 (สารบัญ) ก่อนอัปโหลด SIMS**
- Working tree สะอาด ยกเว้น untracked-by-choice เดิม
- Roadmap: ชุดหลัง-SIMS (8→10→2/4→11 ตามลำดับคุ้ม) รอ submit; ก่อนหน้านั้นไม่มีคิวโค้ดค้าง

## What was NOT done (and why)

- **ส่ง Telegram จริง** — ผู้ใช้ยังไม่สร้าง bot (ต้องทำก่อน demo 21 ส.ค.)
- **แก้ undercoverage สถานี 225585** — บันทึกเป็น limitation ใน MODEL_CARD แล้ว; ทางแก้จริง
  (recalibrate แบบ adaptive/per-year) เป็นงานหลัง-SIMS ถ้าคุ้ม
- NWP blh vs ERA5 blh scale — open question เดียวที่เหลือจาก S13 (ต้องใช้ข้อมูล live)

## Next session must start with (exact first actions)

1. เช็คผู้ใช้: กด F9 + อัปโหลด SIMS แล้วหรือยัง (**เดดไลน์ 17 ก.ค. 17:00 — เหลือ 6 วัน**)
   → ถ้า submit แล้ว: ปลดกติกาเหล็ก เริ่มข้อ 8 (frozen-model eval บน Jan-Apr 2026)
2. ถ้า `0ed94cb` ยังไม่ push: ขอ push (per-push ask) แล้วเช็ค CI
3. เช็คผู้ใช้: Telegram bot (ก่อน demo)

## Critical context to preserve (gotchas, fragile files, decisions)

- **เลข conformal ที่อ้างได้ = held-out เท่านั้น** (0.885/0.882/0.882/0.870); in-sample 0.900
  ห้ามใช้เป็นผลประเมิน — MODEL_CARD §5 เขียนกำกับแล้ว
- `outputs/conformal/holdout_coverage_test2025.json` ถูก commit โดยตั้งใจ (เหมือน intervals);
  ถ้า re-calibrate intervals ต้องรัน script 17 ซ้ำแล้ว commit ทับทั้งคู่
- REPORT_DRAFT.md + generate_report.py ถูกแตะ 1 ครั้งใน `0ed94cb` (เครดิต geoBoundaries)
  โดยผู้ใช้อนุมัติชัดเจน — นอกเหนือจากนี้กติกาเหล็กยังคุมตามเดิมจนกว่า submit
- ตัว geojson ต้องคง license notice (ODbL/CC BY-SA) — ถ้าย้าย/แก้ไฟล์ อย่าลบเครดิตใน
  README/About/DATA_CARD; อ้าง build 2023-12-12 @ `9469f09` (verify แล้ว 2026-07-11)
- Orchestration ที่ work (ยืนยันซ้ำจาก S15): ปลุก agent เดิมด้วย SendMessage สำหรับ follow-up
  ประหยัด context มาก; งานขนานแบ่ง scope ไฟล์ชัด; Fable รัน full gates เองก่อนทุก commit

## Open questions

- NWP blh vs ERA5 blh สเกล — `[NEEDS VERIFICATION]` (ค้างจาก S13, ต้องใช้ข้อมูล live)
- Conformal live mode: mis-coverage จริงใต้ NWP input — ยังไม่มีเลข (MODEL_CARD ระบุเป็น
  limitation แล้ว) `[NEEDS VERIFICATION]`
- geoBoundaries "v6.0.0" tag เป๊ะ ๆ — อ้าง commit hash + build date แทน (ปลอดภัยกว่า);
  แค่ label เวอร์ชัน ไม่กระทบ compliance
