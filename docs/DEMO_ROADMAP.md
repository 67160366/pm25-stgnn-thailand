# Demo Roadmap — ฟีเจอร์สำหรับรอบชิงชนะเลิศ NSC (demo 21 ส.ค. 2569)

สถานะ: **อนุมัติแล้ว — ทำทั้งหมด** (ตกลงกับผู้พัฒนา 2026-07-07, Session 12)
**เริ่มงานแล้ว 2026-07-10** (Session 13) — ผู้พัฒนาอนุมัติให้เริ่มก่อนส่งรายงาน หลังตรวจกติกา NSC แล้ว

## กติกา NSC อนุญาตให้พัฒนาต่อ (ตรวจ booklet แล้ว 2026-07-10)

- Booklet หน้า 33: "หากมีการปรับปรุงเนื้อหาในรายงานฉบับสมบูรณ์ **สามารถจัดส่งใหม่ได้ภายในเวลาที่กำหนด**"
  — กรรมการคาดหวังการพัฒนาต่อระหว่าง 17 ก.ค. → 21 ส.ค. อยู่แล้ว
- เกณฑ์รอบชิง (หน้า 34-35, หมวด 14 นิสิต): Technique 25 / Creativity 25 — ฟีเจอร์ใหม่มีแต่ได้คะแนน
- เงื่อนไขที่คุมจริง: ผลงานริเริ่มเอง (ห้ามลอก), ส่งมอบ core ตาม proposal ได้ครบ (ข้อ 7 — ขอบเขตเป็นขั้นต่ำ
  ไม่ใช่เพดาน), เปลี่ยนข้อมูลทางการ (ชื่อ/ทีม/หมวด) ต้องยื่นฟอร์ม — ฟีเจอร์ซอฟต์แวร์ไม่อยู่ในรายการนี้
- **งานส่งเพิ่มรอบชิง (ตามเวลาที่ประกาศ):** คลิป ≤7 นาที (`28p14e01196.mp4`), ฟอร์มสรุป 1-1.5 หน้า A4
  (ตัวอักษร 14-16), ภาพทีม+ที่ปรึกษา (.pptx) — กันเวลาไว้ทำด้วย

## กติกาเหล็ก (แก้ไข 2026-07-10)

1. ~~ห้ามแตะก่อนส่งรายงาน~~ → **ทำได้ แต่ต้องอยู่บน branch แยก และห้ามแตะไฟล์ที่รายงานพึ่งพา**
   (`outputs/*.json`, `outputs/figures/report/`, `checkpoints*/`, `scripts/generate_report.py`,
   `docs/REPORT_DRAFT.md`, `data/processed/*`) จนกว่าจะกด submit SIMS (17 ก.ค. 17:00) เรียบร้อย
2. ทุกฟีเจอร์ต้องไม่ทำลาย honest thesis: ห้าม claim ความแม่นเพิ่มโดยไม่มีหลักฐาน ต้องคง caveat เดิมทั้งหมด
3. ข้อ 1–4 **ไม่ต้องเทรนใหม่** (inference/engineering ล้วน) — ทำก่อน; ข้อ 5 เป็น experiment ผลไม่การันตี —
   ทำคู่ขนาน **แต่หลังส่งรายงานเท่านั้น** (แตะ pipeline/data ที่รายงานพึ่งพา)

## ลำดับงาน

### 1. โหมดพยากรณ์จริงด้วย NWP สด (Open-Meteo) — ปิดจุดอ่อน "hindcast"
- Scraper ใหม่ `src/data/scrapers/openmeteo.py`: hourly forecast 7 วัน (ฟรี ไม่ต้องมี key,
  non-commercial) ตัวแปร u10/v10/t2m/d2m + `boundary_layer_height`
  `[NEEDS VERIFICATION: endpoint ไหนให้ blh ครบ — docs หลัก หรือ GFS API]`
- ต่อเข้า `app/lib/inference.py` เป็นโหมด "พยากรณ์วันนี้" (คู่กับโหมด hindcast เดิม) + ป้ายบอกชัดว่า
  input เป็น NWP ไม่ใช่ ERA5 พร้อมลิงก์ผล `nwp_sensitivity.json` เป็นหลักฐานความทน noise
- Acceptance: กดใน dashboard แล้วเห็นพยากรณ์ 48 ชม. ข้างหน้าของวันจริง สดต่อหน้ากรรมการ
- อย่าลืม: mock ทุก network call ใน tests (กติกา CLAUDE.md)

### 2. Attribution matrix หลายสถานีชายแดน
- รัน `scripts/10_transboundary_attr.py station_id=<id>` กับสถานีชายแดนทุกตัว (เชียงราย แม่สาย/
  เชียงของ, น่าน, ตาก แม่สอด ฯลฯ — เลือกจาก stations_metadata) ทั้ง 2 checkpoints
- สร้างหน้า/รูป "เหตุการณ์ × สถานี × ประเทศต้นทาง" — หลักฐานเชิงระบบแทน case เดียว
- Acceptance: ตาราง/แผนที่ matrix ใน dashboard + JSON ใหม่ใน outputs/

### 3. Counterfactual แบบโต้ตอบใน dashboard ("ถ้าดับไฟกลุ่มนี้ ฝุ่นลดเท่าไร")
- ใช้ occlusion ที่มีใน `src/explain/gb_ig.py` — ผู้ใช้เลือกประเทศ/กลุ่มไฟ → แสดงค่าพยากรณ์
  ก่อน/หลังลบไฟ (µg/m³) รายสถานี
- ขาย "คันโยกนโยบาย: ในประเทศ vs การทูต" ที่ pitch วางไว้ ให้จับต้องได้จริง
- Acceptance: หน้าตอบสนอง <~5 วิ/ครั้ง (inference CPU 1 sample)

### 4. Attribution uncertainty จาก multi-seed checkpoints
- รัน attribution ซ้ำทุก seed ใน `checkpoints_split2/` (full variant: orig + s0 + s1) →
  รายงาน mean±spread ต่อเหตุการณ์
- เปลี่ยน caveat "ขนาดขึ้นกับโมเดล" จากคำเตือนลอยๆ เป็นตัวเลขที่วัดจริง
- Acceptance: ตาราง/error bar ในหน้า transboundary + อัปเดต qa_defense

### 5. Experiment: scale ERA5 + เทรนใหม่ (งาน B เดิม — ผลไม่การันตี)
- ยืนยันแล้ว (2026-07-07): ERA5 เข้าโมเดลดิบ (`loader.py` fillna(0) — t2m~300K ปนกับ feature ระดับ ±1;
  ชั่วโมงหายกลายเป็น 0 เคลวิน) — คันโยกความแม่นเดียวที่ยังไม่ลอง
- ทำ: standardize ERA5 ต่อฟีเจอร์ (fit จาก train เท่านั้น เก็บ params ข้าง scalers.json),
  fillna เป็นค่ากลางของ train ไม่ใช่ 0, เทรน mtgnn full 3 seeds บน split2 protocol เดิม
- เปรียบเทียบกับ `evaluation_test2025.json` / `baseline_ml_test.json` ชุดเดิม mask เดียวกัน
- ถ้าดีขึ้น → หลักฐานโชว์รอบ demo; ถ้าไม่ → บันทึกเป็น negative result (เสริม governance narrative)
- ห้าม overwrite checkpoints เดิม — ใช้ `checkpoints_scaled/` (gitignore เพิ่ม)

### 6. แจ้งเตือนล่วงหน้า 48 ชม. (Telegram bot)
- แปลงพยากรณ์ → ระดับ AQI ไทย + คำแนะนำกลุ่มเสี่ยง แล้ว push ผ่าน Telegram Bot API
  (หมายเหตุ: LINE Notify ปิดบริการ มี.ค. 2025 แล้ว — ถ้าต้องเป็น LINE ให้ใช้ Messaging API)
- Acceptance: demo สดรับข้อความเตือนบนมือถือ

### 7. Engineering polish
- GitHub Actions CI: pytest + ruff + black ทุก push (ไม่มี network ใน tests อยู่แล้ว — ผ่านได้เลย)
- Model Card + Data Card หนึ่งหน้า (docs/) — เข้าธีม AI governance
- หน้า overview ดึง air4thai realtime สด (scraper มีแล้ว) เป็นค่า "ตอนนี้" คู่กับพยากรณ์

## งานเก็บตกที่ต้องทำช่วง demo prep (ไม่ใช่ฟีเจอร์)

- **อัปเดต `outputs/pitch/qa_defense.md` + presenter script** ให้ตรงตัวเลขป้ายประเทศใหม่
  (เคสเรือธงเปลี่ยนจาก 2025-02-16 [37.3%→36.6%] เป็น **2025-03-18 [72.1%→62.7%]**;
  "128 เท่า" → "59 เท่า"; เลิกอ้าง "เกิดในทั้งสองโมเดล") — เอกสาร AICAMP เก่าปล่อยเป็น historical
- ซ้อม demo script กับ dashboard โหมดใหม่ทั้งหมด

## อ้างอิงบริบท

- เหตุผลการจัดลำดับ + การวิเคราะห์เต็ม: บทสนทนา Session 12 (2026-07-07) และ `docs/SESSION12_NOTES.md`
- ผลป้ายประเทศใหม่ที่ทำให้ตัวเลขเปลี่ยน: `docs/SESSION12_NOTES.md` + `src/data/geocode.py` docstring
