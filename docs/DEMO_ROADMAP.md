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
- หน้า overview ดึง air4thai realtime สด เป็นค่า "ตอนนี้" คู่กับพยากรณ์
  (แก้ข้อมูล 2026-07-10: scraper air4thai **ยังไม่มีในโค้ด** — สร้างใหม่ใน `app/lib/air4thai.py`
  พร้อม mapping รหัสสถานี air4thai↔OpenAQ ที่ `configs/air4thai_station_codes.json`)

## ข้อเสนอเพิ่มเติมรอบสอง (Session 13, 2026-07-10 — ทำต่อหลังชุด 7 ข้อแรก)

ความเห็นเต็มจากการวิเคราะห์ "ถ้าไม่ติดกรอบ scope เดิม อะไรยกระดับโปรเจคได้จริง"
เรียงตามผลตอบแทนต่อแรง (ข้อ 8-10 **ไม่ต้องเทรนใหม่**):

### 8. ประเมินโมเดลกับฤดูเผา 2026 (ม.ค.–เม.ย. 2026) — คุ้มสุด
- Backfill OpenAQ/FIRMS/ERA5 ช่วง 2026 → รัน checkpoint เดิม (แช่แข็ง) พยากรณ์ + attribution
- Out-of-sample แท้ 100% (ข้อมูลไม่มีอยู่ตอนออกแบบ) — ตัดข้อครหา leakage ทุกทาง; ได้เคส demo สดใหม่
- ธีมเราคือ honest evaluation — ผลออกมาไม่สวยก็รายงานตรง ๆ ได้
- ระวัง: เขียนลง `data/` ได้ (gitignored) แต่**ห้ามแตะไฟล์ 2022-2025 เดิมจนกว่าจะ submit SIMS**
  — ใช้ไฟล์ output แยก (เช่น `dataset_2026.parquet`)

### 9. Conformal prediction intervals — ถูกมาก ได้เยอะ
- Calibrate บน val 2024 หลังเทรน (post-hoc ไม่แตะโมเดล โค้ดไม่กี่สิบบรรทัด) → ช่วงพยากรณ์
  พร้อมการันตี coverage เชิงสถิติ เช่น "45–80 µg/m³ ที่ 90%"
- เข้าธีม AI governance/ความซื่อสัตย์ตรง ๆ; แสดงใน dashboard + Telegram alert

### 10. Re-score เป็น exceedance forecasting ("โอกาสฝุ่นเกิน 75 µg/m³ ใน 48 ชม.")
- ใช้ผลพยากรณ์เดิมมา re-score เป็น classification: hit rate / false alarm / Brier vs climatology
- แก้จุดอ่อน "แพ้ persistence" ที่ต้นตอ: persistence อ่อนตอนฝุ่น "เปลี่ยนระดับ" ซึ่งเป็นจังหวะที่
  ข้อมูลไฟ+ลมของเราช่วย — โอกาสได้ headline ที่ชนะแบบซื่อสัตย์; เชื่อมเข้า roadmap ข้อ 6 (alert) พอดี

### 11. Validate attribution ด้วยหลักฐานอิสระ (HYSPLIT back-trajectory)
- รัน HYSPLIT (NOAA, ฟรี/ออนไลน์) หรืออย่างน้อยวิเคราะห์ wind field ERA5 ย้อนหลัง บนเคสเรือธง
  2025-03-18 — ถ้าทิศทางตรงกับ attribution = พยานอิสระ ปิดคำถาม "รู้ได้ไงว่าโมเดลชี้ถูก"
- ยกระดับหลักฐานจาก monotonic tracking (n=5) เป็น cross-validation กับวิธีมาตรฐานของวงการ

### 12. (ถ้าเวลาเหลือ) ขยายกราฟเกิน 18 สถานี — เหตุผลเชิงโครงสร้างที่กราฟไม่ช่วย
- ความเห็นตรง ๆ: multi-seed ablation ที่ null อาจเพราะกราฟ 18 โหนดเล็กเกินกว่า GNN จะโชว์พลัง
  ไม่ใช่เพราะ "กราฟไม่มีประโยชน์" — Air4Thai มีสถานีทั่วประเทศ ~80+; ขยายเป็น ~30-40 สถานี
  (เหนือ+กลางตอนบน) แล้วเทรนใหม่อาจพลิกข้อสรุป
- แรงเยอะ (backfill + กราฟใหม่ + เทรนหลาย seed, เสี่ยง 2-3 สัปดาห์) — ทำเฉพาะถ้าข้อ 8-11 เสร็จเร็ว
- อย่างน้อยที่สุด: เขียน limitation "ข้อสรุป null จำกัดที่ n=18" ลง Model Card + รายงานฉบับปรับปรุง

### 13. เสริมข้อ 5 (ERA5-scaling retrain): เพิ่ม variant residual learning
- ให้โมเดลทำนาย**ส่วนต่างจาก persistence** แทนค่าดิบ — การันตี "แย่สุดเท่า persistence" โดย
  construction แก้ปัญหาแพ้ 6/12h ที่ต้นตอ; เทรนคู่ไปกับ variant ERA5-scaled ใน `checkpoints_scaled/`

### สิ่งที่ตัดสินใจ "ไม่ทำ" (บันทึกไว้กันวนกลับมาคิดใหม่)
- เพิ่มแหล่งข้อมูลใหม่ (เช่น MODIS AOD) / เปลี่ยนสถาปัตยกรรมโมเดล — แรงเยอะ ผลไม่แน่นอน
  เวลา 6 สัปดาห์ไม่พอทำให้สุกงอมพร้อมงานเดิม
- ตกแต่ง dashboard เกินจำเป็น — Look & Feel 20 คะแนน vs Technique+Creativity 50

## งานเก็บตกที่ต้องทำช่วง demo prep (ไม่ใช่ฟีเจอร์)

- **อัปเดต `outputs/pitch/qa_defense.md` + presenter script** ให้ตรงตัวเลขป้ายประเทศใหม่
  (เคสเรือธงเปลี่ยนจาก 2025-02-16 [37.3%→36.6%] เป็น **2025-03-18 [72.1%→62.7%]**;
  "128 เท่า" → "59 เท่า"; เลิกอ้าง "เกิดในทั้งสองโมเดล") — เอกสาร AICAMP เก่าปล่อยเป็น historical
- ซ้อม demo script กับ dashboard โหมดใหม่ทั้งหมด

## อ้างอิงบริบท

- เหตุผลการจัดลำดับ + การวิเคราะห์เต็ม: บทสนทนา Session 12 (2026-07-07) และ `docs/SESSION12_NOTES.md`
- ผลป้ายประเทศใหม่ที่ทำให้ตัวเลขเปลี่ยน: `docs/SESSION12_NOTES.md` + `src/data/geocode.py` docstring
