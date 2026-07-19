# คู่มือการใช้งาน (User Guide)

โครงการ: **ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วย Explainable STGNN สำหรับภาคเหนือของประเทศไทย**
(NSC 2026 หมวด 14)

> สำหรับวิธีติดตั้ง ดู `docs/INSTALL.md` ก่อน · เอกสารนี้อธิบายวิธีใช้งานทั้ง dashboard และ pipeline ประมวลผล

---

## ภาพรวมระบบ

ระบบทำสองสิ่งที่เครื่องมือเดิมทำไม่ได้: **(1) พยากรณ์ PM2.5 ล่วงหน้า 6/12/24/48 ชั่วโมง** ที่ 18 สถานี Air4Thai
ใน 9 จังหวัดภาคเหนือ และ **(2) ระบุสัดส่วนแหล่งกำเนิดฝุ่นแยกตามประเทศ** (ไทย/เมียนมา/ลาว) ด้วย Explainable AI
ผลทั้งหมดแสดงผ่าน dashboard (Streamlit) และมี pipeline สำหรับทำซ้ำผล (reproducible) ผ่านสคริปต์ `01`–`13`

---

## ส่วนที่ 1 — Dashboard (สำหรับผู้ใช้ทั่วไปและผู้พิจารณาผลงาน)

### การเปิดใช้งาน

```bash
UV_NO_SYNC=1 uv run streamlit run app/streamlit_app.py
```

จากนั้นเปิดเบราว์เซอร์ที่ http://localhost:8501 (ปกติจะเปิดให้อัตโนมัติ)

> ต้องมี `data/processed/` และ `checkpoints/mtgnn/` อยู่ในเครื่องก่อน (ดู `docs/INSTALL.md` ข้อ 4)

### 6 หน้าของ dashboard

1. **ภาพรวม (Overview)** — ค่า AQI ปัจจุบันของแต่ละสถานี พร้อมระดับสี/คำแนะนำสุขภาพ, แผนที่ 18 สถานี,
   และแนวโน้ม 48 ชั่วโมงข้างหน้า เหมาะสำหรับประชาชนทั่วไป
2. **พยากรณ์ (Forecast)** — เลือกสถานี ดูค่าพยากรณ์ทั้ง 4 ขอบฟ้า (6/12/24/48 ชม.) เทียบกับ baseline
   persistence พร้อมกราฟค่าจริงย้อนหลังและแถบสี AQI
3. **แหล่งกำเนิด (Attribution)** — สัดส่วนผลกระทบของไฟแยกตามประเทศต่อการพยากรณ์ และความสำคัญของปัจจัย
   (Integrated Gradients) รายเหตุการณ์ เช่น เหตุการณ์เชียงใหม่ มีนาคม 2567
4. **ข้ามแดน (Transboundary)** — แผนที่จุดความร้อน (FIRMS) ใกล้สถานีชายแดนแยกสีตามประเทศ และตารางเหตุการณ์
   การลำเลียงข้ามแดน เช่น แม่ฮ่องสอน (18 มี.ค. 2568)
5. **ผลการทดสอบ (Performance)** — ตาราง RMSE ต่อขอบฟ้า, นัยสำคัญทางสถิติ (CI), และ ablation — **รายงานตามจริง**
6. **เกี่ยวกับโครงการ (About)** — สรุปโครงการ, 3 จุดใหม่, แหล่งข้อมูล, ข้อจำกัด/ความซื่อสัตย์ และข้อตกลงการใช้
   ซอฟต์แวร์ (Disclaimer) ฉบับเต็ม

### การอ่านผลอย่างถูกต้อง (ความซื่อสัตย์ของระบบ)

- ระบบเป็น **hindcast** (ใช้ข้อมูลอากาศ ERA5 ย้อนหลัง) เพื่อสาธิตความสามารถ ไม่ใช่ระบบเรียลไทม์
- **ความได้เปรียบเชิงพยากรณ์เหนือ persistence มีจำกัด** (เด่นที่ 48 ชม. ราว 2% และยังไม่มีนัยสำคัญในปีเดียว)
  และกลไกกราฟไม่ได้ยกระดับความแม่นยำเหนือความผันผวนจากการสุ่ม seed
- **คุณค่าหลักของระบบคือการระบุแหล่งกำเนิดที่อธิบายได้** (XAI) ซึ่ง baseline ทำไม่ได้
- การระบุแหล่งกำเนิดเป็น **การประมาณจากโมเดล** เทียบกับข้อมูลไฟ FIRMS ไม่ใช่การวัดตรง — ไม่ควรใช้กล่าวโทษเชิงการทูต

---

## ส่วนที่ 2 — Pipeline ประมวลผล (สำหรับนักพัฒนาและการทำซ้ำผล)

รันด้วย `UV_NO_SYNC=1 uv run python <script>` (หลังแก้ `.pth` ตาม `docs/INSTALL.md`)

| ลำดับ | สคริปต์ | หน้าที่ |
|---|---|---|
| 1 | `scripts/01_download_all.py` | ดาวน์โหลดข้อมูล (โหมด `realtime` / `discover` / `backfill`) — ต้องมี API keys |
| 2 | `scripts/02_preprocess.py` | ทำความสะอาด + รวม + normalize → `data/processed/{dataset,hotspots,scalers,stations_metadata}` |
| 3 | `scripts/03_train.py model=mtgnn` | เทรนโมเดล → `checkpoints/mtgnn/best_model.pt` |
| 4 | `scripts/04_evaluate.py` | RMSE ต่อขอบฟ้า + hybrid ensemble → `outputs/evaluation_*.json` |
| 5 | `scripts/05_attribution.py` | source attribution (เหตุการณ์ มี.ค. 2567) |
| 6–12 | `scripts/06`–`12` | การทดสอบความเข้มงวด: NWP sensitivity, นัยสำคัญ, ablation, ML baseline, transboundary |
| 13 | `scripts/13_report_figures.py` | สร้างรูปประกอบรายงานจาก `outputs/*.json` |
| — | `scripts/generate_report.py` | สร้างรายงานฉบับสมบูรณ์ (`outputs/NSC2026_Final_Report.docx`) |

ตัวอย่างคำสั่ง:

```bash
# ค้นหาสถานี / ดาวน์โหลดย้อนหลัง (ต้องมี API keys)
uv run python scripts/01_download_all.py discover
uv run python scripts/01_download_all.py backfill --start-year 2022 --end-year 2025

# เทรนและประเมินผล
UV_NO_SYNC=1 uv run python scripts/03_train.py model=mtgnn
UV_NO_SYNC=1 uv run python scripts/04_evaluate.py
```

---

## ส่วนที่ 2.5 — แจ้งเตือนล่วงหน้า 48 ชม. ผ่าน Telegram (สำหรับ demo สด)

`scripts/16_telegram_alert.py` แปลงพยากรณ์สด (โหมด live NWP เดียวกับ dashboard —
`app.lib.inference.live_forecast`) เป็นข้อความภาษาไทย (ระดับ AQI ไทย + คำแนะนำกลุ่มเสี่ยง +
ช่วง 90% หากมีการ calibrate conformal แล้ว) แล้วส่งเข้า Telegram ผ่าน Bot API — ใช้สำหรับสาธิตสดต่อหน้ากรรมการ
(โทรศัพท์ในห้องรับ push notification ทันที)

ต้องตั้งค่า `TELEGRAM_BOT_TOKEN` และ `TELEGRAM_CHAT_ID` ก่อน (ดูวิธีสร้างบอทและหา chat id ใน
`docs/INSTALL.md` ขั้นที่ 5) — ยกเว้นเมื่อใช้ `--dry-run` ซึ่งไม่ต้องตั้งค่าใด ๆ

```bash
# ดูข้อความตัวอย่างก่อน ไม่ส่งจริง
uv run python scripts/16_telegram_alert.py --dry-run

# ตรวจทุกสถานี ส่งเฉพาะสถานีที่พยากรณ์แตะเกณฑ์ (ค่าเริ่มต้น 37.5 µg/m³)
uv run python scripts/16_telegram_alert.py --station all

# บังคับส่งสถานีเดียวเสมอ (โหมด demo ไม่ต้องรอค่าฝุ่นสูงจริง)
uv run python scripts/16_telegram_alert.py --station 225579 --force
```

พารามิเตอร์หลัก: `--station <station_id|all>`, `--threshold <ug/m3>`, `--force` (ส่งเสมอ),
`--dry-run` (พิมพ์ข้อความแทนการส่งจริง) เกณฑ์เริ่มต้น 37.5 µg/m³ ตรงกับเส้นแบ่งระดับ AQI ไทย
"ปานกลาง" → "เริ่มมีผลต่อกลุ่มเสี่ยง" ใน `app/lib/aqi.py` (มาตรฐาน PM2.5 24 ชม. ของไทย)

> โหมดสดใช้ข้อมูลพยากรณ์อากาศ (NWP) แทน ERA5 ย้อนหลัง จึงมีความคลาดเคลื่อนมากกว่าผลประเมินหลักของระบบ
> (ข้อความแจ้งเตือนมีบรรทัด caveat นี้กำกับไว้เสมอ)

---

## ส่วนที่ 3 — การทำซ้ำผลหลัก (Reproduce key results)

- **ความแม่นยำการพยากรณ์:** `UV_NO_SYNC=1 uv run python scripts/04_evaluate.py`
- **การระบุข้ามแดน:** `UV_NO_SYNC=1 uv run python scripts/10_transboundary_attr.py`
- ผลลัพธ์ทั้งหมดเป็นไฟล์ JSON ที่ตรวจสอบซ้ำได้ใน `outputs/`
- ค่าทุกค่าในหน่วย µg/m³ คำนวณผ่านโมดูลร่วม `src/training/evaluation.py` (กัน denormalization drift)

> หมายเหตุการอ่านผล: ชุด validation และ test เป็นข้อมูล **ปี 2568 ทั้งคู่** (ต่างกันที่วิธีเทรนและการเป็น held-out
> ไม่ใช่คนละปี) — รายละเอียดในรายงานฉบับสมบูรณ์ หัวข้อผลการทดสอบ

---

## คำสั่งที่ใช้บ่อย (Quick reference)

```bash
UV_NO_SYNC=1 uv run streamlit run app/streamlit_app.py   # เปิด dashboard
uv run pytest tests/ -q                                  # รันชุดทดสอบ (397 tests)
uv run ruff check src/ && uv run black --check src/      # ตรวจคุณภาพโค้ด
```
