# Data Card — PM2.5 STGNN Thailand

NSC 2026 หมวด 14 · แหล่งข้อมูลทั้งหมดที่ใช้ฝึกและประเมินผล Explainable STGNN สำหรับ
9 จังหวัดภาคเหนือของประเทศไทย (ที่มา: `docs/DESIGN.md`, `CLAUDE.md`)

พื้นที่ครอบคลุมทุกแหล่งข้อมูล: bbox lon 97.0–101.5°E, lat 16.0–21.0°N (9 จังหวัดภาคเหนือ),
ช่วงเวลา 2565–2568 (2022-01-01 ถึง 2025-12-31) (ที่มา: `CLAUDE.md`, `docs/DESIGN.md` §1)

---

## 1. Air4Thai PM2.5 (ผ่าน OpenAQ v3)

| หัวข้อ | รายละเอียด |
|---|---|
| ตัวแปร | PM2.5 (µg/m³) |
| สถานี | 18 สถานี Air4Thai (15 core + 3 extended) คัดเลือกจาก OpenAQ v3 `/locations` ด้วยเกณฑ์ `provider.name == "Air4Thai"`, bbox lon 97.0–101.5/lat 16.0–21.0, `datetimeFirst.utc < 2022-01-01`, `datetimeLast.utc > 2026-01-01` |
| API | OpenAQ v3 `/v3/sensors/{id}/measurements` (ไม่ใช่ `/hours`) — ดาวน์โหลดย้อนหลังเป็นช่วง 7 วันเพื่อกัน timeout 408 |
| ความละเอียด | รายชั่วโมง |
| ช่วงข้อมูล | 2565-01-01 ถึง 2568-12-31 |
| License / attribution | ข้อมูลสาธารณะจาก Air4Thai (กรมควบคุมมลพิษ) เข้าถึงผ่าน OpenAQ (CC BY 4.0 ตามนโยบาย OpenAQ) — ต้องอ้างอิงทั้ง Air4Thai และ OpenAQ เมื่อเผยแพร่ผล |
| Known quirks | OpenAQ response อาจเป็น UTF-16 พร้อม BOM ต้อง decode ด้วย `utf-8-sig`; datetime params ต้องเป็น ISO 8601 เต็มรูปแบบพร้อม timezone; `air4thai` history endpoint มี rolling window ~90 วันเท่านั้น ใช้ cross-validate เรียลไทม์ได้ แต่ **ใช้ฝึกโมเดลไม่ได้** |
| Known quality issue | **ความครอบคลุมข้อมูลปี 2566 ต่ำ** โดยเฉพาะช่วงฤดูเผา ทำให้ตัวอย่างฝึกใน 3-way split เหลือประมาณ 15,110 ตัวอย่าง (ลดลง ~37% จากที่คาด) — อาจกระทบความแม่นยำของโมเดล |

(ที่มา: `docs/DESIGN.md` §3, §4.1, `CLAUDE.md` หัวข้อ "Data sources", `docs/REPORT_DRAFT.md` §7 ข้อ 2)

## 2. NASA FIRMS — จุดความร้อนดาวเทียม (Fire hotspots)

| หัวข้อ | รายละเอียด |
|---|---|
| ผลิตภัณฑ์ | **VIIRS NOAA-20 เท่านั้น** — hybrid ระหว่าง `VIIRS_NOAA20_SP` (Standard Product, validated, ย้อนหลังถึง ~60 วันก่อนปัจจุบัน) กับ `VIIRS_NOAA20_NRT` (Near Real-Time, ~10 วันล่าสุด) ตามค่า default ใน `src/data/scrapers/firms.py`; MODIS เป็นตัวเลือกที่โค้ดรองรับแต่ไม่ได้ใช้ในชุดข้อมูลเทรน (ข้อความ "VIIRS/MODIS" ใน `docs/REPORT_DRAFT.md` §4.2(2) หมายถึงความสามารถของระบบ) |
| ตัวแปร | FRP (Fire Radiative Power, MW) ต่อจุดความร้อน, พิกัด (lat/lon), วันที่ตรวจจับ |
| API | NASA FIRMS API — `day_range` จำกัดสูงสุด 10 วันต่อ request, bbox เป็น lon,lat order |
| การประมวลผลก่อนเข้ากราฟ | จัดกลุ่มจุดความร้อนเชิงพื้นที่เป็น hotspot clusters (`src/data/hotspot_clustering.py`) เก็บ `total_frp`, `centroid_lat`, `centroid_lon` ต่อ cluster ต่อวัน |
| ความละเอียด | รายวัน (cluster) |
| ช่วงข้อมูล | 2565–2568 (ครอบคลุมช่วงเทรน/val/test ทั้งหมด) |
| License / attribution | ข้อมูลสาธารณะจาก NASA FIRMS (LANCE, NASA/GSFC) — ต้องอ้างอิง NASA FIRMS เมื่อเผยแพร่ผล |
| Known quality issue | ความถูกต้องของ attribution ขึ้นกับคุณภาพ/ความครอบคลุมของการตรวจจับไฟดาวเทียม (เมฆบัง/bias ของเซนเซอร์อาจทำให้พลาดจุดไฟ) |

(ที่มา: `docs/DESIGN.md` §4.2, `docs/REPORT_DRAFT.md` §4.2(2), §3.2 ข้อ 6, `CLAUDE.md` หัวข้อ "Data sources")

## 3. ERA5 Reanalysis (Copernicus)

| หัวข้อ | รายละเอียด |
|---|---|
| ตัวแปร | u10, v10 (ลม 10 ม.), t2m (อุณหภูมิ 2 ม.), d2m (จุดน้ำค้าง 2 ม.), blh (ความสูงชั้นบรรยากาศผสม) |
| ใช้งานในโมเดล | เป็น 5 จาก 10 input features ต่อสถานีต่อ timestep และเป็นตัวขับ edge weight ของเส้นเชื่อม type_b (wind-aware) |
| ความละเอียด | รายชั่วโมง |
| ช่วงข้อมูล | 2565–2568 |
| License / attribution | Copernicus Climate Change Service (C3S), ECMWF — ต้องอ้างอิงตามข้อกำหนด Copernicus License เมื่อเผยแพร่ผล (Hersbach et al., 2020) |
| Known quality issue | **ERA5 เป็นข้อมูล reanalysis ที่มีความล่าช้า** (ไม่ใช่ real-time forecast) จึงใช้ฝึกโมเดลได้ แต่การ inference จริงต้องใช้ NWP forecast (เช่น ECMWF HRES/GFS) แทน — ผลกระทบของความคลาดเคลื่อนนี้วัดไว้ใน `outputs/nwp_sensitivity.json` (ความได้เปรียบที่ 24h หายไปเมื่อ noise ถึง 0.25× ของส่วนเบี่ยงเบนธรรมชาติ, 48h ทนได้ถึงต่ำกว่า 0.5×) |

(ที่มา: `docs/DESIGN.md` §4.4, `docs/REPORT_DRAFT.md` §4.3, §4.4, §3.2 ข้อ 2, `CLAUDE.md` หัวข้อ "Data sources")

## 4. Border polygons — `app/assets/borders_th_mm_la.geojson`

| หัวข้อ | รายละเอียด |
|---|---|
| เนื้อหา | ขอบเขตประเทศจริง (real-border polygons) ของไทย/เมียนมา/ลาว (features `id: "THA"` และประเทศข้างเคียง) |
| แหล่งที่มา | geoBoundaries — ระบุไว้ใน `docs/SESSION_HANDOFF_pitch_review.md`: "ขอบเขต TH/MM/LA จาก geoBoundaries (1.5MB)" [NEEDS VERIFICATION: ระดับความละเอียด/เวอร์ชัน ADM ของ geoBoundaries ที่ดาวน์โหลดมาไม่ได้ระบุไว้ในเอกสาร] |
| ขนาดไฟล์ | 1,494,259 bytes (~1.4 MB), ตรวจสอบจริงในโฟลเดอร์ `app/assets/` วันที่ตรวจสอบ 2026-07-10 |
| การใช้งาน | `src/data/geocode.py` ใช้ point-in-polygon (numpy ray-casting) กับไฟล์นี้เป็น single source of truth สำหรับป้ายประเทศต้นทางของกลุ่มไฟ (hotspot clusters); `app/lib/geo.py` re-export ใช้วาดเส้นพรมแดนบน dashboard |
| License | ไม่ได้ระบุใน repo — [NEEDS VERIFICATION: license ของไฟล์ geoBoundaries ที่ใช้ ควรตรวจสอบกับ geoboundaries.org ก่อนเผยแพร่ต่อสาธารณะ] |

(ที่มา: `docs/SESSION_HANDOFF_pitch_review.md`, `src/data/geocode.py`, `docs/SESSION11_NOTES.md`, `docs/SESSION12_NOTES.md`)

### หมายเหตุสำคัญ — การแก้ไขป้ายประเทศ (country-label fix, 2026-07-07)

ป้ายประเทศต้นทางของกลุ่มไฟ (hotspot cluster) คำนวณด้วย **point-in-polygon เทียบกับขอบเขตประเทศจริง**
(`src/data/geocode.py`, คอมมิต `f643a9c`) ซึ่งมาแทนที่วิธี **bounding box** รุ่นแรกที่ตรวจพบว่า
**นับไฟเมียนมา/ลาวขาดไป — mislabeled 6,758 จาก 26,839 กลุ่ม (25.2%)** โดยกลืนไฟชายแดนเข้าไปเป็น
ไฟไทยผิดพลาด การแก้ไขนี้กระทบเฉพาะการจัดกลุ่มประเทศตอนคำนวณ attribution (metadata สำหรับ
รายงานผล) **ไม่กระทบฟีเจอร์ที่เข้าโมเดล** (feature ของโมเดลใช้ `total_frp`, `centroid_lat`,
`centroid_lon` ไม่ใช่ป้ายประเทศ) จึงไม่ต้องเทรนโมเดลใหม่ ผลการระบุแหล่งกำเนิดทั้งหมดในรายงาน
ฉบับปัจจุบัน (`docs/REPORT_DRAFT.md` §6.5) ใช้ป้ายประเทศที่แก้ไขแล้วเท่านั้น

(ที่มา: git commit `f643a9c` "feat: replace bbox country labels with real-border polygon
geocoding", `docs/REPORT_DRAFT.md` §6.5 หมายเหตุความถูกต้องของป้ายประเทศ)

## 5. Preprocessing สรุป

| ขั้นตอน | รายละเอียด | ที่มา |
|---|---|---|
| Normalization | `RobustScaler` แยกต่อสถานี (fit บนชุด train เท่านั้น) สำหรับ `pm25_scaled` | `docs/REPORT_DRAFT.md` §4.3, `CLAUDE.md` โครงสร้าง `src/data/preprocessing.py` |
| นโยบายช่องว่างข้อมูล (Missing Data Policy) | ช่องว่าง < 6 ชม. → linear interpolation · 6–24 ชม. → forward-fill (จำกัดตามความยาวช่องว่าง) · > 24 ชม. → mask ใน loss function (ไม่ imputed) · > 7 วันติดต่อกัน → ตัดทั้งเดือนออกจากชุดฝึก | `docs/DESIGN.md` §4.3 |
| Time encoding | `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` (cyclical encoding) | `docs/DESIGN.md` §6.1 |
| Hotspot clustering | จุดความร้อน FIRMS จัดกลุ่มเชิงพื้นที่เป็น cluster รายวัน ก่อนเข้าเป็นโหนดกราฟ | `src/data/hotspot_clustering.py`, `docs/REPORT_DRAFT.md` §4.2(2) |
| Graph edge construction | 3 ชนิดเส้นเชื่อม (type_a static-spatial ≤100km, type_b wind-aware ≤200km alignment>0.3, type_c fire-bipartite ≤500km alignment>0.4) คำนวณใหม่ทุกชั่วโมงตามลม ERA5 | `docs/DESIGN.md` §5.2 |

---

*เอกสารนี้เป็น one-page data card ตามมาตรฐานความโปร่งใสของชุดข้อมูล ML สร้างขึ้นสำหรับ NSC 2026
ทุกตัวเลขคัดลอกจากไฟล์ `docs/*.md`, git log, หรือไฟล์จริงใน `app/assets/` ที่อ้างอิงไว้ในวงเล็บ
ค่าที่ไม่สามารถยืนยันได้จากไฟล์ในโปรเจกต์ถูกทำเครื่องหมาย `[NEEDS VERIFICATION]` ไว้อย่างชัดเจน
ณ วันที่ตรวจสอบ 2026-07-10*
