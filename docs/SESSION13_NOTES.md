# SESSION 13 NOTES — 2026-07-10
Status: IN_PROGRESS

เริ่มงาน demo roadmap ก่อนส่งรายงาน (ตรวจกติกา NSC แล้ว: ทำได้ — booklet หน้า 33 ให้ส่งรายงาน
ฉบับปรับปรุงใหม่ได้; เกณฑ์รอบชิง Technique 25/Creativity 25 ให้คะแนนการพัฒนาต่อ)
ทำงานแบบ orchestrator: Fable คุม + แจก implementer/architect/reviewer เป็น background agents

## What was accomplished

- **Branch ใหม่ `feat/demo-features`** (แตกจาก feat/session-4-explain-app หลัง commit 943c412
  ที่แก้กติกาเหล็ก roadmap ข้อ 1: ทำก่อนส่งได้บน branch แยก, ห้ามแตะไฟล์ที่รายงานพึ่งพา
  — `outputs/*.json`, `outputs/figures/report/`, `checkpoints*/`, `generate_report.py`,
  `REPORT_DRAFT.md`, `data/processed/*` — จนกว่าจะ submit SIMS 17 ก.ค. 17:00)
- **Roadmap ข้อ 1 ส่วน scraper — COMMITTED `74debd7`:** `src/data/scrapers/openmeteo.py` +
  `tests/test_openmeteo.py` (7 tests) — verify แล้ว blh มีครบ 168 ชม. ไม่มี null,
  `wind_speed_unit=ms` ใช้ได้; แปลง °C→K, speed+direction→u10/v10 (ทิศ "from")
- **Roadmap ข้อ 8-13 — COMMITTED `740afb3`:** ข้อเสนอรอบสอง (eval ฤดูเผา 2026 แบบ frozen model,
  conformal intervals, exceedance re-scoring, HYSPLIT cross-check, ขยายกราฟ >18 โหนด,
  residual-learning variant) + บันทึก "สิ่งที่ไม่ทำ" (AOD, เปลี่ยน arch, ตกแต่ง dashboard เกิน)
- **Roadmap ข้อ 7 (CI + cards) — COMMITTED `7245121`:** `.github/workflows/ci.yml`
  (uv sync --extra dev!, native PyG ผ่าน `uv pip` — venv CI ไม่มี pip, ผมแก้จาก python -m pip),
  `docs/MODEL_CARD.md` (+limitation n=18 ตาม roadmap ข้อ 12), `docs/DATA_CARD.md`
  (FIRMS = VIIRS_NOAA20 SP+NRT เท่านั้น — verify จาก scraper defaults)
- **Roadmap ข้อ 1 ส่วน live mode — เขียนเสร็จ รอ reviewer, ยังไม่ COMMIT:**
  - Modified: `app/lib/inference.py` (LiveDataError, load_air4thai_map, live_forecast),
    `app/views/forecast.py` (radio hindcast ↔ "พยากรณ์วันนี้ (NWP สด)")
  - New: `app/lib/air4thai.py` (history fetch, TLS verify→fallback on SSLError only),
    `app/lib/nwp.py` (Open-Meteo past_days + build_live_window),
    `configs/air4thai_station_codes.json` (mapping 17 สถานี — ทุกแถว distance 0.0 ม. ชื่อตรงเป๊ะ),
    `scripts/14_air4thai_station_map.py` (builder ครั้งเดียว), `tests/test_nwp.py` (11 tests),
    `docs/api_quirks.md` (ไฟล์นี้**ไม่เคยมีจริงใน git** แม้ CLAUDE.md อ้างถึง — architect สร้างให้)
  - Gates ผ่านแล้วโดย architect: **pytest 243 passed** (232+11), ruff clean, black clean,
    end-to-end จริงได้ (18,4) พยากรณ์, TLS fallback ทำงาน
- แก้ข้อมูลผิดใน roadmap ข้อ 7 (scraper air4thai ไม่เคยมี) — commit `c2c9d64`

## Current state (branch, last commit, open work)

- Branch `feat/demo-features`, HEAD = `7245121`; ยังไม่ push (push ต้องขอ confirm ต่อครั้ง)
- **ค้างอยู่: reviewer (Opus) กำลังตรวจ live-mode change set** (8 ไฟล์ด้านบน) — จุดที่สั่งเจาะ:
  denorm path ต้องใช้ evaluation.py helpers, timezone air4thai UTC+7→UTC, feature order ตรง
  loader, empty-hotspot forward, TLS ไม่โดนจาก tests, cache key, เลขต้องห้าม
- ถ้า APPROVE → commit ทั้ง 8 ไฟล์ (message: feat: live NWP forecast mode); ถ้า 🔴 →
  ส่งกลับ architect (agent เดิมยังปลุกต่อได้)

## What was NOT done (and why)

- ยังไม่ push `feat/demo-features` ขึ้น origin (รอ commit live mode ให้ครบชุดก่อน)
- Roadmap ข้อ 2-6, 8-13 ยังไม่เริ่ม — ข้อ 2 (attribution matrix) และข้อ 5 (retrain) ต้องรอ
  หลัง submit SIMS เพราะเขียน outputs/ + แตะ pipeline
- `outputs/pitch/qa_defense.md` ยังตัวเลขเก่า (งานเก็บตก demo prep เหมือนเดิม)
- CI ยังไม่เคยรันจริงบน GitHub (จะรันครั้งแรกตอน push) — เสี่ยงที่รู้: runner ดึง torch cu126
  (~ใหญ่) อาจช้า/เต็ม disk; ถ้า timeout ค่อยปรับ

## Next session must start with (exact first actions)

1. เช็คผล reviewer: ถ้า session นี้ปิดก่อนผลมา ให้รัน gates เอง
   (`UV_NO_SYNC=1 uv run pytest tests/ -q` คาด 243 passed; ruff/black) แล้วอ่าน diff
   `app/lib/inference.py`/`forecast.py` + ไฟล์ใหม่ 6 ไฟล์เอง หรือส่ง reviewer ใหม่
2. Commit live mode (ถ้าผ่าน) → push `feat/demo-features` (ขอ confirm ผู้ใช้) → CI รันครั้งแรก
3. **ก่อน 17 ก.ค.:** ผู้ใช้เปิด docx กด F9 + upload SIMS (ยังเป็น action ค้างของผู้ใช้!)
4. งานถัดไปในคิว: roadmap ข้อ 3 (counterfactual — ไม่แตะไฟล์รายงาน ทำได้เลย),
   ข้อ 6 (Telegram), ข้อ 9 (conformal); หลัง submit: ข้อ 2, 4, 5, 8, 10, 11

## Critical context to preserve

- **กติกาเหล็กใหม่:** งาน demo ทำได้ก่อนส่งรายงาน แต่ห้ามแตะไฟล์ที่รายงานพึ่งพา (รายการใน
  DEMO_ROADMAP กติกาเหล็กข้อ 1) จนกว่า submit SIMS เสร็จ
- **แม่แจ่ม (location_id 225693) ไม่อยู่ในฟีด air4thai** (สถานีใกล้สุด 70 กม.) — โหมดสด:
  center-fill ในกราฟ (โมเดลต้องครบ 18 โหนด) แต่ถอดจาก selector, เปิดเผยใน caption;
  hindcast เลือกได้ปกติ
- โหมดสด v1 **ไม่มีโหนดไฟ** (hotspots ว่าง) + attribution ปิด — บอกใน UI แล้ว; FIRMS NRT
  เข้ากราฟสดเป็นงานอนาคต
- Gap policy โหมดสด: ffill ใน raw space → scale, ปฏิเสธถ้า coverage <70%
  (ต่างจาก hindcast ที่ nan_to_num(0) หลัง scale = center-fill — จดใน comment แล้ว)
- air4thai TLS chain เสีย → fallback verify=False เฉพาะ SSLError (log warning) —
  จดใน docs/api_quirks.md; endpoint ทางการเป็น http:// อยู่แล้ว
- Open-Meteo: `past_days` (≤92) + `forecast_days` ใน call เดียว ครอบ window 24 ชม. + 48 ชม.
- **CI ต้อง `uv sync --extra dev`** (bare uv sync ถอน pytest/ruff/black!) และ `uv pip install`
  (venv ไม่มี pip)
- Orchestration ที่ผู้ใช้ต้องการ: Fable คุม แจกงาน implementer/architect/tester/reviewer
  เป็น background เพื่อประหยัดโทเคน; agent ห้าม commit เอง — Fable review แล้ว commit

## Open questions

- geoBoundaries license/เวอร์ชันของ borders geojson — `[NEEDS VERIFICATION]` ใน DATA_CARD
  ต้องเช็ค geoboundaries.org ก่อนเผยแพร่สาธารณะ
- NWP blh กับ ERA5 blh สเกลตรงกันแค่ไหน — `[NEEDS VERIFICATION]` (โมเดลถูกทดสอบว่าทน noise
  ระดับหนึ่งตาม nwp_sensitivity.json แล้ว แต่ยังไม่ได้ audit ตัวเลขเทียบตรง ๆ)
