# SESSION 15 NOTES — 2026-07-11
Status: PHASE_COMPLETE

ปิด roadmap คิวที่ค้างจาก S14 ครบทั้งสาม (ข้อ 9 → 3 → 6) + งานเก็บตก pitch
ทำแบบ orchestrator: Fable ตรวจ+commit, architect ทำข้อ 9+3 (agent เดิมต่อ context),
implementer ทำ pitch re-sync + ข้อ 6 ขนาน (แบ่ง scope ไฟล์ไม่ให้ชน, gates รวม Fable รันเอง)

## What was accomplished

- **`32bb95a` pitch re-sync:** qa_defense.md + presenter_script_th.md เป็น canon ใหม่
  (เรือธง 2025-03-18: 72.1%→62.7%, "~59 เท่า", เลิก claim "ทั้งสองโมเดล" — เขียนตรง ๆ ว่า
  tracking เป็นของโมเดล report/split2, โมเดล pitch ให้ ~0%) verify กับ JSON จริงก่อน commit
- **`c19a872` Roadmap ข้อ 9 — conformal intervals:** `src/training/conformal.py` (pure,
  k=ceil((n+1)(1−α)) ตาม Lei et al. 2018) + `scripts/15_conformal_calibrate.py` →
  `outputs/conformal/conformal_intervals.json` (**COMMIT ไฟล์นี้ด้วย** — app อ่านตอน runtime;
  .gitignore ไม่ ignore path นี้) calibrate บน val 2024 (8713×18) กับ checkpoint ที่ app ใช้จริง
  (`checkpoints/mtgnn/best_model.pt`, wind_mode=from_field) mode per-station (min cell n=4752,
  fallback pooled) — pooled halfwidth 6h ±5.9 / 12h ±7.8 / 24h ±11.5 / 48h ±15.5 µg/m³ @90%
  แถบใน dashboard ทั้ง hindcast+live, live มี caveat NWP mis-coverage
- **`686251f` Roadmap ข้อ 3 — counterfactual:** `gb_ig.counterfactual_occlusion` คืนกริด (N,H)
  เต็ม; logic occlude แชร์ผ่าน helper ใหม่ (restore ใน finally) โดย behavior ฟังก์ชันเดิมไม่เปลี่ยน;
  UI ในหน้า attribution (dropdown ประเทศ → before/after/Δ µg/m³ รายสถานี + ตาราง + bar chart)
  ~15–40 ms/ครั้ง; caveat what-if-ไม่ใช่-causal ครบ — เคสโชว์: 2024-02-25 ดับไฟเมียนมา 9 จุด
  → Δ สูงสุด +2.07 µg/m³ ขณะที่ดับไฟไทย 32 จุดแทบไม่เปลี่ยน
- **`6bedaf2` Roadmap ข้อ 6 — Telegram alerts:** `app/lib/telegram.py` (format_alert/should_alert
  pure + send_message) + `scripts/16_telegram_alert.py` (--station/--threshold/--force/--dry-run,
  exit codes 0/1/2) + 37 tests (mock หมด); threshold default 37.5 = เส้นแบ่ง band ใน aqi.py;
  docs: INSTALL.md (สร้าง bot) + USER_GUIDE.md (ส่วนที่ 2.5)
- **🔴 ที่ Fable จับเองใน review ข้อ 6:** ข้อความ exception ของ requests ฝัง URL ที่มี bot token
  → แก้ send_message ให้ redact token + `raise ... from None` (ถ้าคง chain, logger.exception
  จะพิมพ์ cause ที่มี token อยู่ดี) + test กันถอยหลัง
- Gates สุดท้ายรวมทุกงาน: **pytest 308 passed**, ruff 4 dirs clean, black clean

## Current state (branch, last commit, open work)

- Branch `feat/demo-features`, HEAD = `6bedaf2` — **ยังไม่ push** (4 commits ค้าง: 32bb95a,
  c19a872, 686251f, 6bedaf2) → CI ยังไม่เคยเห็นชุดนี้
- Working tree สะอาด ยกเว้น untracked-by-choice เดิม (docx/pdf/proposal_text/review_prompt)
- Roadmap ก่อน-submit ครบแล้ว: ข้อ 1, 3, 6, 7, 9 เสร็จ; เหลือชุดหลัง-SIMS (2, 4, 5, 8, 10, 11, 12, 13)

## What was NOT done (and why)

- **ส่ง Telegram จริงยังไม่ได้ทดสอบ** — ไม่มี token ในเครื่อง; ผู้ใช้ต้องสร้าง bot ผ่าน @BotFather
  แล้วใส่ TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID ใน .env (ขั้นตอนใน INSTALL.md) แล้วรัน
  `uv run python scripts/16_telegram_alert.py --station 225579 --force` ก่อน demo
- **Held-out conformal coverage บน test 2025** — เลข 0.900 ที่รายงานเป็น in-sample (calibration
  set) การันตีตรง nominal โดย construction; วัดจริงบน test ยังไม่ทำ (อ่านอย่างเดียว ทำได้เลย
  ถ้าอยากได้เลขซื่อสัตย์ไว้โชว์/ใส่รายงานปรับปรุง)
- `_province_from_name` (data_access.py) คืนชื่อเต็มเมื่อไม่มี comma — workaround ที่ call site
  ใน script 16 แล้ว แต่ตัวฟังก์ชันยังไม่แก้ (view อื่นอาจมีปัญหา cosmetic เดียวกัน) 🟡
- Open questions S13 ยังค้าง (geoBoundaries license, NWP blh vs ERA5 blh scale)

## Next session must start with (exact first actions)

1. **Push `feat/demo-features`** (ขอ confirm ผู้ใช้ต่อครั้งตามกติกา) → เช็ค CI เขียว
   (repo private, ไม่มี gh CLI — ใช้ token จาก `git credential fill` ยิง api.github.com)
2. เช็คผู้ใช้: F9 TOC + upload SIMS แล้วหรือยัง (**เดดไลน์ 17 ก.ค. 17:00**) — ถ้า submit แล้ว
   ปลดกติกาเหล็ก → เริ่มชุดหลัง-SIMS ได้ (ข้อ 8 คุ้มสุด → 10 → 2/4 → 11)
3. ถ้ามีเวลา: วัด held-out conformal coverage บน test 2025 (ข้อ NOT done)

## Critical context to preserve (gotchas, fragile files, decisions)

- **`outputs/conformal/conformal_intervals.json` ถูก commit โดยตั้งใจ** (app พึ่ง) — ถ้า
  re-calibrate ให้รัน `scripts/15_conformal_calibrate.py` แล้ว commit ทับ; ห้ามจัดเป็น
  "ไฟล์รายงาน" (ไม่อยู่ในรายการกติกาเหล็ก และรายงานไม่ได้พึ่ง)
- Conformal ผูกกับ checkpoint `checkpoints/mtgnn/best_model.pt` — ถ้า demo เปลี่ยนไปใช้
  checkpoint อื่น (เช่น split2) ต้อง calibrate ใหม่ ไม่งั้นช่วงไม่ valid
- telegram.send_message ห้ามคืน exception chain ของ requests (token ใน URL) — มี test คุมแล้ว
  (`test_token_never_leaks_into_error_or_chain`)
- กติกาเหล็กเดิมยังคุมจนกว่า submit SIMS: ห้ามแตะ `outputs/*.json` เดิม, `outputs/figures/report/`,
  `checkpoints*/`, `generate_report.py`, `REPORT_DRAFT.md`, `data/processed/*`
- Gates กว้างเท่า CI: ruff src/ app/ tests/ scripts/; black src/ tests/; pytest ตอนนี้ 308
- Orchestration ที่ work: agent เดิมปลุกต่อด้วยงานใหม่ได้ (ประหยัด context re-derive);
  งานขนานต้องแบ่ง scope ไฟล์ชัด และ Fable รัน full gates รวมเองก่อน commit เสมอ

## Open questions

- geoBoundaries license/เวอร์ชัน borders geojson — `[NEEDS VERIFICATION]` (ค้างจาก S13)
- NWP blh vs ERA5 blh สเกล — `[NEEDS VERIFICATION]` (ค้างจาก S13)
- Conformal บน live mode: mis-coverage จริงเท่าไรใต้ NWP input — ยังไม่มีเลขรองรับ (UI บอก
  caveat ไว้แล้ว) `[NEEDS VERIFICATION]`
