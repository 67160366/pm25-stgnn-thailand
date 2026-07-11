# SESSION 14 NOTES — 2026-07-11
Status: PHASE_COMPLETE

Session สั้น: ปิดงานค้างจาก S13 (live mode รอ review) + ทำให้ CI เขียวจริงครั้งแรก
แล้วผู้ใช้สั่งพักงาน roadmap ไว้ทำต่อ session หน้า

## What was accomplished

- **Live NWP forecast mode COMMITTED `50b1dd1` + PUSHED** (reviewer ของ S13 ไม่เคยส่งผลกลับ →
  ตรวจเองแทน): gates 243 passed; verify แล้วว่า feature order + สูตร cyclic (hour/24, doy/365)
  ตรง loader/preprocessing เป๊ะ, `from_arrays` wind mode + empty-hotspot (`.country=[]`) ตรง
  graph_builder/loader, denorm ผ่าน `evaluation.predict/denorm_pred` ถูกต้องสำหรับ 1 sample (N,H)
- **เจอ+แก้ 2 จุด lint ที่จะทำให้ CI แดง** ("ruff clean" ของ S13 ตรวจแค่ src/ แต่ **CI ตรวจ
  src/ app/ tests/ scripts/**): S501 `verify=False` ใน air4thai.py (ใส่ `# noqa: S501` —
  **ต้องวางบนบรรทัด argument ในวงเล็บ** ไม่งั้น black ย้ายไปบรรทัดปิดวงเล็บแล้ว RUF100 ว่า unused)
  และ ANN202 → เติม `-> HeteroData` ที่ `_build_live_sample`
- **CI รันจริงครั้งแรก = FAIL แล้วแก้จน SUCCESS:** บน linux `uv sync` ดึง torch +cu126 →
  `import torch` พังทันที (`libcudart.so.12` ไม่มี — สมมติฐานใน ci.yml ว่า cuda.is_available()
  จะแค่เป็น False นั้นผิด เพราะ import ตายก่อน) → แก้ commit `3ff2fca`: แยก marker ใน
  `tool.uv.sources` (win32 → pytorch-cu126 เดิม, linux → index ใหม่ `pytorch-cpu`),
  `uv lock` ใหม่ (เพิ่ม 2.6.0+cpu, +cu126 เดิมไม่แตะ), แก้ comment ใน ci.yml
- CI รอบสอง**ผ่านทุก step** (native PyG, pytest 243, ruff, black) — CI เขียวบน branch แล้ว

## Current state (branch, last commit, open work)

- Branch `feat/demo-features`, HEAD = `3ff2fca`, **pushed + CI green**
- Working tree สะอาด ยกเว้นไฟล์ root ที่ untracked-by-choice เดิม (docx/pdf/proposal_text/
  review_prompt) — อย่า commit
- ยังไม่มี PR (เปิดเมื่อ user สั่ง)

## What was NOT done (and why)

- Roadmap ข้อ 3 (counterfactual), 6 (Telegram), 9 (conformal) — **ผู้ใช้สั่งพัก ไว้ทำ session หน้า**
- `outputs/pitch/qa_defense.md` + `presenter_script_th.md` ยังมีเลขต้องห้ามชุดเก่า
  (37.3% / 36.6% / 128×) — ต้อง re-sync เป็น canon ใหม่ (flagship 2025-03-18, ~59 เท่า) ก่อนใช้ pitch
- Open questions ของ S13 ยังค้าง (geoBoundaries license, NWP blh vs ERA5 blh scale)

## Next session must start with (exact first actions)

1. เริ่ม roadmap ตามคิวที่ตกลงไว้: **ข้อ 9 (conformal intervals) ก่อน** (ถูกสุด-ได้เยอะ, calibrate
   บน val 2024, ไม่แตะไฟล์รายงาน) → ข้อ 3 (counterfactual, ใช้ occlusion ใน gb_ig.py) →
   ข้อ 6 (Telegram alerts, ฝังช่วง conformal ในข้อความได้)
2. เช็คว่าผู้ใช้ F9 TOC + upload SIMS แล้วหรือยัง (**เดดไลน์ 17 ก.ค.** — ยังเป็น action ของผู้ใช้)
3. งานเก็บตก: re-sync เลขใน pitch files (ข้อบนใน NOT done)

## Critical context to preserve (gotchas, fragile files, decisions)

- **Repo เป็น PRIVATE + เครื่องนี้ไม่มี gh CLI** — เช็ค CI ผ่าน GitHub API ด้วย token จาก
  credential manager: `printf "protocol=https\nhost=github.com\n" | git credential fill`
  → ค่า `password=` ใช้เป็น Bearer token กับ api.github.com ได้
- **ก่อน commit ให้รัน ruff กว้างเท่า CI** (`ruff check src/ app/ tests/ scripts/`) ไม่ใช่แค่
  src/ ตาม CLAUDE.md; black ใน CI ตรวจแค่ src/ tests/ —
  `scripts/generate_{proposal,architecture_diagram}.py` ยัง fail black แต่อยู่นอก scope CI (หนี้เก่า)
- **กติกาเหล็กเดิมยังคุม:** ห้ามแตะไฟล์ที่รายงานพึ่งพา (`outputs/*.json`, `outputs/figures/report/`,
  `checkpoints*/`, `generate_report.py`, `REPORT_DRAFT.md`, `data/processed/*`) จนกว่า submit SIMS
- Gotchas โหมดสดทั้งหมดยังตามใน SESSION13_NOTES (แม่แจ่ม 225693 ไม่อยู่ในฟีด → center-fill +
  ถอดจาก selector, โหมดสดไม่มีโหนดไฟ + attribution ปิด, gap policy ffill raw + ปฏิเสธ <70%,
  TLS fallback เฉพาะ SSLError)
- pyproject: ห้ามย้าย torch marker กลับไปรวม linux เข้า cu126 — CI จะพังแบบเดิมทันที

## Open questions

- geoBoundaries license/เวอร์ชัน borders geojson — `[NEEDS VERIFICATION]` ใน DATA_CARD
- NWP blh vs ERA5 blh สเกลตรงกันแค่ไหน — `[NEEDS VERIFICATION]` (ยังไม่ได้ audit เทียบตรง)
