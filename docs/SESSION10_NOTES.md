# SESSION 10 NOTES — finish the NSC 2026 final report into a submittable artifact

Date: 2026-07-01 · Branch: `feat/session-4-explain-app` · 3 commits (NOT pushed at time of writing)

Mission (from `SESSION10_KICKOFF.md` §9): turn the honest, reviewer-approved Markdown report
(`docs/REPORT_DRAFT.md`) into the submittable **TH Sarabun New 16pt Word document**, roll out the
real NSC Disclaimer, fill the project code, and write the install/user manuals. **No new research —
format + finish only.** Honest thesis unchanged (forecast edge marginal/not-significant; graph does
not robustly help accuracy; demonstrated value = explainable transboundary source attribution).

---

## P10.1 — `scripts/generate_report.py` -> `outputs/NSC2026_Final_Report.docx`  (commit `143d1be`)

- New generator reuses the python-docx helper pattern from `scripts/generate_proposal.py`
  (`_set_cs_font` complex-script trick, `fmt`/`para`/`heading`/`body`/`bullet`/`numbered`,
  `add_table`+`Table Grid`, `add_picture`). **Content transcribed verbatim from `REPORT_DRAFT.md`**;
  the proposal's superseded denorm-bug numbers were NOT reused.
- New helpers added: inline `**bold**`/`*italic*`/`` `code` `` markdown -> runs; A4 page + `Inches(1)`
  margins (booklet rule); built-in Heading 1/2 styles re-fonted to TH Sarabun New so the Word TOC
  field populates in Thai; `add_toc` (Word `TOC` field, updates on open); `add_page_number` (footer
  `PAGE` field, different-first-page so the cover has no number).
- **Cover** follows the booklet final-report template (PDF p.44): รหัสโครงการ, title TH (+EN sub-line),
  หมวด 14 + ระดับนิสิต นักศึกษา, รายงานฉบับสมบูรณ์, เสนอต่อ สวทช., กระทรวง อว., ครั้งที่ 28,
  ประจำปีงบประมาณ 2569, ผู้พัฒนา/อาจารย์/สถาบัน/จังหวัด.
- Body: Ack -> abstract TH -> Abstract EN -> keywords -> TOC -> sections 1-12 -> Disclaimer (TH+EN).
  **8 figures** (`architecture_diagram.png` at `Cm(12)` portrait, `transboundary_map` at `Cm(12)`,
  the other 6 landscape charts at `Cm(16)`), **4 tables** (tools + 2 RMSE + NWP).
- Verified by reopening with python-docx: 8 `inline_shapes`, all 13 sections + Ack + Abstract present,
  default font = TH Sarabun New, TOC + footer PAGE fields present, **all 5 forbidden denorm numbers
  ABSENT** (10.21/14.32/+5.1%/+10.6%/4.46). Numbers independently re-extracted from `outputs/*.json`
  by an Explore agent = ALL MATCH (incl. line-215 main-model GBM 3.83/5.35/8.73/12.86 vs `baseline_ml.json`).
- **Before submitting:** open the .docx in Word once and press **F9 / Update Field** to populate the
  สารบัญ (TOC is a field, empty until updated). The 2 number-heavy figures (rmse_by_horizon,
  significance_48h) were visually re-checked against the authoritative numbers = correct.

## P10.2 — NSC Disclaimer rollout, 63 placeholders / 62 files  (commit `7e228e0`)

- **4 placeholder contexts:** (a) `# [TODO: NSC Disclaimer ...]` header comment, (b) bare
  `[TODO: ...]` line inside a docstring, (c) `"""[TODO: ...]` sharing the docstring-open line
  (`tests/test_loader.py` only), (d) `DISCLAIMER = "[TODO: ...]"` constant (`app/lib/ui.py`).
- Done with a one-off scratchpad script (60 header/docstring files) + targeted edits for `ui.py`,
  `about.py`, `README.md`, `test_loader.py`. **Rollout script was NOT committed.**
- **Lint constraint that drove the design:** `src/**` is NOT ruff-exempt (only `app/**` and
  `scripts/generate_*.py` ignore E501 + RUF001/2/3). So the concise header is **ASCII-punctuation Thai,
  <=99 chars/line, no ambiguous glyphs** (no en/em dash, middot, or curly quotes).
- **Footer vs About decoupling (important):** `ui.py` `DISCLAIMER` is a SHORT one-line pointer (it is
  interpolated into the every-page footer); new `DISCLAIMER_FULL_TH`/`DISCLAIMER_FULL_EN` carry the full
  booklet text and are rendered on the About page (`about.py`). `README.md` got the full TH+EN.
- Gate: `grep "TODO: NSC Disclaimer"` returns only historical docs (CLAUDE.md, SESSION*_KICKOFF,
  SESSION5/7_NOTES); ruff src/app/tests clean; black src/tests clean; **pytest 213 passed**; headless
  `AppTest` of the About page + footer renders with no exception.

## P10.3 — project code + romanization  (in commit `143d1be`)

- NSC project code **`28P14E01196`** filled on the cover (`generate_report.py` `PROJECT_CODE`) and
  `REPORT_DRAFT.md`. .docx regenerated (code on cover, placeholder gone).
- English author name **confirmed by the user as "Mr. Ronnachai Khaosa-ard"**; removed the leftover
  `*[ยืนยันการสะกด...]*` marker from `REPORT_DRAFT.md` §12. (Reviewer flagged "-ard" as a non-standard
  romanization of สะอาด; user confirmed it stands.)

## P10.4 — manuals  (commit `6c831a8`)

- `docs/INSTALL.md` (uv sync, native PyG extensions, the Thai-path `.pth` / `PYTHONUTF8` fix, manual
  `.env` + `~/.cdsapirc` setup, "data/ & checkpoints/ not in repo" note) and `docs/USER_GUIDE.md`
  (launch dashboard, the 6 pages, the 01-13 pipeline + reproduce commands). Report §12 appendix now
  carries concise install+usage summaries; full manuals are the two docs.

---

## Other changes & findings

- **Config:** added `ANN` to the `scripts/generate_*.py` ruff per-file-ignore (one-off Thai
  display-string generators; consistent with the existing E501/RUF ignore; also cleared the
  proposal/pitch_deck ANN debt). **No new dependencies** (`python-docx>=1.2.0` already in pyproject).
- **Stale-doc gaps fixed in README + INSTALL:** `install_native_deps.sh` does NOT exist (only `.ps1`;
  INSTALL gives the manual Linux/macOS commands); `env.example`/`.env.example` do NOT exist (README's
  `cp env.example .env` was wrong; the real `.env` is hand-made with `OPENAQ_API_KEY` + `FIRMS_API_KEY`,
  ERA5 uses `~/.cdsapirc`). CLAUDE.md's repo-structure mention of `.env.example` is also stale vs disk.
- **Reviewer (Opus) audit = PASS** — no honesty/number violations, disclaimer verbatim vs booklet, all
  sections present. 4 non-blocking items: 3 fixed (README `.sh`/`env.example` refs, `ui.py` `footer()`
  "placeholder" docstring), 1 = romanization (user-confirmed).

## Commits (on `feat/session-4-explain-app`, NOT pushed)

- `7e228e0` feat: roll out NSC software disclaimer across the codebase (62 files)
- `143d1be` feat: generate submittable NSC final report (Word) and fill project code (28P14E01196)
- `6c831a8` docs: add detailed install and user guides

## NOT done / handoff

- **`outputs/NSC2026_Final_Report.docx` is left UNTRACKED** by user choice (binary, regenerable via
  `UV_NO_SYNC=1 uv run python scripts/generate_report.py`). It is the primary SIMS upload.
- **Push was blocked** by the local permission policy (agent cannot run `git push`); the user must run
  `git push -u origin feat/session-4-explain-app` themselves, then optionally open a PR.
- Optional **P10.5** rigor (more seeds, other border stations, residual-target training) — none change
  the honest thesis; do only if asked.
- Submission deadline: **17 Jul 2026, 17:00 (SIMS)** — upload the .docx + INSTALL.md + USER_GUIDE.md.
