# SESSION 19 NOTES — 2026-07-12
Status: PHASE_COMPLETE

## What was accomplished

Roadmap item 11 (both packages) + presenter-script demo-prep + the
user-approved pre-SIMS report update. Orchestration: architect specced,
implementers executed, Fable independently re-verified every number and
gate before each commit (caught 1 presenter-script overclaim + 1 report
ambiguity; both fixed before commit).

- **`84dbd74` presenter script (demo prep)**: matrix + multi-seed
  talking points in `outputs/pitch/presenter_script_th.md`. Iron
  phrasing: never quote 63% without the multi-seed range; only Mae Sot
  carries the both-checkpoints-consistent claim (Maesai flagship demo 0%
  vs report 94.3% disclosed — Fable caught this overclaim in review,
  implementer corrected 3 spots).
- **`499a099` item 11 Package A**: `src/explain/backtrajectory.py` +
  `scripts/22_backtrajectory.py` + 21 tests (2 sign-convention oracles)
  → NEW `outputs/backtrajectory_test2025.json`. Kinematic RK2 backward
  48h from ERA5 10m winds at MHS peak-PM2.5 anchors, polygon country
  labels, corridor FRP ≤50km. **Pre-registered binary agreement 2/5;
  BOTH model-foreign events corroborated** — flagship 2025-03-18: 57.1%
  of backward hours over Myanmar (Laos 0.0 on all events), corridor FRP
  MM 12,918.4 vs TH 106.4; 03-13 also agrees. The 3 disagreements:
  air transits Myanmar daily (border station) but corridor foreign FRP
  ≈0 (03-30: MM 0.0) — supports the model's near-zero calls. No
  thresholds tuned post-hoc.
- **`bc3a53a` item 11 Package B**: additive trajectory-map section +
  agreement table + 5 caveats in `app/views/transboundary.py`, cached
  loader in `data_access.py`. AppTest both scenarios 0 exceptions,
  headless boot 200.
- **`5578fd9` REPORT UPDATE (user-approved iron-rule lift, report files
  only)**: REPORT_DRAFT.md + generate_report.py, additive: §3.2(7-8),
  §4.5 system capabilities, §6.7 conformal (±5.9/7.8/11.5/15.5, held-out
  0.885/0.882/0.882/0.870, worst 225585 0.818→0.698), §6.8 exceedance
  (48h BSS 0.306/0.282 @37.5 and 0.190/0.170 @75; base-rate caveat),
  §6.9 2026 OOS (persist 4.08/7.22/11.99/17.52 vs model 5.47/7.67/11.46/
  15.92 → +4.4%/+9.1% @24/48h; magnitude non-comparability caveat),
  §6.10 matrix+multi-seed, §6.11 back-trajectory (2/5 reported plainly),
  refs [9] Lei 2018 [10] Wilks 2011, abstract TH+EN appended 2026
  sentence. docx regenerated + verified (6 new headings, 8 shapes,
  TH Sarabun New 16pt, forbidden numbers absent, load-bearing numbers
  present). Reviewer(opus) audit was cut off by session limit mid-run
  (~25 tool calls, nothing blocking found); Fable completed the
  remaining checklist itself (Laos=0 check, §4.5 file existence,
  2-line-only deletion scope) and fixed the one 🟡 found (24–59% scope
  ambiguity in §6.11, both files, docx regenerated).

Gates before every commit (run by Fable): pytest 397 pass, ruff 4 dirs
clean, black clean.

## Current state (branch, last commit, open work)

- Branch `feat/demo-features` @ `5578fd9` (+ this handoff commit).
  **5 commits NOT pushed** (`371656b..HEAD`); origin at `371656b`
  (session-18 handoff was already pushed — discovered at session start;
  CI SUCCESS there). Per-push ask rule in force.
- SIMS: **still NOT submitted** (user-confirmed this session; deadline
  17 Jul 17:00). The report freeze was lifted DELIBERATELY this session
  for the report files only; everything else (checkpoints, pre-existing
  outputs/*.json, figures/report, data/processed) remains untouched.
- `outputs/NSC2026_Final_Report.docx` regenerated, untracked by choice.
- Telegram bot: still user-todo before demo (21 Aug).

## What was NOT done (and why)

- Push — needs user approval (asked at session end).
- HYSPLIT manual appendix (optional Package B extra in spec) — skipped;
  scripted ERA5 trajectory is the reproducible core.
- Roadmap items 5/13 (retrains) and 12 (graph expansion) — post-SIMS by
  design.
- DEMO_ROADMAP.md not edited (progress tracked in session notes).

## Next session must start with (exact first actions)

1. Ask: SIMS submitted? User must open docx in Word, **F9 to update
   TOC**, upload with INSTALL+USER_GUIDE before 17 Jul 17:00.
2. If push was approved and done: verify CI green via GitHub API
   (credential-fill Bearer token pattern; gh CLI not installed).
3. Clarify the user's mention of "วันที่ 25 ต้องแสดงผลงาน" — not in the
   project calendar (demo final round is 21 Aug); may be a new
   presentation date that needs demo-prep scheduling.
4. Remaining queue: post-SIMS items 5/13 (retrains), 12 (optional),
   demo rehearsal with the new dashboard sections.

## Critical context to preserve (gotchas, fragile files, decisions)

- **Back-trajectory honesty framing** (load-bearing for report + demo):
  agreement is 2/5 under the PRE-REGISTERED binary metric — never
  present as 5/5; the win is "both model-foreign events corroborated +
  disagreements explained by corridor FRP ≈0". Laos hours are 0.0 on
  all 5 events so "ผ่านเมียนมา" is the accurate phrasing.
- `load_era5_window` / plain `xr.open_dataset` FAILS on the Thai cwd —
  must go through `era5._open_nc` (ASCII temp copy). ERA5 lats ship
  DESCENDING; `make_wind_field` flips them.
- Report §6.9 magnitude trap: 2026 burn-season RMSEs are NOT comparable
  to full-year 2025 magnitudes — only % improvements. Never quote 17.52
  next to 12.99 without the caveat.
- `evaluation_2026burn.json` also contains a `mtgnn_demo` block
  (+2.9%/+6.1%) — report cites only `mtgnn_split2`; don't mix.
- Exceedance model numbers live at `checkpoints.primary.model.{h}.{thr}`
  in the JSON (not top-level).
- generate_report.py DOES parse `**bold**`/`` `code` ``/italic via
  `add_md_runs` (architect's plan wrongly said otherwise; implementer
  followed the real convention).
- Session limit can kill subagents mid-run (reviewer died at ~25 calls);
  partial output is recoverable from the task notification — decide
  finish-inline vs re-spawn.

## NEW DEADLINE resolved at session end (user pasted the invitation)

**25 Jul 2026, 08:30, room 11M280, Informatics, Burapha U** — NSC East
regional round (รอบนำเสนอผลงาน). Requirements:
- **A0 poster** (NSC 2026 template on the organizer's Google Drive,
  folder "NSC 2026_TemplatePoster") — board setup from 08:00; bring demo
  equipment.
- Funding paperwork (user admin, hand in at registration): ใบสำคัญรับเงิน,
  ข้อตกลงรับทุน ×2 from GENA signed by head + advisor, ID copies ×2
  certified, no corrections allowed on the agreement.
Priority order now: SIMS report (17 Jul) → A0 poster + presentation
rehearsal (25 Jul) → final round (21 Aug).

## Open questions

- Whether to also cite the 2026 demo-checkpoint numbers (+2.9/+6.1)
  anywhere — currently deliberately excluded to keep §6.9 single-model.
- Poster A0: generate with python-pptx (like the pitch deck) vs Canva —
  next session decision; content should lead with XAI + 2026 OOS result
  + honest-uncertainty story, reusing verified numbers from the report.
