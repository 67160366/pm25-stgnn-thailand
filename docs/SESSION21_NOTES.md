# SESSION 21 NOTES — 2026-07-19
Status: PHASE_COMPLETE

## What was accomplished
- **SIMS confirmed submitted on time 17 Jul** (user confirmation 19 Jul — final report + install/user manuals).
- `0b296e6` — leftover SIMS-manuals work committed (scripts/generate_manuals.py, INSTALL/USER_GUIDE
  test count 213→397, pyproject ruff C901). Gates: 397 pass, ruff/black clean.
- `56edca3` — **A0 poster for 25 Jul regional round**: scripts/generate_poster.py renders the
  organizer template zip (A3-DONE.pdf → 450 dpi jpg background), overlays HTML/CSS content at exact
  A0 (841×1189 mm), prints via Chrome headless → `outputs/NSC2026_Poster_A0.pdf` (2.4 MB) +
  `outputs/poster/poster_preview.png`. Font: Leelawadee UI (machine lacks TH Sarabun New). Build has
  CANON_FORBIDDEN (hard fail) / CANON_REQUIRED (warn) guards.
- `efde52e` — presenter_script_th.md line 130 quoted **val-2025** RMSE (12.68/8.67); fixed to
  held-out **test-2025** (12.66/8.76 per outputs/evaluation_test2025.json). Root cause: val/test mixup.
- Subagent audits before print: reviewer (Opus) verified EVERY poster number vs outputs/*.json → GO;
  caught 1 🔴 (hero tile said "5 seeds"; truth = **3 seeds**, 0.542 = mean(0.627, 0.0, 1.0) per
  transboundary_uncertainty.json) — fixed, and "5 seeds" added to CANON_FORBIDDEN. tester verified
  the rest of the presenter "ตัวเลขสำคัญ" block clean. implementer produced 300-dpi re-renders of 5
  report figures → `outputs/poster/assets_hires/` (untracked; tracked 150-dpi originals untouched).

## Current state (branch, last commit, open work)
- Branch `feat/demo-features`; HEAD after this handoff commit; **commits since `67071c5` are UNPUSHED
  (per-push ask policy)**: 0b296e6, 56edca3, efde52e, + this handoff.
- Poster PDF/assets/zip untracked by design (heavy binaries; PDF rebuildable via
  `uv run --no-sync python scripts/generate_poster.py` — needs organizer zip in repo root, poppler,
  Chrome/Edge).
- Working tree clean apart from long-standing untracked binaries (booklet, proposal docx, manuals
  docx, report docx/pdf, user journeys, proposal_text.txt, review_prompt.md).

## What was NOT done (and why)
- Poster not sent to print — physical step, user's side; needs 1–2 days lead before 25 Jul.
- Telegram bot real delivery still unverified (user must create bot token) — needed before 21 Aug.
- Roadmap items 5/13 (retrains) and 12 (optional) untouched — post-SIMS queue, lower priority than
  regional round.
- No pitch-deck/report changes — poster only.

## Next session must start with (exact first actions)
1. Ask user: poster printed? SDG chips verdict (see Open questions)? push approval given/received?
2. If poster tweaks requested: edit `scripts/generate_poster.py` content constants → rerun the
   build command above → check `outputs/poster/poster_preview.png`.
3. Then: rehearsal support with `outputs/pitch/presenter_script_th.md` (numbers now corrected) +
   `qa_defense.md`; funding paperwork (ใบสำคัญรับเงิน, ข้อตกลงรับทุน ×2, ID copies ×2) is user admin.

## Critical context to preserve (gotchas, fragile files, decisions)
- **Multi-seed ensemble is 3 seeds, never 5** (report_orig / full_s0 / full_s1). On stage: 62.7%
  must always be paired with mean 54.2% (range 0–100%). "5 seeds" is now a forbidden string in the
  poster build.
- Presenter-script numbers were val/test mixed once — if any new number gets quoted anywhere, check
  against outputs/*.json, not other prose files.
- **Booklet facts (reviewer-verified)**: Disclaimer NOT required on posters (only in the software —
  already satisfied); poster is optional appendix "(ถ้ามี)"; **final round 21 Aug is ONLINE** —
  needs ≤7-min video 28p14e01196.mp4, 1–1.5 p summary form, team photos .pptx.
- Poster dates use พ.ศ. (matching report register); EN title stays CE.
- generate_poster.py figure sources: prefers outputs/poster/assets_hires (300 dpi), falls back to
  outputs/figures/report (150 dpi), architecture falls back to tracked repo-root
  architecture_diagram.png (byte-identical to the 300-dpi re-render).
- Poster identity: solo นายรณชัย ขาวสะอาด; advisor ดร.วัชรพงศ์ อยู่ขวัญ; สาขาปัญญาประดิษฐ์ประยุกต์
  และเทคโนโลยีอัจฉริยะ คณะวิทยาการสารสนเทศ ม.บูรพา; official TH title = REPORT_DRAFT.md:9.

## Open questions
- SDG 3/11/13 chips on the poster are an inference (SDG appears nowhere in repo docs) — user keep/drop?
- Swap RMSE figure for transboundary_events.png (300-dpi ready; shows attribution tracking truth
  across all 5 events)? Current choice: RMSE stays.
- Push approval for the 4 unpushed commits.
