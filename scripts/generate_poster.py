"""Build the NSC 2026 A0 poster PDF on top of the organizer template.

Pipeline:
    1. ensure assets in outputs/poster/assets/ — background rendered at 450 dpi
       from the organizer zip (repo root) via poppler pdftoppm.
    2. write outputs/poster/poster.html (A0 = 841 x 1189 mm).
    3. print to outputs/NSC2026_Poster_A0.pdf with Chrome/Edge headless and
       render a preview PNG with pdftoppm.

Charts are built as HTML/CSS by ``scripts/generate_poster_figures.py`` rather than
imported as PNGs: matplotlib drops stacked Thai tone marks (``ที่`` renders as
``ที``) and its A4-scale type is illegible once placed in a poster column.

The build fails if any superseded ("forbidden") result number appears in the
HTML, and warns if a canon number is missing — see CANON_* below.

Run: uv run --no-sync python scripts/generate_poster.py
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_poster_figures import (
    FIGURE_CSS,
    evidence_figure,
    pipeline_figure,
    rmse_figure,
)

REPO = Path(__file__).resolve().parents[1]
POSTER_DIR = REPO / "outputs" / "poster"
ASSETS = POSTER_DIR / "assets"
HTML_PATH = POSTER_DIR / "poster.html"
PDF_PATH = REPO / "outputs" / "NSC2026_Poster_A0.pdf"
PREVIEW_PATH = POSTER_DIR / "poster_preview.png"

POPPLER_BIN = Path(
    r"C:\Users\golfv\AppData\Local\Microsoft\WinGet\Packages"
    r"\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\poppler-25.07.0\Library\bin"
)
# Chrome lays out CSS millimetres at 96 dpi, so 1 mm = 96/25.4 px.
PX_PER_MM = 96 / 25.4

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# Embedded OFL Thai webfonts. Display face is a serif so headings sit at a
# genuinely different texture from the body — a single-family poster reads as
# untouched template output. Base64 so the print is reproducible anywhere.
FONTS = [
    ("Noto Serif Thai", "100 900", "NotoSerifThai-VF.ttf"),
    ("Sarabun", "400", "Sarabun-Regular.ttf"),
    ("Sarabun", "600", "Sarabun-SemiBold.ttf"),
    ("Sarabun", "700", "Sarabun-Bold.ttf"),
]


def _font_faces() -> str:
    """Return @font-face rules with each TTF inlined as a base64 data URI."""
    faces = []
    for family, weight, fname in FONTS:
        b64 = base64.b64encode((ASSETS / "fonts" / fname).read_bytes()).decode()
        faces.append(
            f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:normal;"
            f"font-display:block;src:url(data:font/ttf;base64,{b64}) format('truetype');}}"
        )
    return "".join(faces)


# Superseded numbers must never resurface (see CLAUDE.md / session notes).
CANON_FORBIDDEN = [
    "10.21",
    "14.32",
    "+5.1%",
    "+10.6%",
    "128 เท่า",
    "128:1",
    "37.3",
    "36.6",
    "33.2",
    "2024-02-16",
    "16 ก.พ.",
    "5 seeds",  # the uncertainty ensemble is 3 seeds (transboundary_uncertainty.json)
]
# Current canon numbers the poster is expected to carry. The single-seed 62.7% /
# 51.6 / 11.1 split is deliberately NO LONGER a headline (it cherry-picks one of
# three seeds that gave 0/62.7/100); the honest cross-seed mean 54.2% with its
# range now carries the model's attribution, backed by the robust 72.1% fire /
# 57.1% wind evidence.
CANON_REQUIRED = [
    "54.2",
    "72.1",
    "99.7",
    "59 เท่า",
    "141–144",
    "+9.1",
    "+4.4",
    "2,809",
    "12,918",
    "2/5",
    "57.1",
    "397",
]

# ---------------------------------------------------------------- content --

# Must match the registered project title exactly — do not shorten for layout.
TITLE_TH = (
    "ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วยโครงข่ายกราฟประสาทเทียม"
    "เชิงปริภูมิ-เวลาแบบอธิบายได้ สำหรับภาคเหนือของประเทศไทย"
)
TITLE_EN = (
    "Explainable Spatio-Temporal Graph Neural Network for PM2.5 Forecasting "
    "and Source Attribution in Northern Thailand"
)
PROJECT_CODE = "28P14E01196"
CATEGORY = "หมวด 14 โปรแกรมเพื่องานการพัฒนาด้านวิทยาศาสตร์และเทคโนโลยี"
LEVEL = "ระดับนิสิต นักศึกษา"
TEAM_LINES = ["นายรณชัย ขาวสะอาด — หัวหน้าโครงการ"]
ADVISOR_LINE = "อาจารย์ที่ปรึกษา: ดร.วัชรพงศ์ อยู่ขวัญ"
AFFILIATION = "สาขาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา"

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
@page { size: 841mm 1189mm; margin: 0; }
html, body { width: 841mm; height: 1189mm; }

/* Palette sampled from the organizer template background: the previous navy
   (#0F3D75) sat a few degrees off the template's indigo and read as pasted on.
   Magenta is the NSC logo pink and carries "transboundary" throughout. */
:root {
  --ink:#1B2559; --indigo:#2E3993; --blue:#4B7DC6; --sky:#64CBF4;
  --magenta:#D838A4; --magenta-ink:#A82683; --mute:#5A6A8C;
  --line:#D3DEEE; --tint:#EEF3FB;
  --grey1:#9AA8C4; --grey2:#C3CFE4; --grey3:#7E8AA8;
}

body {
  font-family: 'Sarabun', 'Leelawadee UI', Tahoma, sans-serif;
  color: var(--ink);
  position: relative;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
h1, h2, h3 { font-family: 'Noto Serif Thai', serif; font-weight: 700; }
b, strong { font-weight: 600; }
.bg { position: absolute; inset: 0; width: 841mm; height: 1189mm; }
.content {
  position: absolute;
  left: 46mm; right: 46mm; top: 158mm; height: 848mm;
  display: flex; flex-direction: column; gap: 8mm;
}

/* ---- title band ---- */
.titleband { display: flex; gap: 12mm; align-items: flex-start; }
.titleband .left { flex: 1; min-width: 0; }
.chips { display: flex; gap: 3.5mm; flex-wrap: wrap; margin-bottom: 4.5mm; }
.chip { font-size: 6.2mm; font-weight: 600; padding: 1.4mm 4.6mm; border-radius: 6mm;
  white-space: nowrap; }
.chip.solid { background: var(--indigo); color: #FFFFFF; }
.chip.line { border: 0.5mm solid var(--blue); color: var(--indigo); }
h1 { font-size: 15.6mm; line-height: 1.28; letter-spacing: -0.15mm; color: var(--ink); }
.entitle { font-size: 7.4mm; color: var(--mute); margin-top: 3mm; line-height: 1.35; }
.teamcard { width: 196mm; flex-shrink: 0; border-left: 1mm solid var(--sky); padding-left: 7mm; }
.teamcard .cap { font-size: 5.4mm; font-weight: 700; color: var(--blue);
  letter-spacing: 0.6mm; margin-bottom: 2mm; }
.teamcard .line { font-size: 6.4mm; line-height: 1.5; }
.teamcard .aff { font-size: 5.8mm; color: var(--mute); margin-top: 2mm; line-height: 1.4; }

/* ---- the question + two contrasting cases ---- */
.cases { display: grid; grid-template-columns: 1.12fr 1fr 1fr; gap: 8mm; align-items: stretch; }
.askbox { display: flex; flex-direction: column; justify-content: center; }
.askbox .q { font-family: 'Noto Serif Thai', serif; font-weight: 700; font-size: 14mm;
  line-height: 1.25; color: var(--indigo); }
.askbox .a { font-size: 7.1mm; line-height: 1.45; color: var(--ink); margin-top: 3.5mm; }
.case { border-radius: 5mm; padding: 6mm 7mm 7mm; color: #FFFFFF; }
.case.foreign { background: var(--magenta); }
.case.domestic { background: var(--indigo); }
.case .when { font-size: 6.2mm; opacity: 0.88; }
.case .verdict { font-family: 'Noto Serif Thai', serif; font-weight: 700; font-size: 10.5mm;
  margin-top: 1mm; line-height: 1.2; }
.case .num { font-family: 'Noto Serif Thai', serif; font-weight: 700; font-size: 26mm;
  line-height: 1.05; margin: 2mm 0 1mm; }
.case .sub { font-size: 6.4mm; line-height: 1.4; opacity: 0.95; }

/* ---- columns ---- */
.columns { display: grid; grid-template-columns: 1fr 1.04fr 1.12fr; gap: 9mm;
  flex: 1; min-height: 0; }
.col { display: flex; flex-direction: column; gap: 7mm; min-height: 0;
  justify-content: flex-start; }
/* Without this the flex children are squashed when a column is over-filled and
   their text silently spills out from under them (the footer strip then paints
   over it). Keeping natural heights makes over-fill a real, detectable overflow. */
.col > * { flex-shrink: 0; }
.sechead { display: flex; align-items: baseline; gap: 3.5mm;
  border-bottom: 0.8mm solid var(--indigo); padding-bottom: 2.5mm; }
.sechead .n { font-family: 'Noto Serif Thai', serif; font-weight: 700; font-size: 10.8mm;
  color: var(--sky); }
.sechead h2 { font-size: 15.0mm; line-height: 1.1; color: var(--indigo); }
p, li { font-size: 8.8mm; line-height: 1.5; }
ul { list-style: none; }
ul li { padding-left: 6.5mm; position: relative; margin-bottom: 2.5mm; }
ul li::before { content: ''; position: absolute; left: 0; top: 4mm; width: 2.6mm;
  height: 2.6mm; border-radius: 50%; background: var(--blue); }

/* Panels are now rare and meaningful: one per column, not a grid of boxes. */
.panel { background: var(--tint); border-radius: 4mm; padding: 7mm 8mm; }
.panel.honest { background: #FCF0F7; }
.panel h3 { font-size: 9.7mm; margin-bottom: 2mm; color: var(--indigo); }
.panel.honest h3 { color: var(--magenta-ink); }

/* Numbered innovations: hairline rules instead of three more filled cards. */
.innovs { border-top: 0.4mm solid var(--line); }
.innov { display: flex; gap: 4.5mm; align-items: baseline;
  border-bottom: 0.4mm solid var(--line); padding: 4.8mm 0; }
.innov .k { font-family: 'Noto Serif Thai', serif; font-weight: 700; font-size: 10.3mm;
  color: var(--sky); flex-shrink: 0; width: 8mm; }
.innov p { font-size: 8.3mm; }
.innov b { color: var(--indigo); }

.block h3 { font-size: 9.7mm; margin-bottom: 2mm; color: var(--indigo); }
.block p { font-size: 8.3mm; }

.statline { display: flex; align-items: baseline; gap: 4mm; }
.statline .big { font-family: 'Noto Serif Thai', serif; font-weight: 700; font-size: 19.0mm;
  color: var(--indigo); line-height: 1; }
.statline .txt { font-size: 7.7mm; line-height: 1.4; color: var(--ink); }

.shot { border: 0.5mm solid var(--line); border-radius: 4mm; padding: 4mm;
  background: #FFFFFF; }
.shot img { width: 100%; display: block; border-radius: 2mm; }
.shot .capt { font-size: 7.5mm; color: var(--mute); margin-top: 2.5mm; line-height: 1.35; }

table.mini { border-collapse: collapse; width: 100%; }
table.mini td { font-size: 6.7mm; line-height: 1.45; padding: 2.6mm 0;
  border-bottom: 0.4mm solid var(--line); vertical-align: top; }
table.mini td:first-child { color: var(--mute); width: 36%; padding-right: 4mm; }
table.mini tr:last-child td { border-bottom: none; }

/* ---- bottom strip ---- */
.strip { display: grid; grid-template-columns: 1.35fr 1fr 0.95fr; gap: 9mm;
  background: var(--indigo); color: #fff; border-radius: 5mm; padding: 7mm 9mm; }
.strip h4 { font-family: 'Noto Serif Thai', serif; font-size: 6.8mm; color: var(--sky);
  margin-bottom: 2mm; }
.strip p { font-size: 6.3mm; line-height: 1.45; color: #E4F0FB; }
.strip .chiprow { display: flex; flex-wrap: wrap; gap: 2.5mm; margin-top: 2.5mm; }
.strip .chiprow span { font-size: 5.5mm; border: 0.4mm solid #7FA5D4; border-radius: 4.5mm;
  padding: 1mm 3.8mm; color: #FFFFFF; }
"""


def build_html() -> str:
    team = "".join(f'<div class="line">{t}</div>' for t in TEAM_LINES)
    shot = (
        '<div class="shot"><img src="assets/screens/app_case_map.png" alt="">'
        '<div class="capt">แดชบอร์ด Streamlit ที่ใช้งานจริง — เหตุการณ์ 18 มี.ค. 2568 '
        "ระบบตอบ “ฝุ่นข้ามแดน” พร้อมหลักฐานและแผนที่จุดไฟรายประเทศ</div></div>"
    )
    # second shot backs up the "policy simulator" bullet above it, and shows the
    # tool answering "in-country" on a different event — the same point the two
    # case tiles make at the top of the poster
    shot_policy = (
        '<div class="shot"><img src="assets/screens/app_policy.png" alt="">'
        '<div class="capt">เครื่องมือจำลองนโยบาย — 30 มี.ค. 2568 '
        "ระบบชี้ว่าต้นตออยู่<b>ในประเทศ</b></div></div>"
    )

    return f"""<!DOCTYPE html>
<html lang="th"><head><meta charset="utf-8">
<style>{_font_faces()}{CSS}{FIGURE_CSS}</style></head>
<body>
<img class="bg" src="assets/poster_bg.jpg" alt="">
<main class="content">

  <header class="titleband">
    <div class="left">
      <div class="chips">
        <span class="chip solid">รหัสโครงการ {PROJECT_CODE}</span>
        <span class="chip line">{CATEGORY}</span>
        <span class="chip line">{LEVEL}</span>
      </div>
      <h1>{TITLE_TH}</h1>
      <div class="entitle">{TITLE_EN}</div>
    </div>
    <div class="teamcard">
      <div class="cap">ผู้พัฒนา</div>
      {team}
      <div class="line">{ADVISOR_LINE}</div>
      <div class="aff">{AFFILIATION}</div>
    </div>
  </header>

  <section class="cases">
    <div class="askbox">
      <div class="q">ฝุ่นก้อนนี้<br>“มาจากไหน”</div>
      <div class="a">ระบบตอบเป็น<b>ตัวเลขรายเหตุการณ์</b> พร้อมหลักฐานที่ตรวจสอบได้
      — และตอบต่างกันตามความจริง ไม่ได้โทษต่างชาติเสมอไป</div>
    </div>
    <div class="case foreign">
      <div class="when">แม่ฮ่องสอน · 18 มี.ค. 2568</div>
      <div class="verdict">ฝุ่นข้ามแดน</div>
      <div class="num">72%</div>
      <div class="sub">ของไฟที่ลมพัดมาถึงสถานี อยู่นอกประเทศ ·
      ยืนยันด้วยเส้นทางลมและโมเดล AI</div>
    </div>
    <div class="case domestic">
      <div class="when">เชียงใหม่ · มี.ค. 2567</div>
      <div class="verdict">ฝุ่นในประเทศ</div>
      <div class="num">99.7%</div>
      <div class="sub">เป็นฝุ่นในประเทศ เพราะไฟในไทยแรงกว่าต่างประเทศ ~59 เท่า
      (เฉลี่ย 10 ช่วง · 2 โมเดลตรงกัน)</div>
    </div>
  </section>

  <section class="columns">

    <div class="col">
      <div class="sechead"><div class="n">01</div><h2>ปัญหาและแนวคิด</h2></div>
      <ul>
        <li>ทุกฤดูหมอกควัน (ม.ค.–เม.ย.) PM2.5 ใน 9 จังหวัดภาคเหนือเกินเกณฑ์ต่อเนื่อง
        — มี.ค. 2567 ค่ารายชั่วโมงที่เชียงใหม่แตะ 141–144 µg/m³</li>
        <li>ระบบที่มีอยู่ตอบได้เพียง “ค่าฝุ่นจะเป็นเท่าไร” แต่ไม่ตอบว่า
        <b>“ฝุ่นมาจากไหน”</b> — มาตรการจึงแก้ไม่ตรงจุด</li>
        <li>หมอกควันข้ามพรมแดนเป็นข้อถกเถียงที่ขาดหลักฐานเชิงปริมาณรายเหตุการณ์</li>
      </ul>
      <div class="panel">
        <h3>แนวคิดหลัก</h3>
        <p>กราฟประสาทเทียมเชิงพื้นที่–เวลา (STGNN) ที่<b>พยากรณ์ PM2.5 ล่วงหน้า 6/12/24/48 ชม.</b>
        และ<b>ระบุสัดส่วนแหล่งกำเนิดรายเหตุการณ์</b> พร้อมความไม่แน่นอนที่วัดได้ — ในระบบเดียว</p>
      </div>
      <div class="innovs">
        <div class="innov"><div class="k">01</div>
          <p><b>กราฟพลวัตตามทิศลม</b> — น้ำหนักเส้นเชื่อมเปลี่ยนตามลม ERA5 รายชั่วโมง
          จับการพัดพาฝุ่นข้ามพรมแดนจากเมียนมา/ลาว</p></div>
        <div class="innov"><div class="k">02</div>
          <p><b>จุดความร้อนเป็นโหนดกราฟ</b> — คลัสเตอร์ไฟ NASA FIRMS เป็นโหนดชั้นหนึ่ง
          ไม่ใช่เพียงฟีเจอร์ประกอบ</p></div>
        <div class="innov"><div class="k">03</div>
          <p><b>GB-IG source attribution</b> — Integrated Gradients บนโครงสร้างกราฟ
          แจกแจงสัดส่วนอิทธิพลรายแหล่ง (ไทย/เมียนมา/ลาว) ต่อเหตุการณ์ฝุ่นสูง</p></div>
      </div>
      <div class="block">
        <h3>ขอบเขตการศึกษา</h3>
        <p>9 จังหวัดภาคเหนือตอนบน · 18 สถานีตรวจวัด Air4Thai · ข้อมูลรายชั่วโมง 4 ปี
        (2565–2568) · จุดความร้อน NASA FIRMS · ลมและอุตุนิยมวิทยา ERA5 ·
        ประเมินซ้ำบนข้อมูลอนาคตจริง ม.ค.–เม.ย. 2569</p>
      </div>
      <div class="block">
        <h3>วินัยการแบ่งข้อมูล</h3>
        <p>ฝึกด้วยปี 2565–2566 · เลือกโมเดลบน validation ปี 2567 <b>เท่านั้น</b> ·
        ทดสอบบน held-out ปี 2568 ที่ไม่ถูกแตะระหว่างพัฒนา · เทียบ A3TGCN และ GBM (ไม่ใช้กราฟ)</p>
      </div>
      <div class="panel">
        <h3>ประเมินอย่างเข้มงวดและซื่อสัตย์ — 5 ชั้น</h3>
        <ul>
          <li>ชุดทดสอบ held-out ปี 2568 ไม่ถูกแตะระหว่างพัฒนา</li>
          <li>ประเมินซ้ำบนข้อมูลอนาคตจริง ปี 2569 (นอกช่วงฝึก 100%)</li>
          <li>Ablation หลาย seed แยกผลจริงออกจากความผันผวนสุ่ม</li>
          <li>พยานอิสระ: back-trajectory + จุดความร้อน FIRMS</li>
          <li>ช่วงเชื่อมั่น conformal ตรวจสอบความครอบคลุมจริง</li>
        </ul>
      </div>
    </div>

    <div class="col">
      <div class="sechead"><div class="n">02</div><h2>ระบบและสถาปัตยกรรม</h2></div>
      {pipeline_figure()}
      <div class="block">
        <h3>พร้อมใช้งานจริง</h3>
        <ul>
          <li>แดชบอร์ดอ่านง่ายหน้าเดียว + โหมดวิเคราะห์เชิงลึก · พยากรณ์สดจากข้อมูลจริง</li>
          <li>ช่วงเชื่อมั่น conformal 90% ทุกค่า · แจ้งเตือน Telegram เมื่อคาดว่าจะเกินเกณฑ์</li>
          <li>เครื่องมือ “จำลองนโยบาย” — ประเมินว่าลดไฟฝั่งใดตัดต้นตอฝุ่นได้มากกว่า</li>
          <li>ทดสอบอัตโนมัติ 397 รายการ · ทำซ้ำได้ทั้ง pipeline</li>
        </ul>
      </div>
      {shot}
      {shot_policy}
    </div>

    <div class="col">
      <div class="sechead"><div class="n">03</div><h2>ผลลัพธ์และการตรวจสอบ</h2></div>
      {evidence_figure()}
      <div class="statline">
        <div class="big">+9.1%</div>
        <div class="txt">แม่นกว่าการทายว่า “ค่าเท่าเดิม” ที่ระยะ 48 ชม. (+4.4% ที่ 24 ชม.)
        บนข้อมูลอนาคตจริง ม.ค.–เม.ย. 2569 จำนวน 2,809 ตัวอย่างนอกช่วงฝึกทั้งหมด</div>
      </div>
      {rmse_figure()}
      <table class="mini">
        <tr><td>ช่วงเชื่อมั่น 90%</td>
            <td>±5.9 (6 ชม.) ถึง ±15.5 µg/m³ (48 ชม.) · ครอบคลุมจริง 87.0–88.5%</td></tr>
        <tr><td>เตือนเกินเกณฑ์ 37.5</td>
            <td>Brier Skill Score ที่ 48 ชม. = 0.306 เทียบ persistence 0.282</td></tr>
        <tr><td>ทดสอบลงทะเบียน</td>
            <td>5 เหตุการณ์ · ตรงแบบไบนารี 2/5 — เหตุการณ์ที่ชี้ “ต่างประเทศ” ยืนยันครบ
            อีก 3 เหตุการณ์ไฟต่างชาติ ≈ 0 ตรงกับค่าประเมินใกล้ศูนย์</td></tr>
      </table>
      <div class="panel honest">
        <h3>สิ่งที่ระบบยังทำไม่ได้</h3>
        <p>ที่ระยะสั้น 6–12 ชม. โมเดล<b>ยังแพ้</b>การทายว่า “ค่าเท่าเดิม” — ข้อได้เปรียบเริ่มที่
        24 ชม. และชัดที่ 48 ชม. · โครงสร้างกราฟไม่ได้เพิ่มความแม่นยำอย่างสม่ำเสมอทุก seed ·
        คุณค่าหลักจึงอยู่ที่<b>การอธิบายแหล่งกำเนิดพร้อมความไม่แน่นอนและพยานอิสระ</b>
        ซึ่งระบบพยากรณ์ทั่วไปให้ไม่ได้</p>
      </div>
    </div>

  </section>

  <footer class="strip">
    <div>
      <h4>นวัตกรรมเพื่อความยั่งยืน</h4>
      <p>แยกสัดส่วน “เผาในประเทศ” กับ “หมอกควันข้ามพรมแดน” รายเหตุการณ์ด้วยหลักฐานเชิงปริมาณ
      สนับสนุนการตัดสินใจเชิงนโยบายสาธารณสุขและสิ่งแวดล้อม มาตรการที่ตรงจุด
      และการเจรจาระดับภูมิภาค</p>
      <div class="chiprow"><span>SDG 3 สุขภาพ</span><span>SDG 11 เมืองยั่งยืน</span>
      <span>SDG 13 ภูมิอากาศ</span></div>
    </div>
    <div>
      <h4>เทคโนโลยี</h4>
      <div class="chiprow">
        <span>PyTorch</span><span>PyTorch Geometric</span><span>MTGNN</span>
        <span>GB-IG</span><span>Conformal Prediction</span><span>Streamlit</span>
      </div>
    </div>
    <div>
      <h4>แหล่งข้อมูล</h4>
      <p>OpenAQ / Air4Thai · NASA FIRMS · ERA5 (Copernicus)<br>
      ครอบคลุม 9 จังหวัดภาคเหนือ ปี 2565–2568 + ประเมินนอกช่วงเวลา ปี 2569</p>
    </div>
  </footer>

</main>
</body></html>
"""


def ensure_background() -> None:
    """Render the organizer template PDF to assets/poster_bg.jpg at 450 dpi."""
    target = ASSETS / "poster_bg.jpg"
    if target.exists():
        return
    zips = sorted(REPO.glob("NSC 2026_TemplatePoster*.zip"))
    if not zips:
        sys.exit("organizer template zip not found in repo root")
    pdftoppm = shutil.which("pdftoppm") or str(POPPLER_BIN / "pdftoppm.exe")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zips[0]) as zf:
            member = next(n for n in zf.namelist() if n.endswith("A3-DONE.pdf"))
            zf.extract(member, td)
        subprocess.run(
            [
                pdftoppm,
                "-r",
                "450",
                "-jpeg",
                "-jpegopt",
                "quality=92",
                "-singlefile",
                str(Path(td) / member),
                str(target.with_suffix("")),
            ],
            check=True,
        )
    print(f"background rendered: {target}")


def check_canon(html: str) -> None:
    bad = [s for s in CANON_FORBIDDEN if s in html]
    if bad:
        sys.exit(f"FORBIDDEN superseded numbers present: {bad}")
    missing = [s for s in CANON_REQUIRED if s not in html]
    if missing:
        print(f"WARNING: expected canon numbers missing: {missing}")
    if "รอยืนยัน" in html:
        print("WARNING: placeholder text (รอยืนยัน) still present")


def check_overflow(html: str) -> None:
    """Fail if any column's content is taller than the space it is given.

    Overflow here is invisible in the rendered PDF — the footer strip paints
    over the spilled text rather than moving out of its way — so a pixel scan of
    the output cannot see it. Ask the layout engine directly instead: render a
    probe copy that writes the gap between each column's bottom and its lowest
    child into the title, then read it back out of the dumped DOM.

    ``scrollHeight`` is not usable here: on a flex item with ``overflow:visible``
    Chrome reports it equal to ``clientHeight`` even when content spills, so a
    scrollHeight-based guard silently passes. Measured child rectangles do not.
    """
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        print("WARNING: no Chrome/Edge found, skipping overflow check")
        return
    # measure on 'load', not at parse time: the screenshots are still 0 px tall
    # while they are decoding, which makes an early probe wildly over-optimistic
    probe = html.replace(
        "</body>",
        "<script>window.addEventListener('load',()=>{"
        "document.title='PROBE '+[...document.querySelectorAll('.col')].map((c,i)=>{"
        "const b=c.getBoundingClientRect().bottom;"
        "const m=Math.max(...[...c.children].map(k=>k.getBoundingClientRect().bottom));"
        "return `${i+1}:${Math.round(m-b)}`;}).join(' ');});</script></body>",
    )
    probe_path = POSTER_DIR / "_overflow_probe.html"
    probe_path.write_text(probe, encoding="utf-8")
    try:
        dom = subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--virtual-time-budget=15000",
                "--dump-dom",
                probe_path.as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        ).stdout
    finally:
        probe_path.unlink(missing_ok=True)

    match = re.search(r"PROBE ([^<]*)", dom)
    if match is None:
        print("WARNING: overflow probe produced no measurements")
        return
    spills = []
    slack = []
    for field in match.group(1).split():
        col, _, delta = field.partition(":")
        # positive means the lowest child sits below the column box; 2 px of
        # tolerance absorbs sub-pixel rounding in the layout engine
        over = int(delta)
        if over > 2:
            spills.append(f"col{col} by {over / PX_PER_MM:.0f}mm")
        else:
            slack.append(f"col{col} {-over / PX_PER_MM:.0f}mm")
    if spills:
        sys.exit("COLUMN OVERFLOW (spilled text hides behind the footer): " + ", ".join(spills))
    print(f"overflow check ok — unused space: {', '.join(slack)}")


def render_pdf() -> None:
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        sys.exit("no Chrome/Edge found for PDF rendering")
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_PATH}",
            HTML_PATH.as_uri(),
        ],
        check=True,
        timeout=180,
    )
    print(f"pdf: {PDF_PATH}  ({PDF_PATH.stat().st_size/1e6:.1f} MB)")


def render_preview(dpi: int = 55) -> None:
    pdftoppm = shutil.which("pdftoppm") or str(POPPLER_BIN / "pdftoppm.exe")
    subprocess.run(
        [
            pdftoppm,
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(PDF_PATH),
            str(PREVIEW_PATH.with_suffix("")),
        ],
        check=True,
    )
    print(f"preview: {PREVIEW_PATH}")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    ensure_background()
    html = build_html()
    check_canon(html)
    check_overflow(html)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"html: {HTML_PATH}")
    render_pdf()
    render_preview()


if __name__ == "__main__":
    main()
