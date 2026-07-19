"""Build the NSC 2026 A0 poster PDF on top of the organizer template.

Pipeline:
    1. ensure assets in outputs/poster/assets/ — background rendered at 450 dpi
       from the organizer zip (repo root) via poppler pdftoppm; report figures
       copied from outputs/poster/assets_hires/ (300 dpi re-renders) when
       present, else from outputs/figures/report/.
    2. write outputs/poster/poster.html (A0 = 841 x 1189 mm, Leelawadee UI).
    3. print to outputs/NSC2026_Poster_A0.pdf with Chrome/Edge headless and
       render a preview PNG with pdftoppm.

The build fails if any superseded ("forbidden") result number appears in the
HTML, and warns if a canon number is missing — see CANON_* below.

Run: uv run --no-sync python scripts/generate_poster.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POSTER_DIR = REPO / "outputs" / "poster"
ASSETS = POSTER_DIR / "assets"
HIRES = POSTER_DIR / "assets_hires"
LOWRES = REPO / "outputs" / "figures" / "report"
HTML_PATH = POSTER_DIR / "poster.html"
PDF_PATH = REPO / "outputs" / "NSC2026_Poster_A0.pdf"
PREVIEW_PATH = POSTER_DIR / "poster_preview.png"

POPPLER_BIN = Path(
    r"C:\Users\golfv\AppData\Local\Microsoft\WinGet\Packages"
    r"\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\poppler-25.07.0\Library\bin"
)
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

FIGURES = [
    "architecture.png",
    "transboundary_map.png",
    "rmse_by_horizon.png",
]

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
# Current canon numbers the poster is expected to carry.
CANON_REQUIRED = [
    "62.7",
    "51.6",
    "11.1",
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
    "631,152",
]

# ---------------------------------------------------------------- content --

TITLE_TH = "ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วยโครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลาแบบอธิบายได้ สำหรับภาคเหนือของประเทศไทย"
TITLE_EN = "Explainable Spatio-Temporal Graph Neural Network for PM2.5 Forecasting and Source Attribution in Northern Thailand"
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
body {
  font-family: 'Leelawadee UI', 'Leelawadee', Tahoma, sans-serif;
  color: #16294E;
  position: relative;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
.bg { position: absolute; inset: 0; width: 841mm; height: 1189mm; }
.content {
  position: absolute;
  left: 46mm; right: 46mm; top: 158mm; height: 848mm;
  display: flex; flex-direction: column; gap: 9mm;
}

/* ---- title band ---- */
.titleband { display: flex; gap: 12mm; align-items: stretch; }
.titleband .left { flex: 1; min-width: 0; }
.chips { display: flex; gap: 4mm; flex-wrap: wrap; margin-bottom: 5mm; }
.chip {
  font-size: 6.4mm; font-weight: 600; padding: 1.6mm 5mm;
  border-radius: 6mm; white-space: nowrap;
}
.chip.solid { background: #0F3D75; color: #FFFFFF; }
.chip.line { border: 0.6mm solid #1E6FBF; color: #0F3D75; }
h1 { font-size: 16.8mm; line-height: 1.3; letter-spacing: -0.1mm; }
.entitle { font-size: 8mm; color: #44536E; margin-top: 3mm; line-height: 1.35; }
.teamcard {
  width: 205mm; flex-shrink: 0; background: #EAF3FB;
  border-radius: 5mm; padding: 7mm 9mm; align-self: flex-start;
}
.teamcard .cap {
  font-size: 5.6mm; font-weight: 700; color: #1E6FBF;
  letter-spacing: 0.5mm; margin-bottom: 2.5mm;
}
.teamcard .line { font-size: 6.6mm; line-height: 1.5; }
.teamcard .aff { font-size: 6mm; color: #44536E; margin-top: 2.5mm; line-height: 1.4; }

/* ---- hero band ---- */
.heroband { display: grid; grid-template-columns: 1.18fr 1fr 1fr; gap: 8mm; }
.tile { border-radius: 5mm; padding: 7mm 9mm; }
.tile .label { font-size: 6mm; font-weight: 700; line-height: 1.3; }
.tile .num { font-size: 25mm; font-weight: 700; line-height: 1.1; margin: 1.5mm 0; }
.tile .sub { font-size: 5.7mm; line-height: 1.4; }
.tile.hero { background: #0F3D75; color: #FFFFFF; }
.tile.hero .label { color: #A8CEF2; }
.tile.hero .sub { color: #D7E8FA; }
.tile.contrast { background: #FFFFFF; border: 0.8mm solid #C9D9EA; }
.tile.contrast .num { color: #D9480F; }
.tile.contrast .label { color: #B03A0C; }
.tile.contrast .sub { color: #44536E; }
.tile.forecast { background: #EAF3FB; }
.tile.forecast .num { color: #0F3D75; }
.tile.forecast .label { color: #1E6FBF; }
.tile.forecast .sub { color: #44536E; }

/* ---- columns ---- */
.columns { display: grid; grid-template-columns: 1fr 1.06fr 1.1fr; gap: 9mm; flex: 1; min-height: 0; }
.col { display: flex; flex-direction: column; gap: 6mm; min-height: 0; justify-content: space-between; }
.sechead {
  display: flex; align-items: center; gap: 4mm;
  border-bottom: 1mm solid #1E6FBF; padding-bottom: 2.5mm;
}
.sechead .n {
  width: 12mm; height: 12mm; border-radius: 2.5mm; background: #1E6FBF;
  color: #fff; font-size: 7.5mm; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}
.sechead h2 { font-size: 11.5mm; line-height: 1.1; }
p, li { font-size: 6.9mm; line-height: 1.5; }
ul { list-style: none; }
ul li { padding-left: 7mm; position: relative; margin-bottom: 2.5mm; }
ul li::before { content: ''; position: absolute; left: 0; top: 3.6mm; width: 3.2mm; height: 3.2mm; border-radius: 50%; background: #1E6FBF; }
.card { background: #F4F8FC; border-radius: 4mm; padding: 6mm 7mm; }
.card.accent { background: #EAF3FB; border-left: 2mm solid #1E6FBF; }
.card.warm { background: #FDF1E7; border-left: 2mm solid #D9480F; }
.card h3 { font-size: 7.6mm; margin-bottom: 2mm; color: #0F3D75; }
.card.warm h3 { color: #B03A0C; }
.innov { display: flex; gap: 5mm; align-items: flex-start; }
.innov .k {
  flex-shrink: 0; width: 11mm; height: 11mm; border-radius: 50%;
  background: #0F3D75; color: #fff; font-size: 6.8mm; font-weight: 700;
  display: flex; align-items: center; justify-content: center; margin-top: 1mm;
}
.fig { background: #FFFFFF; border: 0.6mm solid #C9D9EA; border-radius: 4mm; padding: 4mm; }
.fig img { width: 100%; display: block; border-radius: 2mm; }
.fig.fit img { width: 88%; margin: 0 auto; }
.fig .capt { font-size: 5.6mm; color: #44536E; margin-top: 2.5mm; line-height: 1.35; }
.fig .missing {
  height: 120mm; display: flex; align-items: center; justify-content: center;
  color: #B03A0C; font-size: 7mm; border: 1mm dashed #D9480F; border-radius: 2mm;
}
.datachips { display: flex; flex-wrap: wrap; gap: 3mm; }
.datachips span {
  font-size: 6mm; background: #FFFFFF; border: 0.5mm solid #C9D9EA;
  border-radius: 5mm; padding: 1.5mm 4.5mm; color: #16294E;
}
table.mini { border-collapse: collapse; width: 100%; }
table.mini td { font-size: 6.2mm; line-height: 1.4; padding: 2mm 3mm; border-bottom: 0.4mm solid #DCE7F2; vertical-align: top; }
table.mini td:first-child { color: #44536E; width: 38%; }
table.mini tr:last-child td { border-bottom: none; }

/* ---- bottom strip ---- */
.strip {
  display: grid; grid-template-columns: 1.3fr 1fr 0.9fr; gap: 8mm;
  background: #0F3D75; color: #fff; border-radius: 5mm; padding: 7mm 9mm;
}
.strip h4 { font-size: 6.8mm; color: #A8CEF2; margin-bottom: 2mm; }
.strip p { font-size: 6mm; line-height: 1.45; color: #E4F0FB; }
.strip .chiprow { display: flex; flex-wrap: wrap; gap: 2.5mm; margin-top: 2mm; }
.strip .chiprow span {
  font-size: 5.6mm; border: 0.5mm solid #5E92C9; border-radius: 4.5mm;
  padding: 1mm 4mm; color: #FFFFFF;
}
"""


def _fig_tag(name: str, caption: str, extra_class: str = "") -> str:
    """Return a figure card; a dashed placeholder if the asset is absent."""
    if (ASSETS / name).exists():
        body = f'<img src="assets/{name}" alt="">'
    else:
        body = f'<div class="missing">รอไฟล์ {name}</div>'
    cls = f"fig {extra_class}".strip()
    return f'<div class="{cls}">{body}<div class="capt">{caption}</div></div>'


def build_html() -> str:
    team = "".join(f'<div class="line">{t}</div>' for t in TEAM_LINES)
    arch = _fig_tag(
        "architecture.png",
        "ภาพรวมสถาปัตยกรรม: ข้อมูลตรวจวัด + จุดความร้อน + อุตุนิยมวิทยา สู่กราฟพลวัตตามทิศลม, MTGNN multi-horizon และการระบุแหล่งกำเนิดด้วย GB-IG",
    )
    tmap = _fig_tag(
        "transboundary_map.png",
        "เหตุการณ์หลัก 18 มี.ค. 2568 (แม่ฮ่องสอน): สถานี จุดความร้อน FIRMS และระเบียงลมข้ามพรมแดน",
        extra_class="fit",
    )
    rmse = _fig_tag(
        "rmse_by_horizon.png",
        "RMSE รายระยะพยากรณ์ เทียบ baseline (ชุดทดสอบ held-out ปี 2568)",
    )

    return f"""<!DOCTYPE html>
<html lang="th"><head><meta charset="utf-8"><style>{CSS}</style></head>
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

  <section class="heroband">
    <div class="tile hero">
      <div class="label">ระบุแหล่งกำเนิดข้ามพรมแดน — เหตุการณ์ 18 มี.ค. 2568 (แม่ฮ่องสอน)</div>
      <div class="num">62.7%</div>
      <div class="sub">สัดส่วนอิทธิพลต่างประเทศที่โมเดลระบุ (เมียนมา 51.6% · สปป.ลาว 11.1%)
      เทียบสัดส่วนไฟต่างประเทศที่เชื่อมโยงจริง 72.1% · ค่าเฉลี่ยจาก 3 seeds = 54.2%</div>
    </div>
    <div class="tile contrast">
      <div class="label">กรณีตรงข้าม — เชียงใหม่ มี.ค. 2567</div>
      <div class="num">99.7%</div>
      <div class="sub">โมเดลชี้แหล่งกำเนิดในประเทศ เมื่อไฟในประเทศแรงกว่าราว 59 เท่า —
      ไม่ได้ชี้ต่างประเทศเสมอไป</div>
    </div>
    <div class="tile forecast">
      <div class="label">พยากรณ์ข้อมูลอนาคตจริง ม.ค.–เม.ย. 2569</div>
      <div class="num">+9.1%</div>
      <div class="sub">RMSE ดีกว่า persistence ที่ 48 ชม. (และ +4.4% ที่ 24 ชม.)
      จากตัวอย่างนอกช่วงฝึก 2,809 รายการ</div>
    </div>
  </section>

  <section class="columns">

    <div class="col">
      <div class="sechead"><div class="n">1</div><h2>ปัญหาและแนวคิด</h2></div>
      <ul>
        <li>ทุกฤดูหมอกควัน (ม.ค.–เม.ย.) PM2.5 ใน 9 จังหวัดภาคเหนือเกินเกณฑ์มาตรฐานต่อเนื่อง — มี.ค. 2567 ค่ารายชั่วโมงที่เชียงใหม่แตะ 141–144 µg/m³</li>
        <li>ระบบที่มีอยู่ตอบได้เพียง "ค่าฝุ่นจะเป็นเท่าไร" แต่ไม่ตอบว่า <b>"ฝุ่นมาจากไหน"</b> — ทำให้มาตรการแก้ไขไม่ตรงจุด</li>
        <li>หมอกควันข้ามพรมแดนจากประเทศเพื่อนบ้านเป็นข้อถกเถียงที่ขาดหลักฐานเชิงปริมาณรายเหตุการณ์</li>
      </ul>
      <div class="card accent">
        <h3>แนวคิดหลัก</h3>
        <p>กราฟประสาทเทียมเชิงพื้นที่–เวลา (STGNN) ที่<b>พยากรณ์ PM2.5 ล่วงหน้า 6/12/24/48 ชม.</b>
        และ<b>ระบุสัดส่วนแหล่งกำเนิดรายเหตุการณ์</b> พร้อมความไม่แน่นอนที่วัดได้ — ในระบบเดียว</p>
      </div>
      <div class="card">
        <div class="innov"><div class="k">1</div><p><b>กราฟพลวัตตามทิศลม</b> — น้ำหนักขอบกราฟเปลี่ยนตามลม ERA5 รายชั่วโมง จับการพัดพาฝุ่นข้ามพรมแดนจากเมียนมา/ลาว</p></div>
      </div>
      <div class="card">
        <div class="innov"><div class="k">2</div><p><b>จุดความร้อนเป็นโหนดกราฟ</b> — คลัสเตอร์ไฟ NASA FIRMS เป็นโหนดชั้นหนึ่งในกราฟ ไม่ใช่เพียงฟีเจอร์ประกอบ</p></div>
      </div>
      <div class="card">
        <div class="innov"><div class="k">3</div><p><b>GB-IG source attribution</b> — Integrated Gradients บนโครงสร้างกราฟ แจกแจงสัดส่วนอิทธิพลรายแหล่ง (ในประเทศ/เมียนมา/ลาว) ต่อเหตุการณ์ฝุ่นสูง</p></div>
      </div>
      <div class="card accent">
        <h3>การประเมินอย่างเข้มงวดและซื่อสัตย์ — 5 ชั้น</h3>
        <ul>
          <li>ชุดทดสอบ held-out ปี 2568 ไม่ถูกแตะระหว่างพัฒนา</li>
          <li>ประเมินซ้ำบนข้อมูลอนาคตจริง ปี 2569 (out-of-sample 100%)</li>
          <li>Ablation หลาย seed แยกผลจริงออกจากความผันผวนสุ่ม</li>
          <li>พยานอิสระ: back-trajectory + จุดความร้อน FIRMS</li>
          <li>ช่วงเชื่อมั่น conformal ตรวจสอบความครอบคลุมจริง</li>
        </ul>
      </div>
      <div class="card warm">
        <h3>ความโปร่งใสทางวิทยาศาสตร์</h3>
        <p>ที่ระยะ 6–12 ชม. ความแม่นยำยังใกล้เคียง persistence และโครงสร้างกราฟไม่ได้เพิ่มความแม่นยำอย่างสม่ำเสมอทุก seed —
        คุณค่าหลักของระบบคือ<b>การอธิบายแหล่งกำเนิดพร้อมความไม่แน่นอนที่วัดได้และพยานอิสระ</b>
        ซึ่งระบบพยากรณ์ทั่วไปให้ไม่ได้</p>
      </div>
    </div>

    <div class="col">
      <div class="sechead"><div class="n">2</div><h2>สถาปัตยกรรมและระบบ</h2></div>
      {arch}
      <div class="card">
        <h3>ขอบเขตการศึกษา</h3>
        <p>9 จังหวัดภาคเหนือตอนบน · 18 สถานีตรวจวัด (Air4Thai) · ข้อมูลรายชั่วโมง
        ปี 2565–2568 (ค.ศ. 2022–2025) รวม 631,152 แถว · จุดความร้อน NASA FIRMS ·
        ลม–อุตุนิยมวิทยา ERA5 · พยากรณ์ 4 ระยะ (6/12/24/48 ชม.) ·
        ประเมินซ้ำบนข้อมูลอนาคตจริง ม.ค.–เม.ย. 2569</p>
      </div>
      <div class="card">
        <h3>โมเดลและวินัยการแบ่งข้อมูล</h3>
        <p>MTGNN เทียบ A3TGCN และ GBM (ไม่ใช้กราฟ) · ฝึกด้วยปี 2565–2566 ·
        เลือกโมเดลบน validation ปี 2567 เท่านั้น · ทดสอบบน held-out ปี 2568 ·
        ผสาน persistence แบบ hybrid ตามระยะพยากรณ์</p>
      </div>
      <div class="card">
        <h3>ระบบพร้อมใช้งานจริง</h3>
        <ul>
          <li>แดชบอร์ด Streamlit 6 มุมมอง: ภาพรวม · พยากรณ์ · แหล่งกำเนิด · ข้ามพรมแดน · ประสิทธิภาพ · เกี่ยวกับระบบ</li>
          <li>โหมดพยากรณ์สดจากข้อมูลจริง (Air4Thai + ลมพยากรณ์ NWP)</li>
          <li>ช่วงความเชื่อมั่น conformal 90% กำกับทุกค่าพยากรณ์</li>
          <li>แจ้งเตือนอัตโนมัติผ่าน Telegram เมื่อคาดว่าจะเกินเกณฑ์</li>
          <li>ทดสอบอัตโนมัติ 397 รายการ · ทำซ้ำได้ทั้ง pipeline</li>
        </ul>
      </div>
    </div>

    <div class="col">
      <div class="sechead"><div class="n">3</div><h2>ผลลัพธ์และการตรวจสอบ</h2></div>
      {tmap}
      <div class="card accent">
        <h3>พยานอิสระยืนยันการระบุแหล่งกำเนิด</h3>
        <p>ทดสอบแบบลงทะเบียนล่วงหน้า 5 เหตุการณ์ด้วย back-trajectory (ลม ERA5 ย้อนหลัง 48 ชม.):
        เกณฑ์ไบนารีตรง <b>2/5</b> โดยเหตุการณ์ที่โมเดลระบุ "ต่างประเทศ" ได้รับการยืนยัน<b>ทั้งสองเหตุการณ์</b> —
        เหตุการณ์หลัก: มวลอากาศอยู่เหนือเมียนมา <b>57.1%</b> ของชั่วโมงทั้งหมด และ FRP ในระเบียงลมฝั่งเมียนมา
        <b>12,918</b> เทียบฝั่งไทย <b>106 MW</b> · อีก 3 เหตุการณ์ไฟต่างประเทศในระเบียง ≈ 0 สอดคล้องค่าประเมินใกล้ศูนย์ของโมเดล</p>
      </div>
      {rmse}
      <table class="mini">
        <tr><td>ช่วงเชื่อมั่น 90%</td><td>±5.9 (6 ชม.) ถึง ±15.5 µg/m³ (48 ชม.) · ครอบคลุมจริง 87.0–88.5%</td></tr>
        <tr><td>เตือนเกินเกณฑ์ 37.5</td><td>Brier Skill Score 48 ชม. = 0.306 เทียบ persistence 0.282</td></tr>
        <tr><td>ความไม่แน่นอน</td><td>รายงานผลระบุแหล่งพร้อมช่วงจากหลาย seed เสมอ — เหตุการณ์หลัก: เฉลี่ย 54.2% (ช่วง 0–100%)</td></tr>
      </table>
    </div>

  </section>

  <footer class="strip">
    <div>
      <h4>นวัตกรรมเพื่อความยั่งยืน</h4>
      <p>แยกสัดส่วน "เผาในประเทศ กับ หมอกควันข้ามพรมแดน" รายเหตุการณ์ด้วยหลักฐานเชิงปริมาณ
      สนับสนุนการตัดสินใจเชิงนโยบายสาธารณสุขและสิ่งแวดล้อม มาตรการที่ตรงจุด และการเจรจาระดับภูมิภาค</p>
      <div class="chiprow"><span>SDG 3 สุขภาพ</span><span>SDG 11 เมืองยั่งยืน</span><span>SDG 13 ภูมิอากาศ</span></div>
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


def collect_figures() -> None:
    """Copy poster figures into assets/, preferring 300-dpi re-renders."""
    # architecture.png: the tracked repo-root render (300 dpi) is the fallback.
    fallbacks = {"architecture.png": REPO / "architecture_diagram.png"}
    for name in FIGURES:
        candidates = [HIRES / name, LOWRES / name]
        if name in fallbacks:
            candidates.append(fallbacks[name])
        for src in candidates:
            if src.exists():
                shutil.copyfile(src, ASSETS / name)
                print(f"figure: {name}  <-  {src.parent.name}")
                break
        else:
            print(f"figure MISSING (placeholder used): {name}")


def check_canon(html: str) -> None:
    bad = [s for s in CANON_FORBIDDEN if s in html]
    if bad:
        sys.exit(f"FORBIDDEN superseded numbers present: {bad}")
    missing = [s for s in CANON_REQUIRED if s not in html]
    if missing:
        print(f"WARNING: expected canon numbers missing: {missing}")
    if "รอยืนยัน" in html:
        print("WARNING: placeholder text (รอยืนยัน) still present")


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
    collect_figures()
    html = build_html()
    check_canon(html)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"html: {HTML_PATH}")
    render_pdf()
    render_preview()


if __name__ == "__main__":
    main()
