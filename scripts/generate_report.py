# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Generate the NSC 2026 Category 14 FINAL REPORT as a Word document.

Content is transcribed verbatim from ``docs/REPORT_DRAFT.md`` (numbers already verified against
``outputs/*.json``). Formatting follows the NSC booklet rules: TH Sarabun New 16pt, A4, 1-inch
margins, the official final-report cover (booklet PDF p.44), an auto table-of-contents field, the
8 report figures, 3 tables, and footer page numbers.

This reuses the python-docx helper pattern from ``scripts/generate_proposal.py`` but takes NONE of
its content (the proposal carries superseded denorm-bug numbers). Run:

    UV_NO_SYNC=1 uv run python scripts/generate_report.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures" / "report"
OUTPUT = ROOT / "outputs" / "NSC2026_Final_Report.docx"

FONT = "TH Sarabun New"
PROJECT_CODE = "28P14E01196"  # NSC project code
TITLE_TH = (
    "ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วยโครงข่ายกราฟประสาทเทียม"
    "เชิงปริภูมิ-เวลาแบบอธิบายได้ สำหรับภาคเหนือของประเทศไทย"
)
TITLE_EN = (
    "Explainable Spatio-Temporal Graph Neural Network for PM2.5 Forecasting and "
    "Source Attribution in Northern Thailand"
)


# ── Font helpers (pattern from generate_proposal.py) ────────────────────────────


def _set_cs_font(run, name: str = FONT) -> None:
    """Force the complex-script font so Thai glyphs render as TH Sarabun New."""
    rpr = run._r.get_or_add_rPr()
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:cs"), name)
    existing = rpr.find(qn("w:rFonts"))
    if existing is not None:
        rpr.remove(existing)
    rpr.insert(0, rfonts)


def fmt(run, size: int = 16, bold: bool = False, italic: bool = False, font: str = FONT) -> None:
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    _set_cs_font(run, font)


def _apply_cs_to_style(style, name: str = FONT) -> None:
    """Set ascii/hAnsi/cs font on a style definition (so Thai renders in headings/TOC/Normal)."""
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), name)


# ── Inline markdown (**bold**, *italic*, `code`) → runs ─────────────────────────

_INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`.+?`)")


def add_md_runs(p, text: str, size: int = 16, italic: bool = False) -> None:
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2])
            fmt(r, size=size, bold=True, italic=italic)
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1])
            fmt(r, size=size, font="Courier New")
        elif part.startswith("*") and part.endswith("*"):
            r = p.add_run(part[1:-1])
            fmt(r, size=size, italic=True)
        else:
            r = p.add_run(part)
            fmt(r, size=size, italic=italic)


# ── Block helpers ───────────────────────────────────────────────────────────────


def para(
    doc,
    text="",
    size=16,
    bold=False,
    italic=False,
    align=WD_ALIGN_PARAGRAPH.LEFT,
    space_before=0,
    space_after=0,
):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        r = p.add_run(text)
        fmt(r, size=size, bold=bold, italic=italic)
    return p


def body(doc, text, size=16, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent_cm=0.0, space_after=6):
    p = doc.add_paragraph()
    p.alignment = align
    if indent_cm:
        p.paragraph_format.left_indent = Cm(indent_cm)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    add_md_runs(p, text, size=size)
    return p


def h1(doc, text, size=18):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    fmt(r, size=size, bold=True)
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def h2(doc, text, size=16):
    p = doc.add_paragraph(style="Heading 2")
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    fmt(r, size=size, bold=True)
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def numbered(doc, n, text, size=16, indent_cm=0.75):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(indent_cm)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"{n}) ")
    fmt(r, size=size)
    add_md_runs(p, text, size=size)
    return p


def bullet(doc, text, size=16, indent_cm=1.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(indent_cm)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("• ")
    fmt(r, size=size)
    add_md_runs(p, text, size=size)
    return p


def ref_entry(doc, n, text, size=16):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.first_line_indent = Cm(-1.0)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"{n}. {text}")
    fmt(r, size=size)
    return p


def callout(doc, text, size=15):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.right_indent = Cm(0.4)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    add_md_runs(p, text, size=size)
    return p


def caption(doc, text, size=14):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    fmt(r, size=size, italic=True)
    return p


def figure(doc, png: Path, cap: str, width_cm: float = 16.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(png), width=Cm(width_cm))
    caption(doc, cap)


def _bold_cell(cell, text, size=14, bold=False, center=False):
    cp = cell.paragraphs[0]
    if center:
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Whole-cell bold markdown: **...**
    if text.startswith("**") and text.endswith("**"):
        text, bold = text[2:-2], True
    r = cp.add_run(text)
    fmt(r, size=size, bold=bold)


def table(doc, headers, rows, size=14):
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for j, hdr in enumerate(headers):
        _bold_cell(t.rows[0].cells[j], hdr, size=size, bold=True, center=True)
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            _bold_cell(t.rows[i].cells[j], val, size=size, center=(j > 0))
    para(doc, "", size=6)
    return t


# ── Word field helpers (TOC + page number) ──────────────────────────────────────


def mark_fields_dirty(document) -> None:
    """Set w:updateFields so Word refreshes the TOC field on open (no manual F9 needed)."""
    settings = document.settings.element
    upd = settings.find(qn("w:updateFields"))
    if upd is None:
        upd = OxmlElement("w:updateFields")
        settings.append(upd)
    upd.set(qn("w:val"), "true")


def add_toc(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "คลิกขวาที่นี่แล้วเลือก Update Field เพื่อสร้างสารบัญอัตโนมัติ"
    sep.append(placeholder)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, sep, end):
        run._r.append(el)
    fmt(run, size=14, italic=True)


def add_page_number(section):
    section.different_first_page_header_footer = True
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, end):
        run._r.append(el)
    fmt(run, size=14)


# ════════════════════════════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════════════════════════════

doc = Document()

# Page geometry: A4, 1-inch margins all sides (booklet rule)
sec = doc.sections[0]
sec.page_width = Mm(210)
sec.page_height = Mm(297)
sec.top_margin = Inches(1)
sec.bottom_margin = Inches(1)
sec.left_margin = Inches(1)
sec.right_margin = Inches(1)

# Default + heading + TOC styles → TH Sarabun New
normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(16)
_apply_cs_to_style(normal)
for style_name in ("Heading 1", "Heading 2", "Heading 3", "TOC 1", "TOC 2", "TOC 3"):
    try:
        _apply_cs_to_style(doc.styles[style_name])
    except KeyError:
        pass

add_page_number(sec)

# ── COVER (booklet final-report template, PDF p.44) ─────────────────────────────
para(doc, f"รหัสโครงการ  {PROJECT_CODE}", size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
para(doc, TITLE_TH, size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para(doc, f"({TITLE_EN})", size=15, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
para(
    doc,
    "หมวด 14: โปรแกรมเพื่องานการพัฒนาด้านวิทยาศาสตร์และเทคโนโลยี",
    size=16,
    bold=True,
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=2,
)
para(doc, "(ระดับนิสิต นักศึกษา)", size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
para(doc, "รายงานฉบับสมบูรณ์", size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
para(doc, "เสนอต่อ", size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para(
    doc,
    "สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ",
    size=16,
    bold=True,
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=2,
)
para(
    doc,
    "กระทรวงการอุดมศึกษา วิทยาศาสตร์ วิจัยและนวัตกรรม",
    size=16,
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=14,
)
para(
    doc,
    "ได้รับทุนอุดหนุนโครงการวิจัย พัฒนาและวิศวกรรม",
    size=16,
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=2,
)
para(
    doc,
    "โครงการแข่งขันพัฒนาโปรแกรมคอมพิวเตอร์แห่งประเทศไทย ครั้งที่ 28",
    size=16,
    bold=True,
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=2,
)
para(doc, "ประจำปีงบประมาณ 2569", size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
para(doc, "โดย", size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para(
    doc,
    "นายรณชัย ขาวสะอาด (หัวหน้าโครงการ)",
    size=16,
    bold=True,
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=10,
)
para(
    doc,
    "อาจารย์ที่ปรึกษาโครงการ: ดร.วัชรพงศ์ อยู่ขวัญ",
    size=16,
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=2,
)
para(
    doc,
    "คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา  ·  จังหวัดชลบุรี",
    size=16,
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=2,
)
doc.add_page_break()

# ── กิตติกรรมประกาศ ─────────────────────────────────────────────────────────────
h1(doc, "กิตติกรรมประกาศ (Acknowledgement)")
body(
    doc,
    "โครงงานนี้ได้รับทุนอุดหนุนการพัฒนาผลงานจาก **โครงการการแข่งขันพัฒนาโปรแกรมคอมพิวเตอร์"
    "แห่งประเทศไทย ครั้งที่ 28 (NSC 2026)** โดย **สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ "
    "(สวทช.)** ในโครงการ *“ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วยโครงข่ายกราฟ"
    "ประสาทเทียมเชิงปริภูมิ-เวลาแบบอธิบายได้ สำหรับภาคเหนือของประเทศไทย”* คณะผู้พัฒนาขอขอบคุณ "
    "สวทช. สำหรับการสนับสนุน และขอขอบคุณอาจารย์ที่ปรึกษาที่ให้คำแนะนำตลอดการพัฒนา รวมถึงผู้ให้"
    "บริการข้อมูลสาธารณะ ได้แก่ กรมควบคุมมลพิษ (Air4Thai), NASA FIRMS และ Copernicus/ECMWF (ERA5)",
)

# ── บทคัดย่อ ─────────────────────────────────────────────────────────────────────
h1(doc, "บทคัดย่อ")
body(
    doc,
    "ภาคเหนือของประเทศไทยเผชิญวิกฤตฝุ่นละออง PM2.5 เป็นประจำทุกปีในช่วงฤดูแล้ง อันเกิดจากการเผา"
    "ชีวมวลทั้งภายในประเทศและการลำเลียงข้ามพรมแดนจากเมียนมาและลาว ในเดือนมีนาคม 2567 ค่า PM2.5 "
    "รายชั่วโมงที่เชียงใหม่สูงถึง 141–144 µg/m³ เครื่องมือที่มีอยู่เป็นแบบตั้งรับ ไม่พยากรณ์ล่วงหน้า"
    "และไม่ระบุแหล่งกำเนิด โครงงานนี้จึงพัฒนา **โครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลาแบบอธิบายได้ "
    "(Explainable STGNN)** ที่พยากรณ์ PM2.5 ล่วงหน้า 6, 12, 24 และ 48 ชั่วโมง ที่ 18 สถานี Air4Thai "
    "พร้อมกัน และ **ระบุสัดส่วนแหล่งกำเนิดฝุ่นเชิงปริมาณ** ด้วยเทคนิค Graph-based Integrated Gradients "
    "ร่วมกับ occlusion โดยบูรณาการข้อมูลลม/อุตุนิยมวิทยา ERA5 และจุดความร้อนดาวเทียม NASA FIRMS "
    "เป็นโหนดต้นทางไฟในกราฟ",
)
body(
    doc,
    "โครงงานนี้ให้ความสำคัญกับ **การประเมินผลอย่างเข้มงวดและซื่อสัตย์** เป็นพิเศษ: ใช้การแบ่งข้อมูล"
    "แบบ held-out (เทรน 2565–2566, validation 2567 สำหรับเลือกโมเดล, ทดสอบ held-out 2568), "
    "ช่วงความเชื่อมั่นแบบ block-bootstrap, และการทดสอบ ablation แบบหลาย seed ผลการประเมินพบว่า **ความได้เปรียบ"
    "เชิงพยากรณ์เหนือ baseline persistence มีจำกัด** — เด่นเฉพาะที่ 48 ชั่วโมง (RMSE 12.66 เทียบ 12.99 "
    "µg/m³ บนชุด held-out test ปี 2568; ราว 2%) และ **ยังไม่มีนัยสำคัญทางสถิติในข้อมูลปีเดียว** (95% CI "
    "คร่อม 0) อีกทั้ง ablation แบบหลาย seed บ่งชี้ว่า **กลไกกราฟไม่ได้ยกระดับความแม่นยำเหนือความผันผวน"
    "จากการสุ่ม seed** อย่างมีนัย",
)
body(
    doc,
    "**คุณค่าที่พิสูจน์ได้ของระบบคือความสามารถระบุแหล่งกำเนิดที่อธิบายได้** ซึ่ง persistence และโมเดล"
    "ที่ไม่ใช้กราฟทำไม่ได้: บนข้อมูล **held-out ปี 2568** ที่สถานีชายแดนแม่ฮ่องสอน (18 มี.ค. 2568) "
    "ระบบระบุว่าฝุ่นมาจากต่างชาติ (เมียนมา+ลาว) ประมาณ 63% สอดคล้องกับสัดส่วนไฟข้ามแดนที่เชื่อมถึงสถานีจริง "
    "(72.1%) และสัดส่วนที่ระบุแปรผันตามสัดส่วนไฟจริงทั้ง 5 เหตุการณ์ทดสอบ (ทั้งนี้ขนาดของสัดส่วนขึ้นกับโมเดล "
    "ดูรายละเอียดในผลการทดสอบ) ขณะที่เหตุการณ์หมอกควัน"
    "กลางเมืองเชียงใหม่ มีนาคม 2567 (141–144 µg/m³) ระบบระบุว่าเป็นการเผาในไทยเป็นหลัก (~100%; ค่า FRP "
    "ของไฟในไทยสูงกว่าไฟต่างชาติราว 59 เท่า) แสดงให้เห็นว่าระบบสามารถตรวจจับการลำเลียงข้ามแดนได้ตาม"
    "ช่วงเวลาและพื้นที่ที่เกิดขึ้นจริง นอกจากนี้คณะผู้พัฒนายังตรวจพบและแก้ไขข้อผิดพลาดการ denormalize ในขั้นตอนรายงานผลด้วย"
    "ตนเอง และเปิดเผยผลตามจริงทั้งหมด ซึ่งสะท้อนความเข้มงวดทางวิศวกรรมและธรรมาภิบาล AI ระบบทั้งหมด"
    "แสดงผลผ่าน dashboard (Streamlit) เพื่อสนับสนุนการตัดสินใจเชิงนโยบายสาธารณสุขและสิ่งแวดล้อม "
    "ยิ่งกว่านั้น เมื่อนำโมเดลที่แช่แข็งไปประเมินบนข้อมูลฤดูเผา มกราคม–เมษายน 2569 ที่ไม่เคยมีอยู่ตอน"
    "ออกแบบ (out-of-sample 100%) พบรูปแบบเดิมทุกประการ — เอาชนะ persistence ที่ขอบฟ้า 24/48 ชั่วโมง "
    "(+4.4% / +9.1%) และด้อยกว่าที่ขอบฟ้าสั้น — ยืนยันว่าผลไม่ได้เกิดจากการรั่วไหลของข้อมูล",
)

h1(doc, "Abstract")
body(
    doc,
    "Northern Thailand suffers a recurring PM2.5 haze crisis each dry season, driven by biomass "
    "burning and transboundary transport from Myanmar and Laos; in March 2024 hourly PM2.5 in "
    "Chiang Mai reached 141–144 µg/m³. Existing tools are reactive — they neither forecast ahead "
    "nor attribute sources. We present an **Explainable Spatio-Temporal Graph Neural Network "
    "(STGNN)** that jointly forecasts PM2.5 at 18 Air4Thai stations (6/12/24/48 h) and "
    "**quantifies source attribution** via Graph-based Integrated Gradients and occlusion, "
    "integrating ERA5 wind/meteorology and NASA FIRMS fire hotspots as source nodes in a "
    "heterogeneous graph.",
)
body(
    doc,
    "We emphasise **rigorous, honest evaluation**: a held-out split (train 2022–23, val 2024 for "
    "model selection, test 2025), paired block-bootstrap confidence intervals, and a multi-seed "
    "ablation. We find the **forecast advantage over a persistence baseline is marginal** — present "
    "only at 48 h (RMSE 12.66 vs 12.99 µg/m³ on the held-out 2025 test, ~2%) and **not "
    "statistically significant** in a single year (95% CI spans 0); the multi-seed ablation shows "
    "the **graph machinery does not robustly improve accuracy beyond training-seed variance**. The "
    "system's **demonstrated contribution is explainable source attribution**, which persistence "
    "and a non-graph baseline cannot provide: on **held-out 2025** data at the Mae Hong Son border "
    "station (2025-03-18) the model attributes ~63% of pollution to foreign (Myanmar+Laos) burning, "
    "tracking the ~72% foreign share of connected fire activity — and the attributed share tracks "
    "the true foreign share across all five test events — whereas the interior Chiang Mai March-2024 "
    "episode is attributed to predominantly Thai burning (~100%; Thai FRP ~59x the foreign total). "
    "We additionally "
    "caught and fixed a denormalisation reporting bug ourselves and report all results transparently "
    "— reflecting engineering rigour and AI governance. Results are served through a Streamlit "
    "dashboard for public-health and environmental decision support. Moreover, evaluating the frozen "
    "model on the Jan–Apr 2026 burning season — data that did not exist at design time (fully "
    "out-of-sample) — reproduces the same pattern (beating persistence at 24/48 h by +4.4%/+9.1%, "
    "losing at short horizons), confirming the result is not an artefact of leakage.",
)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("คำสำคัญ: ")
fmt(r, size=16, bold=True)
add_md_runs(
    p,
    "PM2.5, โครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลา, การวิเคราะห์แหล่งกำเนิด, Integrated Gradients, "
    "AI ที่อธิบายได้, ภาคเหนือประเทศไทย, FIRMS, ERA5",
)
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(8)
r = p.add_run("Keywords: ")
fmt(r, size=16, bold=True)
r2 = p.add_run(
    "PM2.5, Spatio-Temporal GNN, Source Attribution, Integrated Gradients, Explainable AI, "
    "Northern Thailand, Transboundary Haze, FIRMS, ERA5"
)
fmt(r2, size=16, italic=True)

# ── สารบัญ ───────────────────────────────────────────────────────────────────────
doc.add_page_break()
para(doc, "สารบัญ", size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
add_toc(doc)
doc.add_page_break()

# ── 1. บทนำ ──────────────────────────────────────────────────────────────────────
h1(doc, "1. บทนำ (หลักการและเหตุผล ความสำคัญ และความเป็นมา)")
body(
    doc,
    "ภาคเหนือของประเทศไทยประสบปัญหาหมอกควัน PM2.5 เป็นประจำทุกปีในช่วงฤดูหมอกควัน (มกราคม–เมษายน) "
    "ในปี 2567 ค่า PM2.5 รายชั่วโมงที่เชียงใหม่สูงถึง **141–144 µg/m³** ซึ่งสูงกว่าค่าแนะนำเฉลี่ย 24 "
    "ชั่วโมงขององค์การอนามัยโลก (WHO, 15 µg/m³) เกือบ 10 เท่า และเกินค่ามาตรฐานของประเทศไทย "
    "(50 µg/m³) หลายเท่า ส่งผลกระทบรุนแรงต่อสุขภาพประชาชนหลายล้านคนใน 9 จังหวัด ได้แก่ เชียงใหม่ "
    "เชียงราย ลำปาง ลำพูน แม่ฮ่องสอน น่าน พะเยา แพร่ และตาก ตลอดจนกระทบการท่องเที่ยวและเศรษฐกิจ",
)
body(
    doc,
    "แหล่งกำเนิด PM2.5 หลักในภูมิภาคนี้ ได้แก่ การเผาชีวมวล (ทั้งในไร่นาและไฟป่า) และการลำเลียงมลพิษ"
    "ข้ามพรมแดนจากเมียนมาและลาว ซึ่งพัดพาด้วยรูปแบบลมที่เปลี่ยนแปลงตลอดเวลา การทราบ **สัดส่วนผลกระทบ"
    "ของแต่ละแหล่งกำเนิด** จึงมีความสำคัญต่อการกำหนดนโยบายป้องกันที่ตรงจุด อย่างไรก็ตาม เครื่องมือที่มี"
    "อยู่ในปัจจุบัน เช่น เว็บไซต์ Air4Thai และการแจ้งเตือนของกรมควบคุมมลพิษ เป็นการรายงานค่าปัจจุบันแบบ"
    "ตั้งรับ (reactive) ไม่มีการพยากรณ์ล่วงหน้าระยะสั้น และไม่สามารถระบุสัดส่วนแหล่งกำเนิด (source "
    "attribution) ได้ ทำให้ผู้กำหนดนโยบายขาดข้อมูลทั้งสองด้าน",
)
body(
    doc,
    "งานวิจัยที่ผ่านมาแสดงว่าโครงข่ายกราฟประสาทเทียม (Graph Neural Networks) จับความสัมพันธ์เชิง"
    "ปริภูมิ-เวลาได้อย่างมีประสิทธิภาพ (Wang et al., 2020; Wu et al., 2020) แต่งานส่วนใหญ่ยังไม่"
    "บูรณาการข้อมูลจุดความร้อนเป็นองค์ประกอบของกราฟโดยตรง และขาดความสามารถด้านการระบุแหล่งกำเนิด"
    "ที่อธิบายได้ โครงงานนี้จึงพัฒนาระบบที่ตอบโจทย์ทั้งสองด้านพร้อมกัน — **พยากรณ์ล่วงหน้า** และ "
    "**อธิบายแหล่งที่มา** — โดยให้น้ำหนักเป็นพิเศษกับการประเมินผลอย่างซื่อสัตย์และตรวจสอบได้ ซึ่งเป็น"
    "หัวใจของธรรมาภิบาล AI",
)

# ── 2. วัตถุประสงค์และเป้าหมาย ───────────────────────────────────────────────────
h1(doc, "2. วัตถุประสงค์และเป้าหมาย")
h2(doc, "2.1 วัตถุประสงค์")
for i, t in enumerate(
    [
        "พัฒนาระบบพยากรณ์ PM2.5 รายชั่วโมงล่วงหน้า 6, 12, 24 และ 48 ชั่วโมง สำหรับ 18 สถานีตรวจวัด "
        "Air4Thai ใน 9 จังหวัดภาคเหนือ พร้อมกันเชิงพื้นที่",
        "บูรณาการข้อมูลจุดความร้อนจากดาวเทียม NASA FIRMS เป็นโหนดต้นทางไฟในกราฟ เพื่อจำลองการลำเลียง"
        "ฝุ่นจากแหล่งเผาไหม้สู่สถานี",
        "พัฒนาโมดูล XAI (Graph-based Integrated Gradients ร่วมกับ occlusion) เพื่อระบุสัดส่วนผลกระทบ"
        "เชิงปริมาณจากแหล่งกำเนิดไฟแยกตามประเทศ (ไทย/เมียนมา/ลาว)",
        "ประเมินระบบอย่างเข้มงวดด้วยการแบ่งข้อมูลแบบ held-out (train 2565–66 / val 2567 สำหรับเลือก"
        "โมเดล / test 2568 ที่ไม่เคยเห็นตอนเทรน) เทียบกับ baseline persistence, A3TGCN และโมเดลที่ไม่ใช้"
        "กราฟ พร้อมรายงานช่วงความเชื่อมั่นและนัยสำคัญทางสถิติตามจริง",
        "จัดทำ dashboard (Streamlit) แสดงผลพยากรณ์และการวิเคราะห์แหล่งกำเนิด เพื่อสนับสนุนการตัดสินใจ"
        "ของหน่วยงานที่เกี่ยวข้อง",
    ],
    1,
):
    numbered(doc, i, t)
h2(doc, "2.2 เป้าหมาย")
for i, t in enumerate(
    [
        "ระบบพยากรณ์ PM2.5 ที่ทำงานได้จริง พร้อม dashboard สำหรับผู้ใช้งาน",
        "โมดูล XAI ที่วิเคราะห์แหล่งกำเนิดและแสดงผลแบบ interactive ได้",
        "รายงานการประเมินผล (evaluation report) บน 18 สถานี 4 ขอบฟ้า ที่ตรวจสอบซ้ำได้ (reproducible)",
    ],
    1,
):
    numbered(doc, i, t)

# ── 3. ขอบเขตและข้อจำกัด ─────────────────────────────────────────────────────────
h1(doc, "3. ขอบเขตและข้อจำกัดของโครงการ")
h2(doc, "3.1 ขอบเขต")
for t in [
    "**พื้นที่:** 9 จังหวัดภาคเหนือ (กรอบพื้นที่ lon 97.0–101.5°E, lat 16.0–21.0°N)",
    "**สถานีตรวจวัด:** 18 สถานี Air4Thai (15 สถานีหลัก + 3 สถานีเสริม ที่ผ่านเกณฑ์การคัดเลือกด้าน"
    "ความครบถ้วนของข้อมูล) ครอบคลุมช่วง 2565–2568",
    "**ขอบฟ้าพยากรณ์:** 6, 12, 24, 48 ชั่วโมง ความละเอียดรายชั่วโมง",
    "**ช่วงข้อมูล:** เทรน 2565–2566 / validation 2567 (สำหรับเลือกโมเดล) / ทดสอบแบบ held-out 2568",
    "**แหล่งข้อมูล:** Air4Thai (PM2.5), ERA5 (u10, v10, t2m, d2m, blh), NASA FIRMS (FRP)",
]:
    bullet(doc, t)
h2(doc, "3.2 ข้อจำกัด")
for i, t in enumerate(
    [
        "ข้อมูล PM2.5 มาจาก Air4Thai เท่านั้น (ไม่รวมสถานีอื่นของกรมควบคุมมลพิษในช่วงที่ใช้ฝึก)",
        "**ERA5 เป็นข้อมูล reanalysis ที่มีความล่าช้า** จึงใช้ฝึกได้แต่การพยากรณ์เรียลไทม์ต้องใช้ NWP "
        "forecast แทน — โครงงานได้ **วัดผลกระทบของความคลาดเคลื่อนนี้ไว้แล้ว** (ดูหัวข้อผลการทดสอบ)",
        "**ความได้เปรียบเชิงพยากรณ์เหนือ persistence มีจำกัดและยังไม่มีนัยสำคัญ** และกลไกกราฟไม่ได้ยก"
        "ระดับความแม่นยำเหนือความผันผวนจากการสุ่ม seed (รายงานตามจริงในหัวข้อผลการทดสอบ) — คุณค่าหลัก"
        "ของระบบจึงอยู่ที่การระบุแหล่งกำเนิด มิใช่ความได้เปรียบด้านค่า RMSE",
        "การระบุแหล่งกำเนิดเป็น **การประมาณจากโมเดล** ที่ตรวจสอบเทียบกับข้อมูลไฟจริง FIRMS ไม่ใช่การวัด"
        "โดยตรงในอากาศ และอยู่ในระดับประเทศ ไม่ใช่ระดับพื้นที่ย่อย จึงไม่ควรนำไปใช้อ้างอิงเพื่อกล่าวโทษ"
        "ระหว่างประเทศ",
        "ระบบจำกัดเฉพาะพื้นที่ 9 จังหวัดภาคเหนือ และยังไม่รองรับการลำเลียงฝุ่นจากจีนและอินเดีย",
        "ความถูกต้องของ attribution ขึ้นกับคุณภาพและความครอบคลุมของการตรวจจับไฟของดาวเทียม "
        "(เมฆ/เซ็นเซอร์อาจมี bias)",
        "**ช่วงพยากรณ์แบบ conformal อิงสมมติฐาน exchangeability** — ช่วง 90% ที่รายงาน (§6.7) คาลิเบรต"
        "บนข้อมูล validation 2567 การพยากรณ์ด้วย NWP สดจริงทำให้การกระจายของ input ต่างไป (exchangeability "
        "ไม่เป็นจริง) ความครอบคลุมจริงอาจต่ำกว่า 90% โดยเฉพาะบางสถานี (เช่น 225585 เหลือ ~70% ที่ 48 ชม.)",
        "**ตัวชี้วัดการเกินค่ามาตรฐานขึ้นกับ base rate ของปี** — ผลการพยากรณ์โอกาสเกินค่ามาตรฐาน (§6.8) "
        "คิดบนปี 2568 ซึ่งมีอัตราการเกิดเหตุการณ์ (base rate) ที่เกณฑ์ 75 µg/m³ = 5.5% สูงกว่า climatology "
        "ปี 2565–2567 (4.4%) จึงต้องอ่านคะแนน Brier Skill Score เทียบ climatology ของช่วงฝึกเสมอ ไม่เทียบ"
        "ข้ามปีที่ base rate ต่างกัน",
    ],
    1,
):
    numbered(doc, i, t)

# ── 4. รายละเอียดของการพัฒนา ─────────────────────────────────────────────────────
h1(doc, "4. รายละเอียดของการพัฒนา")
h2(doc, "4.1 ทฤษฎีและงานที่เกี่ยวข้อง (Related Work)")
body(
    doc,
    "**การพยากรณ์ PM2.5 ด้วยโครงข่ายกราฟ** — งาน PM2.5-GNN (Wang et al., 2020) แสดงให้เห็นว่าการ"
    "จำลองความสัมพันธ์ระหว่างเมือง/สถานีเป็นกราฟ และผนวกความรู้เชิงโดเมน (ทิศทางลม ระยะทาง) ช่วยให้"
    "พยากรณ์ PM2.5 ได้ดีขึ้นในขอบฟ้ายาว ขณะเดียวกันงานดังกล่าวยังชี้ว่าในขอบฟ้าสั้น baseline แบบ "
    "persistence มีความแข็งแกร่งสูงเนื่องจาก PM2.5 มี autocorrelation สูง ซึ่งเป็นข้อสังเกตที่สอดคล้อง"
    "กับผลของโครงงานนี้",
)
body(
    doc,
    "**สถาปัตยกรรมหลัก MTGNN** — Wu et al. (2020) เสนอ MTGNN (Multivariate Time series GNN) ซึ่งเป็น"
    "โครงข่ายพยากรณ์อนุกรมเวลาหลายตัวแปรบนกราฟ จุดเด่นคือ *graph learning layer* ที่เรียนรู้ adaptive "
    "adjacency จาก node embeddings ร่วมกับ temporal convolution (TCN) และ graph convolution (GCN) "
    "โครงงานนี้เลือก MTGNN เป็นสถาปัตยกรรมหลักเพราะรองรับการป้อนกราฟภายนอก (3 ชนิดเส้นเชื่อมที่สร้าง"
    "จากข้อมูลจริง) ควบคู่กับ adaptive adjacency ได้",
)
body(
    doc,
    "**โมเดลเปรียบเทียบ A3TGCN** — โครงข่าย Attention Temporal GCN เป็น baseline ขนาดเล็ก (ผสาน GCN "
    "กับ GRU และ attention เชิงเวลา) ใช้เป็นจุดอ้างอิงความคุ้มค่าของความซับซ้อนของโมเดล",
)
body(
    doc,
    "**การอธิบายผลด้วย Integrated Gradients** — Sundararajan et al. (2017) เสนอ Integrated Gradients "
    "ซึ่งเป็นวิธี attribution ที่มี completeness axiom (ผลรวม attribution เท่ากับผลต่างของ output จริง"
    "กับ baseline) โครงงานนี้ขยายแนวคิดสู่ Graph-based Integrated Gradients (GB-IG) ร่วมกับเทคนิค "
    "occlusion เพื่อระบุสัดส่วนผลกระทบของแหล่งกำเนิดไฟแยกตามประเทศ — ความสามารถที่งานพยากรณ์ PM2.5 "
    "ด้วย GNN ส่วนใหญ่ยังไม่มี",
)
body(
    doc,
    "**ช่องว่างที่โครงงานนี้เติมเต็ม:** งานก่อนหน้าส่วนใหญ่ (ก) ไม่บูรณาการจุดความร้อนดาวเทียมเป็น"
    "โหนดต้นทางไฟในกราฟโดยตรง และ (ข) ไม่มีโมดูลระบุแหล่งกำเนิดที่อธิบายได้ โครงงานนี้รวมทั้งสองสิ่ง"
    "เข้าด้วยกัน และเน้นการประเมินผลแบบ held-out + multi-seed + ช่วงความเชื่อมั่น เพื่อรายงานคุณค่าที่"
    "แท้จริงของแต่ละองค์ประกอบอย่างซื่อสัตย์",
)

h2(doc, "4.2 ทฤษฎี หลักการ และเทคนิคที่ใช้")
figure(
    doc,
    ROOT / "architecture_diagram.png",
    "รูปที่ 1 สถาปัตยกรรมระบบ: input 18 สถานี + โหนด hotspot → กราฟ 3 ชนิดเส้นเชื่อม "
    "(พื้นที่/ลม/ไฟ) → MTGNN → พยากรณ์ 6/12/24/48 ชม. + โมดูล GB-IG",
    width_cm=15.0,
)
body(
    doc,
    "**(1) การสร้างกราฟพลวัต 3 ชนิดเส้นเชื่อม** — ระบบสร้างกราฟที่มี 18 station nodes และ M hotspot "
    "cluster nodes เชื่อมด้วยเส้นเชื่อม 3 ประเภทดังนี้:",
)
for t in [
    "**type_a (geographic):** ความใกล้เชิงพื้นที่แบบคงที่ น้ำหนัก `exp(−d/50)` สำหรับระยะ ≤ 100 กม.",
    "**type_b (wind-aware):** เส้นเชื่อมแบบมีทิศทาง ปรับตามลม ERA5 รายชั่วโมงด้วยสูตร "
    "`A_wind[i,j] = max(0, cos(θ_wind, θ_ij)) × exp(−d/λ)` (ระยะ ≤ 200 กม., alignment > 0.3) "
    "เพื่อจับทิศทางการลำเลียงมลพิษ",
    "**type_c (fire bipartite):** เชื่อมโหนด hotspot กับสถานีที่อยู่ปลายลม (ระยะ ≤ 500 กม., "
    "alignment > 0.4) ทำให้อิทธิพลของไฟแต่ละจุดส่งผ่านเส้นทางกราฟที่ชัดเจน",
]:
    bullet(doc, t, indent_cm=1.5)
body(
    doc,
    "**(2) FIRMS hotspot เป็นโหนดต้นทางไฟ** — ข้อมูลไฟ VIIRS/MODIS จาก NASA FIRMS ถูกจัดกลุ่มเชิง"
    "พื้นที่เป็น hotspot clusters และเพิ่มเป็นโหนดพิเศษในกราฟ แต่ละโหนดเก็บ FRP รวมและพิกัดศูนย์กลาง "
    "(`[total_frp, centroid_lat, centroid_lon]`) พร้อมป้ายประเทศต้นทางที่ระบุด้วย point-in-polygon "
    "กับขอบเขตประเทศจริง ซึ่งเป็นพื้นฐานของการระบุแหล่ง"
    "กำเนิดเชิงประเทศ",
)
body(
    doc,
    "**(3) โมเดล MTGNN** — รับหน้าต่างข้อมูลย้อนหลัง 24 ชั่วโมง (18 สถานี × 10 features) ผ่าน graph "
    "learning layer + TCN + GCN (hidden_dim = 64, 3 layers, kernel_size = 7, dropout = 0.3) แล้ว"
    "พยากรณ์ PM2.5 ที่ 18 สถานีพร้อมกันในขอบฟ้า 6/12/24/48 ชั่วโมง การหลอมรวมเชิงพื้นที่ใช้ผลรวมถ่วง"
    "น้ำหนักของเส้นเชื่อมทั้ง 3 ชนิดและ adaptive adjacency",
)
body(
    doc,
    "**(4) Graph-based Integrated Gradients (GB-IG) + Occlusion สำหรับ source attribution** — โมดูล "
    "XAI แบบ post-hoc คำนวณ attribution ด้วยสูตร `IG(x) = (x − b) × ∫₀¹ (∂F/∂x)|_{b+α(x−b)} dα` โดย "
    "F = MTGNN, b = zero baseline, 50 interpolation steps (รับประกัน completeness axiom) ส่วนการระบุ"
    "สัดส่วน **รายประเทศ** ใช้เทคนิค **occlusion**: ปิด (zero) โหนดไฟของแต่ละประเทศแล้ววัดว่าค่าพยากรณ์"
    "ลดลงเท่าใด ซึ่งให้คะแนนไม่ติดลบและรวมกันได้ 1",
)

h2(doc, "4.3 เครื่องมือที่ใช้ในการพัฒนา")
table(
    doc,
    ["องค์ประกอบ", "เทคโนโลยี"],
    [
        ["ภาษา", "Python 3.11 (จัดการแพ็กเกจด้วย uv)"],
        ["GNN Framework", "PyTorch 2.2+, PyG 2.5+, PyG Temporal"],
        ["โมเดลหลัก", "MTGNN (Wu et al., 2020)"],
        ["XAI", "Graph-based Integrated Gradients + Occlusion (พัฒนาเอง)"],
        ["Baseline ไม่ใช้กราฟ", "scikit-learn HistGradientBoostingRegressor"],
        ["การจัดการ Config", "Hydra"],
        ["Experiment Tracking", "Weights & Biases"],
        ["Dashboard", "Streamlit + Plotly"],
        ["คุณภาพโค้ด", "pytest, ruff, black"],
        ["แหล่งข้อมูล", "Air4Thai (PM2.5), NASA FIRMS (FRP), ERA5 (u10, v10, t2m, d2m, blh)"],
        ["Normalization", "RobustScaler ต่อสถานี"],
    ],
)

h2(doc, "4.4 รายละเอียดโปรแกรมเชิงเทคนิค (Software Specification)")
body(doc, "**Input:**")
for t in [
    "ค่า PM2.5 ย้อนหลัง 24 ชั่วโมงจาก 18 สถานี (pm25_scaled ผ่าน RobustScaler ต่อสถานี)",
    "ข้อมูลอุตุนิยมวิทยา ERA5: u10, v10, t2m, d2m, blh (รายชั่วโมง)",
    "กลุ่มจุดความร้อน NASA FIRMS: FRP รวม, พิกัดศูนย์กลาง, ประเทศต้นทาง (รายวัน)",
    "การเข้ารหัสเวลา: hour_sin, hour_cos, doy_sin, doy_cos (รวมเป็น 10 features ต่อสถานี)",
]:
    bullet(doc, t)
body(doc, "**Output:**")
for t in [
    "ค่า PM2.5 พยากรณ์สำหรับ 18 สถานี × 4 ขอบฟ้า (6/12/24/48 ชั่วโมง)",
    "แผนที่ source attribution: สัดส่วน (%) แยกตามประเทศ (ไทย/เมียนมา/ลาว) ต่อการพยากรณ์",
    "ลำดับความสำคัญของปัจจัย (IG feature importance)",
]:
    bullet(doc, t)
body(doc, "**Functional Specification:**")
for i, t in enumerate(
    [
        "ดาวน์โหลดและประมวลผลข้อมูลจาก 3 แหล่ง (Air4Thai, FIRMS, ERA5) แบบอัตโนมัติ",
        "สร้างกราฟพลวัตที่ปรับ edge weights ตามทิศทางลมทุกชั่วโมง",
        "พยากรณ์ PM2.5 ที่ 18 สถานีพร้อมกันใน 4 ขอบฟ้าด้วย MTGNN",
        "คำนวณ source attribution แยกตามประเทศด้วย GB-IG + occlusion",
        "แสดงผลผ่าน dashboard (Streamlit) พร้อมแผนที่ interactive, กราฟ time series และผลการประเมิน",
        "รายงานผลแบบตรวจสอบซ้ำได้ (JSON) ผ่านสคริปต์ pipeline",
    ],
    1,
):
    numbered(doc, i, t)
body(
    doc,
    "**โครงสร้างของซอฟต์แวร์ (Design):** repository แบ่งเป็น `src/data/` (scrapers, preprocessing, "
    "`graph_builder.py`, loader), `src/models/` (MTGNN, A3TGCN), `src/explain/` (`gb_ig.py`, "
    "`attribution.py`), `src/training/` (trainer, losses, metrics, `evaluation.py`), `app/` "
    "(dashboard), `configs/` (Hydra), `scripts/` (pipeline `01`–`12`)",
)
body(
    doc,
    "**ส่วนที่พัฒนาเอง vs. นำมาใช้ (Built vs. Reused):**",
)
for t in [
    "*พัฒนาเอง:* `src/data/graph_builder.py` (กราฟ 3 ชนิดเส้นเชื่อม + wind-aware), `src/explain/"
    "gb_ig.py` และ `attribution.py` (GB-IG + occlusion), `src/training/evaluation.py` (denorm/"
    "persistence/มาสก์/RMSE ที่ใช้ร่วมกันทุกสคริปต์เพื่อป้องกันความคลาดเคลื่อน), สคริปต์ประเมินผลและทดสอบ"
    "ความเข้มงวด `scripts/04`–`12`, และ dashboard ทั้งหมด",
    "*นำมาใช้ (อ้างอิงชัดเจน):* สถาปัตยกรรม MTGNN (Wu et al., 2020), HistGradientBoostingRegressor "
    "(scikit-learn), ไลบรารี PyTorch/PyG; ข้อมูลสาธารณะจาก Air4Thai, NASA FIRMS, ERA5",
]:
    bullet(doc, t)

h2(doc, "4.5 ความสามารถของระบบที่พัฒนาเพิ่มเติม (System Capabilities)")
body(
    doc,
    "นอกเหนือจากแกนพยากรณ์และการระบุแหล่งกำเนิด ระบบได้พัฒนาความสามารถเชิงวิศวกรรมเพิ่มเติมต่อไปนี้ "
    "(เป็นการพัฒนาระบบ ไม่ใช่การอ้างความแม่นยำเพิ่ม — ตัวเลขความแม่นยำทั้งหมดคงเดิมตาม §6):",
)
for i, t in enumerate(
    [
        "**โหมดพยากรณ์สดจากเวลาปัจจุบัน (Live NWP forecast)** — dashboard เพิ่มโหมด “พยากรณ์วันนี้” ที่"
        "ดึงค่าฝุ่นล่าสุดจาก Air4Thai realtime ร่วมกับพยากรณ์อากาศ (NWP) จาก Open-Meteo เพื่อพยากรณ์"
        "ล่วงหน้า 48 ชั่วโมงของวันจริง โดย **ระบุชัดเจนว่า input เป็น NWP ไม่ใช่ ERA5** และแนบลิงก์"
        "หลักฐานความทนต่อ noise ของ NWP (§6.4) โหมดนี้ช่วยลดข้อจำกัดของระบบแบบ hindcast "
        "โดยไม่กล่าวอ้างเกินจริง",
        "**การวิเคราะห์แบบโต้ตอบ “ถ้าดับไฟกลุ่มนี้” (interactive counterfactual/occlusion)** — ผู้ใช้"
        "เลือกประเทศ/กลุ่มไฟ แล้วระบบแสดงค่าพยากรณ์ก่อน/หลังปิดโหนดไฟ (µg/m³) ต่อสถานี ใช้กลไก occlusion "
        "เดียวกับ §6.5 ทำให้เห็นผลของมาตรการเชิงนโยบายอย่างเป็นรูปธรรม",
        "**การแจ้งเตือนล่วงหน้า 48 ชั่วโมงผ่าน Telegram** — แปลงค่าพยากรณ์เป็นระดับ AQI ไทยพร้อมคำแนะนำ"
        "กลุ่มเสี่ยง แล้วส่งผ่าน Telegram Bot API",
        "**ช่วงความเชื่อมั่นแบบ conformal ใน dashboard** — แสดงช่วงพยากรณ์ 90% (§6.7) ควบคู่ค่าจุด พร้อม "
        "caveat เรื่อง exchangeability",
        "**ธรรมาภิบาลทางวิศวกรรม** — เพิ่ม Continuous Integration (GitHub Actions: pytest + ruff + black "
        "ทุก push โดยไม่มี network call ในการทดสอบ) และจัดทำ Model Card + Data Card หนึ่งหน้า "
        "(`docs/MODEL_CARD.md`, `docs/DATA_CARD.md`) ตามแนวทาง AI governance",
    ],
    1,
):
    numbered(doc, i, t)

# ── 5. กลุ่มผู้ใช้โปรแกรม ─────────────────────────────────────────────────────────
h1(doc, "5. กลุ่มผู้ใช้โปรแกรม")
body(
    doc,
    "ระบบออกแบบให้รองรับผู้ใช้หลายกลุ่ม โดย dashboard นำเสนอข้อมูลแบบเข้าใจง่ายสำหรับสาธารณชน และมี"
    "หน้ารายละเอียดเชิงเทคนิคสำหรับผู้เชี่ยวชาญ:",
)
for i, t in enumerate(
    [
        "**หน่วยงานด้านสาธารณสุขและสิ่งแวดล้อม** (เช่น กรมควบคุมมลพิษ สำนักงานสาธารณสุขจังหวัด) — ใช้"
        "พยากรณ์ล่วงหน้า 48 ชั่วโมงเพื่อเตือนภัยและวางแผนมาตรการ และใช้ผลการระบุแหล่งกำเนิดประกอบการ"
        "ตัดสินใจเชิงนโยบาย",
        "**หน่วยงานปกครองท้องถิ่นและบรรเทาสาธารณภัย** — ใช้ข้อมูลเชิงพื้นที่ 18 สถานีและแนวโน้มเพื่อ"
        "จัดการพื้นที่เสี่ยง",
        "**นักวิจัยและสถาบันการศึกษา** — ใช้ผลที่ตรวจสอบซ้ำได้และโมดูล XAI เพื่อศึกษาการลำเลียงฝุ่นและ"
        "พัฒนาต่อยอด",
        "**ประชาชนทั่วไปและผู้ประกอบการท่องเที่ยว** — ดูค่าฝุ่นปัจจุบัน ดัชนี AQI คำแนะนำสุขภาพ และแนวโน้ม "
        "48 ชั่วโมง เพื่อวางแผนกิจกรรมกลางแจ้ง",
    ],
    1,
):
    numbered(doc, i, t)

# ── 6. ผลการทดสอบและประเมินผล ───────────────────────────────────────────────────
h1(doc, "6. ผลการทดสอบและประเมินผล")
body(
    doc,
    "รายงานนี้ยึดผลบน **ชุดทดสอบ held-out ปี 2568** เป็นหลัก จากการแบ่งข้อมูล 3 ส่วน (เทรน 2565–2566, "
    "validation 2567 สำหรับเลือกโมเดล, ทดสอบ 2568 ที่โมเดลไม่เคยเห็น) ผลทุกค่าตรวจสอบซ้ำได้ผ่านสคริปต์ "
    "`scripts/04`–`12` และคำนวณค่า µg/m³ ผ่านโมดูลร่วม `src/training/evaluation.py` (ป้องกันความคลาด"
    "เคลื่อนจากการ denormalize) baseline persistence ใช้ค่าที่สังเกตล่าสุด (carry-forward) และทุกวิธีถูกประเมินบนตำแหน่งที่"
    "ถูกต้องชุดเดียวกัน",
)
callout(
    doc,
    "**หมายเหตุการอ่านผล (ความซื่อสัตย์):** ผลในหัวข้อนี้อยู่บน**ข้อมูลปี 2568 ทั้งหมด** ตารางหลักคือ "
    "**held-out test** จากโมเดลที่เทรนใหม่บนปี 2565–2566 (ไม่เคยเห็นปี 2568) ส่วนตารางอ้างอิงคือ "
    "**โมเดลหลักที่ใช้ใน dashboard/การนำเสนอ** ซึ่งเทรนถึงปี 2567 และใช้ปี 2568 เป็น validation — ทั้ง"
    "สองเป็นข้อมูล*ปีเดียวกัน (2568)* ต่างกันที่วิธีเทรนและการเป็น held-out ไม่ใช่คนละปี",
)

h2(doc, "6.1 ความแม่นยำการพยากรณ์ (RMSE, µg/m³ — ยิ่งต่ำยิ่งดี)")
body(
    doc,
    "**ผลหลัก — ชุด held-out test ปี 2568** (โมเดลรายงาน เทรน 2565–66 ไม่เคยเห็นปี 2568) — แหล่ง "
    "`evaluation_test2025.json`, `baseline_ml_test.json` (3,637 ตัวอย่าง, 18 สถานี):",
)
table(
    doc,
    ["วิธี", "6h", "12h", "24h", "48h"],
    [
        ["Persistence", "2.96", "5.19", "8.70", "12.99"],
        ["**MTGNN (กราฟ)**", "4.52", "5.93", "8.76", "**12.66**"],
        ["A3TGCN", "6.93", "7.86", "9.92", "13.34"],
        ["GBM (ไม่ใช้กราฟ)", "3.78", "5.58", "9.55", "14.22"],
    ],
)
body(
    doc,
    "**อ้างอิง — โมเดลหลัก (ใช้ใน dashboard/การนำเสนอ) ประเมินบนปี 2568** (เทรนถึงปี 2567, ใช้ปี 2568 "
    "เป็น validation) — แหล่ง `evaluation_val2025.json`:",
)
table(
    doc,
    ["วิธี", "6h", "12h", "24h", "48h"],
    [
        ["Persistence", "2.96", "5.19", "8.70", "12.99"],
        ["**MTGNN (กราฟ)**", "4.31", "5.80", "**8.67**", "**12.68**"],
        ["A3TGCN", "6.56", "7.48", "9.63", "13.18"],
        ["Hybrid (ใช้งานจริง)", "2.96", "5.19", "8.67", "12.68"],
    ],
)
body(doc, "ทั้งสองชุดเป็นข้อมูลปี 2568 และให้ผลใกล้เคียงกัน (เช่น 48h: 12.66 เทียบ 12.68 µg/m³)")
body(
    doc,
    "**การอ่านผลอย่างตรงไปตรงมา:** ในขอบฟ้าสั้น (6h, 12h) persistence เหนือกว่าโมเดลอย่างชัดเจน "
    "เนื่องจาก PM2.5 มี autocorrelation สูง (สอดคล้องกับวรรณกรรม PM2.5-GNN) MTGNN เหนือกว่า persistence "
    "เพียงเล็กน้อยที่ 48h (12.66 เทียบ 12.99 บน held-out test; และ 12.68 จากโมเดลหลักบนปี 2568) ส่วนที่ "
    "24h ใกล้เคียงกันหรือด้อยกว่าเล็กน้อย คณะผู้พัฒนาจึงออกแบบ **ระบบ hybrid** (persistence ช่วงสั้น + "
    "MTGNN ช่วงยาว) ที่"
    "ให้ผล “ดีเท่าหรือดีกว่า baseline ทุกขอบฟ้า” สำหรับการใช้งานจริง เทียบกับโมเดลที่ไม่ใช้กราฟ (GBM) "
    "พบว่า GBM ใกล้เคียง MTGNN มาก (โมเดลหลักบนปี 2568 GBM ได้ 3.83/5.35/8.73/12.86) บ่งชี้ว่ากลไก"
    "กราฟให้ประโยชน์เพิ่มเพียงเล็กน้อย โดย MTGNN มี 252,588 พารามิเตอร์ (best epoch 15) ขณะที่ A3TGCN "
    "มี 27,164 พารามิเตอร์ (epoch 97) และ MTGNN ดีกว่า A3TGCN ที่ 24h ราว 7.1% ในเชิง normalized RMSE "
    "(0.4576 เทียบ 0.4928)",
)
figure(
    doc,
    FIG / "rmse_by_horizon.png",
    "รูปที่ 2 RMSE ต่อขอบฟ้าบนชุด held-out test 2568 (ยิ่งต่ำยิ่งดี) — "
    "Persistence / MTGNN / A3TGCN / GBM",
)

h2(doc, "6.2 นัยสำคัญทางสถิติ (paired block-bootstrap)")
body(
    doc,
    "ใช้ block-bootstrap แบบจับคู่ตามวันต้นทางการพยากรณ์ (B = 2000) เปรียบเทียบ MTGNN กับ persistence "
    "— แหล่ง `significance_val2025.json`, `significance_test2025.json`:",
)
for t in [
    "**48h (held-out test 2568):** ดีขึ้น +2.54% แต่ 95% CI = **[−1.44%, +6.28%]** ซึ่งคร่อม 0",
    "**48h (โมเดลหลัก บนปี 2568):** ดีขึ้น +2.37% แต่ 95% CI = **[−3.31%, +8.10%]** ซึ่งคร่อม 0",
    "**24h:** เสมอ (CI คร่อม 0 เช่นกัน)",
]:
    bullet(doc, t)
body(
    doc,
    "**สรุป:** ความได้เปรียบที่ 48h **ยังไม่มีนัยสำคัญทางสถิติในข้อมูลปีเดียว** คณะผู้พัฒนาจึงไม่กล่าวอ้าง"
    "ว่าโมเดลเหนือกว่า persistence อย่างมีนัยสำคัญ และนำเสนอคุณค่าหลักของระบบด้วยการระบุแหล่งกำเนิด "
    "(XAI) ไม่ใช่ตัวเลข RMSE",
)
figure(
    doc,
    FIG / "significance_48h.png",
    "รูปที่ 3 ความได้เปรียบที่ 48 ชม. เทียบ persistence พร้อม 95% CI — "
    "ทั้งสองชุดคร่อม 0 (ยังไม่มีนัยสำคัญ)",
)

h2(doc, "6.3 การวิเคราะห์ผลของ 3 จุดใหม่ (Multi-seed Ablation)")
body(
    doc,
    "เพื่อตอบว่า “3 จุดใหม่ช่วยความแม่นยำจริงหรือไม่” คณะผู้พัฒนาเทรนโมเดลใหม่ 3 seeds ต่อ variant "
    "บนชุด held-out test แล้วถอดทีละองค์ประกอบ — แหล่ง `ablation_multiseed.json`:",
)
for t in [
    "โมเดลเต็ม (full) RMSE เฉลี่ย = 5.01 / 6.19 / 8.93 / 12.70 µg/m³ (6/12/24/48h)",
    "ถอด**กราฟทั้งหมด** (temporal-only) ต่างจาก full เพียง **+0.04 / +0.13 / +0.18 / +0.14** µg/m³ "
    "ซึ่งอยู่ในช่วงความผันผวนจากการสุ่ม seed",
    "**ไม่มี variant ใด** (ถอด type_b ลม / type_c ไฟ / adaptive / กราฟทั้งหมด) ที่ให้ผลต่างเกินความ"
    "ผันผวนของ seed ที่ขอบฟ้าใด ๆ (`robust_beyond_noise = false` ทุกช่อง)",
]:
    bullet(doc, t)
body(
    doc,
    "**ข้อสรุปที่ซื่อสัตย์:** กลไกกราฟ**ไม่ได้ยกระดับความแม่นยำการพยากรณ์อย่างมีนัยเหนือความผันผวนจาก"
    "การสุ่ม seed** ผลจากการทดสอบด้วย seed เดียวก่อนหน้านี้ ซึ่งดูเสมือนว่ากลไกกราฟช่วยเพิ่มความแม่นยำ "
    "แท้จริงเป็นเพียงผลจากความผันผวนของ seed (artifact) การรายงาน"
    "ผลลบนี้อย่างเปิดเผยคือจุดแข็งด้านธรรมาภิบาล AI ของโครงงาน",
)
figure(
    doc,
    FIG / "ablation_multiseed.png",
    "รูปที่ 4 Ablation หลาย seed: ผลต่างของแต่ละองค์ประกอบเทียบ full (±combined std) "
    "อยู่ในช่วง noise ทุกขอบฟ้า",
)

h2(doc, "6.4 ความทนต่อความคลาดเคลื่อนของพยากรณ์อากาศ (NWP Sensitivity)")
body(
    doc,
    "ERA5 เป็น reanalysis ที่มีความล่าช้า ระบบจริงจึงต้องใช้ NWP forecast คณะผู้พัฒนาจึงเติม noise "
    "(σ = สัดส่วน × ส่วนเบี่ยงเบนธรรมชาติของแต่ละตัวแปร) ลง ERA5 ทั้ง 5 ตัว เฉลี่ย 3 seeds — แหล่ง "
    "`nwp_sensitivity.json`:",
)
table(
    doc,
    ["ระดับ noise (× ธรรมชาติ)", "MTGNN 24h", "MTGNN 48h"],
    [
        ["0.00 (ERA5 จริง)", "8.67", "12.68"],
        ["0.25", "8.79", "12.76"],
        ["0.50", "9.22", "13.03"],
        ["1.00", "11.21", "14.45"],
    ],
)
body(
    doc,
    "(persistence: 24h = 8.70, 48h = 12.99) ความได้เปรียบที่ **24h หายไปเมื่อ noise ถึง 0.25×** ส่วน "
    "**48h ทนได้เมื่อ noise ต่ำกว่า 0.5× และเริ่มเสียเปรียบที่ 0.5×** การออกแบบเชิงรับมือคือ (1) พึ่ง"
    "โมเดลที่ขอบฟ้า 48h และใช้ hybrid กับช่วงสั้น (2) ระบบจริงควรใช้ NWP คุณภาพสูง (ECMWF HRES/GFS) "
    "(3) ความล่าช้าของ ERA5 กระทบเฉพาะ inference ไม่กระทบการเทรน",
)
figure(
    doc,
    FIG / "nwp_sensitivity.png",
    "รูปที่ 5 ความทนต่อ noise ของ ERA5: ความได้เปรียบ 24h เสียที่ 0.25×, 48h ที่ 0.5×",
)

h2(doc, "6.5 การระบุแหล่งกำเนิด (XAI Results) — คุณค่าหลักของระบบ")
body(
    doc,
    "**หมายเหตุความถูกต้องของป้ายประเทศ:** ผลการระบุแหล่งกำเนิดทั้งหมดในหัวข้อนี้ใช้ป้ายประเทศของ"
    "กลุ่มไฟที่คำนวณด้วย point-in-polygon กับขอบเขตประเทศจริง (`src/data/geocode.py`) ซึ่งแทนที่วิธี "
    "bounding box หยาบรุ่นแรกที่คณะผู้พัฒนาตรวจพบว่านับไฟเมียนมา/ลาวขาดไปราว 25% ของกลุ่มไฟทั้งหมด "
    "(ป้ายเปลี่ยน 6,758 จาก 26,839 กลุ่ม) การแก้ไขนี้กระทบเฉพาะการจัดกลุ่มประเทศตอนคำนวณ attribution "
    "ไม่กระทบฟีเจอร์ที่เข้าโมเดล จึงไม่ต้องเทรนใหม่",
)
body(
    doc,
    "**(ก) เหตุการณ์กลางเมืองเชียงใหม่ มีนาคม 2567 (141–144 µg/m³)** — แหล่ง "
    "`attribution_march2024.json`: ระบบระบุว่าฝุ่นมาจากการเผาในไทยเป็นหลัก (**Thailand ~99.7%, Myanmar "
    "0.3%, Laos 0%**) ซึ่งสอดคล้องกับข้อมูลดาวเทียม โดยค่า FRP รวมของไฟในไทย (108,569) สูงกว่าไฟต่างชาติ"
    "ทั้งหมดรวมกัน (เมียนมา 246 + ลาว 1,603 ≈ 1,849) ราว "
    "**59 เท่า** ช่อง hotspot เพิ่มค่าพยากรณ์เฉลี่ย **3.49 µg/m³** (ช่วง 2.5–5.6) เหนือ baseline ที่ไม่มี"
    "ไฟ และ Integrated Gradients ชี้ว่าปัจจัยความชื้น/อุณหภูมิ (d2m 0.0345, t2m 0.0344) มีผลเด่นกว่า"
    "ปัจจัยอื่น เมื่อเทรนใหม่บน split ที่ทำให้ มี.ค. 2567 เป็น held-out ผลยังคงเป็น Thailand ~100% "
    "(`attribution_march2024_val.json`: 99.6%) ยืนยันว่าไม่ใช่การจดจำข้อมูล (memorization)",
)
body(
    doc,
    "**(ข) การระบุข้ามแดนบนข้อมูล held-out 2568 ที่สถานีชายแดนแม่ฮ่องสอน** — แหล่ง "
    "`transboundary_attr_test_split2.json` (โมเดลรายงาน): ในวันที่ **18 มี.ค. 2568** (PM2.5 สูงสุด 69.1 "
    "µg/m³) ไฟที่เชื่อมถึงสถานีเป็นไฟต่างชาติ **72.1%** ของ FRP (เมียนมา 677.8 + ลาว 406.1 เทียบไทย "
    "419.5) และระบบระบุว่าฝุ่นมาจากต่างชาติ **62.7%** (เมียนมา 51.6%, ลาว 11.1%) — สอดคล้องกับสัดส่วน"
    "ไฟจริงอย่างดี ยิ่งกว่านั้น สัดส่วนที่โมเดลระบุ**แปรผันตามสัดส่วนไฟจริงอย่างเป็นระบบทั้ง 5 เหตุการณ์**"
    "ที่มีไฟต่างชาติสูงสุดของปีทดสอบ: 72.1%→62.7%, 52.8%→40.3% (กรณีไฟลาว 13 มี.ค.), 11.5%→0.5%, "
    "3.7%→0%, 1.4%→0% อย่างไรก็ตาม **ขนาดของสัดส่วนขึ้นกับโมเดล**: โมเดลหลัก (pitch) ระบุต่างชาติเพียง "
    "~0–6% ในเหตุการณ์ชุดเดียวกัน (`transboundary_attr_test.json`) ความสามารถติดตามไฟข้ามแดนที่ "
    "validate ได้นี้จึงเป็นของโมเดลรายงาน (ที่เทรนบน split เข้มงวด) และต้องรายงานพร้อม caveat นี้เสมอ",
)
body(
    doc,
    "**บทสรุป XAI:** ระบบตรวจจับการลำเลียงข้ามแดนได้ตามช่วงเวลาและพื้นที่ที่เกิดขึ้นจริง กล่าวคือ "
    "เมื่อไฟข้ามแดนอยู่ใกล้สถานีชายแดน (แม่ฮ่องสอน) ระบบระบุสัดส่วนต่างชาติสูง ขณะที่เหตุการณ์ในเมือง"
    "ที่อยู่ลึกเข้ามาในแผ่นดิน (เชียงใหม่) ระบบระบุว่าเป็นไฟภายในประเทศเป็นหลัก — ความสามารถนี้ "
    "persistence และโมเดลที่ไม่ใช้กราฟ**ไม่สามารถทำได้** และคือคุณค่าที่พิสูจน์ได้ของระบบ",
)
figure(
    doc,
    FIG / "transboundary_map.png",
    "รูปที่ 6 จุดความร้อนใกล้สถานีแม่ฮ่องสอน (18 มี.ค. 2568) แยกสีตามประเทศ — "
    "ไฟเมียนมาลูกใหญ่อยู่ประชิดสถานีทางทิศตะวันตก",
    width_cm=12.0,
)
figure(
    doc,
    FIG / "transboundary_events.png",
    "รูปที่ 7 สัดส่วนไฟต่างชาติจริง (FRP) เทียบกับที่โมเดลระบุ รายเหตุการณ์ — "
    "แปรผันตามกันทั้ง 5 เหตุการณ์ (เช่น 18 มี.ค. 2568: 72%→63%)",
)
figure(
    doc,
    FIG / "ig_feature_importance.png",
    "รูปที่ 8 ความสำคัญของปัจจัย (Integrated Gradients) กรณีเชียงใหม่ มี.ค. 2567 — "
    "ความชื้น/อุณหภูมิ (d2m, t2m) เด่น",
)

h2(doc, "6.6 ความซื่อสัตย์ของการรายงานผล (Engineering Integrity)")
body(
    doc,
    "ระหว่างการพัฒนา คณะผู้พัฒนา **ตรวจพบข้อผิดพลาดในขั้นตอน denormalization** ของการรายงานผล (ใช้ "
    "scale ผิด) ซึ่งทำให้ค่า RMSE ที่รายงานในข้อเสนอโครงการฉบับส่ง (พฤษภาคม 2568) **ต่ำกว่าจริงและ"
    "เปอร์เซ็นต์การเอาชนะ persistence สูงเกินจริง** ทีมได้แก้ไขด้วยตนเอง รวมศูนย์การคำนวณไว้ที่ "
    "`src/training/evaluation.py` และเขียนสคริปต์ที่ reproduce ผลได้ ทั้งนี้ **โมเดลและการเทรนถูกต้องอยู่"
    "แล้ว** (normalized metric และลำดับของโมเดลไม่เปลี่ยน) ค่าที่ถูกต้องคือชุดในตารางข้างต้นทั้งหมด การ"
    "ตรวจพบและเปิดเผยข้อผิดพลาดด้วยตนเองสะท้อนความเข้มงวดทางวิศวกรรมและธรรมาภิบาล AI ซึ่งเป็นหลักการ"
    "สำคัญของโครงงานนี้",
)

h2(doc, "6.7 ช่วงพยากรณ์แบบ Conformal (90% coverage)")
body(
    doc,
    "เพื่อสื่อสารความไม่แน่นอนของค่าพยากรณ์อย่างมีการรับประกันเชิงสถิติ ระบบใช้ **split-conformal "
    "prediction** (Lei et al., 2018) คาลิเบรตค่าครึ่งความกว้าง (half-width) จากส่วนที่เหลือสัมบูรณ์ "
    "(absolute residual) ต่อสถานี บนข้อมูล validation ปี 2567 (α = 0.1, เป้าหมายความครอบคลุม 90%, "
    "n = 8,713) แล้ว **ทดสอบความครอบคลุมจริงบนชุด held-out test 2568** ที่ไม่เห็นตอนคาลิเบรต — แหล่ง "
    "`conformal/conformal_intervals.json`, `conformal/holdout_coverage_test2025.json`:",
)
table(
    doc,
    ["ขอบฟ้า", "ครึ่งความกว้าง ±µg/m³", "ความครอบคลุมจริง (เป้า 0.90)"],
    [
        ["6h", "±5.9", "0.885"],
        ["12h", "±7.8", "0.882"],
        ["24h", "±11.5", "0.882"],
        ["48h", "±15.5", "0.870"],
    ],
)
body(
    doc,
    "**การอ่านผลอย่างซื่อสัตย์:** ความครอบคลุมจริงบน held-out ต่ำกว่าเป้า 90% เล็กน้อยและลดลงตามขอบฟ้า "
    "(88.5% → 87.0%) ซึ่งเป็นพฤติกรรมที่คาดได้เมื่อการกระจายของข้อมูลทดสอบต่างจากช่วงคาลิเบรต ในระดับ"
    "สถานี ความครอบคลุมมีความแปรปรวน โดยสถานีที่แย่สุดคือ 225585 เหลือ 81.8% ที่ 6 ชม. และ 69.8% ที่ 48 "
    "ชม. จึงต้องรายงานช่วงเป็น “โดยประมาณ” และเตือนว่าเมื่อใช้ NWP สด (ไม่ใช่ ERA5) สมมติฐาน "
    "exchangeability จะไม่เป็นจริงและความครอบคลุมอาจคลาดเคลื่อนได้ (ดู §3.2 ข้อ 7)",
)

h2(doc, "6.8 การพยากรณ์โอกาสเกินค่ามาตรฐาน (Exceedance Forecasting)")
body(
    doc,
    "นอกจากค่าตัวเลข RMSE ระบบยัง re-score ค่าพยากรณ์เดิมเป็นปัญหา **จำแนกโอกาสเกินเกณฑ์** (37.5 และ "
    "75 µg/m³) ซึ่งตรงกับการใช้งานเตือนภัยจริงมากกว่า — แหล่ง `exceedance_test2025.json` (test 2568, "
    "n = 3,637) วัดด้วย Brier Skill Score (BSS) เทียบ climatology ของช่วงฝึก 2565–2567 (ไม่รั่วข้อมูล):",
)
table(
    doc,
    ["เกณฑ์ / ขอบฟ้า", "BSS โมเดล", "BSS persistence"],
    [
        ["37.5 µg/m³ @ 48h", "0.306", "0.282"],
        ["75 µg/m³ @ 48h", "0.190", "0.170"],
    ],
)
body(
    doc,
    "**การอ่านผลอย่างตรงไปตรงมา:** โมเดล **มีคะแนน BSS สูงกว่า persistence ทั้งสองเกณฑ์ที่ขอบฟ้า 48 "
    "ชั่วโมง** (ด้วยระยะห่างเล็กน้อย; ที่ 24 ชั่วโมง โมเดลนำเฉพาะเกณฑ์ 75 µg/m³) ซึ่งสอดคล้องกับผล RMSE "
    "(§6.1) ที่โมเดลเด่นเฉพาะขอบฟ้ายาว ส่วนขอบฟ้าสั้น "
    "persistence เหนือกว่าและโมเดลมี **อัตราเตือนพลาด (False Alarm Rate) สูงกว่า** (เช่น 6 ชม. ที่เกณฑ์ "
    "37.5: FAR โมเดล 0.088 เทียบ persistence 0.051) ทั้งนี้ตัวเลขทั้งหมดคิดบนปี 2568 ซึ่งมี base rate ที่"
    "เกณฑ์ 75 µg/m³ = 5.5% สูงกว่า climatology 2565–2567 (4.4%) จึงต้องอ่าน BSS เทียบ climatology ของ"
    "ช่วงฝึกเสมอ (ดู §3.2 ข้อ 8) ผลนี้ยืนยันแนวทางการใช้งานแบบ hybrid: พึ่งโมเดลที่ขอบฟ้ายาวเพื่อการ"
    "เตือนภัยล่วงหน้า 48 ชั่วโมง",
)

h2(doc, "6.9 การประเมินนอกกลุ่มตัวอย่างจริง — ฤดูเผา 2569")
body(
    doc,
    "เพื่อขจัดข้อสงสัยเรื่องการรั่วไหลของข้อมูล (data leakage) อย่างสิ้นเชิง คณะผู้พัฒนานำ "
    "**โมเดลรายงานที่แช่แข็งไว้** "
    "(ไม่เทรนใหม่ ไม่ปรับ scaler) มาประเมินบนข้อมูลฤดูเผา **มกราคม–เมษายน 2569** ซึ่งเป็นข้อมูลที่ **ไม่มี"
    "อยู่เลยตอนออกแบบและเทรนโมเดล** (out-of-sample 100%) — แหล่ง `evaluation_2026burn.json` (n = 2,809, "
    "ครบทั้ง 18 สถานี, ความครอบคลุม 80.6–100%):",
)
table(
    doc,
    ["วิธี", "6h", "12h", "24h", "48h"],
    [
        ["Persistence", "4.08", "7.22", "11.99", "17.52"],
        ["**MTGNN (โมเดลรายงาน)**", "5.47", "7.67", "**11.46**", "**15.92**"],
        ["ดีขึ้นเทียบ persistence", "−34.1%", "−6.2%", "**+4.4%**", "**+9.1%**"],
    ],
)
body(
    doc,
    "**การอ่านผลอย่างซื่อสัตย์:** บนข้อมูลจริงที่ไม่เคยเห็นมาก่อน โมเดลเอาชนะ persistence ที่ 24 และ 48 "
    "ชั่วโมง (+4.4% และ +9.1%) และด้อยกว่าที่ขอบฟ้าสั้น — รูปแบบเดียวกับผล held-out 2568 ทุกประการ ซึ่งเป็น"
    "หลักฐานที่หนักแน่นว่าความได้เปรียบขอบฟ้ายาวไม่ได้มาจากการจดจำข้อมูล **ข้อควรระวังสำคัญ:** ค่า RMSE "
    "สัมบูรณ์ในตารางนี้ **สูงกว่า** ชุด full-year 2568 (เช่น persistence 48h 17.52 เทียบ 12.99 µg/m³) "
    "เพราะเป็นช่วง **ฤดูเผาโดยเฉพาะ** ที่ค่าฝุ่นสูงและผันผวนกว่า จึง **เทียบขนาดสัมบูรณ์ข้ามตารางไม่ได้** "
    "— เทียบได้เฉพาะ *เปอร์เซ็นต์การเอาชนะ* เท่านั้น (โปรโตคอลการวัดเหมือนกับ `evaluation_test2025.json` "
    "ทุกประการ: โมเดลแช่แข็ง ไม่เทรนใหม่ ไม่ refit scaler)",
)

h2(doc, "6.10 เมทริกซ์การระบุข้ามแดนหลายสถานี และความไม่แน่นอนหลาย seed (ขยายผล §6.5)")
body(
    doc,
    "**(ก) เมทริกซ์ สถานีชายแดน × checkpoint** — เพื่อยกระดับหลักฐานจากกรณีเดียวเป็นเชิงระบบ คณะผู้พัฒนา"
    "รัน occlusion country attribution ที่ **สถานีชายแดน 5 แห่ง** (ระยะ ≤ 50 กม. จากพรมแดนเมียนมา/ลาว) "
    "บน **2 checkpoint** (demo และ report/split2) สำหรับเหตุการณ์หลัก 18 มี.ค. 2568 — แหล่ง "
    "`transboundary_matrix.json`:",
)
table(
    doc,
    [
        "สถานี (ระยะพรมแดน)",
        "สัดส่วนไฟต่างชาติเชื่อมถึง",
        "attribution (demo)",
        "attribution (report)",
    ],
    [
        ["แม่สาย 225567 (2.1 กม.)", "0.811", "0.0", "0.943"],
        ["แม่สอด 225626 (6.5 กม.)", "0.820", "0.957", "0.994"],
        ["แม่ฮ่องสอน 225648 (13.5 กม.)", "0.721", "0.0", "0.627"],
        ["เชียงราย 2328 (40.7 กม.)", "0.753", "0.156", "0.507"],
        ["น่าน 225674 (45.5 กม.)", "0.738", "0.0", "0.655"],
    ],
)
body(
    doc,
    "สถานีชายแดนแท้ (แม่สาย แม่สอด) แสดงสัดส่วนต่างชาติสูงเมื่อไฟต่างชาติเชื่อมถึงจริง อย่างไรก็ตาม "
    "**ขนาดของสัดส่วนขึ้นกับ checkpoint**: มีเพียง **แม่สอด** ที่สัดส่วนต่างชาติของเหตุการณ์หลักสูง"
    "ภายใต้ **ทั้งสอง checkpoint** (0.957 และ 0.994) ขณะที่แม่สายและแม่ฮ่องสอนเปลี่ยนจาก 0 (demo) เป็น"
    "ค่าสูง (report) — คณะผู้พัฒนาแสดงผลของทั้งสอง checkpoint โดยไม่คัดเลือกรายงานเฉพาะผลที่เป็นคุณ "
    "ตามหลักการรายงานผลตามจริง",
)
body(
    doc,
    "**(ข) ความไม่แน่นอนของ attribution จากหลาย seed** — เพื่อเปลี่ยน caveat “ขนาดขึ้นกับโมเดล” จาก"
    "คำเตือนเชิงนามธรรมให้เป็นตัวเลขที่วัดได้จริง คณะผู้พัฒนารัน attribution ซ้ำด้วยโมเดล full variant "
    "**3 seed** ที่แม่ฮ่องสอน — แหล่ง `transboundary_uncertainty.json`: เหตุการณ์หลัก 18 มี.ค. 2568 ให้สัดส่วน"
    "ต่างชาติราย seed = **0.627 / 0.0 / 1.0** (ค่าเฉลี่ย **0.542**, ส่วนเบี่ยงเบนมาตรฐาน 0.505, ช่วง "
    "min–max **0.0–1.0**) โดย **ทิศทาง (ต่างชาติ > 0) คงอยู่ 2 ใน 3 seed** ส่วน **ขนาดแกว่งเต็มช่วง** "
    "ค่าเฉลี่ยของช่วง min–max ต่อเหตุการณ์ทั้ง 5 = 0.397 ดังนั้นเมื่ออ้างตัวเลข 62.7% ต้องรายงานช่วง "
    "0.0–1.0 (เฉลี่ย 54.2%) ควบคู่เสมอ ข้อสรุปเชิงคุณภาพ (การระบุของระบบแปรผันตามไฟข้ามแดนที่สถานี"
    "ชายแดน) มั่นคงกว่าตัวเลขขนาดเชิงปริมาณ ซึ่งคณะผู้พัฒนารายงานพร้อมความไม่แน่นอนนี้อย่างเปิดเผย",
)

h2(doc, "6.11 พยานอิสระเชิงทิศทาง: วิถีลมย้อนหลัง (Back-trajectory) ที่แม่ฮ่องสอน (ขยายผล §6.5)")
body(
    doc,
    "เพื่อตรวจสอบผลระบุแหล่งกำเนิดของโมเดล (§6.5, §6.10) ด้วยพยานที่ **ไม่ขึ้นกับโมเดล** คณะผู้พัฒนา"
    "คำนวณ **วิถีลมย้อนหลังเชิงจลนศาสตร์ (kinematic backward trajectory)** จากลมผิว 10 เมตร (u10, v10) "
    "ของ ERA5 ด้วยวิธี RK2 ย้อนหลัง 48 ชั่วโมงจากชั่วโมงที่ PM2.5 สูงสุดของแต่ละเหตุการณ์ที่สถานีแม่ฮ่องสอน "
    "(จุดปล่อยเดียวกับที่ใช้ทำ attribution) แล้วติดป้ายประเทศทุกจุดตามวิถีด้วย point-in-polygon กับขอบเขต"
    "ประเทศจริง และจับคู่กับจุดความร้อน FIRMS ในรัศมี 50 กม. รอบแต่ละจุด (“corridor FRP”) เป็น **พยานอิสระ"
    "เชิงทิศทางแบบไบนารี** (ต่างชาติ/ในประเทศ) — metric และเกณฑ์ทั้งหมดกำหนดไว้ **ก่อนเห็นผล** ไม่ปรับจูน"
    "ย้อนหลัง — แหล่ง `backtrajectory_test2025.json` (สร้างโดย `scripts/22_backtrajectory.py`)",
)
body(
    doc,
    "ทั้ง 5 เหตุการณ์ วิถีลม integrate ครบ 48/48 ชั่วโมงโดยไม่หลุดออกนอกโดเมน ERA5:",
)
table(
    doc,
    [
        "เหตุการณ์",
        "attribution ต่างชาติ (โมเดล)",
        "สัดส่วนชั่วโมงอยู่เหนือต่างชาติ (วิถีลม)",
        "corridor FRP ต่างชาติ : ไทย",
        "เห็นตรงกับโมเดล",
    ],
    [
        ["2025-03-18 (เหตุการณ์หลัก)", "0.627", "57.1%", "เมียนมา 12,918.4 : ไทย 106.4", "ตรงกัน"],
        ["2025-03-13", "0.403", "63.3%", "เมียนมา 52.5 : ไทย 36.6", "ตรงกัน"],
        ["2025-03-30", "0.005", "59.2%", "เมียนมา 0.0 : ไทย 5.2", "ไม่ตรง"],
        ["2025-03-11", "0.0", "24.5%", "เมียนมา 1.2 : ไทย 173.0", "ไม่ตรง"],
        ["2025-03-17", "0.0", "46.9%", "เมียนมา 8.8 : ไทย 25.7", "ไม่ตรง"],
    ],
)
body(
    doc,
    "รวม agreement ระหว่างพยานอิสระกับโมเดล = **2/5 (0.4)**",
)
body(
    doc,
    "**การอ่านผลอย่างซื่อสัตย์:** (1) ทั้ง 2 เหตุการณ์ที่โมเดลชี้ต่างชาติ (18 และ 13 มี.ค.) ได้พยานอิสระ"
    "ยืนยันครบทั้งทิศทางวิถีลมและ corridor FRP (2) อีก 3 เหตุการณ์ที่ “ไม่ตรง” ชี้ข้อจำกัดของการใช้ "
    "“ทิศลมอย่างเดียว” เป็นพยาน: วิถีลมผ่านเมียนมาในทุกเหตุการณ์เช่นเดียวกัน (24–59% ของชั่วโมง"
    "ย้อนหลังสำหรับ 3 เหตุการณ์นี้; แม่ฮ่องสอน"
    "ติดชายแดน อากาศย่อมพัดผ่านเมียนมาแทบทุกวันไม่ว่าจะมีไฟหรือไม่) แต่ corridor FRP ต่างชาติตามวิถีนั้น"
    "แทบเป็นศูนย์ (03-30: เมียนมา 0.0; 03-11: เมียนมา 1.2 เทียบไทย 173.0; 03-17: เมียนมา 8.8 เทียบไทย "
    "25.7) กล่าวคือ โมเดลซึ่งใช้ข้อมูลไฟจริงให้ค่าต่างชาติ ~0 ในวันที่ไม่มีไฟ active ตามวิถี ซึ่ง corridor "
    "FRP สนับสนุนการตัดสินของโมเดลมากกว่าจะขัดแย้ง (3) ข้อจำกัดของวิธีนี้: ใช้เฉพาะลมผิว 10 เมตร ไม่มีการ"
    "เคลื่อนที่แนวดิ่งหรือชั้นบรรยากาศบน และเป็นวิถีเดียวต่อเหตุการณ์ ไม่ใช่ ensemble — จึงเป็นหลักฐานเชิง"
    "บ่งชี้ ไม่เทียบเท่าแบบจำลอง HYSPLIT เต็มรูปแบบ และ “เห็นตรงกัน” ในที่นี้วัดเฉพาะทิศทางแบบไบนารี "
    "ไม่ยืนยันขนาดของ attribution",
)

# ── 7. ปัญหาและอุปสรรค ──────────────────────────────────────────────────────────
h1(doc, "7. ปัญหาและอุปสรรค")
for i, t in enumerate(
    [
        "**ข้อผิดพลาดการ denormalization ในการรายงานผล** — พบหลังส่งข้อเสนอโครงการ ทำให้ตัวเลข µg/m³ "
        "คลาดเคลื่อน ทีมแก้ไขโดยรวมศูนย์การคำนวณไว้ที่ `src/training/evaluation.py` และตรวจสอบซ้ำได้ (ดู §6.6)",
        "**ช่องว่างของข้อมูล PM2.5 ปี 2566** — ความครอบคลุมของข้อมูล Air4Thai ในปี 2566 ต่ำ (โดยเฉพาะ"
        "ช่วงฤดูเผา) ส่งผลให้ข้อมูลฝึกในรูปแบบ 3-way split เหลือประมาณ 15,110 ตัวอย่าง (ลดลง ~37%) ซึ่งอาจ"
        "กระทบความแม่นยำ",
        "**การออกแบบการประเมินครั้งแรกขาดชุดทดสอบ held-out ที่แท้จริง** — เดิมประเมินบนชุดที่ใช้เลือก"
        "โมเดล จึงปรับเป็นการแบ่ง 3 ส่วน (train/val/test) และเทรนโมเดลใหม่เพื่อให้ปี 2568 เป็น held-out จริง",
        "**ข้อมูลปี 2568 มีจำนวนจำกัดและกระจายตัว** — ทำให้ช่วงความเชื่อมั่นกว้างและอำนาจการทดสอบ (statistical "
        "power) จำกัด ซึ่งเป็นเหตุผลหนึ่งที่ความได้เปรียบ 48h ยังไม่ถึงนัยสำคัญ",
        "**ข้อจำกัดด้านสภาพแวดล้อมการเทรน** — บน Windows ต้องตั้ง `num_workers = 0` ทำให้การเทรนช้าลง "
        "และ ERA5 มีความล่าช้าทำให้ระบบเป็น hindcast ไม่ใช่เรียลไทม์",
    ],
    1,
):
    numbered(doc, i, t)

# ── 8. แนวทางการพัฒนา ───────────────────────────────────────────────────────────
h1(doc, "8. แนวทางการพัฒนาและประยุกต์ใช้ในขั้นต่อไป")
for i, t in enumerate(
    [
        "**เพิ่มสถานีตรวจวัด** ของกรมควบคุมมลพิษและเครือข่ายอื่นเมื่อได้รับข้อมูล เพื่อเพิ่มความหนาแน่นเชิงพื้นที่",
        "**ต่อ NWP forecast จริง** (เช่น ECMWF HRES/GFS) เข้า pipeline เพื่อให้พยากรณ์เรียลไทม์ได้ พร้อม"
        "ประเมินผลกระทบของคุณภาพ NWP",
        "**การเทรนแบบ residual target** เพื่อพยายามเอาชนะ persistence ในช่วงสั้นให้ชัดเจนขึ้น",
        "**เพิ่มจำนวน seed และสถานีชายแดนอื่น** (เช่น แม่สอด แม่สาย) เพื่อยืนยันความสามารถ transboundary "
        "ให้แน่นขึ้น",
        "**Uncertainty quantification สำหรับ attribution** เพื่อสื่อสารความไม่แน่นอนของสัดส่วนแหล่งกำเนิด"
        "อย่างเป็นระบบ",
        "**ขยายไปภูมิภาคอื่น/ประเทศเพื่อนบ้าน** และรองรับการลำเลียงฝุ่นจากแหล่งที่อยู่นอกกรอบพื้นที่ปัจจุบัน",
    ],
    1,
):
    numbered(doc, i, t)

# ── 9. ข้อสรุปและข้อเสนอแนะ ─────────────────────────────────────────────────────
h1(doc, "9. ข้อสรุปและข้อเสนอแนะ")
body(
    doc,
    "โครงงานนี้พัฒนาโครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลาแบบอธิบายได้ สำหรับพยากรณ์ PM2.5 และระบุ"
    "แหล่งกำเนิดใน 9 จังหวัดภาคเหนือ จากการประเมินอย่างเข้มงวด (held-out, multi-seed, ช่วงความเชื่อมั่น) "
    "พบว่า **ความได้เปรียบเชิงพยากรณ์เหนือ baseline persistence มีจำกัด** (เด่นเฉพาะ 48h ราว 2% และยัง"
    "ไม่มีนัยสำคัญ) และ **กลไกกราฟไม่ได้ยกระดับความแม่นยำเหนือความผันผวนจากการสุ่ม seed** — "
    "คณะผู้พัฒนารายงานผลเหล่านี้ตามจริงทั้งหมด",
)
body(
    doc,
    "**คุณค่าที่พิสูจน์ได้** ของระบบคือ **ความสามารถระบุแหล่งกำเนิดที่อธิบายได้** ซึ่งกราฟไฟ/ลมเป็นตัว"
    "เปิดทาง: ระบบระบุการลำเลียงข้ามแดน (เมียนมา) ที่สอดคล้องกับความใกล้ของไฟจริงบนข้อมูล held-out — "
    "สิ่งที่ persistence และโมเดลที่ไม่ใช้กราฟทำไม่ได้ ข้อเสนอแนะคือควรนำระบบไปใช้ในบทบาท **การเตือนภัย"
    "ล่วงหน้า 48 ชั่วโมงควบคู่กับการระบุแหล่งกำเนิด** มากกว่าการมุ่งแข่งขันด้านค่า RMSE และพึงสื่อสารผลการระบุ"
    "แหล่งกำเนิดพร้อมความไม่แน่นอนเสมอ ทั้งนี้ ความซื่อสัตย์เชิงระเบียบวิธี (held-out, CI, multi-seed, "
    "การตรวจพบข้อผิดพลาดด้วยตนเอง) คือจุดแข็งด้านธรรมาภิบาล AI ที่โครงงานนี้ยึดถือ",
)

# ── 10. เอกสารอ้างอิง ────────────────────────────────────────────────────────────
h1(doc, "10. เอกสารอ้างอิง (References)")
for i, t in enumerate(
    [
        "Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020). Connecting the Dots: "
        "Multivariate Time Series Forecasting with Graph Neural Networks. KDD 2020, 753–763. "
        "https://arxiv.org/abs/2005.11650",
        "Wang, S., Li, Y., Zhang, J., Meng, Q., Meng, L., & Gao, F. (2020). PM2.5-GNN: A Domain "
        "Knowledge Enhanced Graph Neural Network for PM2.5 Forecasting. ACM SIGSPATIAL 2020. "
        "https://arxiv.org/abs/2002.12898",
        "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph "
        "Attention Networks. ICLR 2018. https://arxiv.org/abs/1710.10903",
        "Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML "
        "2017, 3319–3328. https://arxiv.org/abs/1703.01365",
        "NASA FIRMS — Fire Information for Resource Management System (VIIRS/MODIS Active Fire Data). "
        "https://firms.modaps.eosdis.nasa.gov/",
        "Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal "
        "Meteorological Society, 146(730), 1999–2049. https://doi.org/10.24381/cds.adbb2d47",
        "Graph-based Integrated Gradients for source attribution (2025). https://arxiv.org/abs/2509.07648",
        "Runfola, D. et al. (2020). geoBoundaries: A global database of political administrative "
        "boundaries. PLoS ONE 15(4): e0231866. https://doi.org/10.1371/journal.pone.0231866 — "
        "ข้อมูลเส้นพรมแดนประเทศ (ไทย/เมียนมา/ลาว) จาก geoBoundaries (gbOpen, ADM0 simplified); "
        "ข้อมูลต้นทาง © ผู้ร่วมพัฒนา OpenStreetMap (ODbL 1.0 / CC BY-SA 2.0)",
        "Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). "
        "Distribution-Free Predictive Inference for Regression. Journal of the American Statistical "
        "Association, 113(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116",
        "Wilks, D. S. (2011). Statistical Methods in the Atmospheric Sciences (3rd ed.), sec. 8.4. "
        "Academic Press. — เกณฑ์ Brier score / Brier Skill Score สำหรับการพยากรณ์โอกาสเกินค่ามาตรฐาน "
        "(Brier, 1950)",
    ],
    1,
):
    ref_entry(doc, i, t)

# ── 11. สถานที่ติดต่อ ────────────────────────────────────────────────────────────
h1(doc, "11. สถานที่ติดต่อของผู้พัฒนาและอาจารย์ที่ปรึกษา")
for t in [
    "**ผู้พัฒนา:** นายรณชัย ขาวสะอาด — คณะวิทยาการสารสนเทศ สาขาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยี"
    "อัจฉริยะ มหาวิทยาลัยบูรพา วิทยาเขตบางแสน · อีเมล 67160366@go.buu.ac.th · โทร 0645607957",
    "**อาจารย์ที่ปรึกษา:** ดร.วัชรพงศ์ อยู่ขวัญ — คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา · อีเมล "
    "wyookwan@informatics.buu.ac.th · โทร 038103061",
    "**ที่อยู่สถาบัน:** 169 ถนนลงหาดบางแสน ตำบลแสนสุข อำเภอเมืองชลบุรี จังหวัดชลบุรี 20130",
    "**ที่เก็บซอร์สโค้ด:** https://github.com/67160366/pm25-stgnn-thailand",
]:
    bullet(doc, t)

# ── 12. ภาคผนวก ──────────────────────────────────────────────────────────────────
h1(doc, "12. ภาคผนวก (Appendix)")
body(
    doc,
    "คู่มือฉบับเต็มจัดทำเป็นเอกสารแยกและแนบในภาคผนวก: `docs/INSTALL.md` (คู่มือการติดตั้งอย่างละเอียด) "
    "และ `docs/USER_GUIDE.md` (คู่มือการใช้งานอย่างละเอียด) ด้านล่างเป็นสรุปย่อ",
)
h2(doc, "ก. คู่มือการติดตั้งโดยสรุป")
for i, t in enumerate(
    [
        "ความต้องการ: Python 3.11, uv, (ทางเลือก) NVIDIA GPU + CUDA 12.6; รองรับ Windows และ Linux/macOS",
        "ติดตั้ง dependencies หลัก: `uv sync`",
        "ติดตั้ง native PyG extensions (torch-scatter/sparse/geometric-temporal): บน Windows ใช้ "
        "`./install_native_deps.ps1`",
        "Windows + พาธอักษรไทย: ตั้ง `PYTHONUTF8=1` (สคริปต์ตั้งให้แล้ว ให้เปิด terminal ใหม่) หรือแก้ไฟล์ "
        "editable `.pth` ให้เป็น `../../..` แล้วรันด้วย `UV_NO_SYNC=1 uv run ...`",
        "API keys (เฉพาะกรณีดาวน์โหลดข้อมูลใหม่): `.env` (OPENAQ_API_KEY, FIRMS_API_KEY) และ `~/.cdsapirc` "
        "(ERA5) — dashboard และการประเมินผลใช้ข้อมูลที่เตรียมไว้แล้ว ไม่ต้องใช้ key",
        "ตรวจสอบการติดตั้ง: `uv run pytest tests/ -q` (213 tests)",
    ],
    1,
):
    numbered(doc, i, t)
h2(doc, "ข. คู่มือการใช้งานโดยสรุป")
for i, t in enumerate(
    [
        "เปิด dashboard: `UV_NO_SYNC=1 uv run streamlit run app/streamlit_app.py` แล้วเปิด "
        "http://localhost:8501 (ต้องมี `data/processed/` และ `checkpoints/mtgnn/`)",
        "dashboard มี 6 หน้า: ภาพรวม / พยากรณ์ / แหล่งกำเนิด / ข้ามแดน / ผลการทดสอบ / เกี่ยวกับโครงการ",
        "pipeline (ทำซ้ำผลได้): `scripts/01` ดาวน์โหลด → `02` เตรียมข้อมูล → `03` เทรน → `04` ประเมินผล "
        "→ `05`–`12` การทดสอบความเข้มงวด → `13` สร้างรูปรายงาน",
        "การทำซ้ำผลหลัก: `scripts/04_evaluate.py` (ความแม่นยำ) และ `scripts/10_transboundary_attr.py` "
        "(ข้ามแดน); ผลทั้งหมดเป็น JSON ใน `outputs/`",
    ],
    1,
):
    numbered(doc, i, t)
h2(doc, "ข้อตกลงในการใช้ซอฟต์แวร์ (Disclaimer)")
body(
    doc,
    "ซอฟต์แวร์นี้เป็นผลงานที่พัฒนาขึ้นโดย **นายรณชัย ขาวสะอาด** จาก **มหาวิทยาลัยบูรพา** ภายใต้การดูแล"
    "ของ **ดร.วัชรพงศ์ อยู่ขวัญ** ภายใต้โครงการ **“ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิด"
    "ด้วยโครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลาแบบอธิบายได้ สำหรับภาคเหนือของประเทศไทย”** ซึ่งสนับสนุน"
    "โดยสำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ โดยมีวัตถุประสงค์เพื่อส่งเสริมให้นักเรียนและ"
    "นักศึกษาได้เรียนรู้และฝึกทักษะในการพัฒนาซอฟต์แวร์ ลิขสิทธิ์ของซอฟต์แวร์นี้จึงเป็นของผู้พัฒนา ซึ่งผู้"
    "พัฒนาได้อนุญาตให้สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติเผยแพร่ซอฟต์แวร์นี้ตาม “ต้นฉบับ” "
    "โดยไม่มีการแก้ไขดัดแปลงใด ๆ ทั้งสิ้น ให้แก่บุคคลทั่วไปได้ใช้เพื่อประโยชน์ส่วนบุคคลหรือประโยชน์ทาง"
    "การศึกษาที่ไม่มีวัตถุประสงค์ในเชิงพาณิชย์ โดยไม่คิดค่าตอบแทนการใช้ซอฟต์แวร์ ดังนั้น สำนักงานพัฒนา"
    "วิทยาศาสตร์และเทคโนโลยีแห่งชาติจึงไม่มีหน้าที่ในการดูแล บำรุงรักษา จัดการอบรมการใช้งาน หรือพัฒนา"
    "ประสิทธิภาพซอฟต์แวร์ รวมทั้งไม่รับรองความถูกต้องหรือประสิทธิภาพการทำงานของซอฟต์แวร์ ตลอดจนไม่"
    "รับประกันความเสียหายต่าง ๆ อันเกิดจากการใช้ซอฟต์แวร์นี้ทั้งสิ้น",
)
body(
    doc,
    "**License Agreement.** This software is a work developed by **Mr. Ronnachai Khaosa-ard** from "
    "**Burapha University** under the provision of **Dr. Watcharapong Yookwan** under the project "
    "*“Explainable Spatio-Temporal Graph Neural Network for PM2.5 Forecasting and Source Attribution "
    "in Northern Thailand”*, which has been supported by the National Science and Technology "
    "Development Agency (NSTDA), in order to encourage pupils and students to learn and practice "
    "their skills in developing software. Therefore, the intellectual property of this software shall "
    "belong to the developer and the developer gives NSTDA a permission to distribute this software "
    "as an “as is” and non-modified software for a temporary and non-exclusive use without "
    "remuneration to anyone for his or her own purpose or academic purpose, which are not commercial "
    "purposes. In this connection, NSTDA shall not be responsible to the user for taking care, "
    "maintaining, training, or developing the efficiency of this software. Moreover, NSTDA shall not "
    "be liable for any error, software efficiency and damages in connection with or arising out of "
    "the use of the software.",
)

mark_fields_dirty(doc)
doc.save(str(OUTPUT))
print("Saved: outputs/NSC2026_Final_Report.docx")
