"""Generate NSC 2026 Category 14 proposal following the 512_1.pdf template structure."""

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_cs_font(run, name="TH Sarabun New"):
    r_pr = run._r.get_or_add_rPr()
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), name)
    r_fonts.set(qn("w:hAnsi"), name)
    r_fonts.set(qn("w:cs"), name)
    existing = r_pr.find(qn("w:rFonts"))
    if existing is not None:
        r_pr.remove(existing)
    r_pr.insert(0, r_fonts)


def fmt(run, size=16, bold=False, italic=False, font="TH Sarabun New"):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    _set_cs_font(run, font)


def para(doc, text="", size=16, bold=False, italic=False,
         align=WD_ALIGN_PARAGRAPH.LEFT, indent_cm=0.0, space_before=0, space_after=0):
    p = doc.add_paragraph()
    p.alignment = align
    if indent_cm:
        p.paragraph_format.left_indent = Cm(indent_cm)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        r = p.add_run(text)
        fmt(r, size=size, bold=bold, italic=italic)
    return p


def heading(doc, text, size=16, bold=True, space_before=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    fmt(r, size=size, bold=bold)
    return p


def body(doc, text, size=16, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent_cm=0.0):
    p = doc.add_paragraph()
    p.alignment = align
    if indent_cm:
        p.paragraph_format.left_indent = Cm(indent_cm)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    fmt(r, size=size, bold=bold)
    return p


def bullet(doc, text, size=16, indent_cm=1.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(indent_cm)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"• {text}")
    fmt(r, size=size)
    return p


def numbered(doc, number, text, size=16, indent_cm=0.75):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(indent_cm)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"{number}) {text}")
    fmt(r, size=size)
    return p


def person_block(doc, number, label, name="[ชื่อ นามสกุล]",
                 dob="[วว/ดด/ปปปป]", edu_level="ปริญญาตรี",
                 institution="[ชื่อมหาวิทยาลัย / สถาบัน]",
                 home_addr="[ที่อยู่ตามทะเบียนบ้าน]",
                 contact_addr="[สถานที่ติดต่อ]",
                 tel="[0X-XXXX-XXXX]", mobile="[0XX-XXX-XXXX]",
                 fax="-", email="[email@domain.com]"):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"{number}. ชื่อ-นามสกุล(นาย/นาง/น.ส.)  {name}")
    fmt(r, size=16)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(
        f"วัน/เดือน/ปีเกิด  {dob}    "
        f"ระดับการศึกษา  {edu_level}    "
        f"สถานศึกษา  {institution}"
    )
    fmt(r, size=16)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"ที่อยู่ตามทะเบียนบ้าน  {home_addr}")
    fmt(r, size=16)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"สถานที่ติดต่อ  {contact_addr}")
    fmt(r, size=16)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(
        f"โทรศัพท์  {tel}    มือถือ  {mobile}    โทรสาร  {fax}    E-mail  {email}"
    )
    fmt(r, size=16)

    # signature line
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(
        "ลงชื่อ  …………………………………………………………………….    "
        f"({label})"
    )
    fmt(r, size=16)


def signature_line(doc, role):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(f"ลงชื่อ  …………………………………………………………………….    ({role})")
    fmt(r, size=16)


def ref_entry(doc, number, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.first_line_indent = Cm(-1.0)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"{number}. {text}")
    fmt(r, size=16)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════

doc = Document()
for sec in doc.sections:
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.0)

# ══════════════════════════════════════════════════════════════════════════════
# COVER PAGE  (follows 512_1.pdf page 2 layout exactly)
# ══════════════════════════════════════════════════════════════════════════════

para(doc, "แบบฟอร์มข้อเสนอโครงการ", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "รหัสโครงการ  NSC [รหัสโครงการ]", size=16, align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "", size=16)

para(doc, "ข้อเสนอโครงการ", size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "การแข่งขันพัฒนาโปรแกรมคอมพิวเตอร์แห่งประเทศไทย ครั้งที่ 28", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "(National Software Contest 2026 — NSC 2026)", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "หมวด 14: โปรแกรมเพื่องานการพัฒนาด้านวิทยาศาสตร์และเทคโนโลยี", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

para(doc, "", size=14)

# Project title block
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("ชื่อโครงการ")
fmt(r, size=16, bold=True)
r2 = p.add_run(
    "     (ภาษาไทย)  ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วย"
    "โครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลา สำหรับภาคเหนือของประเทศไทย"
)
fmt(r2, size=16)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
r = p.add_run(
    "                      (ภาษาอังกฤษ)  Explainable Spatio-Temporal Graph Neural Network "
    "for PM2.5 Forecasting and Source Attribution in Northern Thailand"
)
fmt(r, size=16, italic=True)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(8)
r = p.add_run("ประเภทโปรแกรมที่เสนอ")
fmt(r, size=16, bold=True)
r2 = p.add_run("     ระบบพยากรณ์และวิเคราะห์ข้อมูลด้วย Graph Neural Network (Web Dashboard)")
fmt(r2, size=16)

para(doc, "", size=12)

# ── Team ──────────────────────────────────────────────────────────────────────

para(doc, "ทีมพัฒนา", size=16, bold=True)

para(doc, "หัวหน้าโครงการ", size=16, bold=True)
person_block(doc, number="1", label="หัวหน้าโครงการ")

para(doc, "ผู้ร่วมโครงการ", size=16, bold=True)
person_block(doc, number="2", label="ผู้ร่วมโครงการ")
person_block(doc, number="3", label="ผู้ร่วมโครงการ")

# ── Advisor ───────────────────────────────────────────────────────────────────

para(doc, "อาจารย์ที่ปรึกษาโครงการ", size=16, bold=True)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("ชื่อ-นามสกุล(นาย/นาง/น.ส.)  [ชื่อ นามสกุล อาจารย์]")
fmt(r, size=16)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("สังกัด/สถาบัน  [ชื่อคณะ / ภาควิชา]    มหาวิทยาลัย/สถาบัน  [ชื่อมหาวิทยาลัย]")
fmt(r, size=16)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("สถานที่ติดต่อ  [ที่อยู่สถาบัน]")
fmt(r, size=16)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
r = p.add_run("โทรศัพท์  [0X-XXXX-XXXX]    มือถือ  [0XX-XXX-XXXX]    โทรสาร  -    E-mail  [email@domain.com]")
fmt(r, size=16)

body(
    doc,
    "คำรับรอง \"โครงการนี้เป็นความคิดริเริ่มของนักพัฒนาโครงการและไม่ได้ลอกเลียนแบบมาจาก"
    "ผู้อื่นผู้ใด ข้าพเจ้าขอรับรองว่าจะให้คำแนะนำและสนับสนุนให้นักพัฒนาในความดูแลของ"
    "ข้าพเจ้าดำเนินการศึกษา/วิจัย/พัฒนาตามหัวข้อที่เสนอและจะทำหน้าที่ประเมินผลงาน"
    "ดังกล่าวให้กับโครงการฯ ด้วย\"",
    size=16,
)
signature_line(doc, "อาจารย์ที่ปรึกษา")

# ── Head of institution ───────────────────────────────────────────────────────

para(
    doc,
    "หัวหน้าสถาบัน (อธิการบดี/คณบดี/หัวหน้าภาควิชา/ผู้อำนวยการ/อาจารย์ใหญ่/หัวหน้าหมวด)",
    size=16,
    bold=True,
)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("ชื่อ-นามสกุล(นาย/นาง/น.ส.)  [ชื่อ นามสกุล]    ตำแหน่ง  [ตำแหน่งบริหาร]")
fmt(r, size=16)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("สถาบัน  [ชื่อมหาวิทยาลัย / สถาบัน]")
fmt(r, size=16)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("สถานที่ติดต่อ  [ที่อยู่สถาบัน]")
fmt(r, size=16)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
r = p.add_run("โทรศัพท์  [0X-XXXX-XXXX]    มือถือ  [0XX-XXX-XXXX]    โทรสาร  -    E-mail  [email@domain.com]")
fmt(r, size=16)

body(
    doc,
    "คำรับรอง \"ข้าพเจ้าขอรับรองว่าผู้พัฒนามีสิทธิ์ขอรับทุนสนับสนุนตามเงื่อนไขที่โครงการฯ"
    "กำหนดและอนุญาตให้ดำเนินการศึกษา/วิจัย/พัฒนาตามหัวข้อที่ได้เสนอมานี้ในสถาบันได้"
    "ภายใต้การบังคับบัญชาของข้าพเจ้า\"",
    size=16,
)
signature_line(doc, "หัวหน้าสถาบัน")

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 2. สาระสำคัญของโครงการ
# ══════════════════════════════════════════════════════════════════════════════

heading(doc, "สาระสำคัญของโครงการ", size=16, bold=True)
body(
    doc,
    (
        "โครงงานนี้พัฒนาระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดฝุ่น "
        "สำหรับ 9 จังหวัดภาคเหนือของประเทศไทย โดยใช้โครงข่ายกราฟประสาทเทียม "
        "เชิงปริภูมิ-เวลาแบบอธิบายได้ (Explainable Spatio-Temporal Graph Neural Network) "
        "ระบบพยากรณ์ค่า PM2.5 ล่วงหน้า 6, 12, 24 และ 48 ชั่วโมง จาก 18 สถานีตรวจวัด "
        "Air4Thai พร้อมกันในคราวเดียว โดยบูรณาการข้อมูลอุตุนิยมวิทยา ERA5 "
        "และข้อมูลจุดความร้อนจากดาวเทียม NASA FIRMS เป็น node พิเศษในกราฟ "
        "เพื่อจำลองการลำเลียงฝุ่นจากแหล่งเผาไหม้ทั้งภายในและข้ามพรมแดน\n\n"
        "จุดเด่นสำคัญของระบบคือโมดูล XAI (Graph-based Integrated Gradients) "
        "ที่สามารถระบุสัดส่วนผลกระทบจากแหล่งกำเนิดไฟแยกตามประเทศ (ไทย/เมียนมา/ลาว) "
        "ได้เป็นรายสถานีและรายเหตุการณ์ โมเดล MTGNN ที่พัฒนาขึ้นมีค่า RMSE = 10.21 µg/m³ "
        "ที่ขอบฟ้า 24 ชั่วโมง และ 14.32 µg/m³ ที่ 48 ชั่วโมง "
        "ดีกว่า persistence baseline ร้อยละ 5.1 และ 10.6 ตามลำดับ "
        "การวิเคราะห์เหตุการณ์หมอกควันมีนาคม 2024 ที่เชียงใหม่ (141–144 µg/m³) "
        "พบว่าเกิดจากการเผาไหม้ในประเทศไทยเป็นหลัก โดยค่า Fire Radiative Power (FRP) "
        "สูงกว่าแหล่งเมียนมาถึง 128 เท่า ระบบแสดงผลผ่าน Streamlit dashboard "
        "เพื่อสนับสนุนการตัดสินใจเชิงนโยบายสาธารณสุขและสิ่งแวดล้อม"
    ),
)

para(doc, "", size=12)

# คำสำคัญ
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
r = p.add_run("คำสำคัญ: ")
fmt(r, size=16, bold=True)
r2 = p.add_run(
    "PM2.5, โครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลา, การวิเคราะห์แหล่งกำเนิด, "
    "Integrated Gradients, ภาคเหนือประเทศไทย, FIRMS, ERA5"
)
fmt(r2, size=16)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(8)
r = p.add_run("Keywords: ")
fmt(r, size=16, bold=True)
r2 = p.add_run(
    "PM2.5, Spatio-Temporal GNN, Source Attribution, Integrated Gradients, "
    "Northern Thailand, FIRMS, ERA5"
)
fmt(r2, size=16, italic=True)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 3. หลักการและเหตุผล
# ══════════════════════════════════════════════════════════════════════════════

heading(doc, "หลักการและเหตุผล", size=16, bold=True)
body(
    doc,
    (
        "ภาคเหนือของประเทศไทยประสบปัญหาหมอกควัน PM2.5 เป็นประจำทุกปีในช่วงฤดูหมอกควัน "
        "(มกราคม–เมษายน) ในปี 2024 ค่า PM2.5 รายชั่วโมงที่เชียงใหม่พุ่งสูงถึง 141–144 µg/m³ "
        "ซึ่งสูงกว่าค่ามาตรฐาน 24 ชั่วโมงขององค์การอนามัยโลก (WHO) ที่ 15 µg/m³ ถึง 9–10 เท่า "
        "และเกินมาตรฐานประเทศไทยที่ 50 µg/m³ ถึง 3 เท่า ส่งผลกระทบรุนแรงต่อสุขภาพประชาชน "
        "ใน 9 จังหวัดภาคเหนือ ได้แก่ เชียงใหม่ เชียงราย ลำปาง ลำพูน แม่ฮ่องสอน น่าน พะเยา แพร่ และตาก"
    ),
)
body(
    doc,
    (
        "แหล่งกำเนิดฝุ่น PM2.5 หลักในภูมิภาคนี้ได้แก่ การเผาชีวมวล (ทั้งในไร่นาและไฟป่า) "
        "และการลำเลียงมลพิษข้ามพรมแดนจากเมียนมาและสาธารณรัฐประชาธิปไตยประชาชนลาว "
        "ซึ่งพัดพาโดยรูปแบบลมที่เปลี่ยนแปลงตลอดเวลา ทำให้การระบุสัดส่วนผลกระทบจาก"
        "แต่ละแหล่งกำเนิดมีความสำคัญอย่างยิ่งต่อการวางนโยบายป้องกัน"
    ),
)
body(
    doc,
    (
        "เครื่องมือติดตามคุณภาพอากาศที่มีอยู่ในปัจจุบัน เช่น Air4Thai และการแจ้งเตือน "
        "ของกรมควบคุมมลพิษ (PCD) เป็นการรายงานค่าปัจจุบันแบบ reactive "
        "ไม่มีระบบพยากรณ์ล่วงหน้าระยะสั้น และไม่สามารถระบุสัดส่วนผลกระทบจากแหล่งกำเนิด "
        "เฉพาะ (source attribution) ได้ ทำให้หน่วยงานด้านนโยบายขาดข้อมูลในการ "
        "ดำเนินมาตรการป้องกันอย่างตรงจุด"
    ),
)
body(
    doc,
    (
        "งานวิจัยล่าสุดแสดงให้เห็นว่าโครงข่ายกราฟประสาทเทียม (Graph Neural Networks) "
        "สามารถจับความสัมพันธ์เชิงปริภูมิ-เวลาได้อย่างมีประสิทธิภาพ และให้ผลการพยากรณ์ "
        "ที่ดีกว่า persistence และ Numerical Weather Prediction (NWP) "
        "ในขอบฟ้า 24 ชั่วโมงขึ้นไป (Wang et al., 2020; Wu et al., 2020) "
        "แต่งานส่วนใหญ่ยังไม่ได้บูรณาการข้อมูลจุดความร้อนเป็นส่วนประกอบกราฟโดยตรง "
        "และขาดความสามารถด้าน source attribution โครงงานนี้จึงพัฒนาระบบที่ตอบโจทย์ "
        "ทั้งสองความต้องการพร้อมกัน"
    ),
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 4. วัตถุประสงค์
# ══════════════════════════════════════════════════════════════════════════════

heading(doc, "วัตถุประสงค์", size=16, bold=True)
objectives = [
    "พัฒนาระบบพยากรณ์ PM2.5 รายชั่วโมงล่วงหน้า 6, 12, 24 และ 48 ชั่วโมง "
    "สำหรับ 18 สถานีตรวจวัด Air4Thai ใน 9 จังหวัดภาคเหนือ",

    "บูรณาการข้อมูลจุดความร้อนจากดาวเทียม NASA FIRMS (VIIRS/MODIS) "
    "เป็น node พิเศษในกราฟ เพื่อจำลองการลำเลียงฝุ่นจากแหล่งเผาไหม้สู่สถานีตรวจวัด",

    "พัฒนาโมดูล XAI (Graph-based Integrated Gradients) "
    "เพื่อระบุสัดส่วนผลกระทบเชิงปริมาณจากแหล่งกำเนิดไฟแยกตามประเทศ (ไทย/เมียนมา/ลาว)",

    "ตรวจสอบความถูกต้องของระบบบนชุดข้อมูล validation ปี 2025 "
    "และเปรียบเทียบกับ persistence baseline และโมเดล A3TGCN",

    "พัฒนา dashboard แบบ interactive สำหรับแสดงผลการพยากรณ์และการวิเคราะห์แหล่งกำเนิด "
    "เพื่อสนับสนุนการตัดสินใจของหน่วยงานที่เกี่ยวข้อง",
]
for i, obj in enumerate(objectives, 1):
    numbered(doc, i, obj)

para(doc, "", size=12)

# ══════════════════════════════════════════════════════════════════════════════
# 5. ปัญหาหรือประโยชน์ที่เป็นเหตุผลให้ควรพัฒนาโปรแกรม
# ══════════════════════════════════════════════════════════════════════════════

heading(doc, "ปัญหาหรือประโยชน์ที่เป็นเหตุผลให้ควรพัฒนาโปรแกรม", size=16, bold=True)
problems = [
    "ไม่มีระบบพยากรณ์ PM2.5 ล่วงหน้าในระดับภูมิภาค: เครื่องมือที่มีอยู่รายงานค่าปัจจุบัน "
    "เท่านั้น ไม่มีความสามารถในการแจ้งเตือนล่วงหน้าเพื่อให้ประชาชนเตรียมตัวรับมือ",

    "ขาดระบบระบุแหล่งกำเนิด (source attribution): ผู้กำหนดนโยบายไม่มีข้อมูลว่าฝุ่น "
    "มาจากการเผาในประเทศหรือข้ามพรมแดน ทำให้ไม่สามารถออกมาตรการได้ตรงจุด",

    "การพัฒนาองค์ความรู้ด้าน GNN ประยุกต์: โครงงานนี้นำหลักการ Graph Neural Networks "
    "มาประยุกต์ใช้กับปัญหาสิ่งแวดล้อมจริงในภูมิภาคเอเชียตะวันออกเฉียงใต้",

    "ความสำเร็จเชิงวิชาการที่พิสูจน์แล้ว: MTGNN มีค่า RMSE ดีกว่า persistence "
    "ที่ขอบฟ้า 24h (+5.1%) และ 48h (+10.6%) บน validation set ปี 2025 "
    "และดีกว่า A3TGCN ร้อยละ 7.1 ที่ 24h",

    "ประโยชน์ต่อสาธารณะ: dashboard แบบ real-time สำหรับเจ้าหน้าที่สาธารณสุข "
    "นักวิจัย และประชาชนทั่วไปในภาคเหนือ",
]
for i, p_text in enumerate(problems, 1):
    numbered(doc, i, p_text)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 6. เป้าหมายและขอบเขตของโครงการ
# ══════════════════════════════════════════════════════════════════════════════

heading(doc, "เป้าหมายและขอบเขตของโครงการ", size=16, bold=True)
body(doc, "เป้าหมาย:")
goals = [
    "ระบบพยากรณ์ PM2.5 ที่ทำงานได้จริงบน validation set ปี 2025 "
    "พร้อม dashboard สำหรับผู้ใช้งาน",
    "โมดูล XAI ที่วิเคราะห์แหล่งกำเนิดและแสดงผลแบบ interactive ได้",
    "รายงานการตรวจสอบความถูกต้อง (evaluation report) บน 18 สถานี 4 ขอบฟ้า",
]
for i, g in enumerate(goals, 1):
    numbered(doc, i, g)

para(doc, "", size=12)
body(doc, "ขอบเขต:")
scopes = [
    "พื้นที่: 9 จังหวัดภาคเหนือ — เชียงใหม่ เชียงราย ลำปาง ลำพูน แม่ฮ่องสอน น่าน พะเยา แพร่ ตาก "
    "(กรอบพื้นที่ lon 97.0–101.5°E, lat 16.0–21.0°N)",
    "สถานีตรวจวัด: 18 สถานี Air4Thai (ระดับมหาวิทยาลัย) ที่มีข้อมูลครบ 2022–2025",
    "ขอบฟ้าพยากรณ์: 6h, 12h, 24h, 48h ความละเอียดรายชั่วโมง",
    "ข้อมูลฝึกโมเดล: 2022–2024 (3 ฤดูกาล), ข้อมูล validation: 2025",
    "แหล่งข้อมูล: Air4Thai (PM2.5), ERA5 (u10, v10, t2m, d2m, blh), NASA FIRMS (FRP)",
]
for i, s in enumerate(scopes, 1):
    numbered(doc, i, s)

para(doc, "", size=12)
body(doc, "ข้อจำกัด:")
limits = [
    "ข้อมูล PM2.5 มาจาก Air4Thai เท่านั้น (ไม่รวมสถานี PCD ในช่วง 2022–2024)",
    "ERA5 มีความล่าช้า ~6 ชั่วโมงในโหมด near-real-time",
    "โมเดลยังไม่รองรับการลำเลียงฝุ่นจากจีนและอินเดีย",
    "Attribution อยู่ในระดับประเทศ ไม่ใช่ระดับพื้นที่ย่อย",
]
for i, lim in enumerate(limits, 1):
    numbered(doc, i, lim)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 7. รายละเอียดของการพัฒนา
# ══════════════════════════════════════════════════════════════════════════════

heading(doc, "รายละเอียดของการพัฒนา", size=16, bold=True)

# ── 7.1 Storyboard ───────────────────────────────────────────────────────────

heading(doc, "7.1 เนื้อเรื่องย่อ (Storyboard) และกระบวนการทำงานของระบบ", size=16, bold=True)

steps = [
    ("ขั้นที่ 1: การรวบรวมข้อมูล (Data Collection)",
     "ระบบดาวน์โหลดข้อมูล PM2.5 รายชั่วโมงจาก Air4Thai 18 สถานี, "
     "ข้อมูลจุดความร้อน VIIRS/MODIS รายวันจาก NASA FIRMS "
     "ภายในกรอบพื้นที่ภาคเหนือ (lon 97.0–101.5°E, lat 16.0–21.0°N) "
     "และข้อมูลอุตุนิยมวิทยา ERA5 (u10, v10, t2m, d2m, blh) ทุก 6 ชั่วโมง"),

    ("ขั้นที่ 2: การสร้างกราฟพลวัต (Dynamic Graph Construction)",
     "ระบบสร้างกราฟที่มี 18 station nodes และ M hotspot cluster nodes "
     "เชื่อมด้วย edge 3 ประเภท: "
     "(a) geo — k-NN ระยะทางทางภูมิศาสตร์ (k=5, คงที่), "
     "(b) wind — ปรับตามทิศทางลมจาก ERA5 ทุกชั่วโมง, "
     "(c) bipartite — เชื่อม hotspot node กับสถานีที่อยู่ปลายลมในรัศมี 300 กม."),

    ("ขั้นที่ 3: การพยากรณ์ (PM2.5 Forecasting)",
     "โมเดล MTGNN รับ input ข้อมูลย้อนหลัง 24 ชั่วโมง (18 สถานี × 10 features) "
     "และพยากรณ์ค่า PM2.5 ที่ 18 สถานีพร้อมกันในขอบฟ้า 6h, 12h, 24h, 48h"),

    ("ขั้นที่ 4: การวิเคราะห์แหล่งกำเนิด (Source Attribution)",
     "โมดูล GB-IG คำนวณสัดส่วนผลกระทบ (%) ของแต่ละ hotspot node แยกตามประเทศ "
     "เช่น เหตุการณ์มีนาคม 2024 ที่เชียงใหม่: ไฟในไทย 100%, เมียนมา 0%"),

    ("ขั้นที่ 5: การแสดงผล (Dashboard Visualization)",
     "เจ้าหน้าที่สาธารณสุขหรือนักวิจัยเปิด Streamlit dashboard "
     "เพื่อดูแผนที่พยากรณ์ PM2.5 แบบ interactive, "
     "กราฟ source attribution แยกตามประเทศ และ feature importance ranking"),
]

for title, desc in steps:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    fmt(r, size=16, bold=True)
    body(doc, desc, indent_cm=0.75)

para(doc, "", size=12)

# Architecture diagram — insert real PNG
body(doc, "แผนภาพสถาปัตยกรรมระบบ:", bold=False)
DIAGRAM_PNG = r"D:\งาน\NSC\pm25-stgnn-thailand\architecture_diagram.png"
p_img = doc.add_paragraph()
p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_img.paragraph_format.space_after = Pt(6)
run_img = p_img.add_run()
run_img.add_picture(DIAGRAM_PNG, width=Cm(14.5))

doc.add_page_break()

# ── 7.2 เทคนิคและเทคโนโลยี ───────────────────────────────────────────────────

heading(doc, "7.2 เทคนิคหรือเทคโนโลยีที่ใช้", size=16, bold=True)

body(doc, "1) โมเดล MTGNN (Multivariate Time Series Graph Neural Network)", bold=False)
body(
    doc,
    "MTGNN (Wu et al., KDD 2020) เป็นโมเดลพยากรณ์ time series หลายตัวแปรบนกราฟ "
    "ประกอบด้วย 3 ส่วนหลัก:",
    indent_cm=0.75,
)
bullet(
    doc,
    "Graph Learning Layer: เรียนรู้ adaptive adjacency matrix จาก node embeddings "
    "เพื่อค้นพบความสัมพันธ์ที่ซ่อนอยู่ระหว่างสถานี",
    indent_cm=1.5,
)
bullet(
    doc,
    "Temporal Convolution Network (TCN): จับรูปแบบเชิงเวลาด้วย dilated causal convolutions",
    indent_cm=1.5,
)
bullet(
    doc,
    "Graph Convolution Module (GCN): แพร่กระจายข้อมูลระหว่าง node ตาม edge weights "
    "จากทั้ง adaptive adjacency และ 3 edge types ที่สร้างจากข้อมูลจริง",
    indent_cm=1.5,
)

para(doc, "", size=12)
body(doc, "2) Wind-aware Dynamic Graph Adjacency", bold=False)
body(
    doc,
    "น้ำหนักเส้นเชื่อมระหว่างสถานีปรับปรุงทุกชั่วโมงตามทิศทางลม ERA5 (u10, v10) "
    "ด้วยสูตร A_wind[i,j] = max(0, cos(θ_wind, θ_ij)) × exp(−d(i,j)/λ) "
    "ทำให้กราฟจับทิศทางการลำเลียงมลพิษได้แบบ real-time",
    indent_cm=0.75,
)

para(doc, "", size=12)
body(doc, "3) FIRMS Fire Hotspot Clusters as Graph Nodes", bold=False)
body(
    doc,
    "ข้อมูลไฟ VIIRS/MODIS จาก NASA FIRMS จัดกลุ่มเชิงพื้นที่เป็น hotspot clusters "
    "และเพิ่มเป็น node พิเศษในกราฟ แต่ละ node เก็บ FRP รวม และพิกัดศูนย์กลาง "
    "เชื่อมกับสถานีที่อยู่ปลายลมในรัศมี 300 กม. ผ่าน bipartite edge (type_c)",
    indent_cm=0.75,
)

para(doc, "", size=12)
body(doc, "4) Graph-based Integrated Gradients (GB-IG) สำหรับ Source Attribution", bold=False)
body(
    doc,
    "วิธี XAI แบบ post-hoc คำนวณ attribution ด้วยสูตร: "
    "IG(x) = (x − b) × ∫₀¹ (∂F/∂x)|_{b+α(x−b)} dα "
    "โดย F = MTGNN, b = zero baseline, 50 interpolation steps "
    "รับประกัน completeness axiom ว่าผลรวม attribution = output จริง − output จาก baseline",
    indent_cm=0.75,
)

doc.add_page_break()

# ── 7.3 เครื่องมือ ────────────────────────────────────────────────────────────

heading(doc, "7.3 เครื่องมือที่ใช้ในการพัฒนา", size=16, bold=True)

tools = [
    ("ภาษาโปรแกรม", "Python 3.11 (via uv package manager)"),
    ("GNN Framework", "PyTorch 2.2, PyG 2.5 (PyTorch Geometric), PyG Temporal"),
    ("โมเดลหลัก", "MTGNN implementation (Wu et al., KDD 2020)"),
    ("XAI Module", "Graph-based Integrated Gradients (GB-IG) — custom implementation"),
    ("การจัดการ Config", "Hydra framework"),
    ("Experiment Tracking", "Weights & Biases (WandB)"),
    ("Dashboard", "Streamlit"),
    ("Code Quality", "pytest (testing), ruff (linting), black (formatting)"),
]
for i, (comp, tech) in enumerate(tools, 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Cm(0.75)
    r = p.add_run(f"{i}) {comp}: ")
    fmt(r, size=16, bold=True)
    r2 = p.add_run(tech)
    fmt(r2, size=16)

para(doc, "", size=12)

# ── 7.4 Software Specification ───────────────────────────────────────────────

heading(doc, "7.4 รายละเอียดโปรแกรมที่จะพัฒนา (Software Specification)", size=16, bold=True)

# Input/Output
body(doc, "Input/Output Specification", bold=True)
body(doc, "Input:", indent_cm=0.75)
inputs = [
    "ค่า PM2.5 ย้อนหลัง 24 ชั่วโมงจาก 18 สถานี (pm25_scaled via RobustScaler per station)",
    "ข้อมูลอุตุนิยมวิทยา ERA5: u10, v10, t2m, d2m, blh (รายชั่วโมง)",
    "กลุ่มจุดความร้อน NASA FIRMS: FRP รวม, lat/lon ศูนย์กลาง, ประเทศต้นทาง (รายวัน)",
    "การเข้ารหัสเวลา: hour_sin, hour_cos, doy_sin, doy_cos",
]
for item in inputs:
    bullet(doc, item, indent_cm=1.5)

body(doc, "Output:", indent_cm=0.75)
outputs = [
    "ค่า PM2.5 พยากรณ์สำหรับ 18 สถานี × 4 ขอบฟ้า (6h, 12h, 24h, 48h)",
    "แผนที่ source attribution: สัดส่วน (%) แยกตามประเทศ (ไทย/เมียนมา/ลาว) ต่อการพยากรณ์ 24h",
    "Feature importance ranking (GB-IG attribution แยกตามประเภท feature)",
]
for item in outputs:
    bullet(doc, item, indent_cm=1.5)

para(doc, "", size=12)

# Functional Specification
body(doc, "Functional Specification", bold=True)
funcs = [
    "ดาวน์โหลดและประมวลผลข้อมูลจาก 3 แหล่ง (Air4Thai, FIRMS, ERA5) แบบอัตโนมัติ",
    "สร้างกราฟพลวัตที่ปรับปรุง edge weights ตามทิศทางลมทุกชั่วโมง",
    "พยากรณ์ PM2.5 ที่ 18 สถานีพร้อมกันใน 4 ขอบฟ้าด้วยโมเดล MTGNN",
    "คำนวณ source attribution แยกตามประเทศด้วย GB-IG",
    "แสดงผลผ่าน Streamlit dashboard พร้อมแผนที่ interactive และกราฟ time series",
    "รายงานผลการพยากรณ์และ attribution แบบ exportable (JSON/CSV)",
]
for i, f in enumerate(funcs, 1):
    numbered(doc, i, f, indent_cm=0.75)

para(doc, "", size=12)

# Design
body(doc, "โครงสร้างของซอฟต์แวร์ (Design)", bold=True)
body(doc, "โครงสร้างโค้ดแบ่งตามหน้าที่ดังนี้:", indent_cm=0.75)

tree = (
    "pm25-stgnn-thailand/\n"
    "│\n"
    "├── src/                         โค้ดหลักของระบบ\n"
    "│   ├── data/\n"
    "│   │   ├── scrapers/            air4thai.py · firms.py · era5.py\n"
    "│   │   ├── preprocessing.py     normalize, gap-fill\n"
    "│   │   ├── graph_builder.py     wind-aware dynamic graph (3 edge types)\n"
    "│   │   └── loader.py            PyG dataset builder\n"
    "│   │\n"
    "│   ├── models/\n"
    "│   │   ├── mtgnn.py             MTGNN — โมเดลหลัก (Wu et al. 2020)\n"
    "│   │   └── a3tgcn.py            A3TGCN — baseline\n"
    "│   │\n"
    "│   ├── explain/\n"
    "│   │   ├── gb_ig.py             Graph-based Integrated Gradients\n"
    "│   │   └── attribution.py       source attribution per country\n"
    "│   │\n"
    "│   └── training/\n"
    "│       ├── trainer.py           training loop + early stopping\n"
    "│       ├── losses.py            MAE + RMSE\n"
    "│       └── metrics.py           denormalized evaluation\n"
    "│\n"
    "├── app/\n"
    "│   └── streamlit_app.py         interactive dashboard\n"
    "│\n"
    "├── configs/                     Hydra YAML (model / data / training)\n"
    "├── scripts/                     01_download → 03_train → 04_evaluate → 05_explain\n"
    "└── outputs/                     evaluation_val2025.json · attribution_march2024.json"
)
p_tree = doc.add_paragraph()
p_tree.alignment = WD_ALIGN_PARAGRAPH.LEFT
p_tree.paragraph_format.left_indent = Cm(0.75)
p_tree.paragraph_format.space_after = Pt(6)
r_tree = p_tree.add_run(tree)
fmt(r_tree, size=10, font="Courier New")

doc.add_page_break()

# ── 7.5 ขอบเขตและข้อจำกัด (ในส่วนพัฒนา) ────────────────────────────────────

heading(doc, "7.5 ขอบเขตและข้อจำกัดของโปรแกรมที่พัฒนา", size=16, bold=True)
body(
    doc,
    "โปรแกรมนี้พัฒนาสำหรับระบบปฏิบัติการ Linux และ Windows โดยต้องการ Python 3.11 "
    "และ GPU (NVIDIA) สำหรับการฝึกโมเดล การพยากรณ์แบบ inference สามารถรันบน CPU ได้ "
    "dashboard ใช้งานผ่าน web browser (Streamlit) ไม่ต้องติดตั้งซอฟต์แวร์พิเศษ",
)
limits_prog = [
    "โปรแกรมทำงานกับพื้นที่ภาคเหนือ 9 จังหวัดเท่านั้น (bounding box ที่กำหนดไว้)",
    "ต้องการข้อมูล PM2.5 อย่างน้อย 24 ชั่วโมงต่อเนื่องก่อนจะพยากรณ์ได้",
    "ERA5 มีความล่าช้าประมาณ 5–7 วันในโหมด reanalysis; "
    "สำหรับการพยากรณ์แบบ real-time ต้องใช้ ERA5 forecast ที่มีข้อมูลล่าช้าน้อยกว่า",
    "ความแม่นยำของ source attribution ขึ้นกับคุณภาพและความครอบคลุมของข้อมูล FIRMS",
    "ไม่รองรับการลำเลียงฝุ่นจากจีนและอินเดีย (อยู่นอกกรอบพื้นที่)",
]
for i, lim in enumerate(limits_prog, 1):
    numbered(doc, i, lim)

para(doc, "", size=12)

# ── ผลการทดสอบ ───────────────────────────────────────────────────────────────

heading(doc, "ผลการทดสอบและประเมินผล", size=16, bold=True)
body(
    doc,
    "ทดสอบบน validation set ปี 2025 (2025-01-01 ถึง 2025-12-31) "
    "จาก 18 สถานี รวม 3,637 samples:",
)

# Results table
t = doc.add_table(rows=5, cols=5)
t.style = "Table Grid"
headers = ["ขอบฟ้า", "Persistence RMSE", "MTGNN RMSE", "ปรับปรุง (%)", "A3TGCN RMSE"]
rows_data = [
    ["6 ชั่วโมง", "3.79 µg/m³", "4.92 µg/m³", "−29.8%", "7.01 µg/m³"],
    ["12 ชั่วโมง", "6.51 µg/m³", "6.87 µg/m³", "−5.5%", "8.26 µg/m³"],
    ["24 ชั่วโมง ★", "10.76 µg/m³", "10.21 µg/m³", "+5.1% ✓", "10.79 µg/m³"],
    ["48 ชั่วโมง ★", "16.02 µg/m³", "14.32 µg/m³", "+10.6% ✓", "14.76 µg/m³"],
]
for j, h in enumerate(headers):
    p_cell = t.rows[0].cells[j].paragraphs[0]
    r = p_cell.add_run(h)
    fmt(r, size=14, bold=True)
    p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
for i, row_data in enumerate(rows_data, 1):
    for j, val in enumerate(row_data):
        p_cell = t.rows[i].cells[j].paragraphs[0]
        r = p_cell.add_run(val)
        fmt(r, size=14)

para(doc, "", size=12)
body(
    doc,
    "ผล XAI — เหตุการณ์มีนาคม 2024 ที่เชียงใหม่ (141–144 µg/m³): "
    "Attribution ไทย 100%, เมียนมา 0% (FRP อัตราส่วน 128:1) "
    "Fire hotspot channel เพิ่มค่า PM2.5 เฉลี่ย 4.46 µg/m³ เหนือ baseline ที่ไม่มีไฟ",
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 8. บรรณานุกรม
# ══════════════════════════════════════════════════════════════════════════════

heading(doc, "บรรณานุกรม (Bibliography)", size=16, bold=True)

references = [
    (
        "Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020). "
        "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks. "
        "Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery "
        "& Data Mining (KDD 2020), 753–763. https://arxiv.org/abs/2005.11650"
    ),
    (
        "Wang, S., Li, Y., Zhang, J., Meng, Q., Meng, L., & Gao, F. (2020). "
        "PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting. "
        "Proceedings of the 28th ACM SIGSPATIAL International Conference on Advances "
        "in Geographic Information Systems (SIGSPATIAL 2020). "
        "https://arxiv.org/abs/2002.12898"
    ),
    (
        "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). "
        "Graph Attention Networks. "
        "Proceedings of the International Conference on Learning Representations (ICLR 2018). "
        "https://arxiv.org/abs/1710.10903"
    ),
    (
        "Sundararajan, M., Taly, A., & Yan, Q. (2017). "
        "Axiomatic Attribution for Deep Networks. "
        "Proceedings of the 34th International Conference on Machine Learning "
        "(ICML 2017), 70, 3319–3328. https://arxiv.org/abs/1703.01365"
    ),
    (
        "NASA FIRMS — Fire Information for Resource Management System. (2024). "
        "Active Fire Data (VIIRS/MODIS). "
        "National Aeronautics and Space Administration. "
        "https://firms.modaps.eosdis.nasa.gov/"
    ),
    (
        "Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., "
        "Muñoz-Sabater, J., ... & Thépaut, J.-N. (2020). "
        "The ERA5 global reanalysis. "
        "Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049. "
        "Data: https://doi.org/10.24381/cds.adbb2d47"
    ),
]

for i, ref in enumerate(references, 1):
    ref_entry(doc, i, ref)

# ── Save ──────────────────────────────────────────────────────────────────────

OUTPUT = r"D:\งาน\NSC\pm25-stgnn-thailand\NSC2026_PM25_STGNN_Proposal.docx"
doc.save(OUTPUT)
print(f"Saved: {OUTPUT}")
