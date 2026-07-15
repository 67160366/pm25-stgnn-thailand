# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Generate the system-architecture figure (รูปที่ 1) for the NSC 2026 report.

Design notes (v2, regional-round revision):
- Light tinted fills + dark text (soft, print-friendly) instead of saturated
  fills + white text; fewer/larger boxes so the figure stays legible when
  embedded at ~15 cm width in the Word report.
- NO performance numbers in this figure — results live in report §6 only
  (the v1 figure carried superseded pre-Session-8 numbers by accident).
- Edge-type parameters mirror src/data/graph_builder.py _DEFAULT_CONFIG
  (type_a ≤100 km exp(-d/50); type_b ≤200 km align>0.3 hourly;
  type_c ≤500 km align>0.4 downwind), DBSCAN eps 25 km from
  src/data/hotspot_clustering.py, ERA5 hourly + bilinear from scrapers/era5.py.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Thai-capable font available on Windows
plt.rcParams["font.family"] = "Leelawadee UI"
plt.rcParams["axes.unicode_minus"] = False

INK = "#1a1a19"  # primary text
INK2 = "#52514e"  # secondary text
ARROW = "#8a8880"  # subdued connectors
BG = "#ffffff"

# fill / border per band (light tint + mid-tone border, dark text everywhere)
BLUE = ("#e3eefc", "#2a78d6")  # [1] data sources
PURPLE = ("#eceafb", "#4a3aa7")  # [2] preprocessing
AMBER = ("#fdf3dc", "#c98500")  # [3] graph builder container
ORANGE = ("#fce7da", "#d95926")  # hotspot nodes
RED = ("#fbe3e3", "#c73e3d")  # [4] MTGNN
GREEN = ("#dcf3ea", "#199e70")  # [5] forecast output
PINK = ("#f9e3ec", "#c2416f")  # [5] XAI output
GRAY = ("#efedea", "#52514e")  # [6] dashboard

EDGE_A = "#52514e"
EDGE_B = "#d95926"
EDGE_C = "#199e70"

FS_TITLE = 13.0
FS_CHIP = 8.6
FS_BOX = 10.0
FS_SUB = 8.2
FS_NOTE = 7.8

fig, ax = plt.subplots(figsize=(7.0, 8.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.15)
ax.axis("off")
fig.patch.set_facecolor(BG)


def box(x, y, w, h, colors, title, sub=None, title_fs=FS_BOX, sub_fs=FS_SUB, dashed=False):
    fill, border = colors
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=fill,
            edgecolor=border,
            linewidth=1.1,
            linestyle=(0, (4, 2)) if dashed else "solid",
            zorder=3,
        )
    )
    if title:
        cy = y + h / 2 + (0.155 if sub else 0)
        ax.text(
            x + w / 2,
            cy,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            color=INK,
            fontweight="bold",
            zorder=4,
        )
    if sub:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.175,
            sub,
            ha="center",
            va="center",
            fontsize=sub_fs,
            color=INK2,
            zorder=4,
        )


def chip(x, y, text, border):
    ax.text(
        x,
        y,
        text,
        fontsize=FS_CHIP,
        color=border,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=5,
    )


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            color=ARROW,
            lw=1.3,
            zorder=2,
            shrinkA=0,
            shrinkB=1,
        )
    )


# ── Title ────────────────────────────────────────────────────────────────────
ax.text(
    5.0,
    11.93,
    "สถาปัตยกรรมระบบ Explainable STGNN",
    ha="center",
    va="center",
    fontsize=FS_TITLE,
    fontweight="bold",
    color=INK,
)
ax.text(
    5.0,
    11.60,
    "พยากรณ์ PM2.5 และวิเคราะห์แหล่งกำเนิด · 9 จังหวัดภาคเหนือ",
    ha="center",
    va="center",
    fontsize=9.0,
    color=INK2,
)

COLS = [(0.25, 3.05), (3.48, 3.05), (6.71, 3.05)]  # (x, w) — 3 aligned columns

# ── [1] Data sources ─────────────────────────────────────────────────────────
chip(0.25, 11.18, "[1] แหล่งข้อมูล (DATA SOURCES)", BLUE[1])
box(COLS[0][0], 10.28, COLS[0][1], 0.72, BLUE, "Air4Thai", "PM2.5 · 18 สถานี · รายชั่วโมง")
box(COLS[1][0], 10.28, COLS[1][1], 0.72, BLUE, "NASA FIRMS", "จุดความร้อน VIIRS/MODIS · FRP")
box(COLS[2][0], 10.28, COLS[2][1], 0.72, BLUE, "ERA5 (CDS API)", "u10 v10 t2m d2m blh · รายชั่วโมง")

for cx, cw in COLS:
    arrow(cx + cw / 2, 10.28, cx + cw / 2, 9.92)

# ── [2] Preprocessing ────────────────────────────────────────────────────────
chip(0.25, 9.72, "[2] การเตรียมข้อมูล (PREPROCESSING)", PURPLE[1])
box(
    COLS[0][0],
    8.82,
    COLS[0][1],
    0.72,
    PURPLE,
    "เตรียมข้อมูลสถานี",
    "RobustScaler ต่อสถานี · เติมช่องว่าง",
)
box(
    COLS[1][0],
    8.82,
    COLS[1][1],
    0.72,
    PURPLE,
    "จัดกลุ่มจุดความร้อน",
    "DBSCAN (eps 25 กม.) · รวม FRP",
)
box(
    COLS[2][0],
    8.82,
    COLS[2][1],
    0.72,
    PURPLE,
    "ประมาณค่า ERA5",
    "bilinear สู่พิกัดสถานี · รายชั่วโมง",
)

for cx, cw in COLS:
    arrow(cx + cw / 2, 8.82, cx + cw / 2, 8.46)

# ── [3] Dynamic graph builder ────────────────────────────────────────────────
chip(0.25, 8.26, "[3] ตัวสร้างกราฟพลวัต (DYNAMIC GRAPH BUILDER) — G(V, E, t)", AMBER[1])
box(0.25, 5.62, 9.51, 2.42, AMBER, None, dashed=True)

box(0.55, 7.06, 4.45, 0.74, AMBER, "โหนดสถานี 18 โหนด", "PM2.5 + ERA5 + เวลา (10 features)")
box(
    5.31,
    7.06,
    4.15,
    0.74,
    ORANGE,
    "โหนดจุดความร้อน M โหนด",
    "FRP · พิกัดศูนย์กลาง · ประเทศ TH/MM/LA",
)

edge_rows = [
    (EDGE_A, "type_a — เชิงพื้นที่ (คงที่):", "ระยะ ≤ 100 กม. · น้ำหนัก exp(−d/50)"),
    (EDGE_B, "type_b — ตามทิศลม (มีทิศทาง, รายชั่วโมง):", "ระยะ ≤ 200 กม. · alignment > 0.3"),
    (EDGE_C, "type_c — จุดความร้อนสู่สถานี (ปลายลม):", "ระยะ ≤ 500 กม. · alignment > 0.4"),
]
for i, (col, head, tail) in enumerate(edge_rows):
    ry = 6.60 - i * 0.42
    ax.add_patch(
        FancyBboxPatch(
            (0.55, ry - 0.14),
            0.34,
            0.28,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=col,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(
        1.05,
        ry,
        head,
        ha="left",
        va="center",
        fontsize=FS_SUB,
        color=INK,
        fontweight="bold",
        zorder=4,
    )
    ax.text(4.75, ry, tail, ha="left", va="center", fontsize=FS_SUB, color=INK2, zorder=4)

arrow(5.0, 5.62, 5.0, 5.26)

# ── [4] MTGNN ────────────────────────────────────────────────────────────────
chip(0.25, 5.06, "[4] โมเดล MTGNN (Wu et al., KDD 2020)", RED[1])
box(0.25, 3.42, 9.51, 1.42, RED, None, dashed=True)

mt = [
    ("Graph Learning", "adaptive adjacency"),
    ("TCN", "dilated causal conv"),
    ("GCN", "hidden 64 · 3 ชั้น"),
    ("Output heads", "6 / 12 / 24 / 48 ชม."),
]
mx, mw, gap = 0.55, 2.13, 0.23
for i, (t, s) in enumerate(mt):
    x = mx + i * (mw + gap)
    box(x, 4.02, mw, 0.72, RED, t, s, title_fs=9.2, sub_fs=7.6)
    if i:
        arrow(x - gap + 0.02, 4.38, x - 0.02, 4.38)
ax.text(
    5.0,
    3.68,
    "อินพุต X ขนาด N × 24 ชม. × 10 features · N = 18+M โหนด · รวมน้ำหนักเส้นเชื่อม 3 ชนิด + adaptive",
    ha="center",
    va="center",
    fontsize=FS_NOTE,
    color=INK2,
)

arrow(2.60, 3.42, 2.60, 3.06)
arrow(7.40, 3.42, 7.40, 3.06)

# ── [5] Outputs ──────────────────────────────────────────────────────────────
chip(0.25, 2.86, "[5] ผลลัพธ์ (OUTPUTS)", GREEN[1])
box(
    0.25,
    1.72,
    4.63,
    0.94,
    GREEN,
    "พยากรณ์ PM2.5",
    "18 สถานี × 4 ขอบฟ้า · hybrid กับ persistence\nพร้อมช่วงความเชื่อมั่น conformal 90%",
)
box(
    5.13,
    1.72,
    4.63,
    0.94,
    PINK,
    "XAI: GB-IG + Occlusion",
    "สัดส่วนแหล่งกำเนิดรายประเทศ (ไทย/เมียนมา/ลาว)\nและความสำคัญของปัจจัย (IG)",
)

arrow(2.60, 1.72, 3.80, 1.28)
arrow(7.40, 1.72, 6.20, 1.28)

# ── [6] Dashboard ────────────────────────────────────────────────────────────
chip(0.25, 1.46, "[6] แดชบอร์ด (STREAMLIT DASHBOARD)", GRAY[1])
box(
    0.25,
    0.30,
    9.51,
    0.98,
    GRAY,
    "Streamlit Dashboard",
    "แผนที่พยากรณ์ · แหล่งกำเนิด · ข้ามแดน + back-trajectory · counterfactual "
    "“ถ้าดับไฟกลุ่มนี้” · Live NWP · แจ้งเตือน Telegram",
)

fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
out = r"D:\งาน\NSC\pm25-stgnn-thailand\architecture_diagram.png"
plt.savefig(out, dpi=300, facecolor=BG)
print(f"Saved: {out}")
