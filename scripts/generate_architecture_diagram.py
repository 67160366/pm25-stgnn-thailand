"""Generate system architecture diagram for NSC 2026 proposal."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Use Leelawadee UI — Thai-capable font available on this Windows system
plt.rcParams["font.family"] = "Leelawadee UI"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

# ── Color palette ─────────────────────────────────────────────────────────────
C_DATA    = "#2196F3"   # blue  — data sources
C_PROC    = "#9C27B0"   # purple — processing
C_GRAPH   = "#FF9800"   # orange — graph builder
C_MODEL   = "#F44336"   # red   — MTGNN model
C_OUTPUT  = "#4CAF50"   # green — outputs
C_XAI     = "#00BCD4"   # cyan  — XAI
C_DASH    = "#795548"   # brown — dashboard
C_EDGE_A  = "#607D8B"
C_EDGE_B  = "#FF5722"
C_EDGE_C  = "#8BC34A"
WHITE     = "#FFFFFF"
DARK      = "#212121"
LIGHT_BG  = "#F5F5F5"


def box(ax, x, y, w, h, color, text, fontsize=11, text_color=WHITE,
        bold=False, radius=0.3, alpha=1.0, sub_text=None, sub_size=9):
    fancy = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor=WHITE, linewidth=2, alpha=alpha, zorder=3
    )
    ax.add_patch(fancy)
    weight = "bold" if bold else "normal"
    cy = y + h / 2 + (0.15 if sub_text else 0)
    ax.text(x + w / 2, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=weight,
            zorder=4, wrap=True)
    if sub_text:
        ax.text(x + w / 2, y + h / 2 - 0.28, sub_text, ha="center", va="center",
                fontsize=sub_size, color=text_color, alpha=0.85, zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#546E7A", lw=2.5, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle="arc3,rad=0.0"
        ),
        zorder=2,
    )


def section_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=10, color=color, fontweight="bold",
            va="center", ha="left", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15,
                      edgecolor=color, linewidth=1.2))


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DATA SOURCES  (y: 18.2 – 19.6)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 19.3, "[1] DATA SOURCES", C_DATA)

box(ax, 0.3,  18.0, 4.5, 1.1, C_DATA,
    "Air4Thai API", fontsize=11, bold=True,
    sub_text="PM2.5 · 18 สถานี · รายชั่วโมง")

box(ax, 5.55, 18.0, 4.9, 1.1, C_DATA,
    "NASA FIRMS", fontsize=11, bold=True,
    sub_text="VIIRS/MODIS Fire FRP · รายวัน")

box(ax, 11.0, 18.0, 4.7, 1.1, C_DATA,
    "ERA5 (CDS API)", fontsize=11, bold=True,
    sub_text="u10, v10, t2m, d2m, blh · ทุก 6h")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — PREPROCESSING  (y: 15.8 – 17.1)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 17.5, "[2] PREPROCESSING", C_PROC)

box(ax, 0.3,  15.8, 4.5, 1.1, C_PROC,
    "Station Preprocessing", fontsize=10, bold=True,
    sub_text="RobustScaler per station · gap fill")

box(ax, 5.55, 15.8, 4.9, 1.1, C_PROC,
    "Hotspot Clustering", fontsize=10, bold=True,
    sub_text="DBSCAN spatial cluster · FRP aggregate")

box(ax, 11.0, 15.8, 4.7, 1.1, C_PROC,
    "ERA5 Interpolation", fontsize=10, bold=True,
    sub_text="Bilinear interp to station coords")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — GRAPH BUILDER  (y: 12.8 – 15.0)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 15.2, "[3] DYNAMIC GRAPH BUILDER", C_GRAPH)

# Main graph box
box(ax, 0.3, 12.8, 15.4, 2.0, C_GRAPH,
    "", fontsize=10, alpha=0.12, text_color=DARK, radius=0.4)

# Nodes
box(ax, 0.7,  13.55, 4.5, 1.0, C_GRAPH,
    "18 Station Nodes", fontsize=10, bold=True,
    sub_text="PM2.5 (scaled) + ERA5 + time encoding")

box(ax, 5.8,  13.55, 4.5, 1.0, "#FF6F00",
    "M Hotspot Nodes", fontsize=10, bold=True,
    sub_text="FRP, lat/lon, country (TH/MM/LA)")

# Edge type boxes
box(ax, 0.7,  12.9, 3.2, 0.55, C_EDGE_A,
    "type_a: k-NN Geographic  (k=5, fixed)", fontsize=8.5, bold=False)
box(ax, 4.2,  12.9, 3.8, 0.55, C_EDGE_B,
    "type_b: Wind-aligned  (cosine sim, hourly)", fontsize=8.5, bold=False)
box(ax, 8.3,  12.9, 4.5, 0.55, C_EDGE_C,
    "type_c: Hotspot-to-Station bipartite  (<300 km, downwind)", fontsize=8.5, bold=False)

# Arrow label
ax.text(13.1, 14.05, "Dynamic\nGraph\nG(V,E,t)", fontsize=9,
        color=C_GRAPH, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                  edgecolor=C_GRAPH, linewidth=1.5))

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — MTGNN MODEL  (y: 10.0 – 12.3)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 12.2, "[4] MTGNN MODEL  (Wu et al., KDD 2020)", C_MODEL)

box(ax, 0.3, 10.0, 15.4, 1.9, C_MODEL, "", alpha=0.08, radius=0.4)

box(ax, 0.7, 10.55, 3.3, 1.1, C_MODEL,
    "Graph Learning", fontsize=10, bold=True,
    sub_text="Adaptive adjacency\nfrom node embeddings")

box(ax, 4.3, 10.55, 3.3, 1.1, C_MODEL,
    "TCN × 3", fontsize=10, bold=True,
    sub_text="Dilated causal conv\ntemporal patterns")

box(ax, 7.9, 10.55, 3.3, 1.1, C_MODEL,
    "GCN × 3", fontsize=10, bold=True,
    sub_text="Spatial propagation\nhidden_dim=64")

box(ax, 11.5, 10.55, 3.9, 1.1, C_MODEL,
    "Skip Connections", fontsize=10, bold=True,
    sub_text="252,588 params\nBest epoch: 15")

# inner arrows
arrow(ax, 4.0, 11.1, 4.3, 11.1, color=WHITE, lw=2)
arrow(ax, 7.6, 11.1, 7.9, 11.1, color=WHITE, lw=2)
arrow(ax, 11.2, 11.1, 11.5, 11.1, color=WHITE, lw=2)

# Input label
ax.text(8.0, 10.1, "Input: X in R^(N x T x F)   N=18+M nodes, T=24h, F=10 features",
        fontsize=9, color=C_MODEL, ha="center", va="center", fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — OUTPUTS  (y: 7.0 – 9.5)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 9.4, "[5] OUTPUTS", C_OUTPUT)

# Forecast output
box(ax, 0.3, 7.1, 7.2, 2.0, C_OUTPUT,
    "PM2.5 FORECAST", fontsize=12, bold=True,
    sub_text="18 สถานี × 4 ขอบฟ้า\n6h · 12h · 24h · 48h")

# XAI output
box(ax, 8.5, 7.1, 7.2, 2.0, C_XAI,
    "XAI: GB-IG Attribution", fontsize=12, bold=True,
    sub_text="Source % (TH / MM / LA)\nFeature importance ranking")

# Performance note
ax.text(3.9, 7.55, "RMSE 24h: 10.21 µg/m³ (+5.1% vs persistence)\nRMSE 48h: 14.32 µg/m³ (+10.6% vs persistence)",
        fontsize=8.5, color=WHITE, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#2E7D32", alpha=0.85))

ax.text(12.1, 7.55, "Mar 2024 Chiang Mai peak:\nTH=100%, MM=0% (FRP ratio 128:1)\nHotspot adds ~4.46 µg/m³ at 24h",
        fontsize=8.5, color=WHITE, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#00838F", alpha=0.85))

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — DASHBOARD  (y: 4.8 – 6.6)
# ══════════════════════════════════════════════════════════════════════════════
section_label(ax, 0.3, 6.5, "[6] STREAMLIT DASHBOARD", C_DASH)

box(ax, 0.3, 4.9, 15.4, 1.4, C_DASH,
    "Streamlit Dashboard (Web Browser)", fontsize=13, bold=True,
    sub_text="Interactive forecast map · Source attribution chart · Feature importance · Time series viewer")

# ══════════════════════════════════════════════════════════════════════════════
# VERTICAL ARROWS BETWEEN LAYERS
# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 → 2
for cx in [2.55, 8.0, 13.35]:
    arrow(ax, cx, 18.0, cx, 17.0, color=C_DATA, lw=2.5)

# Layer 2 → 3 (converge)
arrow(ax, 2.55, 15.8, 5.0, 14.85, color=C_PROC, lw=2.5)
arrow(ax, 8.0,  15.8, 8.0, 14.85, color=C_PROC, lw=2.5)
arrow(ax, 13.35,15.8, 11.0,14.85, color=C_PROC, lw=2.5)

# Layer 3 → 4
arrow(ax, 8.0, 12.8, 8.0, 11.95, color=C_GRAPH, lw=3)

# Layer 4 → 5 (split)
arrow(ax, 4.5, 10.0, 3.9, 9.15,  color=C_MODEL, lw=3)
arrow(ax, 11.5,10.0, 12.1, 9.15, color=C_MODEL, lw=3)

# Layer 5 → 6
arrow(ax, 3.9, 7.1, 5.5, 6.35,  color=C_OUTPUT, lw=2.5)
arrow(ax, 12.1,7.1, 10.5, 6.35, color=C_XAI,    lw=2.5)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE & LEGEND
# ══════════════════════════════════════════════════════════════════════════════
ax.text(8.0, 19.7,
        "System Architecture: Explainable STGNN for PM2.5 Forecasting & Source Attribution",
        ha="center", va="center", fontsize=14, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                  edgecolor="#90A4AE", linewidth=1.5))

# Legend for edge types
legend_x, legend_y = 0.3, 4.3
ax.text(legend_x, legend_y, "Edge types:", fontsize=9, fontweight="bold", color=DARK)
patches = [
    mpatches.Patch(color=C_EDGE_A, label="type_a: Geographic k-NN (k=5, fixed)"),
    mpatches.Patch(color=C_EDGE_B, label="type_b: Wind-aligned (cosine sim, hourly update)"),
    mpatches.Patch(color=C_EDGE_C, label="type_c: Hotspot-to-Station bipartite (downwind <=300 km)"),
]
ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(0.0, 0.0),
          fontsize=8.5, framealpha=0.9, edgecolor="#B0BEC5")

plt.tight_layout()
out = r"D:\งาน\NSC\pm25-stgnn-thailand\architecture_diagram.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Saved: {out}")
