"""Poster-scale figures for the NSC 2026 A0 poster, emitted as HTML/CSS.

Why not matplotlib: matplotlib cannot stack Thai tone marks over an upper
vowel, so ``ที่`` (ท + ◌ี + ◌่) silently renders as ``ที`` — a visible spelling
error. Chrome shapes Thai correctly, and the poster is already printed through
headless Chrome, so every figure here is built as markup and rendered by the
same engine. Side benefit: the bars stay vector in the PDF instead of being
resampled raster at A0.

Numbers are read from ``outputs/*.json`` — never hard-coded — so the poster
cannot drift from the evaluation artefacts.

Each builder returns an HTML fragment; ``FIGURE_CSS`` carries their styles and
is concatenated into the poster stylesheet by ``generate_poster.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"

EVENT_DATE = "2025-03-18"  # the Mae Hong Son transboundary case carried on the poster
HORIZONS = ["6h", "12h", "24h", "48h"]
HORIZON_TH = ["6 ชม.", "12 ชม.", "24 ชม.", "48 ชม."]


def _load(name: str) -> dict:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


# Palette sampled from the organizer template background (poster_bg.jpg): using
# the template's own hues stops the content reading as pasted on top of it.
FIGURE_CSS = """
/* ---- shared figure chrome ---- */
.figpanel { background:#FFFFFF; border:0.5mm solid var(--line); border-radius:4mm; padding:7mm 8mm 6mm; }
.figpanel .figtitle { font-family:'Noto Serif Thai',serif; font-weight:700; font-size:10.5mm;
  color:var(--indigo); line-height:1.25; margin-bottom:1.5mm; }
.figpanel .figsub { font-size:6.5mm; color:var(--mute); line-height:1.35; margin-bottom:5mm; }
.figpanel .fignote { font-size:6.9mm; color:var(--mute); line-height:1.4; margin-top:4mm; }

/* ---- horizontal evidence bars ---- */
.evrow { display:grid; grid-template-columns: 74mm 1fr 26mm; align-items:center;
  gap:4mm; margin-bottom:5mm; }
.evrow .evlab { font-size:6.6mm; line-height:1.3; color:var(--ink); text-align:right; }
.evrow .evlab i { display:block; font-style:normal; font-size:6.4mm; color:var(--mute); }
.evtrack { position:relative; height:12.3mm; background:#F1F5FB; border-radius:1.5mm; }
.evtrack .evfill { position:absolute; left:0; top:0; bottom:0; border-radius:1.5mm; }
.evtrack .evfill.foreign { background:var(--magenta); }
.evtrack .evfill.model { background:var(--blue); }
.evtrack .evhalf { position:absolute; top:-1.5mm; bottom:-1.5mm; width:0;
  border-left:0.6mm dashed var(--mute); }
.evtrack .evrange { position:absolute; top:50%; height:0; border-top:0.7mm solid var(--ink); }
.evtrack .evcap { position:absolute; top:50%; width:0; height:5mm;
  border-left:0.7mm solid var(--ink); transform:translateY(-50%); }
.evrow .evval { font-family:'Noto Serif Thai',serif; font-weight:700; font-size:9.4mm;
  color:var(--ink); text-align:right; }
.evscale { display:grid; grid-template-columns: 74mm 1fr 26mm; gap:4mm; }
/* ticks are absolutely placed at their true percentage: flex space-between
   aligns box edges, not label centres, which slid "50" to roughly 34% */
.evscale .ticks { position:relative; height:9.0mm; border-top:0.4mm solid var(--line); }
.evscale .ticks span { position:absolute; top:1.5mm; transform:translateX(-50%);
  font-size:5.6mm; color:var(--mute); white-space:nowrap; }

/* ---- grouped RMSE columns ---- */
.rmseplot { position:relative; height:85.1mm; margin:2mm 0 0 16mm;
  border-bottom:0.5mm solid var(--line); }
.rmseplot .grid { position:absolute; left:0; right:0; border-top:0.4mm solid var(--line); }
.rmseplot .grid span { position:absolute; left:-16mm; top:-3mm; width:13mm; text-align:right;
  font-size:5.6mm; color:var(--mute); }
.rmsegroups { position:absolute; inset:0; display:flex; }
.rmsegroup { flex:1; display:flex; align-items:flex-end; justify-content:center; gap:1.6mm; }
.rmsegroup b { width:10.1mm; border-radius:1mm 1mm 0 0; display:block; }
.rmsex { display:flex; margin-left:16mm; }
.rmsex span { flex:1; text-align:center; font-size:6.2mm; color:var(--ink); padding-top:2mm; }
.rmsekey { display:flex; flex-wrap:wrap; gap:2mm 7mm; margin-bottom:3mm; }
.rmsekey span { font-size:6.2mm; color:var(--ink); display:flex; align-items:center; gap:2mm; }
.rmsekey i { width:5mm; height:5mm; border-radius:1mm; display:block; }

/* ---- pipeline diagram ---- */
.pipe { display:flex; flex-direction:column; gap:0; }
.pipestage { border-radius:3.5mm; padding:5mm 6mm; text-align:center; }
.pipestage .st { font-family:'Noto Serif Thai',serif; font-weight:700; font-size:8.3mm;
  line-height:1.25; }
.pipestage .sb { font-size:5.7mm; line-height:1.4; color:var(--ink); margin-top:1.5mm; }
.pipestage.data { background:#F2F6FC; border:0.5mm solid var(--line); }
.pipestage.data .st { color:var(--indigo); }
.pipestage.wind { background:#FBEDF7; border:0.5mm solid #F0C8E4; }
.pipestage.wind .st { color:var(--magenta-ink); }
.pipestage.model { background:#EDF1FB; border:0.5mm solid #C9D4EE; }
.pipestage.model .st { color:var(--indigo); }
.pipearrow { align-self:center; width:0; height:0; border-left:3.4mm solid transparent;
  border-right:3.4mm solid transparent; border-top:4mm solid var(--blue); margin:3mm 0; }
.pipesplit { display:flex; gap:5mm; }
.pipesplit > div { flex:1; }
.pipefork { display:flex; justify-content:space-around; }
.pipefork div { width:0; height:0; border-left:3.4mm solid transparent;
  border-right:3.4mm solid transparent; border-top:4mm solid var(--blue); margin:3mm 0; }
"""


def evidence_figure() -> str:
    """Three independent witnesses, all on one 0–100% 'foreign share' scale.

    This is the poster's primary evidence graphic. The argument for the
    18 Mar 2025 Mae Hong Son event is that two physical witnesses and the model
    agree — and that the model is the *weakest* of the three, its seed spread
    covering the full range. Putting them on a shared axis makes both facts
    readable in one glance.
    """
    unc = _load("transboundary_uncertainty.json")
    bt = _load("backtrajectory_test2025.json")
    event = next(e for e in unc["events"] if e["date"] == EVENT_DATE)
    traj = next(e for e in bt["events"] if e["date"] == EVENT_DATE)

    fire = event["connected_foreign_fraction"] * 100
    wind = traj["foreign_hours_fraction"] * 100
    seeds = [v["foreign_attribution"] * 100 for v in event["per_seed"].values()]
    model = sum(seeds) / len(seeds)
    lo, hi = min(seeds), max(seeds)

    # fires the wind graph actually connects to the station (drives the 72.1%)
    conn = event["connected_frp_by_country"]
    conn_foreign = round(conn.get("Myanmar", 0) + conn.get("Laos", 0))
    conn_thai = round(conn.get("Thailand", 0))
    # fires sitting inside the 50 km back-trajectory corridor (drives the 57.1%)
    corr = traj["corridor_frp_by_country"]
    corr_mm = round(corr.get("Myanmar", 0))
    corr_th = round(corr.get("Thailand", 0))

    rows = [
        ("ไฟที่ลมพัดถึงสถานี", "จุดความร้อน NASA FIRMS", fire, "foreign", ""),
        ("เส้นทางมวลอากาศ 48 ชม.", "back-trajectory จากลม ERA5", wind, "foreign", ""),
        (
            "โมเดล AI (GB-IG)",
            "เฉลี่ย 3 seed",
            model,
            "model",
            f'<div class="evrange" style="left:{lo}%;width:{hi - lo}%"></div>'
            f'<div class="evcap" style="left:{lo}%"></div>'
            f'<div class="evcap" style="left:{hi}%"></div>',
        ),
    ]

    body = "".join(
        f'<div class="evrow"><div class="evlab">{lab}<i>{sub}</i></div>'
        f'<div class="evtrack"><div class="evfill {cls}" style="width:{val:.1f}%"></div>'
        f'<div class="evhalf" style="left:50%"></div>{extra}</div>'
        f'<div class="evval">{val:.1f}%</div></div>'
        for lab, sub, val, cls, extra in rows
    )

    return f"""<div class="figpanel">
  <div class="figtitle">หลักฐาน 3 ชั้น ชี้ตรงกัน</div>
  <div class="figsub">เหตุการณ์แม่ฮ่องสอน 18 มี.ค. 2568 · PM2.5 สูงสุด {event["peak_pm25_ug_m3"]:.0f} µg/m³</div>
  {body}
  <div class="evscale"><div></div>
    <div class="ticks"><span style="left:0">0</span><span style="left:25%">25</span>
    <span style="left:50%">50</span><span style="left:75%">75</span>
    <span style="left:100%">100%</span></div>
    <div></div></div>
  <div class="fignote">ความแรงไฟ (FRP) ที่ลมเชื่อมถึงสถานี: ต่างประเทศ <b>{conn_foreign:,}</b>
  เทียบไทย <b>{conn_thai:,} MW</b> · ไฟในแนวเส้นทางลม: เมียนมา <b>{corr_mm:,}</b>
  เทียบไทย <b>{corr_th:,} MW</b><br>
  โมเดล AI ผันตาม seed ถึง <b>{lo:.0f}–{hi:.0f}%</b> จึงยึดหลักฐานฟิสิกส์เป็นหลัก
  และรายงานช่วงความไม่แน่นอนเสมอ</div>
</div>"""


def rmse_figure() -> str:
    """Held-out 2025 RMSE per horizon; ours is the only saturated series."""
    ev = _load("evaluation_test2025.json")
    gbm = _load("baseline_ml_test.json")
    series = [
        ("ค่าล่าสุด (persistence)", ev["persistence_rmse_ug_m3"], "var(--grey1)"),
        ("MTGNN + กราฟ (ของเรา)", ev["mtgnn"]["rmse_ug_m3"], "var(--indigo)"),
        ("A3TGCN", ev["a3tgcn"]["rmse_ug_m3"], "var(--grey2)"),
        ("GBM (ไม่ใช้กราฟ)", gbm["baseline_ml_rmse_ug_m3"], "var(--grey3)"),
    ]
    top = 15.0  # µg/m³ — clears the 14.22 max with headroom for the axis label

    key = "".join(f'<span><i style="background:{c}"></i>{lab}</span>' for lab, _, c in series)
    gridlines = "".join(
        f'<div class="grid" style="bottom:{v / top * 100:.1f}%"><span>{v}</span></div>'
        for v in (0, 5, 10, 15)
    )
    groups = "".join(
        '<div class="rmsegroup">'
        + "".join(
            f'<b style="height:{d[h] / top * 100:.1f}%;background:{c}"></b>' for _, d, c in series
        )
        + "</div>"
        for h in HORIZONS
    )
    xlabels = "".join(f"<span>{t}</span>" for t in HORIZON_TH)

    return f"""<div class="figpanel">
  <div class="figtitle">ความแม่นยำเทียบเกณฑ์เปรียบเทียบ</div>
  <div class="figsub">แท่งเตี้ยกว่า = แม่นกว่า · ค่าคลาดเคลื่อน RMSE (µg/m³) · ชุดทดสอบ held-out ปี 2568</div>
  <div class="rmsekey">{key}</div>
  <div class="rmseplot">{gridlines}<div class="rmsegroups">{groups}</div></div>
  <div class="rmsex">{xlabels}</div>
</div>"""


def pipeline_figure() -> str:
    """Four-stage pipeline, legible at poster scale.

    The detailed 20-box report diagram stays in the report: placed in a 245 mm
    poster column its stage-3 legend renders under 3 mm and cannot be read.
    """
    return """<div class="figpanel">
  <div class="figtitle">ระบบทำงานอย่างไร</div>
  <div class="figsub">จากข้อมูลดิบ 3 แหล่ง สู่คำตอบ 2 อย่างในระบบเดียว</div>
  <div class="pipe">
    <div class="pipestage data">
      <div class="st">1 · ข้อมูลเข้า</div>
      <div class="sb">Air4Thai 18 สถานี (PM2.5 รายชั่วโมง) · จุดความร้อน NASA FIRMS
      · ลมและอุตุนิยมวิทยา ERA5</div>
    </div>
    <div class="pipearrow"></div>
    <div class="pipestage wind">
      <div class="st">2 · กราฟพลวัตตามทิศลม</div>
      <div class="sb">โหนด = สถานีตรวจวัด <b>และกลุ่มจุดไฟ</b> ·
      เส้นเชื่อมเปลี่ยนน้ำหนักทุกชั่วโมงตามทิศลมจริง จึงจับการพัดข้ามพรมแดนได้</div>
    </div>
    <div class="pipearrow"></div>
    <div class="pipestage model">
      <div class="st">3 · โมเดล MTGNN</div>
      <div class="sb">เรียนรู้เชิงพื้นที่และเวลาพร้อมกัน · พยากรณ์ 4 ระยะ 6 / 12 / 24 / 48 ชม.</div>
    </div>
    <div class="pipefork"><div></div><div></div></div>
    <div class="pipesplit">
      <div class="pipestage model">
        <div class="st">4a · ค่าพยากรณ์</div>
        <div class="sb">PM2.5 ล่วงหน้าถึง 48 ชม.<br>พร้อมช่วงเชื่อมั่น 90%</div>
      </div>
      <div class="pipestage wind">
        <div class="st">4b · แหล่งกำเนิด</div>
        <div class="sb">GB-IG แยกสัดส่วน<br>ไทย / เมียนมา / ลาว รายเหตุการณ์</div>
      </div>
    </div>
  </div>
</div>"""
