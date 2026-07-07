# PART A — ภาพรวมโปรเจค (BIG PICTURE)

> สรุปตามคำสั่งใน `docs/Explain.md` — อธิบายภาษาไทย ใช้โค้ด/ข้อมูลจริงจากโปรเจค

## A1. โปรเจคคืออะไร

**เป้าหมาย / ปัญหาที่แก้:**
ผลงานส่ง **NSC 2026 ประเภทที่ 14** (ระดับนิสิต นักศึกษา) สร้าง **Explainable
Spatio-Temporal Graph Neural Network (STGNN)** เพื่อ 2 อย่างพร้อมกันใน 9 จังหวัดภาคเหนือ:
1. **พยากรณ์ PM2.5** ล่วงหน้า 6h / 12h / 24h / 48h
2. **ระบุแหล่งที่มา (source attribution)** ของฝุ่น — โดยเฉพาะฝุ่นข้ามพรมแดนจากพม่า/ลาว

จุดขาย 3 อย่าง (จาก `CLAUDE.md`):
| # | Novelty | แนวคิด |
|---|---|---|
| 1 | Wind-aware dynamic adjacency | สร้าง edge ของกราฟตามทิศลม (ฝุ่นข้ามแดน) |
| 2 | FIRMS hotspot เป็น node | จุดความร้อนไฟ (ดาวเทียม NASA) เป็น node ในกราฟโดยตรง |
| 3 | GB-IG source attribution | อธิบายที่มาฝุ่นด้วย Graph-based Integrated Gradients |

**Deliverables จริง:**
- โมเดล 2 ตัว: **MTGNN** (หลัก) และ **A3TGCN** (`src/models/`)
- **Streamlit dashboard** หลายหน้า (`app/`)
- **NSC Final Report (Word)** + pitch deck + รูปผลการทดลอง (`outputs/`)
- Pipeline ข้อมูลครบ (scrapers → preprocess → graph → train → evaluate → attribution)

**Thesis จริง (จาก `docs/CRITICAL_REVIEW.md`):**
โปรเจคนี้ **ซื่อสัตย์กับผลลบของตัวเอง** ซึ่งเป็นจุดเด่น:
- ในฐานะ **ผลงาน NSC** (เทคนิคขั้นสูง + ปัญหาจริง + pipeline หลายแหล่ง + XAI + eval ทำซ้ำได้):
  **แข็งแรงและป้องกันได้**
- ในฐานะ **"ระบบที่พยากรณ์ดีขึ้นจริง / attribution ใช้เชิงนโยบายได้":** **หลักฐานยังไม่รองรับ**

ผลบน held-out test 2025 (`outputs/evaluation_test2025.json`) เทียบ persistence:

| RMSE (µg/m³) | 6h | 12h | 24h | 48h |
|---|---|---|---|---|
| Persistence | **2.96** | **5.19** | 8.70 | 12.99 |
| MTGNN | 4.52 | 5.93 | 8.76 | **12.66** |

→ MTGNN **แพ้** persistence ระยะสั้น ดีขึ้นเล็กน้อยเฉพาะ 48h และหายไปเมื่อใช้ NWP จริง
= ความหมายของ "forecast edge marginal" และ "graph does not robustly help accuracy"

## A2. โครงสร้าง Repository

```
pm25-stgnn-thailand/
├── CLAUDE.md, README.md          ← memory โปรเจค + คู่มือ
├── pyproject.toml, uv.lock       ← dependency (จัดการด้วย uv)
├── configs/                      ← Hydra YAML
├── docs/                         ← DESIGN.md, CRITICAL_REVIEW.md, SESSION{1..10}_NOTES.md
├── src/                          ← ★ หัวใจ
│   ├── data/
│   │   ├── graph_builder.py      ← ★★ หัวใจที่สุด (heterograph)
│   │   ├── hotspot_clustering.py, preprocessing.py, loader.py
│   │   └── scrapers/{openaq,firms,era5}.py
│   ├── models/{base,a3tgcn,mtgnn}.py
│   ├── training/{trainer,losses,metrics,evaluation}.py
│   ├── explain/{gb_ig,gnn_explainer,attribution}.py
│   └── viz/{maps,timeseries}.py
├── app/{streamlit_app.py, lib/*, views/*}
├── scripts/01..13_*.py + generate_report/pitch
├── tests/                        ← pytest 10 ไฟล์
└── outputs/                      ← ★ generated: eval, ablation, figures, report
```

**ไฟล์หัวใจ:** `graph_builder.py`, `models/mtgnn.py`, `explain/gb_ig.py`, `data/loader.py`
**Generated/ไม่แก้มือ:** ทั้ง `outputs/`, `uv.lock`, `data/` (gitignored), `architecture_diagram.png`

## A3. Tech Stack

| Library | ทำอะไร | ใช้ตรงไหน | ตัดออกแล้วพัง |
|---|---|---|---|
| PyTorch 2.2+ | DL core | models, training | โมเดลพังหมด |
| PyG + PyG Temporal | Graph NN + temporal layers | graph_builder, models | STGNN สร้างไม่ได้ |
| torch-scatter/sparse | native ops ให้ PyG | ใต้ PyG | message passing พัง (ติดตั้งแยก) |
| Hydra | config YAML | scripts/03 + configs/ | สั่ง train แบบ override ไม่ได้ |
| Weights & Biases | experiment tracking | training | แค่ log หาย |
| Streamlit | dashboard | app/ | เดโมพัง |
| xarray | wind field (time,lat,lon,2) | ERA5 / build_graph | รับ wind_field ไม่ได้ |
| pandas / scikit-learn | ตาราง + RobustScaler | preprocessing | feature เตรียมไม่ได้ |
| requests / requests-mock | API + mock ใน test | scrapers, tests | ดาวน์โหลด/เทสต์พัง |
| cdsapi | ดึง ERA5 | scrapers/era5.py | ไม่มีลม/อุณหภูมิจริง |
| pytest + ruff + black | test + lint + format | tests/, คุณภาพ | คุณภาพไม่ถูกบังคับ |

`uv` (Python 3.11) จัดการ environment — **ห้ามแก้ `pyproject.toml` เอง** ใช้ `uv add`

## ⚠️ หมายเหตุความถูกต้องของเอกสาร
`docs/DESIGN.md` (v2.1) อธิบายสถาปัตยกรรมกราฟดี แต่ **ล้าสมัยเรื่องผลลัพธ์** —
ความจริงปัจจุบัน (test set จริง, ablation, ผลลบ) อยู่ใน `CRITICAL_REVIEW.md` และ
`SESSION{7..10}_NOTES.md` ยึดตัวหลังเป็นหลักเมื่อเล่าเรื่องผล
