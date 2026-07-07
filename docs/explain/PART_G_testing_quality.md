# PART G — TESTING & CODE QUALITY

> อ้างอิงจริง: `tests/`, `pyproject.toml` (`[tool.ruff]`, `[tool.black]`, `[tool.pytest]`),
> SESSION notes (จำนวน test)

## G1. Tests

**ไฟล์ test ทั้งหมด** (`tests/`, pytest):

| ไฟล์ | ครอบคลุม |
|---|---|
| `test_openaq.py` | OpenAQ scraper — discover, backfill, curate, decode UTF-16 (mock ทั้งหมด) |
| `test_firms.py` | FIRMS scraper — bbox order, SP 5-day cap, NRT/SP hybrid, clustering input |
| `test_era5.py` | ERA5 — coord normalise, alias, Windows temp-file roundtrip |
| `test_preprocessing.py` | gap policy (<6/6-24/>24/>168h), RobustScaler fit, cyclic features |
| `test_graph_builder.py` | haversine, wind alignment, Type A/B/C edges, build_graph HeteroData (+integration) |
| `test_loader.py` | sliding window, anchor index, split, feature stack, ERA5 fallback zero |
| `test_models.py` | MTGNN/A3TGCN forward shape, empty edge, clip ≥0 |
| `test_training.py` | trainer loop, NaN guard, losses/metrics |
| `test_explain.py` | IG completeness, occlusion country attribution, gradient×input |
| `test_app_lib.py` | ฟังก์ชัน helper ของ Streamlit (`app/lib/`) |

**รันยังไง:**
```bash
uv run pytest tests/ -v
```
`addopts = "-v --cov=src --cov-report=term-missing --cov-report=html"` → รายงาน coverage อัตโนมัติ

**ผลปัจจุบัน (จาก SESSION notes):** Session 1 = 25/25 green, Session 2 = **70/70 green**
(หลังจากนั้นเพิ่ม test ของ ERA5/models/training/explain/app-lib) — แนวทางคือทุก session ต้องเขียว
ก่อน commit

**หลักการเทสต์สำคัญ (CLAUDE.md hard restriction):**
> **ห้าม network call จริงในเทส** — mock ด้วย `requests-mock` เสมอ
ทุก scraper test จึง mock HTTP response ไม่แตะ API จริง (เร็ว + ทำซ้ำได้ + ไม่กิน quota)

**อะไรที่ยังไม่ได้ test / test บาง:**
- pipeline end-to-end จริง (ดึง→เทรน→eval) ไม่มี integration test เต็ม — แยกเป็น unit ราย module
- สคริปต์ generate report/pitch/proposal (`scripts/generate_*.py`) เป็น one-off ไม่มี test
- ผลเชิงตัวเลขของโมเดล (RMSE) ไม่ได้ assert ในเทส — วัดผ่าน `outputs/*.json` แทน

## G2. Code Quality Tools

**linter/formatter** (`pyproject.toml`):

| tool | ตั้งค่า |
|---|---|
| **black** | `line-length = 100` |
| **ruff** | `line-length = 100`; lint select = `["E","W","F","I","B","C90","N","UP","S","ANN","RUF"]`, `ignore = []` |

ความหมาย rule set: `E/W/F` (pycodestyle+pyflakes), `I` (isort), `B` (bugbear), `C90` (complexity),
`N` (naming), `UP` (pyupgrade), `S` (bandit security), `ANN` (type annotation บังคับ), `RUF` (ruff)

**ข้อยกเว้น (`per-file-ignores`) และเหตุผล:**
```toml
"tests/**/*.py"          = ["S101", "ANN"]   # ใช้ assert ได้ (S101), ไม่ต้อง annotate เต็ม
"notebooks/**/*.py"      = ["ANN", "E402"]   # EDA — import กลางไฟล์/ไม่ annotate ได้
"scripts/**/*.py"        = ["S101"]          # assert ในสคริปต์ได้
"scripts/generate_*.py"  = ["E501","RUF001/2/3","ANN"]  # สตริงไทยยาว + อักขระไทยกำกวม (× dash) + throwaway
"app/**/*.py"            = ["E501","RUF001/2/3"]         # UI ไทย: บรรทัดยาว + glyph ไทยตั้งใจ
```
เหตุผลหลัก: **เนื้อหาภาษาไทย** ทำให้ ruff เตือน E501 (บรรทัดยาว) และ RUF001/2/3 (อักขระที่หน้าตา
คล้าย ASCII เช่น `×` U+00D7, en/em dash) — ในบริบท display/UI/เอกสารเป็นความตั้งใจ จึง ignore เฉพาะไฟล์

**บังคับก่อน commit** (CLAUDE.md working style):
```bash
uv run black src/ tests/ && uv run ruff check --fix src/    # format + fix
uv run ruff check src/ && uv run black --check src/          # verify
```
"resolve all warnings before commit" — ห้ามเหลือ warning; ห้าม `print()` ใน `src/` (ใช้ logging);
ห้าม `except Exception:` ลอยๆ

---

## สรุป PART G

- **10 ไฟล์ test** ครอบคลุมทุก module หลัก (scraper→preprocess→graph→loader→model→train→explain→app)
  ทุก network call mock ด้วย requests-mock; ประวัติ 70/70 green (S2)
- ยังขาด: integration end-to-end, test ของ generate scripts, assert ผลตัวเลขโมเดล
- **black + ruff** line-length 100; ruff select กว้าง (รวม S/ANN); ignore เฉพาะไฟล์ที่มี **เนื้อหาไทย**
  (E501/RUF001-3) และ tests/notebooks/scripts (S101/ANN)

> **ต้องการลงลึก PART ไหน หรือไปต่อ PART H (Session Summary — Part สุดท้าย)?**
