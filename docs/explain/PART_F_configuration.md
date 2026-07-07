# PART F — PROJECT CONFIGURATION

> อ้างอิงจริง: `README.md`, `docs/INSTALL.md`, `.env` usage ใน `scrapers/{openaq,firms,era5}.py`,
> `pyproject.toml`, `configs/`, memory `.pth encoding fix`

## F1. Environment & Config

**ต้องตั้งค่าอะไรก่อนรัน (2 ขั้น — จาก CLAUDE.md):**
```powershell
uv sync                    # ติดตั้ง torch + pure-Python deps ทั้งหมด
./install_native_deps.ps1  # ติดตั้ง torch-scatter/sparse/geometric-temporal (native, ต้องมี torch ก่อน)
```
เหตุผลที่ native แยก: uv resolve ทุก dep ก่อนติดตั้ง แต่ 3 ตัวนี้ต้อง compile ทับ torch ที่ลงแล้ว →
ใส่ใน `pyproject.toml` ไม่ได้ ต้องลงจาก wheel index ของ PyG ทีหลัง

**`.env` ต้องมี key อะไร** (ราก repo; scraper เรียก `load_dotenv()` + `os.getenv`):
```dotenv
OPENAQ_API_KEY=<key จาก https://docs.openaq.org/>     # openaq.py → header X-API-Key
FIRMS_API_KEY=<MAP_KEY จาก https://firms.modaps.eosdis.nasa.gov/api/>   # firms.py URL path
```
- `.env.example` ในโปรเจค **ว่างเปล่า** — ต้องสร้าง `.env` เอง (README/`docs/INSTALL.md` บอกวิธี)
- จำเป็นเฉพาะตอน **re-download ข้อมูล** (`scripts/01`); dashboard + evaluation ใช้ข้อมูลที่เตรียมไว้แล้ว
  จึงรันได้โดยไม่ต้องมี key
- ห้าม commit `.env`/keys (Hard restriction ใน CLAUDE.md)

**`~/.cdsapirc` คืออะไร** (`era5.py:14`): ไฟล์ credential ของ **Copernicus CDS API** (ERA5) วางที่ home
```
url: https://cds.climate.copernicus.eu/api
key: <UID:API-KEY>
```
ใช้ตอนดึง ERA5 (ลม/อุณหภูมิ/blh) เท่านั้น — ถ้าไม่มี `download_era5_year` จะ error พร้อมข้อความ
ชี้ว่า "Ensure ~/.cdsapirc is present..."

**Config (Hydra) — `configs/`:**
```
config.yaml       ← รวม (defaults: data, model, trainer)
data.yaml         ← path, split, window_in, horizons
model/{mtgnn,a3tgcn}.yaml   ← n_features:10, hidden_dim, k_hop ฯลฯ
trainer.yaml      ← lr, epochs, batch, W&B
```
override ตอนรัน: `uv run python scripts/03_train.py model=mtgnn`

## F2. Known Issues & Workarounds

| ปัญหา | อาการ | workaround | สถานะ |
|---|---|---|---|
| **Windows Thai path** (repo อยู่ใต้ `D:\งาน\...`) | netCDF4/h5py เปิด path อักษรไทยไม่ได้ | `era5.py` `_open_nc/_write_nc` copy ไป temp ASCII ก่อน | ✅ แก้ถาวรในโค้ด |
| PYTHONUTF8 | อักษรไทย/encoding เพี้ยนบน Windows | `install_native_deps.ps1` ตั้ง `PYTHONUTF8=1` เป็น User env ถาวร | ✅ ถาวร |
| **`.pth` encoding** (หลัง `uv sync`) | `.venv/**/*.pth` มี abs-path ไทย → Python crash ตอน import | แทน abs-path ใน `.pth` ด้วย `../../..` (relative) | ⚠️ **manual fix ทุกครั้งหลัง sync** (ดู memory) |
| CDS v2 request ทั้งปีใหญ่ไป | request ถูกปฏิเสธ | `download_era5_year` ดึงทีละเดือน 12 ครั้งแล้ว merge | ✅ ในโค้ด |
| CDS v2 rename coord | `valid_time/latitude` ไม่ตรง | `_normalise_coords` → `time/lat/lon` | ✅ ในโค้ด |
| OpenAQ UTF-16+BOM | decode พัง | `response.encoding="utf-8-sig"` | ✅ ในโค้ด |
| OpenAQ 408 timeout ช่วงยาว | ดึงล้ม | แบ่ง 7-วัน chunk + retry (tenacity) | ✅ ในโค้ด |
| FIRMS SP day_range ≤ 5 (ไม่ใช่ 10) | request เกิน limit | `_SP_WINDOW_DAYS=5` chunk อัตโนมัติ | ✅ ในโค้ด |
| denorm reporting bug (S7) | ตัวเลข µg/m³ สูงเกินจริง | รวม eval ไว้ที่ `evaluation.py` ตัวเดียว | ✅ แก้แล้ว (PART E) |

> ปัญหาที่ยัง **ต้อง fix มือ**: `.pth` encoding หลัง `uv sync` — นอกนั้นแก้ถาวรในโค้ด/สคริปต์แล้ว

## F3. คำสั่งสำคัญ (จาก CLAUDE.md)

| ขั้น | คำสั่ง | ทำอะไร |
|---|---|---|
| 1. ติดตั้ง | `uv sync` → `./install_native_deps.ps1` | env + native deps |
| (fix มือ) | แก้ `.pth` เป็น relative | กัน crash import |
| 2. ค้นสถานี | `uv run python scripts/01_download_all.py discover` | curate 18 สถานี |
| 3. ดึงข้อมูล | `... 01_download_all.py backfill --start-year 2022 --end-year 2025` | PM2.5 + FIRMS ย้อนหลัง |
| 4. preprocess | `scripts/02_preprocess.py` | `dataset.parquet` + merge ERA5 |
| 5. เทรน | `uv run python scripts/03_train.py model=mtgnn` | train (Hydra override) |
| 6. eval | `scripts/04_evaluate.py` | RMSE vs persistence |
| 7. dashboard | `uv run streamlit run app/streamlit_app.py` | เดโม |
| คุณภาพ | `uv run pytest tests/ -v` / `uv run ruff check src/ && uv run black --check src/` | test/lint |
| snapshot สด | `scripts/01_download_all.py realtime` | ค่าสด air4thai |

**ลำดับที่ถูกต้อง:** install → (fix `.pth`) → discover → backfill → preprocess → train → evaluate →
dashboard. แต่ถ้าจะแค่ดู demo/ผล: มีข้อมูล+checkpoint เตรียมไว้ ข้ามขั้น 2–3 ได้ (ไม่ต้องมี API key)

---

## สรุป PART F

- 2 ขั้นติดตั้ง: `uv sync` + `install_native_deps` (native deps ลงทับ torch ไม่ได้ใน pyproject)
- `.env` = `OPENAQ_API_KEY` + `FIRMS_API_KEY` (เฉพาะตอน re-download); `~/.cdsapirc` = ERA5
- ปัญหาเด่น = **Windows Thai path** (แก้ถาวรในโค้ด) และ **`.pth` encoding** (ยังต้อง fix มือทุกครั้งหลัง sync)
- ลำดับรัน: install → discover → backfill → preprocess → train → evaluate → dashboard

> **ต้องการลงลึก PART ไหน หรือไปต่อ PART G (Testing & Code Quality)?**
