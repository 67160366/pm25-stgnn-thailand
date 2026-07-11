# โปรแกรมพยากรณ์ PM2.5 ด้วย Spatio-Temporal GNN สำหรับภาคเหนือของประเทศไทย
# PM2.5 Forecasting with Explainable Spatio-Temporal GNN for Northern Thailand

โปรแกรมนี้เป็นส่วนหนึ่งของโครงงานนักศึกษาสำหรับการแข่งขัน NSC 2026 (การแข่งขันพัฒนาโปรแกรมคอมพิวเตอร์แห่งชาติ ครั้งที่ 28) ประเภทที่ 14 โดยนำเสนอโมเดล Graph Neural Network เชิงอธิบายได้ สำหรับพยากรณ์และระบุแหล่งที่มาของ PM2.5 ใน 9 จังหวัดภาคเหนือของประเทศไทย

This project is an NSC 2026 (28th National Software Contest, Thailand) Category 14 entry. It presents an Explainable Spatio-Temporal Graph Neural Network (STGNN) for PM2.5 forecasting and source attribution across 9 Northern Thai provinces, incorporating wind-aware dynamic graph adjacency and NASA FIRMS fire hotspot nodes.

---

## สารบัญ / Table of Contents

- [การติดตั้ง / Quickstart](#การติดตั้ง--quickstart)
- [คำสั่งที่ใช้บ่อย / Common Commands](#คำสั่งที่ใช้บ่อย--common-commands)
- [ข้อกำหนดการใช้งาน / Disclaimer](#ข้อกำหนดการใช้งาน--disclaimer)
- [กิตติกรรมประกาศ / Acknowledgements](#กิตติกรรมประกาศ--acknowledgements)

---

## การติดตั้ง / Quickstart

ต้องการ Python 3.11 และ [uv](https://github.com/astral-sh/uv) ก่อนเริ่มต้น

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

### ขั้นตอนที่ 1 — ติดตั้ง dependencies หลัก / Step 1 — Install core dependencies

```bash
uv sync
```

### ขั้นตอนที่ 2 — ติดตั้ง native PyG extensions / Step 2 — Install native PyG extensions

ขั้นตอนนี้ติดตั้ง `torch-scatter`, `torch-sparse` และ `torch-geometric-temporal`
ซึ่งเป็น extensions ที่มี native C++/CUDA code และต้องการให้ `torch` ถูกติดตั้งก่อน
(จึงไม่สามารถรวมใน `pyproject.toml` ได้)

This step installs `torch-scatter`, `torch-sparse`, and `torch-geometric-temporal`,
which are native C++/CUDA extensions that require `torch` to already exist before
they can build. They cannot be listed in `pyproject.toml` because uv resolves all
dependencies before installing any of them.

**Windows:**

```powershell
./install_native_deps.ps1
```

> **Note (Windows):** The script sets `PYTHONUTF8=1` as a permanent User environment variable.
> **Restart your shell** after running it for the change to take effect. This is required when
> the repo is checked out to a path containing non-ASCII characters (e.g. Thai).

**Linux / macOS** (ยังไม่มีสคริปต์ `.sh` — รันคำสั่งเทียบเท่า / no `.sh` script yet; run the equivalent):

```bash
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.','') if torch.cuda.is_available() else 'cpu')")
pip install torch-scatter torch-sparse --find-links "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
pip install "torch-geometric-temporal>=0.54"
```

### ตัวแปรสภาพแวดล้อม / Environment variables

จำเป็นเฉพาะเมื่อจะดาวน์โหลดข้อมูลใหม่ (`scripts/01`) — dashboard และการประเมินผลใช้ข้อมูลที่เตรียมไว้แล้ว ไม่ต้องใช้ key
สร้างไฟล์ `.env` ที่รากของโปรเจกต์ด้วยตนเอง (ดูรายละเอียดใน `docs/INSTALL.md`)

Only needed to re-download data (`scripts/01`); the dashboard and evaluation use prepared data. Create `.env` manually at the repo root:

```
OPENAQ_API_KEY=<key from https://docs.openaq.org/>
FIRMS_API_KEY=<MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/>
```

สำหรับ ERA5 (Copernicus) วางไฟล์ `~/.cdsapirc` / For ERA5, place `~/.cdsapirc` (see `docs/INSTALL.md`).

---

## คำสั่งที่ใช้บ่อย / Common Commands

### ข้อมูล / Data

```bash
# ดาวน์โหลดข้อมูลแบบ realtime snapshot / Realtime snapshot
uv run python scripts/01_download_all.py realtime

# ค้นหาสถานีตรวจวัด / Discover monitoring stations
uv run python scripts/01_download_all.py discover

# ดาวน์โหลดข้อมูลย้อนหลัง / Backfill historical data
uv run python scripts/01_download_all.py backfill --start-year 2022 --end-year 2025
```

### การฝึกโมเดล / Training

```bash
# ฝึกโมเดล MTGNN / Train MTGNN model
uv run python scripts/03_train.py model=mtgnn
```

### แดชบอร์ด / Dashboard

```bash
# รัน Streamlit dashboard / Run Streamlit dashboard
uv run streamlit run app/streamlit_app.py
```

### การทดสอบและตรวจสอบคุณภาพโค้ด / Testing and Code Quality

```bash
# รัน unit tests / Run unit tests
uv run pytest tests/ -v

# ตรวจสอบ linting / Lint check
uv run ruff check src/ && uv run black --check src/

# จัดรูปแบบโค้ด / Format code
uv run black src/ tests/ && uv run ruff check --fix src/
```

---

## ข้อกำหนดการใช้งาน / Disclaimer

ซอฟต์แวร์นี้เป็นผลงานที่พัฒนาขึ้นโดย **นายรณชัย ขาวสะอาด** จาก **มหาวิทยาลัยบูรพา** ภายใต้การดูแลของ **ดร.วัชรพงศ์ อยู่ขวัญ** ภายใต้โครงการ *“ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วยโครงข่ายกราฟประสาทเทียมเชิงปริภูมิ-เวลาแบบอธิบายได้ สำหรับภาคเหนือของประเทศไทย”* ซึ่งสนับสนุนโดยสำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ โดยมีวัตถุประสงค์เพื่อส่งเสริมให้นักเรียนและนักศึกษาได้เรียนรู้และฝึกทักษะในการพัฒนาซอฟต์แวร์ ลิขสิทธิ์ของซอฟต์แวร์นี้จึงเป็นของผู้พัฒนา ซึ่งผู้พัฒนาได้อนุญาตให้สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติเผยแพร่ซอฟต์แวร์นี้ตาม “ต้นฉบับ” โดยไม่มีการแก้ไขดัดแปลงใด ๆ ทั้งสิ้น ให้แก่บุคคลทั่วไปได้ใช้เพื่อประโยชน์ส่วนบุคคลหรือประโยชน์ทางการศึกษาที่ไม่มีวัตถุประสงค์ในเชิงพาณิชย์ โดยไม่คิดค่าตอบแทนการใช้ซอฟต์แวร์ ดังนั้น สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติจึงไม่มีหน้าที่ในการดูแล บำรุงรักษา จัดการอบรมการใช้งาน หรือพัฒนาประสิทธิภาพซอฟต์แวร์ รวมทั้งไม่รับรองความถูกต้องหรือประสิทธิภาพการทำงานของซอฟต์แวร์ ตลอดจนไม่รับประกันความเสียหายต่าง ๆ อันเกิดจากการใช้ซอฟต์แวร์นี้ทั้งสิ้น

**License Agreement.** This software is a work developed by **Mr. Ronnachai Khaosa-ard** from **Burapha University** under the provision of **Dr. Watcharapong Yookwan** under the project *“Explainable Spatio-Temporal Graph Neural Network for PM2.5 Forecasting and Source Attribution in Northern Thailand”*, which has been supported by the National Science and Technology Development Agency (NSTDA), in order to encourage pupils and students to learn and practice their skills in developing software. Therefore, the intellectual property of this software shall belong to the developer and the developer gives NSTDA a permission to distribute this software as an “as is” and non-modified software for a temporary and non-exclusive use without remuneration to anyone for his or her own purpose or academic purpose, which are not commercial purposes. In this connection, NSTDA shall not be responsible to the user for taking care, maintaining, training, or developing the efficiency of this software. Moreover, NSTDA shall not be liable for any error, software efficiency and damages in connection with or arising out of the use of the software.

---

## กิตติกรรมประกาศ / Acknowledgements

โครงงานนี้เป็นส่วนหนึ่งของการแข่งขัน NSC 2026 (การแข่งขันพัฒนาโปรแกรมคอมพิวเตอร์แห่งชาติ ครั้งที่ 28)
ประเภทที่ 14 — โปรแกรมเพื่องานการพัฒนาด้านวิทยาศาสตร์และเทคโนโลยี ระดับนิสิต นักศึกษา
ภายใต้ธีม "Sustainability Innovation"

This is an NSC 2026 (28th National Software Contest, Thailand) entry,
Category 14 — Science and Technology Development Software, University level,
under the theme "Sustainability Innovation".

ข้อมูลอ้างอิง / References:
- Wang et al. (2020). PM2.5-GNN. https://arxiv.org/abs/2002.12898
- Wu et al. (2020). MTGNN (KDD 2020).
- PyG Temporal. https://pytorch-geometric-temporal.readthedocs.io/
- OpenAQ API v3. https://docs.openaq.org/
- NASA FIRMS API. https://firms.modaps.eosdis.nasa.gov/api/
- ERA5 via CDS API. https://cds.climate.copernicus.eu/how-to-api
- Country borders (`app/assets/borders_th_mm_la.geojson`): geoBoundaries gbOpen ADM0 simplified
  (data build 2023-12-12, wmgeolab/geoBoundaries@9469f09); boundary data © OpenStreetMap
  contributors — Thailand/Laos under ODbL 1.0, Myanmar under CC BY-SA 2.0 (the geojson file
  remains under these licenses). Runfola, D. et al. (2020). geoBoundaries: A global database of
  political administrative boundaries. PLoS ONE 15(4): e0231866.
  https://doi.org/10.1371/journal.pone.0231866
