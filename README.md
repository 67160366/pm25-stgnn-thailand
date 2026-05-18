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

**Linux / macOS:**

```bash
./install_native_deps.sh
```

### ตัวแปรสภาพแวดล้อม / Environment variables

คัดลอก `env.example` เป็น `.env` แล้วกรอก API keys ที่จำเป็น

Copy `env.example` to `.env` and fill in the required API keys.

```bash
cp env.example .env
```

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

```
[TODO: NSC Disclaimer Thai + English]
```

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
