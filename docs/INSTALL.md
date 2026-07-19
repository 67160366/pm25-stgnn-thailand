# คู่มือการติดตั้ง (Installation Guide)

โครงการ: **ระบบพยากรณ์ฝุ่นละออง PM2.5 และวิเคราะห์แหล่งกำเนิดด้วย Explainable STGNN สำหรับภาคเหนือของประเทศไทย**
(NSC 2026 หมวด 14) · repo: https://github.com/67160366/pm25-stgnn-thailand

> เอกสารนี้เป็นคู่มือการติดตั้งอย่างละเอียด (ภาคผนวกของรายงานฉบับสมบูรณ์) สำหรับวิธีใช้งานหลังติดตั้งเสร็จ ดู `docs/USER_GUIDE.md`

---

## 1. ความต้องการของระบบ (Prerequisites)

| รายการ | รายละเอียด |
|---|---|
| ระบบปฏิบัติการ | Windows 10/11 (หลัก) หรือ Linux / macOS |
| Python | 3.11 (รองรับ 3.11–3.12) |
| ตัวจัดการแพ็กเกจ | [uv](https://github.com/astral-sh/uv) |
| Git | สำหรับโคลนซอร์สโค้ด |
| GPU (ทางเลือก) | NVIDIA GPU + CUDA 12.6 สำหรับการเทรน — การ inference และ dashboard รันบน CPU ได้ |
| เนื้อที่ดิสก์ | ~10 GB (torch + CUDA wheels + ข้อมูล) |

---

## 2. ขั้นตอนการติดตั้ง

### ขั้นที่ 1 — โคลนซอร์สโค้ด

```bash
git clone https://github.com/67160366/pm25-stgnn-thailand
cd pm25-stgnn-thailand
```

### ขั้นที่ 2 — ติดตั้ง dependencies หลัก

```bash
uv sync
```

คำสั่งนี้ติดตั้ง `torch==2.6.0` (จากดัชนี wheel ของ PyTorch CUDA 12.6) พร้อม dependencies แบบ pure-Python ทั้งหมด

### ขั้นที่ 3 — ติดตั้ง native PyG extensions

แพ็กเกจ `torch-scatter`, `torch-sparse`, และ `torch-geometric-temporal` เป็น extension ที่มี native C++/CUDA code
และ **ต้องการให้ `torch` ถูกติดตั้งก่อน** จึงรวมไว้ใน `pyproject.toml` ไม่ได้ (เพราะ uv resolve dependencies ทั้งหมดก่อนติดตั้ง)

**Windows:**

```powershell
./install_native_deps.ps1
```

**Linux / macOS** (ยังไม่มีสคริปต์ `.sh` — รันคำสั่งเทียบเท่า ซึ่งเป็นสิ่งที่สคริปต์ `.ps1` ทำ):

```bash
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.','') if torch.cuda.is_available() else 'cpu')")
pip install torch-scatter torch-sparse --find-links "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
pip install "torch-geometric-temporal>=0.54"
```

### ขั้นที่ 4 — (Windows + พาธที่มีอักษรไทย) แก้ปัญหา editable `.pth` crash ⚠️

หาก clone โปรเจกต์ไว้ในพาธที่มีอักษรไทย (เช่น `D:\งาน\...`) `uv sync` จะเขียน **พาธสัมบูรณ์ (มีอักษรไทย)**
ลงไฟล์ `.venv/Lib/site-packages/_editable_impl_pm25_stgnn_thailand.pth` ทำให้ `site.py` ของ Python อ่านไฟล์นี้
ด้วย encoding cp874 ไม่ได้ → เกิด `UnicodeDecodeError` ตอน import (อาการ: `black`/`python` พัง แต่ `ruff` ยังทำงาน)

แก้ได้ 2 วิธี:

**(ก) อัตโนมัติ** — สคริปต์ `install_native_deps.ps1` ตั้ง `PYTHONUTF8=1` เป็น User environment variable ให้แล้ว
**ต้องเปิด terminal ใหม่** หลังรันสคริปต์เพื่อให้มีผล

**(ข) สำรอง** — หากยัง crash ให้แก้ไฟล์ `.pth` ให้เหลือพาธสัมพัทธ์ แล้วรันทุกคำสั่งด้วย `UV_NO_SYNC=1 uv run ...`
(เพื่อกัน uv เขียนทับไฟล์ `.pth` ตอน sync ครั้งถัดไป):

```bash
printf '../../..\n' > .venv/Lib/site-packages/_editable_impl_pm25_stgnn_thailand.pth
```

> หมายเหตุ: ทุกครั้งที่รัน `uv sync` / `uv add` / แก้ `pyproject.toml` uv จะเขียนไฟล์ `.pth` ทับใหม่
> ให้ทำซ้ำวิธี (ข) หรือพึ่ง `PYTHONUTF8=1` จากวิธี (ก)

### ขั้นที่ 5 — ตั้งค่า API keys (เฉพาะกรณีจะดาวน์โหลดข้อมูลใหม่)

> dashboard และการประเมินผลใช้ **ข้อมูลที่เตรียมไว้แล้ว** จึง **ไม่ต้องใช้ API key**
> ขั้นตอนนี้จำเป็นเฉพาะเมื่อจะรัน `scripts/01_download_all.py` เพื่อดาวน์โหลดข้อมูลดิบใหม่เท่านั้น

สร้างไฟล์ `.env` ที่รากของโปรเจกต์ (ไฟล์นี้ถูก gitignore ไว้):

```
OPENAQ_API_KEY=<API key จาก https://docs.openaq.org/>
FIRMS_API_KEY=<MAP_KEY จาก https://firms.modaps.eosdis.nasa.gov/api/>
TELEGRAM_BOT_TOKEN=<token จาก @BotFather — เฉพาะกรณีใช้ scripts/16_telegram_alert.py>
TELEGRAM_CHAT_ID=<chat id ปลายทาง — เฉพาะกรณีใช้ scripts/16_telegram_alert.py>
```

สำหรับ ERA5 (Copernicus): วางไฟล์ `~/.cdsapirc` ที่มี UID และ API key ตามคู่มือ
https://cds.climate.copernicus.eu/how-to-api

**สำหรับ Telegram Bot (การแจ้งเตือนล่วงหน้า 48 ชม.):** `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`
จำเป็นเฉพาะเมื่อจะรัน `scripts/16_telegram_alert.py` แบบส่งจริง (ไม่ใช้กับ dashboard หลัก และไม่จำเป็นเมื่อใช้ `--dry-run`)

1. เปิดแชทกับ [@BotFather](https://t.me/BotFather) ใน Telegram แล้วพิมพ์ `/newbot` ทำตามขั้นตอน
   (ตั้งชื่อบอท) จะได้ **token** รูปแบบ `123456789:AAExampleTokenString` — นี่คือค่า `TELEGRAM_BOT_TOKEN`
2. หา **chat id**:
   - ส่งข้อความใด ๆ ให้บอทที่สร้างไว้ก่อน (ต้องกด Start หรือพิมพ์อะไรก็ได้ในแชทกับบอท)
   - เปิด `https://api.telegram.org/bot<TOKEN>/getUpdates` ในเบราว์เซอร์ (แทน `<TOKEN>` ด้วย token จริง)
   - หาค่า `"chat":{"id": ...}` ในผลลัพธ์ JSON — ตัวเลขนั้นคือ `TELEGRAM_CHAT_ID`
   - หากต้องการส่งเข้ากลุ่ม ให้เชิญบอทเข้ากลุ่มก่อน แล้ว chat id ของกลุ่มจะเป็นเลขติดลบ
3. ทดสอบไม่ส่งจริงก่อน: `uv run python scripts/16_telegram_alert.py --dry-run`

---

## 3. ตรวจสอบการติดตั้ง (Verify)

```bash
uv run pytest tests/ -q          # ควรผ่านครบ 397 tests
```

หรือเปิด dashboard (ดูรายละเอียดใน `docs/USER_GUIDE.md`):

```bash
UV_NO_SYNC=1 uv run streamlit run app/streamlit_app.py
```

---

## 4. หมายเหตุเรื่องข้อมูลและโมเดล (สำคัญ)

โฟลเดอร์ `data/` และ `checkpoints/` **ไม่ได้อยู่ใน repository** (gitignore ไว้เพราะมีขนาดใหญ่)
เพื่อให้ระบบทำงานได้เต็มรูปแบบ เลือกอย่างใดอย่างหนึ่ง:

- **รับชุดข้อมูล + checkpoint ที่เตรียมไว้** จากผู้พัฒนา แล้ววางไว้ที่ `data/processed/` และ `checkpoints/mtgnn/`
- **หรือสร้างใหม่จาก pipeline** (ต้องมี API keys ในขั้นที่ 5): รัน `scripts/01` → `02` → `03`
  ตามลำดับ (ดู `docs/USER_GUIDE.md` ส่วนที่ 2)

---

## 5. ปัญหาที่พบบ่อย (Troubleshooting)

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `UnicodeDecodeError` ตอน import / `black` พังแต่ `ruff` ทำงาน | ปัญหาไฟล์ `.pth` พาธไทย → ทำขั้นที่ 4 |
| `ImportError: torch_scatter` / `torch_sparse` | ยังไม่ได้ติดตั้ง native extensions → ทำขั้นที่ 3 ซ้ำ |
| `cdsapi authentication failed` | ยังไม่ได้ตั้ง `~/.cdsapirc` → ทำขั้นที่ 5 (ERA5) |
| dashboard ขึ้น error หาไฟล์ข้อมูลไม่เจอ | ยังไม่มี `data/processed/` หรือ `checkpoints/` → ดูข้อ 4 |
