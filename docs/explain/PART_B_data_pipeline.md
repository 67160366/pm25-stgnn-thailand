# PART B — DATA PIPELINE (เส้นทางของข้อมูล)

> อ้างอิงโค้ดจริง: `src/data/scrapers/{openaq,firms,era5}.py`, `preprocessing.py`, `loader.py`

## B1. แหล่งข้อมูล

| แหล่ง | ให้อะไร | Endpoint จริง | สถานะ |
|---|---|---|---|
| **OpenAQ v3** | PM2.5 รายชั่วโมง (แกนหลัก training) | `api.openaq.org/v3` | ★ primary |
| **NASA FIRMS** | จุดความร้อนไฟ (FRP, lat/lon) | `firms.modaps.eosdis.nasa.gov/api/area/csv` | ใช้จริง |
| **ERA5 (Copernicus CDS)** | ลม u10/v10, t2m, d2m, blh | `cdsapi` | Session 3 |
| **air4thai realtime** | ค่าสด ใช้ cross-check | `getNewAQI_JSON.php` | live เท่านั้น |

**OpenAQ — endpoint + param จริง** (`openaq.py`):
- ค้นสถานี: `GET /v3/locations` → `iso=TH`, `parameters_id=2` (2 = PM2.5), `bbox="97.0,16.0,101.5,21.0"`, `limit=100`, `page`
- ดึงข้อมูลย้อนหลัง: `GET /v3/sensors/{sensor_id}/measurements` (ไม่ใช่ `/hours`) → `datetime_from/to` เป็น ISO 8601 มี `Z` เสมอ (`strftime("%Y-%m-%dT%H:%M:%SZ")`), `limit=1000`

**FIRMS — endpoint จริง** (`firms.py`):
`{BASE}/{api_key}/{source}/{bbox}/{day_range}[/{date}]` — bbox เรียง **lon,lat** (`f"{lon_w},{lat_s},{lon_e},{lat_n}"`), source `VIIRS_NOAA20_NRT` / `_SP`

**Quirk / limit ที่ต้องระวัง (จากโค้ดจริง):**
| แหล่ง | ปัญหา | วิธีจัดการในโค้ด |
|---|---|---|
| OpenAQ | response อาจเป็น UTF-16+BOM | `response.encoding = "utf-8-sig"` |
| OpenAQ | ช่วงยาว → HTTP 408 timeout | `backfill_station` แบ่งเป็น **7-วัน chunk** + `time.sleep(1)` |
| OpenAQ | rate/error ชั่วคราว | `tenacity` retry 3 ครั้งบน 408/429/500/502/503/504, exponential backoff |
| FIRMS | `day_range` NRT ≤ 10, **SP ≤ 5** | `_NRT_WINDOW_DAYS=10`, `_SP_WINDOW_DAYS=5` chunk อัตโนมัติ |
| FIRMS | SP ล่าช้า ~60 วัน; NRT ย้อนได้ ~10 วัน | `fetch_hotspots_hybrid` = SP (อดีต) + NRT (ล่าสุด) |
| FIRMS | ช่องว่าง ~48 วันระหว่าง SP↔NRT | **เติมไม่ได้** (ข้อจำกัด NASA) — log WARNING |
| air4thai history | window ~90 วัน | ไม่ใช้ train ใช้แค่ cross-validate |

## B2. Station Selection (`curate_training_stations` + DESIGN §3)

เกณฑ์คัด (โค้ดจริงใน `openaq.py`):
```python
mask = (
    (df_locations["provider"] == "Air4Thai")
    & (datetime_first < 2022-01-01)   # มีข้อมูลก่อนเริ่ม train
    & (datetime_last  > 2026-01-01)   # ยังส่งข้อมูลหลังจบ test
)
```
+ อยู่ใน bbox lon 97.0–101.5, lat 16.0–21.0 + parameter = PM2.5

**ผล: 18 สถานี** (15 core + 3 extended) ครอบคลุมครบ 9 จังหวัดภาคเหนือ
ทำไม 18 ไม่ใช่จำนวนอื่น? = จำนวน station Air4Thai ที่ "ผ่านทุกเกณฑ์" พอดี
(ต้องมีประวัติต่อเนื่องคลุมทั้ง train+val+test 2022–2025) — ไม่ได้เลือกตัวเลข แต่เป็นผลจากเกณฑ์คุณภาพข้อมูล

## B3. Data Flow (ต้นจนจบ)

```
OpenAQ /v3        FIRMS area/csv
   │                   │
   ▼ openaq.py         ▼ firms.py
data/raw/openaq/     data/raw/firms/
sensor_{id}_{yr}       firms_*.csv
.parquet                   │
   │                       ▼ hotspot_clustering.py
   │                  hotspots.parquet (centroid_lat/lon, total_frp, date, country)
   ▼ preprocessing.py (scripts/02)
data/processed/dataset.parquet  +  scalers.json
   │  (timestamp, station_id, pm25_raw, pm25_scaled, mask_in_loss,
   │   exclude_from_training, hour_sin/cos, doy_sin/cos [+ ERA5])
   ▼ loader.py — PM25GraphDataset
   │  pivot เป็น wide (T_full=35064, N=18) ต่อ feature
   │  sliding window: x=(N, T_in=24, F=10), y/mask=(N, H=4)
   ▼ graph_builder.build_graph / _assemble_graph
HeteroData {station.x, hotspot.x, type_a/b/c edges}
   ▼
   model (MTGNN / A3TGCN)
```

**Input/output แต่ละ step:**
| Step | ไฟล์ | input | output |
|---|---|---|---|
| Scrape PM2.5 | `openaq.py` | sensor_id, ช่วงเวลา | `sensor_{id}_{year}.parquet` (timestamp_utc, value…) |
| Scrape ไฟ | `firms.py` | bbox, day_range | CSV FIRMS ดิบ |
| Cluster ไฟ | `hotspot_clustering.py` | FIRMS CSV | `hotspots.parquet` (centroid, total_frp, date) |
| Preprocess | `preprocessing.py` | raw parquet + metadata | `dataset.parquet` + `scalers.json` |
| Load | `loader.py` | dataset+hotspots+metadata | `HeteroData` ต่อ sample |

**Feature array shape** (`loader.py`): `_FULL_INDEX` = 2022-01-01→2025-12-31 รายชั่วโมง = **35,064 steps**; pivot เป็น `(35064, 18)` ต่อ feature; หน้าต่าง input default `window_in=24` ชม., horizons `(6,12,24,48)` → x = `(18, 24, 10)`, y/mask = `(18, 4)`

## B4. Missing Data Policy (`preprocessing._apply_gap_policy`)

ใช้กับ **gap เดิม** (ก่อนเติม) ต่อสถานี:
| ความยาว gap | ทำอะไร | ทำไม | ค่าคงที่ในโค้ด |
|---|---|---|---|
| < 6 ชม. | linear interpolate | gap สั้น เติมเชิงเส้นแม่นพอ | `_GAP_INTERPOLATE_H=6` |
| 6–24 ชม. | forward-fill (จำกัดความยาว) | ยาวไป interpolate เพี้ยน; ค่าเดิมยังพอเชื่อ | `_GAP_FFILL_H=24` |
| > 24 ชม. | **ปล่อย NaN + `mask_in_loss=True`** | ไม่มโนค่า — ตัดออกจาก loss แทน | — |
| > 7 วันติด | **`exclude_from_training=True` ทั้งเดือน** | ข้อมูลเสียมากทั้งเดือน ไม่น่าเชื่อถือ | `_GAP_EXCLUDE_H=168` |

จุดสำคัญของโค้ด: interpolate/ffill อ่านจาก series **ต้นฉบับ** เพื่อไม่ให้ค่าที่เติมใน gap สั้นรั่วไป gap ข้างเคียง; target training ใช้ `pm25_scaled` (ไม่ใช่ raw) เพื่อกัน gradient explosion แต่ตรวจ validity ด้วย `pm25_raw` finite

**ถ้าข้ามขั้นนี้:** โมเดลจะ (1) เรียนค่าที่มโนขึ้น (imputed) เป็นความจริง, (2) loss ระเบิดจากค่า outlier ฤดูเผา >500 µg/m³, (3) `_build_anchor_index` เลือก sample ที่ target เป็น NaN → NaN ในการเทรน

## หมายเหตุ
- Scaler fit เฉพาะปี **2022–2023** (`_SCALER_FIT_END_YEAR=2023`) แล้วใช้พารามิเตอร์เดิมกับทุกปี — กัน data leakage
- ใช้ **RobustScaler** (median + IQR) ไม่ใช่ StandardScaler เพราะ PM2.5 มี outlier หนักช่วงเผา (รายละเอียด PART D)

---

# ภาคผนวก PART B (เจาะลึก) — Hotspot Clustering & ERA5

## B5. Hotspot Clustering (`hotspot_clustering.py`)

FIRMS ให้ **จุดไฟรายจุด** (แต่ละ pixel ดาวเทียม) ต้องยุบเป็น "กลุ่มไฟ" ก่อนเป็น node กราฟ
ใช้ **DBSCAN + haversine** รายวัน:

```python
_DBSCAN_EPS_KM = 25.0
_DBSCAN_EPS_RAD = 25.0 / 6371.0      # แปลง กม.→เรเดียน (haversine ต้องการเรเดียน)
_DBSCAN_MIN_SAMPLES = 2
labels = DBSCAN(eps=_DBSCAN_EPS_RAD, min_samples=2, metric="haversine").fit_predict(coords)
```

- **eps = 25 กม.** → จุดไฟห่างกัน ≤25 กม. รวมเป็นกลุ่มเดียว
- **min_samples = 2** → ต้องมี ≥2 จุดถึงเป็นคลัสเตอร์
- **จุดโดด (noise, label −1) ไม่ทิ้ง** — โค้ดตั้งใจแปลงให้เป็นคลัสเตอร์เดี่ยว ID ใหม่:
  ```python
  adjusted[labels == -1] = noise_ids  # กันข้อมูลไฟหายเงียบๆ
  ```
- ต่อคลัสเตอร์คำนวณ: `centroid_lat/lon` (mean), `total_frp` (**sum** ของ FRP), `point_count`, `country`

**Country attribution** — geocode ด้วย bbox แบบง่าย (priority Thailand > Myanmar > Laos > other):
```python
_COUNTRY_BBOXES = [
    ("Thailand", 97.3, 5.6, 105.7, 20.5),
    ("Myanmar",  92.2, 9.8, 101.2, 28.5),
    ("Laos",    100.1,13.9, 107.6, 22.5),
]
```
> ⚠️ ข้อจำกัดที่โค้ดยอมรับเอง: **ไม่ใช้ shapely** (ดู SESSION2_NOTES) → เขตชายแดนอาจจัดประเทศผิด
> `country` นี้แหละคือ field ที่ dashboard/attribution ใช้เล่าเรื่อง "ฝุ่นข้ามแดน" (PART E)

output: `hotspots.parquet` (date, cluster_id, centroid_lat/lon, total_frp, point_count, country)
loader อ่านแบบ index-by-date → sample ที่ anchor วันไหน หยิบไฟของวันนั้น (รายวัน ไม่ใช่รายชั่วโมง)

## B6. ERA5 (`scrapers/era5.py`) — Session 3

ดึง 5 ตัวแปรจาก CDS `reanalysis-era5-single-levels`:
`10m_u/v_component_of_wind`, `2m_temperature`, `2m_dewpoint_temperature`, `boundary_layer_height`
→ column `u10, v10, t2m, d2m, blh`

> 📌 **DESIGN.md ระบุ ERA5 เป็น u10/v10/t2m/rh (4 ตัว) แต่โค้ดจริงเป็น u10/v10/t2m/d2m/blh (5 ตัว)**
> — ใช้ d2m (dewpoint) + blh (boundary-layer height) แทน rh; DESIGN.md ตรงนี้ล้าสมัย ยึดโค้ด

**Quirk ที่โค้ดจัดการจริง (สำคัญมากบน Windows path ไทย):**
| ปัญหา | วิธีแก้ในโค้ด |
|---|---|
| netCDF4/h5py เปิดไฟล์ path ที่มีอักษรไทยไม่ได้ (Windows) | `_open_nc/_write_nc` copy ไป temp ASCII ก่อน |
| CDS v2 ปฏิเสธ request ทั้งปี (ใหญ่ไป) | `download_era5_year` ดึง **ทีละเดือน 12 ครั้ง** แล้ว merge |
| CDS v2 เปลี่ยนชื่อ coord (`valid_time`, `latitude`) | `_normalise_coords` rename → `time/lat/lon` |
| ชื่อ short-name ต่างเวอร์ชัน (u10/U10M/10u) | `_ERA5_ALIASES` ลองหลายชื่อ |
| grid ERA5 หยาบกว่าตำแหน่งสถานี | `interp(method="linear")` — **bilinear** ลงพิกัด 18 สถานี |

- t2m/d2m เก็บเป็น **เคลวิน** (native ERA5 ไม่แปลง)
- `load_wind_field(nc, dt)` → คืน `xr.DataArray` shape `(1, lat, lon, 2)` ป้อนเข้า `build_graph` (PART C)
- ในทางปฏิบัติ ERA5 ถูก interp ลงสถานีแล้ว เก็บใน `dataset.parquet` เป็นคอลัมน์ → `wind_mode="from_field"`
  ใน loader ใช้ค่า per-station ตรงๆ (ดู `_build_graph_full`)
