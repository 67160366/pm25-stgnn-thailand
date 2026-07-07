# PART D — FEATURES & MODEL INPUT

> อ้างอิงโค้ดจริง: `src/data/preprocessing.py`, `src/data/loader.py`, `src/models/{mtgnn,a3tgcn}.py`

## D1. Node Features

### Station node — feature อะไรบ้าง?

ต้องแยก 2 ระดับให้ชัด (จุดที่คนสับสน):

| ระดับ | ที่ไหน | มิติ | feature |
|---|---|---|---|
| `build_graph` default | `_DEFAULT_CONFIG['station_features']` | **5** | `[pm25_scaled, hour_sin, hour_cos, doy_sin, doy_cos]` |
| **model input จริง** | loader `_FEATURE_COLS` → `data['station'].x` | **10** | 5 ข้างบน **+** `[u10, v10, t2m, d2m, blh]` (ERA5) |

`build_graph` ตั้ง `data['station'].x` ด้วย 5 คอลัมน์เป็น placeholder แต่ตอนเทรน **loader เขียนทับ**
ด้วย tensor เต็มรูป `(N, T_in, F)` (`loader.py:524`):
```python
data["station"].x = torch.from_numpy(x_ntf).contiguous().float()   # (18, 24, 10)
```
โดย `x_ntf` มาจากการ stack 10 array ตาม `_FEATURE_COLS` (`loader.py:476`):
```python
x_features = np.stack([
    self._pm25_scaled[sl], self._hour_sin[sl], self._hour_cos[sl],
    self._doy_sin[sl], self._doy_cos[sl],
    self._u10[sl], self._v10[sl], self._t2m[sl], self._d2m[sl], self._blh[sl],
], axis=-1)                                    # (T_in, N, F)
x_ntf = np.nan_to_num(x_features.transpose(1, 0, 2), nan=0.0)   # (N, T_in, F=10)
```

**ความหมายแต่ละ feature:**

| feature | ความหมาย | ที่มา |
|---|---|---|
| `pm25_scaled` | PM2.5 หลัง RobustScaler (แกนหลัก) | preprocessing `_normalize` |
| `hour_sin/cos` | ชั่วโมงในวัน encode เป็นวงกลม | `_add_cyclic_features` |
| `doy_sin/cos` | วันในปี (ฤดูกาล) encode เป็นวงกลม | `_add_cyclic_features` |
| `u10` / `v10` | ลม 10 ม. ตะวันออก / เหนือ (m/s) | ERA5 |
| `t2m` / `d2m` | อุณหภูมิ / จุดน้ำค้าง 2 ม. (**เคลวิน**) | ERA5 |
| `blh` | ความสูงชั้นผสมของบรรยากาศ (m) | ERA5 |

> `nan_to_num(..., nan=0.0)` เติม 0 ให้ค่าว่าง — สำคัญเพราะ gap policy ปล่อย NaN ไว้ (PART B)
> โมเดลรับ NaN ไม่ได้ แต่ target ยังถูกกันด้วย `mask` แยกต่างหาก จึงไม่เรียนค่า 0 ปลอมเป็นความจริง

### Hotspot node — feature อะไรบ้าง?

3 มิติ (`graph_builder.py:409` และ loader `_assemble_graph`):
```python
hotspot_x = df_hotspots[["total_frp", "centroid_lat", "centroid_lon"]]   # (N_hotspot, 3)
```
- `total_frp` = ผลรวม Fire Radiative Power ของคลัสเตอร์ (ความแรงไฟ)
- `centroid_lat/lon` = พิกัดศูนย์กลางกลุ่มไฟ
- ถ้าไม่มีไฟวันนั้น → `(0, 3)` (node ว่าง) + `data['hotspot'].country = []`

### ทำไม sin/cos encoding? (`_add_cyclic_features` docstring)

> "Cyclic encoding maps periodic features onto a unit circle so the model sees
> continuity across midnight (hour 23 → 0) and year boundaries."

ปัญหาถ้า encode ชั่วโมงเป็นเลข 0–23 ตรงๆ: โมเดลจะเห็น **23:00 กับ 00:00 ห่างกัน 23 หน่วย**
ทั้งที่จริงห่างกันแค่ 1 ชั่วโมง → เกิด "รอยขาด" ที่เที่ยงคืน/สิ้นปี
```python
df["hour_sin"] = np.sin(2 * math.pi * hour / 24)
df["hour_cos"] = np.cos(2 * math.pi * hour / 24)
```
แมปลงวงกลมหนึ่งหน่วย → 23:00 กับ 00:00 อยู่ติดกันบนวง ระยะใกล้กันจริง โมเดลเรียน pattern
รายวัน (peak ฝุ่นเช้า/เย็น) และรายฤดู (ฤดูเผา ก.พ.–เม.ย.) ได้ต่อเนื่อง ต้องใช้ **คู่ sin+cos**
เพราะค่าเดียวไม่ unique (sin ค่าเดียวตรงกับ 2 เวลา)

### RobustScaler ทำไมไม่ใช้ StandardScaler? (`_fit_scalers` docstring)

> "Robust scaling (median + IQR) is preferred over z-score because PM2.5 has
> heavy outliers during burning season (>500 µg/m³)."

```python
center = np.median(vals)                    # แทน mean
scale  = np.percentile(vals,75) - np.percentile(vals,25)   # IQR แทน std
# pm25_scaled = (raw - center) / scale
```
- **StandardScaler** ใช้ mean/std ซึ่งถูก outlier ฤดูเผา (>500) ดึงเบี้ยว → ค่าปกติถูกบีบจนโมเดลแยกไม่ออก
- **RobustScaler** ใช้ median/IQR ทนต่อ outlier — ค่าสุดโต่งไม่ทำให้สเกลของวันปกติพัง
- fit เฉพาะ **2022–2023** (`_SCALER_FIT_END_YEAR`) แล้วใช้พารามิเตอร์เดิมกับ val/test → กัน data leakage
- guard: `scale==0` (สัญญาณคงที่) → ตั้ง `scale_=1` กันหารศูนย์; สถานีที่ไม่มีข้อมูลเลย → `center_=0, scale_=1`

## D2. Edge Attributes

**แต่ละ edge type มี attribute อะไร** (สร้างใน `graph_builder.py`, ทวนจาก PART C):

| edge | `edge_attr` shape | คอลัมน์ |
|---|---|---|
| Type A | `(E_a, 1)` | `[w]` |
| Type B | `(E_b, 3)` | `[w, alignment, wind_speed]` |
| Type C | `(E_c, 3)` | `[w, alignment, frp]` |

**ถูกใช้ใน model อย่างไร — จุดสำคัญที่ต้องซื่อสัตย์:**
โมเดลใช้ **เฉพาะคอลัมน์ 0 (`w`)** เป็น `edge_weight` ในการ message-passing เท่านั้น
คอลัมน์ `alignment`, `wind_speed`, `frp` (คอลัมน์ 1–2) **ไม่ถูกป้อนเข้าโมเดล** — ถูกพกไว้เพื่อ
inspect/อธิบาย (XAI, dashboard) เท่านั้น

MTGNN (`mtgnn.py:342–353`) — ใช้ครบทั้ง 3 edge types:
```python
ew_a = data["station","type_a","station"].edge_attr[:, 0].contiguous()   # (E_a,)
ew_b = data["station","type_b","station"].edge_attr[:, 0].contiguous()   # (E_b,)
ew_c = data["hotspot","type_c","station"].edge_attr[:, 0].contiguous()   # (E_c,)
```
A3TGCN (`a3tgcn.py:108`) — **ใช้แค่ Type A** (spatial ล้วน ไม่รับ Type B/C):
```python
ew = data["station","type_a","station"].edge_attr[:, 0].contiguous()   # (E_a,)
h = self.encoder(X=x, edge_index=ei, edge_weight=ew)
```
→ นัยสำคัญ: A3TGCN ไม่ได้รับข้อมูลลม/ไฟผ่านกราฟเลย เป็น baseline ที่เบากว่า MTGNN โดยตั้งใจ

## D3. Feature ที่ยังขาด

**ERA5 5 คอลัมน์ (`u10, v10, t2m, d2m, blh`) คือส่วนที่ "ขาด" ตอนยังไม่รัน Session 3**
loader ตรวจและเติม 0 แทน (`loader.py:395–406`):
```python
_era5_missing = [f for f in ("u10","v10","t2m","d2m","blh") if f not in wide]
...
self._u10 = wide.get("u10", _zero)      # ถ้าไม่มีคอลัมน์ → array ศูนย์
self._v10 = wide.get("v10", _zero)
...
```
เพราะ `preprocess()` docstring ระบุเอง: *"Final DataFrame matching DESIGN.md §6.1 schema
(minus ERA5 weather columns, which are added in Session 3)."*

**ผลตอนยังไม่มี ERA5:**
- feature 5 มิติ (index 5–9 ของ x) เป็น 0 ทั้งหมด → โมเดลไม่มีสัญญาณลม/อุณหภูมิใน node feature
- Type B/C ยังสร้างได้ด้วย **ลมจำลอง `constant_ne` (3.5, 3.5)** (PART C) แต่ไม่ใช่ลมจริง

**เมื่อ ERA5 มาใน Session 3 จะเพิ่มอะไร:**
1. เติมค่าจริงใน 5 คอลัมน์ node feature (index 5–9)
2. สลับ graph ไปใช้ลมจริง: `wind_mode='from_field'` → loader แปลงเป็น `from_arrays`
   ใช้ค่า ERA5 ที่ interp ลง 18 สถานีล่วงหน้า (`loader.py:625–629`) สร้าง Type B/C ด้วยลมจริงต่อ timestep

> ⚠️ หมายเหตุจาก PART A/B: แม้เติม ERA5 จริงแล้ว ผลลัพธ์ก็ยังไม่ได้ช่วยความแม่นยำอย่างมั่นคง
> (ดู CRITICAL_REVIEW) — "graph does not robustly help accuracy" (จะขยายใน PART E)

---

## สรุป PART D

- **Station node feature = 10 มิติ** (loader เขียนทับ default 5 ของ build_graph):
  `pm25_scaled` + cyclic 4 (hour/doy sin·cos) + ERA5 5 (u10,v10,t2m,d2m,blh)
- **Hotspot node feature = 3 มิติ**: `total_frp, centroid_lat, centroid_lon`
- **sin/cos** ทำให้เวลาต่อเนื่องข้ามเที่ยงคืน/สิ้นปี; **RobustScaler** (median/IQR) ทน outlier
  ฤดูเผา ที่ StandardScaler แพ้
- **edge_attr**: โมเดลใช้แค่คอลัมน์ weight; alignment/wind_speed/frp เก็บไว้อธิบายเท่านั้น
  A3TGCN ใช้แค่ Type A, MTGNN ใช้ครบ 3 type
- **feature ที่ขาด** = ERA5 5 คอลัมน์ (เติม 0 จนกว่า Session 3) และลมจริง

> **ต้องการลงลึก PART ไหน หรือไปต่อ PART E (Output & Results)?**
