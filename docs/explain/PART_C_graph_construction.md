# PART C — GRAPH CONSTRUCTION (หัวใจของ STGNN)

> อ้างอิงโค้ดจริงทั้งหมด: `src/data/graph_builder.py` (+ `loader.py` สำหรับ wind_mode)
> ค่า default ทุกตัวมาจาก `_DEFAULT_CONFIG` และ DESIGN.md §5.2

## C1. Graph คืออะไรในโปรเจคนี้

กราฟถูกสร้าง **ต่อ 1 timestep** (1 ชั่วโมง) โดยฟังก์ชัน `build_graph(...)` → คืน `HeteroData`
การซ้อนเป็น time-series เป็นหน้าที่ของ loader (PART B) ไม่ใช่ของไฟล์นี้

**Node — มี 2 ประเภท (heterogeneous):**

| ประเภท node | คืออะไร | จำนวน | feature (`data[...].x`) |
|---|---|---|---|
| `station` | สถานีวัด PM2.5 | 18 (fixed) | `[pm25_scaled, hour_sin, hour_cos, doy_sin, doy_cos]` |
| `hotspot` | กลุ่มจุดไฟ (DBSCAN cluster) ของ**วันนั้น** | ผันแปรรายวัน (0..N) | `[total_frp, centroid_lat, centroid_lon]` |

```python
data["station"].x = station_x       # (18, 5)
data["hotspot"].x = hotspot_x       # (N_hotspot, 3) หรือ (0,3) ถ้าไม่มีไฟ
```

**Edge — มี 3 ประเภท** (Type A / B / C) — รายละเอียด C2

**ทำไมต้อง heterogeneous graph?**
- node สองชนิดนี้ "คนละธรรมชาติกัน": สถานีคือ *ตัวรับ/ตัววัด* ฝุ่น มี time-series; hotspot คือ *แหล่งปล่อย* ฝุ่น มีแค่ FRP+พิกัดรายวัน
- feature dimension ต่างกัน (5 vs 3) และความสัมพันธ์ก็คนละทิศทาง (ไฟ→สถานี ทางเดียว) — homogeneous graph บังคับ node ทุกตัวมี feature เดียวกัน ทำแบบนี้ไม่ได้
- PyG `HeteroData` เก็บ edge เป็น triple `(src_type, relation, dst_type)` จึงแยก 3 ความสัมพันธ์ออกจากกันสะอาด ให้โมเดล MTGNN/A3TGCN เรียน message-passing แต่ละชนิดแยกกัน

## C2. Edge Types ทั้ง 3 แบบ

ตารางสรุป (ทุกค่าจากโค้ด `_build_type_*` + `_DEFAULT_CONFIG`):

| | **Type A** (static spatial) | **Type B** (wind-aware dynamic) | **Type C** (hotspot influence) |
|---|---|---|---|
| เชื่อม | station ↔ station | station → station | hotspot → station |
| triple | `('station','type_a','station')` | `('station','type_b','station')` | `('hotspot','type_c','station')` |
| เงื่อนไขระยะ | `d ≤ 100 km` | `d ≤ 200 km` | `d ≤ 500 km` |
| เงื่อนไขลม | — (ไม่ใช้ลม) | `alignment > 0.3` | `alignment > 0.4` |
| ทิศทาง | **undirected** (ใส่ทั้ง i→j และ j→i) | **directed** (i→j เมื่อลมพัดจาก i ไป j) | **directed** (ไฟ→สถานี) |
| สูตร weight | `exp(-d/50)` | `alignment · wind_speed · exp(-d/100)` | `alignment · total_frp · exp(-d/200)` |
| `edge_attr` shape | `[E, 1]` = `[w]` | `[E, 3]` = `[w, alignment, wind_speed]` | `[E, 3]` = `[w, alignment, frp]` |

**Type A — static spatial** (`_build_type_a_edges`)
```python
for i in range(n):
    for j in range(i + 1, n):          # คู่ที่ไม่ซ้ำ
        d = _haversine_km(...)
        if d <= max_km:                 # 100 km
            w = math.exp(-d / 50.0)
            src_idx += [i, j]; tgt_idx += [j, i]   # ใส่สองทิศ → undirected
            weights += [w, w]
```
- ตัวแปรในสูตร: `d` = ระยะ great-circle (km), `50` = length-scale (km) ยิ่งไกลยิ่งลด weight เร็ว
- undirected เพราะ "ความใกล้ทางภูมิศาสตร์" สมมาตร — A ใกล้ B ก็แปลว่า B ใกล้ A

**Type B — wind-aware dynamic** (`_build_type_b_edges`) — **จุดขาย novelty #1**
```python
for i in range(n):
    u, v = _get_wind_uv(wind_field, config, lats[i], lons[i])
    wind_speed = math.sqrt(u**2 + v**2)
    for j in range(n):
        if i == j: continue
        d = _haversine_km(...)
        if d > max_km: continue                    # 200 km
        alignment = _wind_alignment(u, v, ...)
        if alignment <= min_align: continue        # 0.3
        w = alignment * wind_speed * math.exp(-d / 100.0)
        attrs.append([w, alignment, wind_speed])
```
- ตัวแปร: `alignment` = cos(θ) ความสอดคล้องทิศลม (ดู C3), `wind_speed` = |ลม| m/s, `100` = length-scale
- directed เพราะ **ลมพัดทางเดียว**: ถ้าลมพัด i→j ฝุ่นไหลจาก i ไป j ไม่ใช่ทางกลับ → edge j→i จะถูกสร้างก็ต่อเมื่อ alignment ของทิศตรงข้ามผ่าน threshold ด้วย (ปกติไม่ผ่านเพราะ cos กลับเครื่องหมาย)

**Type C — hotspot influence** (`_build_type_c_edges`) — **จุดขาย novelty #2**
```python
for h_idx in range(len(df_hotspots)):
    u, v = _get_wind_uv(wind_field, config, h_lats[h_idx], h_lons[h_idx])
    for s_idx in range(len(df_stations)):
        d = _haversine_km(...)
        if d > max_km: continue                    # 500 km (ไกลสุด เพราะไฟข้ามแดน)
        alignment = _wind_alignment(u, v, ...)
        if alignment <= min_align: continue        # 0.4 (เข้มกว่า B)
        w = alignment * frp * math.exp(-d / 200.0)
        attrs.append([w, alignment, frp])
```
- ตัวแปร: `frp` = Fire Radiative Power (ความแรงไฟ) แทนที่ `wind_speed` ของ Type B → ไฟแรง = อิทธิพลมาก
- `edge_index` row 0 = index ของ hotspot, row 1 = index ของ station (ทิศ ไฟ→สถานี เท่านั้น — สถานีไม่ส่งอะไรกลับไฟ)
- ระยะ 500 km และ threshold 0.4 เข้มกว่า Type B: เพราะ Type C คือเรื่อง **ฝุ่นข้ามพรมแดน** (ไฟพม่า/ลาว) ต้องมั่นใจว่าลมพัดตรงเข้าหาสถานีจริงๆ ถึงจะนับ

**หมายเหตุ empty case:** ทุก builder คืน `edge_index` shape `(2,0)` และ `edge_attr` shape ที่ถูกต้อง (`(0,1)` หรือ `(0,3)`) เมื่อไม่มี edge — กันไม่ให้ downstream พังเรื่อง shape; Type C คืนว่างทันทีถ้า `df_hotspots.empty`

## C3. Wind Alignment (`_wind_alignment` + `_bearing_rad`)

```python
def _wind_alignment(u, v, src_lat, src_lon, tgt_lat, tgt_lon):
    bearing = _bearing_rad(src_lat, src_lon, tgt_lat, tgt_lon)  # ทิศจาก src→tgt
    wind_dir = math.atan2(u, v)          # ทิศที่ลม "พัดไปหา" (0=เหนือ, π/2=ตะวันออก)
    return math.cos(bearing - wind_dir)  # cos(θ) ของมุมต่าง
```

**`alignment = cos(θ)` หมายความว่าอะไร?**
θ = มุมระหว่าง (ทิศจาก source ไป target) กับ (ทิศที่ลมกำลังพัดไป)

| ค่า | มุม θ | ความหมาย | จะเกิด edge ไหม |
|---|---|---|---|
| **+1** | 0° | ลมพัด**ตรง**จาก source → target พอดี (ฝุ่นไหลถึงเต็มๆ) | ✅ ผ่านแน่ |
| **0** | 90° | ลมพัด**ตั้งฉาก** ไม่พาฝุ่นไปทาง target | ❌ (≤0.3/0.4) |
| **−1** | 180° | ลมพัด**สวนทาง** (ฝุ่นไหลออกจาก target) | ❌ |

**ตัวอย่างจริงจากค่า default `constant_ne` (u=3.5, v=3.5):**
- `wind_dir = atan2(3.5, 3.5) = 45°` = ลมพัดไปทาง **ตะวันออกเฉียงเหนือ (NE)**
- สถานี source ที่อยู่ทาง SW ของ target (target อยู่ทาง NE) → bearing ≈ 45° → `cos(45°−45°)=1` → เกิด edge
- ถ้า target อยู่ทางใต้ (bearing 180°) → `cos(180°−45°)=cos(135°)≈−0.71` → ไม่เกิด edge (ฝุ่นไม่ไปทางนั้น)

> convention สำคัญ (ตรงกับ ERA5/ECMWF): `u` = องค์ประกอบไปทางตะวันออก, `v` = ไปทางเหนือ,
> และ `wind_dir` คือทิศที่ลม **พัดไปหา** (blowing-to) ไม่ใช่ทิศที่ลมพัดมาจาก

**ทำไม threshold 0.3 (Type B) และ 0.4 (Type C)?**
- ต้อง `> 0` เพื่อตัด edge ที่ลมตั้งฉาก/สวนทางออกทั้งหมด (ฝุ่นไม่ไหลไปทางนั้น)
- ยกให้ **0.3** = ยอมรับเฉพาะลมที่พัดไปทาง target ภายใน ~72° (cos⁻¹0.3) → กรอง edge อ่อนๆ ทิ้ง ไม่ให้กราฟรก
- Type C เข้มกว่าเป็น **0.4** (~66°) เพราะระยะไกลถึง 500 km ความผิดพลาดของการประมาณทิศลมสะสมมาก จึงต้องมั่นใจทิศมากกว่า

**หมายเหตุ bearing:** ใช้ equirectangular approximation `atan2(dlon·cos(src_lat), dlat)` — docstring ระบุว่าที่ระยะสุดของ bbox (~500 km, lat 16–21°) คลาดเคลื่อน <1.5° = alignment error <0.02 ซึ่งอยู่ในเกณฑ์ยอมรับ

## C4. `build_graph` API

```python
def build_graph(
    df_stations: pd.DataFrame,        # ต้องมี station_id, lat, lon + คอลัมน์ feature
    df_hotspots: pd.DataFrame,        # centroid_lat/lon, total_frp (empty ได้)
    wind_field: xr.DataArray | None,  # (time, lat, lon, component); None ได้ถ้าใช้ลมสังเคราะห์
    config: dict[str, Any],
) -> HeteroData
```

**Parameter แต่ละตัว:**

| param | รับอะไร | format/shape | หมายเหตุ |
|---|---|---|---|
| `df_stations` | ตารางสถานี | 18 แถว; **ลำดับแถว = node index** | อ่าน feature จาก `config['station_features']` เท่าที่มีคอลัมน์จริง |
| `df_hotspots` | กลุ่มไฟของวันเป้าหมาย | N แถว หรือว่าง | ว่าง = วันไม่มีไฟ → ไม่มี hotspot node/Type C |
| `wind_field` | สนามลม ERA5 | `(time, lat, lon, 2)` comp0=u comp1=v | ใช้เฉพาะ `wind_mode='from_field'` |
| `config` | dict override | คีย์ตาม `_DEFAULT_CONFIG` | merge `{**_DEFAULT_CONFIG, **config}` — คีย์ที่ไม่ตั้ง fallback default |

**Return — `HeteroData` มีอะไรบ้าง:**
```
data['station'].x                                → (18, 5)
data['hotspot'].x                                → (N, 3) หรือ (0, 3)
data['station','type_a','station'].edge_index    → (2, E_a)  .edge_attr (E_a, 1)
data['station','type_b','station'].edge_index    → (2, E_b)  .edge_attr (E_b, 3)
data['hotspot','type_c','station'].edge_index    → (2, E_c)  .edge_attr (E_c, 3)
```

**wind_mode — 4 โหมด (`_get_wind_uv`):** วิธีหาลม (u,v) ต่อพิกัด

| mode | ทำอะไร | ต้องมี wind_field? |
|---|---|---|
| `constant_ne` (default) | คืน `(synthetic_u, synthetic_v)=(3.5, 3.5)` คงที่ = ลม NE จำลอง | ไม่ |
| `random` | สุ่ม u,v ∈ [−5,5] (ทดสอบ/ablation) | ไม่ |
| `from_field` | `sel(time=ts, nearest)` แล้ว `sel(lat,lon, nearest)` จาก DataArray | **ต้องมี** (ไม่งั้น `RuntimeError`) |
| `from_arrays` | lookup สถานีใกล้สุดจาก array ERA5 ที่ loader interp ไว้ล่วงหน้า (`_u10_per_station`) | ไม่ (ใช้ array ใน config) |

**"ตอนยังไม่มี ERA5 ทำยังไง?"**
ส่ง `wind_field=None` และ `config={"wind_mode": "constant_ne"}` (ค่า default อยู่แล้ว) →
กราฟใช้ลม NE คงที่ 3.5/3.5 m/s แทนลมจริง ยังสร้าง Type B/C ได้ครบ ไม่ crash
เมื่อ ERA5 พร้อม (Session 3) loader จะ interp ลง 18 สถานีล่วงหน้าแล้วสลับไปโหมด `from_arrays`
(ดู `loader._build_graph_full` ใน PART B ภาคผนวก B6) — นี่คือ path ที่ใช้จริงในการเทรน

**helper สาธารณะ** (ท้ายไฟล์) ให้ loader เรียก precompute แบบ fast-path:
```python
build_type_a_edges = _build_type_a_edges
build_type_b_edges = _build_type_b_edges
build_type_c_edges = _build_type_c_edges
```
Type A ขึ้นกับพิกัดสถานีล้วน (คงที่ทุก timestep) → คำนวณครั้งเดียว cache ได้;
Type B/C ขึ้นกับลม/ไฟรายเวลา → ต้องสร้างใหม่ต่อ sample

---

## สรุป PART C

- กราฟ = **heterogeneous**: node `station` (18, feature 5 มิติ) + `hotspot` (รายวัน, feature 3 มิติ)
- edge 3 ชนิด: **A** ระยะคงที่ undirected (≤100 km), **B** ลมพลวัต directed (≤200 km, align>0.3),
  **C** ไฟ→สถานี directed (≤500 km, align>0.4)
- หัวใจ novelty = **wind alignment = cos(θ)** ระหว่างทิศ source→target กับทิศลม → ใช้ทั้งเป็น
  เงื่อนไขสร้าง edge และเป็น weight
- `build_graph` สร้าง 1 timestep คืน `HeteroData`; ก่อนมี ERA5 ใช้ `wind_mode='constant_ne'`
  (ลม NE จำลอง, `wind_field=None`) ตอนเทรนจริงใช้ `from_arrays` จาก ERA5 ที่ interp ไว้

> **ต้องการลงลึก PART ไหน หรือไปต่อ PART D (Features & Model Input)?**

---

# ภาคผนวก PART C (เจาะลึก) — ฟีเจอร์ในกราฟ + จุดที่เป็นปัญหา feature-engineering

> โฟกัสว่า `graph_builder.py` "ใส่อะไรเป็นฟีเจอร์" จริง และอะไรที่เป็น footgun/ข้อควรระวัง
> ที่ไม่เห็นจากผิวโค้ด — อ้าง `graph_builder.py`, `loader.py`, `models/mtgnn.py`

## C5. Station node feature — ของที่ `build_graph` ใส่ ≠ ของที่โมเดลเห็นจริง

```python
# graph_builder.py — สร้าง station.x จาก config
feat_cols = cfg.get("station_features", [])              # default 5 คอลัมน์
available = [c for c in feat_cols if c in df_stations.columns]   # ← เอาเฉพาะที่ "มีจริง"
if available:
    station_x = torch.tensor(df_stations[available].values.astype("float32"))
else:
    station_x = torch.zeros((len(df_stations), 0))       # ← ไม่มีเลย = (N, 0)
```

**2 footgun ที่ต้องรู้:**
1. **Silent column drop** — ถ้าคอลัมน์ใน `station_features` ไม่มีใน `df_stations` มัน**เงียบๆ ตัดทิ้ง** ไม่ error → มิติ `station.x` อาจ < 5 โดยไม่มีคำเตือน (เช่นถ้าลืม feature ตัวหนึ่ง โมเดลก็ยังรันแต่ได้ input ผิดมิติ)
2. **snapshot 1 timestep เท่านั้น** — `build_graph` อ่านค่า ณ แถวเดียวของ `df_stations` = ฟีเจอร์ของ **1 ชั่วโมง** ไม่มีมิติเวลา

**แล้วโมเดลเห็นอะไรจริง?** ตอนเทรน loader **เขียนทับ** `station.x` ด้วย tensor เต็ม `(N, T_in=24, F=10)`:
```python
data["station"].x = torch.from_numpy(x_ntf).contiguous().float()   # loader.py:524
```
→ **`station.x` ที่ `build_graph` ใส่ (5 มิติ, 1 timestep) ถูกทิ้งทั้งหมด** ในเส้นทางเทรนจริง
มันมีความหมายเฉพาะกรณีเรียก `build_graph` เดี่ยวๆ (เทสต์/inspect) เท่านั้น
→ MTGNN รับ 10 มิติผ่าน `start_conv = Conv1d(n_features=10, hidden, 1)`

## C6. Hotspot node feature — 3 มิติดิบ และปัญหา scale ที่ซ่อนอยู่

```python
hotspot_x = df_hotspots[["total_frp", "centroid_lat", "centroid_lon"]]   # (K, 3) — ดิบ ไม่ scale
```

**ปัญหาที่มองไม่เห็น: 3 คอลัมน์นี้อยู่คนละสเกลกันมหาศาล**
| feature | ช่วงค่าจริง | ปัญหา |
|---|---|---|
| `total_frp` | ~10 ถึง **หลักพัน** | ค่าใหญ่ ไม่มี upper bound, ไม่ normalize |
| `centroid_lat` | 16–21 | เกือบคงที่ (ช่วงแคบ) ให้ข้อมูลน้อย |
| `centroid_lon` | 97–101 | เกือบคงที่ + **ซ้ำซ้อนกับ edge** (ตำแหน่งถูกใช้ในระยะ/alignment อยู่แล้ว) |

- `total_frp` **กลบ** lat/lon เชิงตัวเลข → `hotspot_encoder` (MLP `Linear(3→hidden)`) ต้องเรียนชดเชยเอง โดยไม่มี BatchNorm/scaling ช่วย
- ป้อน lat/lon เป็น node feature ทั้งที่ **ตำแหน่งถูกเข้ารหัสใน edge แล้ว** (Type C weight = f(ระยะ, alignment)) = redundancy + ค่าคงที่ใหญ่ๆ ที่อาจกวน encoder มากกว่าช่วย
- ว่างเมื่อไม่มีไฟ → `(0, 3)` และ MTGNN มี guard: `if hx.size(0)==0 → c = zeros` (ไม่ให้พัง)

> นี่คือจุดที่ต่างจาก station feature ชัด: **station ผ่าน RobustScaler มาแล้ว (PART D) แต่ hotspot ดิบ 100%**

## C7. Edge feature (`edge_attr`) — โครงสร้าง 3 คอลัมน์ + ปัญหา magnitude ข้าม type

| edge | `edge_attr` | คอลัมน์ | สูตร col 0 (`w`) |
|---|---|---|---|
| A | (E,1) | `[w]` | `exp(-d/50)` |
| B | (E,3) | `[w, alignment, wind_speed]` | `alignment·wind_speed·exp(-d/100)` |
| C | (E,3) | `[w, alignment, frp]` | `alignment·frp·exp(-d/200)` |

**โมเดลใช้แค่ col 0** (`edge_attr[:, 0]` เป็น `edge_weight`) — col 1,2 เก็บให้ XAI/dashboard ถอดอ่าน
(เพราะ `w` เป็นผลคูณ ถอดกลับเป็น alignment/wind_speed แยกไม่ได้ — ดูที่คุยกันเรื่องนี้)

**ปัญหาที่ซ่อน: magnitude ของ `w` ต่างกันมหาศาลระหว่าง type**
```
Type A:  w ∈ (0, 1]                        ← exp decay ล้วน
Type B:  w ~ alignment(0.3–1)·|ลม|(~5)     → ระดับ ~หน่วยเดียว
Type C:  w ~ alignment·frp(หลักร้อย–พัน)   → ระดับ ~หลักร้อย+  ← ใหญ่กว่ามาก
```
- Type C weight **ไม่มี upper bound** เพราะ `frp` ดิบ → edge_weight ของ Type C อาจใหญ่กว่า Type A/B หลายร้อยเท่า
- ตัวช่วย: MTGNN แยก conv คนละตัวต่อ type (`type_a_conv`, `type_b_conv`, `type_c_conv`) แล้วค่อย fuse + `LayerNorm` → **บรรเทา** การปะทะข้าม type ได้บ้าง แต่**ภายใน Type C เอง** ค่ายังกระจายกว้างเพราะ frp ไม่ normalize
- นี่เป็นอีกเหตุผลเชิงเทคนิคว่าทำไม Type C (ไฟ) ถึงจูนยากและ ablation ไม่โชว์ประโยชน์ชัด

## C8. สรุป: อะไร "เข้าโมเดล" จริง vs "เก็บไว้อธิบาย"

| สิ่ง | มิติ | เข้าโมเดล? | หมายเหตุ |
|---|---|---|---|
| `station.x` (จาก loader) | (N, 24, 10) | ✅ ผ่าน `start_conv` | scaled แล้ว |
| `hotspot.x` | (K, 3) ดิบ | ✅ ผ่าน `hotspot_encoder` (ถ้ามี edge) | ไม่ scale, มีปัญหา magnitude |
| `edge_attr[:, 0]` (w) | (E,) | ✅ เป็น `edge_weight` | ผลคูณ physics |
| `edge_attr[:, 1:]` (alignment, wind_speed, frp) | (E, 2) | ❌ | XAI/dashboard เท่านั้น |
| `station.x` จาก `build_graph` (5 มิติ) | (N, 5) | ❌ ในเส้นทางเทรน | ถูก loader เขียนทับ |
| ป้าย `country` ของ hotspot | — | ❌ | metadata สำหรับ attribution ไม่ใช่ feature |

**หัวใจภาคผนวก:** ฟีเจอร์ที่ "ดูเหมือนอยู่ในกราฟ" หลายตัว**ไม่ได้เข้าโมเดลจริง** (station.x 5 มิติ, edge_attr col 1-2, country) — และฟีเจอร์ที่เข้าจริงบางตัว (`hotspot.x`, Type C `w`) **ดิบและ scale ไม่ดี** ซึ่งเป็นหนี้ทางเทคนิคที่สอดคล้องกับผล ablation ที่ว่าไฟ/ลมไม่ช่วย accuracy อย่างมั่นคง
