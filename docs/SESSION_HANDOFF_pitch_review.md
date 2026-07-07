# Handoff — เตรียม AI Camp Pitch + แก้ Dashboard (สำหรับโมเดลที่มาตรวจสอบต่อ)
> วันที่: 2026-07-02 · Branch: `feat/session-4-explain-app`
> เป้าของเซสชันนี้: ทำความเข้าใจโปรเจกต์ให้ลึก → เตรียมพิตช์ AI Camp (2-3 ก.ค.) อย่างซื่อสัตย์
> → แก้ dashboard (bounding box + layout + ลูกศรลม)
>
> **หน้าที่ผู้ตรวจสอบ:** ตรวจว่า (ก) positioning พิตช์ทนคำถามได้จริงไหม (ข) ตัวเลขถูกไหม
> (ค) การแก้โค้ด dashboard สมเหตุผล/ไม่พังไหม (ง) มี claim ไหนยัง overclaim อยู่

---

## 1. สถานะจริงของโปรเจกต์ (ยึดอันนี้ ไม่ใช่ DESIGN.md)
- **พยากรณ์ PM2.5 แพ้ persistence** ที่ 6h/12h, ดีกว่านิดที่ 48h แต่ **ไม่ผ่านนัยสำคัญ 95%**
  (`outputs/evaluation_test2025.json`, `significance_test2025.json`)
- **กราฟ wind-aware ไม่ช่วย accuracy อย่างมั่นคง** — ablation ตัด Type B/C แทบไม่เปลี่ยน RMSE,
  noise ERA5 ก็ไม่ขยับ (`ablation_multiseed.json`, `nwp_sensitivity.json`); ใช้ลม ERA5 จริง (from_field)
- **Attribution รายประเทศยังไม่ validate** — เคส 2025-03-25 โมเดลให้พม่า 100% ทั้งที่ FRP เชื่อมต่อ
  พม่าแค่ 27% (`transboundary_attr_test.json`) = artifact ของ normalize + ป้ายประเทศหยาบ
- แหล่งความจริง: `CRITICAL_REVIEW.md`, `SESSION{7,8,10}_NOTES.md`, `outputs/*.json`

## 2. ข้อสรุปเชิงกลยุทธ์พิตช์ (ผ่านการถูกท้าทายจนตกผลึก)
- rubric AI Camp น้ำหนัก **10/5/10/5** (เทคนิค/สร้างสรรค์/สังคม-governance/นำเสนอ) — ไม่ให้คะแนน
  "โปรดักขายได้" หรือ "ความแม่น" ตรงๆ → เข้าทางงานซื่อสัตย์
- **ขายโปรเจ็ค/decision-support ไม่ขายโปรดัก/ออราเคิล**
- claim ที่ถูกพิสูจน์ว่า**ไปไม่รอด** (อย่าใช้): "พยากรณ์แม่นกว่า", "AI ระบุแหล่งเก่งกว่าคน",
  "กราฟช่วยความแม่น", "พร้อม deploy ตัดสินนโยบาย"
- claim ที่**ทนคำถามได้**: สเกล/อัตโนมัติ/ทำซ้ำได้/objective, เครื่องมือช่วยตัดสินใจรวมหลายแหล่ง,
  บอก "คันโยก" (ในประเทศ vs การทูต), วิศวกรรม+ความซื่อสัตย์+governance
- **จุดพลิกสำคัญ:** ผู้ใช้ท้วงว่า "ดูแผนที่จุดไฟก็รู้ว่าไฟประเทศข้างๆ จะใช้ AI ทำไม" →
  ตรวจข้อมูลจริงแล้วพบว่า**ใน 5 เหตุการณ์ ตากับโมเดลมักได้คำตอบเดียวกัน** (เคสต่าง=25 มี.ค.
  ซึ่งเป็น artifact) → **เราไม่มีหลักฐานว่า AI ระบุแหล่งเก่งกว่าตา** ต้องเลิกเคลมข้อนี้เด็ดขาด
  ค่าที่เหลือคือ สเกล+consistency+forecast+counterfactual (ไม่ใช่ single-case accuracy)

## 3. สิ่งที่แก้ใน dashboard เซสชันนี้ (โค้ดจริง)
> ทั้งหมด **display-layer เท่านั้น ไม่แตะโมเดล/pipeline ไม่ retrain**

| ไฟล์ | แก้อะไร |
|---|---|
| `.venv/.../_editable_impl_*.pth` | แก้ Thai abs-path → `../../..` (bug cp874 crash หลัง uv sync) |
| `app/lib/geo.py` (**ใหม่**) | point-in-polygon pure-Python (ไม่ใช้ shapely) — geocode ประเทศจากขอบเขตจริง; test 11/11 |
| `app/assets/borders_th_mm_la.geojson` (**ใหม่**) | ขอบเขต TH/MM/LA จาก geoBoundaries (1.5MB) |
| `app/lib/data_access.py` | `hotspots_for_date` override country ด้วย geo.country_of; เพิ่ม `wind_for_date` |
| `app/views/transboundary.py` | วาดเส้นพรมแดนจริง + ลูกศรลม ERA5 (18 สถานี) + สี "อื่นๆ" + caption |
| `app/views/performance.py` | แก้ layout legend RMSE ทับ title |
| `src/viz/maps.py` | แก้ colorbar/legend แผนที่ overview ทับกัน |

**ค้นพบจากการแก้ geocode:** bbox เดิมนับพม่า/ลาวขาดจริง (18 มี.ค. เดิมพม่า 5 → จริง 13+ลาว 4;
16 ก.พ. เดิมพม่า 10 → จริง 20)

## 4. ⚠️ ค้าง/ต้องระวัง (open items)
1. **Desync:** แผนที่ (geocode ใหม่ ถูก) vs ตัวเลข % ในตาราง/JSON (geocode เดิม bbox) — ยังไม่ตรงกัน
   ต้อง **รัน attribution ซ้ำด้วยป้ายใหม่** (inference ไม่ retrain) ให้ตรงกัน (งานหลังพิตช์) —
   ใส่ caption เตือนไว้บนหน้าเว็บแล้ว
2. **ERA5 = reanalysis** ไม่ใช่ forecast → deployment จริงต้องใช้ NWP + ยังไม่ validate กับ NWP
3. รูปที่ต้องทำใหม่: Hook หน้าปก + กราฟ 73/27 vs 100/0 (สเปกใน `AICAMP_pitch_governance_QA.md`)
   + diagram operational
4. เคยค้างเช็ก: **ERA5 features (u10..blh) ถูก scale ก่อนเข้าโมเดลไหม** — ยังไม่ได้ verify
   (t2m~300K/blh~500m อาจกลบ feature อื่นใน start_conv) — จุดที่ผู้ตรวจอาจเจาะ

## 5. ไฟล์เอกสารที่สร้างในเซสชันนี้
- `docs/explain/PART_{A..H}_*.md` — อธิบายโปรเจกต์ทั้งหมด (verify กับซอร์สแล้ว; PART_C มีภาคผนวกฟีเจอร์)
- `docs/AICAMP_pitch_deck_content.md` (v1) + `docs/AICAMP_pitch_deck_v2.md` (**ล่าสุด ใช้ตัวนี้**)
- `docs/AICAMP_pitch_governance_QA.md` — สไลด์ governance + สคริปต์ Q&A + สเปกกราฟ 73/27

## 6. ตัวเลขหลัก (verify แล้ว — ผู้ตรวจเช็กซ้ำได้)
| อ้าง | ค่า | ไฟล์ |
|---|---|---|
| RMSE persistence/MTGNN (test) | 2.96/5.19/8.70/12.99 · 4.52/5.93/8.76/12.66 | evaluation_test2025.json |
| 6h ไม่ significant | prob_model_better=0.0 | significance_test2025.json |
| 2025-03-25 | peak 126.7 · FRP ไทย1305.9/พม่า478.2 · attr พม่า100% | transboundary_attr_test.json |
| geocode test | 11/11 (รวมเมืองชายแดน) | app/lib/geo.py |

## 7. คำถามที่อยากให้ผู้ตรวจสอบตอบ
1. Positioning v2 ("decision-support + honest, ไม่ใช่ oracle") ยังมี claim ไหน overclaim อยู่ไหม?
2. คำตอบ Q4/Q5 (ทำไมใช้ AI / แก้อะไรได้) ป้องกันได้จริงไหม หรือยังอ่อน?
3. การแก้ geocode ที่ read-time (map ถูก แต่ JSON เดิม) — ยอมรับได้สำหรับพิตช์ หรือควรรัน attribution
   ซ้ำก่อน?
4. มีมุมไหนที่กรรมการสายเทคนิค/นโยบายจะเจาะแล้วเรายังไม่มีคำตอบ?
