# SESSION 7 — AI Camp Pitch Prep + Report Pipeline
# Claude Code Session Initialization Prompt
# ใช้พรอมพ์นี้เป็นข้อความแรกในแต่ละ Claude Code session ใหม่

---

## ขั้นตอนบังคับก่อนทำอะไรทั้งนั้น

**อ่านไฟล์ทั้งหมดนี้ตามลำดับ ก่อนเสนอแผนหรือเขียนโค้ดใด ๆ:**

```
1. docs/DESIGN.md               ← architecture decisions (source of truth)
2. CLAUDE.md                    ← project memory, conventions, constraints
3. docs/SESSION6_NOTES.md       ← สถานะล่าสุด (attribution fix, A3TGCN done)
4. docs/PROPOSAL_MEMO.md        ← เนื้อหาเทคนิคสำเร็จรูปทั้งหมด
5. outputs/evaluation_val2025.json   ← ตัวเลขผลลัพธ์จริง
6. outputs/attribution_march2024.json ← attribution findings จริง
```

---

## ขั้นตอนที่ 1 — เลือก Skills ก่อนเริ่มงาน

อ่าน SKILL.md ของทุก skill ต่อไปนี้ตามลำดับ และอธิบายสั้น ๆ ว่าแต่ละ skill จะใช้ทำอะไรในโปรเจคนี้โดยเฉพาะ:

### บังคับ (ต้องติดตั้งก่อน)
- `/mnt/skills/examples/skill-creator/SKILL.md` — สร้าง skill ใหม่สำหรับโปรเจคนี้
  → จะใช้สร้าง 2 skills: (1) `pm25-pitch-qa` สำหรับ Q&A prep, (2) `pm25-session` สำหรับ session workflow

### สำหรับผลลัพธ์ที่ต้องการ
- `/mnt/skills/public/pptx/SKILL.md` — สร้าง pitching deck สำหรับ AI Camp 2-3 ก.ค.
- `/mnt/skills/public/docx/SKILL.md` — สร้าง script/notes สำหรับ presenter
- `/mnt/skills/public/pdf-reading/SKILL.md` — อ่าน example proposals ใน ex/ folder

### อ่านถ้าเวลาเหลือ
- `/mnt/skills/public/pdf/SKILL.md` — export สุดท้ายเป็น PDF

**หลังอ่านครบ ให้ระบุ: "Skills ที่จะใช้: X, Y, Z" พร้อมเหตุผล 1 บรรทัดต่อ skill**

---

## ขั้นตอนที่ 2 — วิเคราะห์ข้อดีข้อเสียที่ได้รับ

หลังอ่านไฟล์โปรเจคครบแล้ว ให้วิเคราะห์ข้อดีข้อเสียต่อไปนี้เทียบกับ rubric จริง

### ข้อดีที่ผู้ประเมินให้มา
1. ใช้เทคนิคขั้นสูงที่สอดคล้องกับ state-of-the-art ในงาน air quality forecasting จริง
2. มีประโยชน์ต่อสังคม
3. ขอบเขตเหมาะสมกับระยะเวลาที่กำหนด
4. มีความลึกทางวิชาการและเทคโนโลยีสูง
5. ผลลัพธ์สามารถนำไปใช้สนับสนุนการตัดสินใจเชิงนโยบายได้จริง

### ข้อเสียที่ผู้ประเมินให้มา
1. ERA5 มี latency 5-7 วัน → production ต้องใช้ NWP forecast แทน reanalysis ซึ่งจะส่งผลต่อ accuracy
2. ไม่ระบุรายละเอียดการวัดผลความถูกต้อง (per-horizon RMSE ยังขาดอยู่)
3. ความสามารถในการพยากรณ์ดีขึ้นจาก baseline ยังไม่สูงมาก (6h/12h ยังแพ้ persistence)
4. การระบุสัดส่วนฝุ่นจากแต่ละประเทศยังเป็นการอนุมาน ไม่ใช่การวัดโดยตรง

### Rubric AI Camp (2-3 ก.ค. 2569) — น้ำหนักคะแนน
| มิติ | น้ำหนัก | หมายเหตุ |
|---|---|---|
| ความถูกต้องและความสมบูรณ์ทางเทคนิค | 10 | **สูงสุด — ต้องแก้ Con #1, #2, #3** |
| ผลกระทบต่อสังคมและการใช้ AI อย่างมีธรรมาภิบาล | 10 | **สูงสุด — ต้องแก้ Con #4 + AI governance** |
| ความคิดสร้างสรรค์และศักยภาพในการต่อยอด | 5 | 3 novelties มีอยู่แล้ว |
| ทักษะการนำเสนอและการตอบคำถาม | 5 | ทุกคนต้องมีส่วนร่วม |
| **รวม** | **30** | |

**สำหรับแต่ละข้อเสีย ให้ระบุ:**
- มิติ rubric ที่ได้รับผลกระทบ (Technical / Social / Creativity / Presentation)
- ระดับความรุนแรง (blocking คะแนน / framing เท่านั้น / ต้องแก้โค้ด)
- วิธีแก้ที่เป็นรูปธรรม (แก้โค้ด / แก้ framing / เพิ่มเอกสาร)

---

## ขั้นตอนที่ 3 — วางแผนก่อนลงมือ (ห้ามข้ามขั้นตอนนี้)

วางแผนในรูปแบบต่อไปนี้ แล้วรอให้ user approve ก่อนทำ:

### Timeline ที่เหลือ
- **2-3 ก.ค. 2569** = AI Camp Pitching (อีก ~4 วัน) ← CRITICAL
- **17 ก.ค. 2569** = รายงาน NSC (อีก ~19 วัน)

### แผนงาน (format ที่ต้องการ)
```
## Plan Session 7

### Deliverable 1: [ชื่อ] — ต้องทำภายใน [วันที่]
Agent: [architect/implementer/reviewer]
Files: [รายการไฟล์ที่จะสร้างหรือแก้ไข]
Rubric impact: [มิติที่ได้คะแนนจาก deliverable นี้]
Estimated effort: [S/M/L]

### Deliverable 2: ...
```

**กฎสำหรับ Session 7:**
- วางแผนก่อนเสมอ รอ approve แล้วค่อยเริ่ม
- แต่ละ deliverable ต้อง map ไปยัง rubric มิติที่ชัดเจน
- ห้ามเริ่มโค้ดก่อน plan ได้รับ approve
- อ้างอิง DESIGN.md section number ทุกครั้งที่ตัดสินใจ architectural decision
- หากไม่แน่ใจเรื่องผลลัพธ์จริง (RMSE, attribution numbers) ให้อ่านจาก outputs/ ก่อนเสมอ อย่า hallucinate

---

## Context ที่สำคัญสำหรับ session นี้

### สถานะโปรเจค (ณ วันที่ 28 มิ.ย. 2569)
- Session 1-6 เสร็จแล้ว
- MTGNN trained: val_rmse_24h = 0.4576 (normalized) ≈ 9.1 µg/m³
- A3TGCN trained: val_rmse_24h = 0.4928 ≈ 9.8 µg/m³  
- Occlusion bug fixed (SESSION6_NOTES §2)
- Attribution march2024: Thailand=100%, Myanmar=0% (physically correct, 128× FRP ratio)
- IG feature importance: d2m (0.0345) > t2m (0.0344) > pm25_scaled (0.0140)
- **ยังขาด:** scripts/04_evaluate.py (per-horizon RMSE table), Streamlit end-to-end test

### Branch ปัจจุบัน
`feat/session-3-era5-models` — ยังไม่ merge ไป main

### ข้อห้าม (จาก CLAUDE.md)
- ห้ามเพิ่ม model เกิน A3TGCN / PM2.5-GNN / MTGNN
- ห้าม sub-hourly forecast
- ห้ามเพิ่ม station เกิน 15 core + 10 extended
- ห้ามเพิ่ม dependency โดยไม่ขออนุญาต user ก่อน
- Reply in Thai for explanations, English for code/identifiers

