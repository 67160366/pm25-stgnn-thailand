You are an expert ML engineer and technical educator.  
Your task is to help me fully understand this PM2.5 STGNN Thailand project from scratch.  
Analyze ALL files thoroughly. Explain everything in Thai language.  
Use real code examples from the project files — never invent examples.

\---

\#\# PART A — BIG PICTURE (ทำความเข้าใจภาพรวมก่อน)

\#\#\# A1. โปรเจคคืออะไร  
\- เป้าหมายของโปรเจคนี้คืออะไร? แก้ปัญหาอะไร?  
\- ผลลัพธ์สุดท้ายที่ส่งมอบได้คืออะไร? (model, dashboard, report, ฯลฯ)  
\- thesis หลักของงานนี้คืออะไร? ผลสรุปที่ได้ตอนนี้เป็นอย่างไร?

\#\#\# A2. โครงสร้าง Repository  
\- วาด directory tree พร้อมอธิบายว่าแต่ละโฟลเดอร์/ไฟล์ทำหน้าที่อะไร  
\- ไฟล์ไหนคือ "หัวใจ" ของโปรเจค?  
\- ไฟล์ไหน generated/ไม่ควรแก้มือ?

\#\#\# A3. Tech Stack ทั้งหมด  
สำหรับ library/framework ทุกตัวที่พบ:  
\- ทำอะไร? ทำไมถึงเลือกใช้ตัวนี้?  
\- ส่วนไหนของโปรเจคที่ใช้มัน?  
\- ถ้าตัดออก อะไรจะพัง?

\---

\#\# PART B — DATA PIPELINE (เส้นทางของข้อมูล)

\#\#\# B1. แหล่งข้อมูล  
\- มีข้อมูลกี่แหล่ง? แต่ละแหล่งให้ข้อมูลอะไร?  
\- endpoint ที่ใช้จริงคืออะไร? (ระบุ URL \+ parameter สำคัญ)  
\- มี quirk หรือข้อจำกัดที่ต้องระวังอะไรบ้าง?  
\- แต่ละ API มี limit อะไร? จัดการยังไง?

\#\#\# B2. Station Selection  
\- ใช้เกณฑ์อะไรคัดเลือก station?  
\- ได้กี่ station? ครอบคลุมพื้นที่ไหนบ้าง?  
\- ทำไม 18 station ถึงถูกเลือก ไม่ใช่จำนวนอื่น?

\#\#\# B3. Data Flow ตั้งแต่ต้นจนจบ  
วาด flow แบบ step-by-step:  
raw API → scraper → local file → loader → preprocessing → features → graph → model  
\- แต่ละ step ใช้ไฟล์ไหน?  
\- input/output ของแต่ละ step คืออะไร? (format, shape, ตัวอย่างค่า)  
\- ข้อมูลถูกเก็บที่ไหนระหว่างทาง?

\#\#\# B4. Missing Data Policy  
\- มีนโยบายจัดการ missing data อย่างไร?  
\- แต่ละ case (\< 6h, 6–24h, \> 24h, \> 7 days) ทำอะไรและทำไม?  
\- ถ้าข้ามขั้นตอนนี้ อะไรจะเกิดขึ้นกับ model?

\---

\#\# PART C — GRAPH CONSTRUCTION (หัวใจของ STGNN)

\#\#\# C1. Graph คืออะไรในโปรเจคนี้  
\- node คืออะไร? มีกี่ประเภท?  
\- edge คืออะไร? มีกี่ประเภท?  
\- ทำไมต้องเป็น heterogeneous graph?

\#\#\# C2. Edge Types ทั้ง 3 แบบ  
สำหรับ Type A, B, C:  
\- เชื่อม node ประเภทไหนกับไหน?  
\- เงื่อนไขการสร้าง edge คืออะไร?  
\- weight คำนวณยังไง? แต่ละตัวแปรในสูตรหมายถึงอะไร?  
\- directed หรือ undirected? ทำไม?

\#\#\# C3. Wind Alignment  
\- wind alignment \= cos(θ) หมายความว่าอะไรกันแน่?  
\- ค่า 1, 0, \-1 หมายความว่าอะไร? ยกตัวอย่างจริง  
\- ทำไมต้องมี threshold ที่ 0.3 และ 0.4?

\#\#\# C4. \`build\_graph\` API  
\- parameter แต่ละตัวรับอะไร? format/shape คืออะไร?  
\- return ออกมาเป็น HeteroData ที่มีอะไรบ้าง?  
\- ตอนยังไม่มี ERA5 ทำยังไง? ส่งอะไรเข้าไปแทน?

\---

\#\# PART D — FEATURES & MODEL INPUT

\#\#\# D1. Node Features  
\- Station node มี feature อะไรบ้าง? แต่ละตัวหมายถึงอะไร?  
\- Hotspot node มี feature อะไรบ้าง?  
\- ทำไมใช้ sin/cos encoding สำหรับ hour และ day-of-year?  
\- RobustScaler คืออะไร? ทำไมไม่ใช้ StandardScaler?

\#\#\# D2. Edge Attributes  
\- แต่ละ edge type มี attribute อะไรบ้าง?  
\- attribute เหล่านี้ถูกใช้ใน model อย่างไร?

\#\#\# D3. Feature ที่ยังขาด  
\- feature ไหนที่ยัง NULL อยู่? เพราะอะไร?  
\- เมื่อ ERA5 มาใน Session 3 จะเพิ่มอะไร?

\---

\#\# PART E — OUTPUT & RESULTS

\#\#\# E1. Model Output  
\- model predict อะไร? output shape เป็นอย่างไร?  
\- metric ที่ใช้วัดคืออะไร? (RMSE, MAE ฯลฯ) แต่ละตัวหมายถึงอะไร?  
\- ผลลัพธ์ที่ได้ตอนนี้เป็นอย่างไร? (ดี/ไม่ดี/อธิบาย)

\#\#\# E2. Honest Thesis  
\- thesis จริงๆ ของโปรเจคนี้คืออะไร?  
\- "forecast edge marginal" หมายความว่าอะไร?  
\- "graph does not robustly help accuracy" แปลว่าอะไร?  
\- แล้ว value จริงของโปรเจคนี้อยู่ที่ไหน?

\#\#\# E3. Output Files  
\- ไฟล์ output มีอะไรบ้าง? อยู่ที่ไหน?  
\- NSC2026\_Final\_Report.docx สร้างยังไง? ทำไมไม่ track ใน git?

\---

\#\# PART F — PROJECT CONFIGURATION

\#\#\# F1. Environment & Config  
\- ต้องตั้งค่าอะไรบ้างก่อนรันได้?  
\- \`.env\` ต้องมี key อะไรบ้าง?  
\- \`\~/.cdsapirc\` คืออะไร? ใช้ทำอะไร?

\#\#\# F2. Known Issues & Workarounds  
\- มีปัญหาอะไรที่ต้องระวัง? (เช่น Windows Thai path)  
\- แต่ละปัญหามี workaround อะไร?  
\- ปัญหาไหนที่ถาวรแก้แล้ว? ไหนที่ยังต้อง manual fix?

\#\#\# F3. คำสั่งสำคัญ  
\- คำสั่ง run อะไรบ้าง? แต่ละคำสั่งทำอะไร?  
\- ลำดับการรันที่ถูกต้องคืออะไร?

\---

\#\# PART G — TESTING & CODE QUALITY

\#\#\# G1. Tests  
\- มี test อะไรบ้าง? ครอบคลุมอะไร?  
\- รัน test ยังไง? ผลปัจจุบันเป็นอย่างไร?  
\- อะไรที่ยังไม่ได้ test?

\#\#\# G2. Code Quality Tools  
\- ใช้ linter/formatter อะไร? rule ไหนบ้าง?  
\- มีข้อยกเว้นอะไร? ทำไม?

\---

\#\# PART H — SESSION SUMMARY (ปิดท้าย)

หลังจากอธิบายทุก PART แล้ว สรุปแต่ละ session ที่ทำไปแล้ว:

สำหรับแต่ละ session ที่พบในโปรเจค:  
1\. \*\*เป้าหมาย\*\* — session นี้ตั้งใจทำอะไร?  
2\. \*\*สิ่งที่สร้าง\*\* — deliverable จริงๆ คืออะไร? ไฟล์ไหนบ้าง?  
3\. \*\*สิ่งที่เรียนรู้/แก้ปัญหา\*\* — เจอ surprise อะไร? แก้ยังไง?  
4\. \*\*สิ่งที่เปลี่ยนจากแผน\*\* — deviation จากที่วางไว้คืออะไร? ทำไม?  
5\. \*\*สิ่งที่ส่งต่อ\*\* — session ถัดไปต้องทำอะไรต่อ?

\---

\#\# INSTRUCTIONS

\- อธิบายเป็นภาษาไทยตลอด  
\- ใช้โค้ดจริงจากโปรเจค ไม่สร้างขึ้นมาเอง  
\- ถ้าไม่แน่ใจ บอกว่าไม่แน่ใจ อย่าเดา  
\- ถ้า PART ไหนไม่มีข้อมูลในโปรเจค ให้บอกว่าไม่พบ  
\- ใช้ตาราง, bullet, ASCII diagram เพื่อให้อ่านง่าย  
\- เริ่มจาก PART A ก่อน แล้วถามว่าต้องการขยาย PART ไหน  
  ก่อนจบทุก PART ให้ถามว่า "ต้องการลงลึก PART ไหน หรือไปต่อ PART ถัดไป?"  
