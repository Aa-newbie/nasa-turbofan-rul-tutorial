# NASA Turbofan Engine RUL Prediction (บทเรียน ML เบื้องต้น)

บทเรียนสำหรับผู้เริ่มต้นเรียน Machine Learning โดยใช้ข้อมูลจริงจาก NASA
เพื่อทำนาย **Remaining Useful Life (RUL)** หรือ "จำนวนรอบการทำงานที่เหลือ
ก่อนเครื่องยนต์เจ็ทจะเสีย" — เป็นตัวอย่างของงาน Predictive Maintenance
(ซ่อมบำรุงเชิงพยากรณ์) ที่ใช้จริงในอุตสาหกรรม

Dataset ต้นทาง: [NASA Turbofan Engine Degradation Simulation (Kaggle)](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation)
(อ้างอิงจาก C-MAPSS dataset ของ NASA Prognostics Center of Excellence)

## เนื้อหา

- [`data/`](data/) — ข้อมูลดิบ (train/test/RUL) ของชุด FD001
- [`rul_tutorial.py`](rul_tutorial.py) — สคริปต์บทเรียนแบบ step-by-step
  (ใช้ `# %%` แบ่ง cell รันทีละส่วนได้ใน VS Code + Python extension)
- [`plots/`](plots/) — กราฟผลลัพธ์ตัวอย่างจากการรันสคริปต์

## การติดตั้ง (สำหรับมือใหม่)

### 1. ติดตั้ง Python

ถ้ายังไม่มี Python ในเครื่อง โหลดได้จาก [python.org/downloads](https://www.python.org/downloads/)
**ตอนติดตั้งอย่าลืมติ๊ก "Add python.exe to PATH"** ไม่งั้นจะเรียกคำสั่ง `python` จาก
terminal ไม่ได้ ตรวจสอบว่าติดตั้งสำเร็จด้วยคำสั่ง:

```bash
python --version
```

### 2. โหลดโค้ดชุดนี้ลงเครื่อง

เลือกวิธีใดวิธีหนึ่ง:

**แบบไม่ต้องลง Git (ง่ายสุด)** — กดปุ่ม "Code" → "Download ZIP" ที่หน้า repo นี้
บน GitHub แล้วแตกไฟล์ zip ออกมา

**แบบใช้คำสั่ง (ต้องลง Git ก่อน)** — โหลด Git จาก
[git-scm.com/download/win](https://git-scm.com/download/win) หรือถ้ามี winget:

```bash
winget install --id Git.Git -e
```

จากนั้น clone repo:

```bash
git clone https://github.com/Aa-newbie/nasa-turbofan-rul-tutorial.git
cd nasa-turbofan-rul-tutorial
```

### 3. ติดตั้งไลบรารีที่ต้องใช้

เปิด terminal ในโฟลเดอร์ที่โหลดมา แล้วรัน:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 4. รันสคริปต์

รันทั้งไฟล์:

```bash
python rul_tutorial.py
```

หรือ (แนะนำสำหรับการเรียนรู้) เปิดโฟลเดอร์นี้ด้วย **VS Code** แล้วติดตั้ง
extension **2 ตัว** จากช่อง Extensions (`Ctrl+Shift+X`):
- **"Python"** (โดย Microsoft)
- **"Jupyter"** (โดย Microsoft) — ตัวนี้จำเป็นด้วย ถ้าลงแค่ Python extension
  เฉยๆ จะไม่เห็นปุ่ม "Run Cell"

ลงเสร็จแล้วปิด-เปิดไฟล์ `rul_tutorial.py` ใหม่ (หรือกด `Ctrl+Shift+P` พิมพ์
"Reload Window") จะเห็นปุ่ม "Run Cell" ลอยอยู่เหนือแต่ละ block
(คั่นด้วย `# %%`) กดไล่ทีละ block เพื่อดูผลลัพธ์และกราฟทีละขั้นตอน

## สิ่งที่ได้เรียนรู้

1. การโหลด/สำรวจข้อมูล time-series แบบ multivariate
2. การสร้าง label (RUL) เองจากข้อมูลดิบ
3. การเลือกฟีเจอร์ (ตัดเซ็นเซอร์ที่ไม่มีประโยชน์)
4. การเทรนและเปรียบเทียบโมเดล Linear Regression กับ Random Forest
5. การประเมินผลด้วย RMSE/MAE และดู feature importance
