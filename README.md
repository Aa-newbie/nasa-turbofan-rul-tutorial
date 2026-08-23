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
- [`rul_lstm.py`](rul_lstm.py) — บทเรียนต่อยอด ทำนาย RUL ด้วย LSTM (PyTorch)
- [`requirements.txt`](requirements.txt) — รายการไลบรารีพร้อมเวอร์ชันที่ตรึงไว้
- [`requirements-lstm.txt`](requirements-lstm.txt) — ไลบรารีเพิ่มเติมสำหรับบทเรียน LSTM

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

### 3. สร้าง virtual environment แล้วติดตั้งไลบรารี

เปิด terminal ในโฟลเดอร์ที่โหลดมา แล้วรัน:

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.venv` คือ *virtual environment* — โฟลเดอร์ที่เก็บไลบรารีของโปรเจกต์นี้แยกไว้
ต่างหาก ไม่ปนกับ Python ตัวหลักของเครื่อง ข้อดีคือเวอร์ชันไลบรารีไม่ตีกับ
โปรเจกต์อื่น และถ้าพังเมื่อไหร่ก็แค่ลบโฟลเดอร์ `.venv` ทิ้งแล้วสร้างใหม่

เวอร์ชันของไลบรารีทั้งหมดถูกตรึงไว้ใน [`requirements.txt`](requirements.txt)
แล้ว เพื่อให้รันซ้ำเมื่อไหร่ก็ได้ผลลัพธ์เหมือนเดิม

> **ถ้าเครื่องมี MSYS2 / MinGW อยู่** — อย่าใช้ `python` ที่มาจาก `C:\msys64\...`
> เพราะตัวนั้นไม่มี `pip` ติดมา และถูกล็อกให้ติดตั้งแพ็กเกจผ่าน `pacman`
> เท่านั้น ให้ใช้ Python จาก python.org ตามข้อ 1

### 4. รันสคริปต์

รันทั้งไฟล์ (ต้อง activate `.venv` ไว้ก่อน):

```bash
python rul_tutorial.py
```

หรือ (แนะนำสำหรับการเรียนรู้) เปิดโฟลเดอร์นี้ด้วย **VS Code** แล้วติดตั้ง
extension **2 ตัว** จากช่อง Extensions (`Ctrl+Shift+X`):
- **"Python"** (โดย Microsoft)
- **"Jupyter"** (โดย Microsoft) — ตัวนี้จำเป็นด้วย ถ้าลงแค่ Python extension
  เฉยๆ จะไม่เห็นปุ่ม "Run Cell"

จากนั้น **บอก VS Code ให้ใช้ `.venv`** โดยกด `Ctrl+Shift+P` พิมพ์
"Python: Select Interpreter" แล้วเลือกตัวที่ขึ้นว่า `.venv` ขั้นตอนนี้ห้ามข้าม —
ถ้า VS Code ไปหยิบ Python ตัวอื่นของเครื่องมาใช้ จะขึ้น error ว่า
`requires the ipykernel package` ตอนกด Run Cell

ปิด-เปิดไฟล์ `rul_tutorial.py` ใหม่ (หรือกด `Ctrl+Shift+P` พิมพ์
"Reload Window") จะเห็นปุ่ม "Run Cell" ลอยอยู่เหนือแต่ละ block
(คั่นด้วย `# %%`) กดไล่ทีละ block เพื่อดูผลลัพธ์และกราฟทีละขั้นตอน

## สิ่งที่ได้เรียนรู้

1. การโหลด/สำรวจข้อมูล time-series แบบ multivariate
2. การสร้าง label (RUL) เองจากข้อมูลดิบ และเทคนิค clip แบบ piecewise linear
3. การเลือกฟีเจอร์ (ตัดเซ็นเซอร์ที่ไม่มีประโยชน์)
4. การเทรนและเปรียบเทียบโมเดล 3 แบบ — Linear Regression, Random Forest,
   HistGradientBoosting
5. การประเมินผลด้วย RMSE/MAE และดู feature importance
6. การแบ่ง validation set ให้ถูกหลัก (แบ่งตามเครื่องยนต์ ไม่ใช่ตามแถว) และ
   เหตุผลที่ห้ามใช้ test set เลือกโมเดล
7. Feature engineering แบบ rolling window เพื่อให้โมเดลเห็นแนวโน้มตามเวลา
8. การผสมโมเดล (ensemble) และการรายงานผลเป็น ค่าเฉลี่ย ± ส่วนเบี่ยงเบน จากหลาย
   seed แทนตัวเลขเดี่ยวจากการรันครั้งเดียว

## ผลลัพธ์

วัดบน test set ของ FD001 (100 เครื่องยนต์) — ยิ่ง RMSE น้อยยิ่งดี

| โมเดล | ฟีเจอร์ | RMSE |
|---|---|---|
| Linear Regression | ดิบ 18 ตัว | 20.83 |
| Random Forest | ดิบ 18 ตัว | 17.48 |
| HistGradientBoosting | + window 72 ตัว | 13.89 ± 0.33 |
| Random Forest | + window 72 ตัว | 13.56 ± 0.10 |
| Extra Trees | + window 72 ตัว | 13.26 ± 0.06 |
| **Ensemble (เฉลี่ย 3 ตัว)** | **+ window 72 ตัว** | **13.02 ± 0.13** |
| LSTM *(rul_lstm.py)* | ลำดับ 30 รอบ | 12.94 *(seed เดียว)* |

ค่า ± คือส่วนเบี่ยงเบนมาตรฐานจากการรัน 5 seed สองแถวแรกเป็นโมเดลพื้นฐานที่ไม่มี
การสุ่มภายใน จึงไม่มีแถบความคลาดเคลื่อน

ข้อสังเกตที่เป็นบทเรียนหลัก: การเพิ่มฟีเจอร์ที่จับแนวโน้มตามเวลา ช่วยได้มากกว่า
การเปลี่ยนอัลกอริทึมหลายเท่า — RMSE ลดจาก 17.48 เหลือราว 13 ด้วย feature
engineering อย่างเดียว ส่วนการผสมโมเดล (ensemble) ช่วยเพิ่มอีกเล็กน้อยและทำให้
ผลนิ่งขึ้นชัดเจน

ข้อควรระวังที่บทเรียนสาธิตไว้ด้วย: test set มีเพียง 100 เครื่องยนต์ ช่วงความเชื่อมั่น
95% ของ RMSE จึงกว้างราว ±2.3 ซึ่งกว้างกว่าความต่างระหว่างโมเดลส่วนใหญ่ในตาราง
**ถ้าสองโมเดลต่างกันไม่ถึง 1 RMSE ยังสรุปไม่ได้ว่าตัวไหนดีกว่า**

และที่น่าสนใจกว่านั้น LSTM ซึ่งกินเวลาเทรนหลักนาทีและต้องลง PyTorch เพิ่มอีก
200 MB ทำได้ 12.94 — ดีกว่าเพียงเล็กน้อยเท่านั้น สำหรับ dataset ขนาดนี้
โมเดลที่เรียบง่ายกว่าและอธิบายให้วิศวกรเข้าใจได้ง่ายกว่า จึงอาจเป็นทางเลือก
ที่เหมาะกว่าในทางปฏิบัติ

## บทเรียน LSTM (ไม่บังคับ)

ถ้าอยากลองแนวทาง deep learning ต่อ:

```bash
pip install -r requirements-lstm.txt
python rul_lstm.py
```

ไฟล์ [`rul_lstm.py`](rul_lstm.py) แยกจากบทเรียนหลักเพราะใช้โครงสร้างข้อมูล
คนละแบบ (3 มิติ: ตัวอย่าง × เวลา × เซ็นเซอร์) และอธิบายเรื่องที่เฉพาะกับ
neural network เช่น การ normalize, sliding window, วงจรการเทรนของ PyTorch
และการอ่านกราฟ train/validation
