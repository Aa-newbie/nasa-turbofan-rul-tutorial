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

## วิธีรัน

ติดตั้งไลบรารีที่ต้องใช้:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

รันทั้งไฟล์:

```bash
python rul_tutorial.py
```

หรือเปิดใน VS Code แล้วกด "Run Cell" ไล่ทีละ block เพื่อดูผลลัพธ์และกราฟทีละขั้น

## สิ่งที่ได้เรียนรู้

1. การโหลด/สำรวจข้อมูล time-series แบบ multivariate
2. การสร้าง label (RUL) เองจากข้อมูลดิบ
3. การเลือกฟีเจอร์ (ตัดเซ็นเซอร์ที่ไม่มีประโยชน์)
4. การเทรนและเปรียบเทียบโมเดล Linear Regression กับ Random Forest
5. การประเมินผลด้วย RMSE/MAE และดู feature importance
