# %% [markdown]
# # บทเรียน ML: ทำนายอายุการใช้งานที่เหลือของเครื่องยนต์ (RUL Prediction)
#
# Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)
# โจทย์: จากข้อมูลเซ็นเซอร์ของเครื่องยนต์ ให้ทำนายว่า "ยังเหลือรอบการทำงานอีกกี่รอบ
# ก่อนเครื่องยนต์จะเสีย" (Remaining Useful Life = RUL)
#
# นี่คือปัญหาแบบ Regression (ทำนายค่าตัวเลขต่อเนื่อง) ที่ใช้จริงในงาน
# Predictive Maintenance (ซ่อมบำรุงเชิงพยากรณ์)

# %%
import pandas as pd
import numpy as np
import sys
import matplotlib

# สคริปต์นี้เซฟกราฟลงไฟล์อย่างเดียว ไม่ได้เปิดหน้าต่างโชว์กราฟ ตอนรันทั้งไฟล์
# จาก terminal จึงสั่งใช้ backend "Agg" ที่ไม่ต้องใช้หน้าต่าง ไม่งั้น Tk จะพ่น
# RuntimeError ตอนปิดโปรแกรม ส่วนตอนกด Run Cell ปล่อยให้ Jupyter เลือก backend
# เองเพื่อให้กราฟแสดง inline ได้ตามปกติ
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

OUT_DIR = "plots"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ใช้ฟอนต์ที่รองรับภาษาไทย ไม่งั้นตัวอักษรไทยในกราฟจะกลายเป็นกล่องว่าง
plt.rcParams["font.family"] = "Leelawadee UI"
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1) โหลดข้อมูล
#
# ไฟล์เป็น text ไม่มี header คั่นด้วยช่องว่าง มี 26 คอลัมน์ต่อแถว:
# - unit_number: หมายเลขเครื่องยนต์ (มี 100 เครื่องใน FD001)
# - time_cycles: รอบการทำงานที่ผ่านมา (1, 2, 3, ...) ของเครื่องนั้น
# - setting_1..3: ค่าการตั้งค่าการทำงาน (operating condition)
# - s_1..s_21: ค่าที่อ่านได้จากเซ็นเซอร์ 21 ตัว (อุณหภูมิ, แรงดัน, ความเร็วรอบ ฯลฯ)
#
# ข้อมูล "train" คือประวัติการทำงานตั้งแต่เครื่องยนต์ใหม่ จนกระทั่ง "เสีย" จริง ๆ
# (run-to-failure) ส่วนข้อมูล "test" จะถูกตัดจบก่อนที่เครื่องจะเสีย แล้วให้เราทำนายว่า
# เหลืออีกกี่รอบ คำตอบจริงอยู่ในไฟล์ RUL_FD001.txt

index_names = ["unit_number", "time_cycles"]
setting_names = ["setting_1", "setting_2", "setting_3"]
sensor_names = [f"s_{i}" for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

train = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)
test = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=col_names)
true_rul = pd.read_csv("data/RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])

print("train shape:", train.shape)
print("test shape:", test.shape)
print("จำนวนเครื่องยนต์ใน train:", train["unit_number"].nunique())
print(train.head())

# %% [markdown]
# ## 2) สำรวจข้อมูล (EDA)
#
# ดูก่อนว่าเครื่องยนต์แต่ละเครื่องมีอายุ (จำนวนรอบก่อนเสีย) เท่าไหร่ และเซ็นเซอร์
# ตัวไหน "นิ่ง" ไม่เปลี่ยนแปลงเลย (ถ้าค่าคงที่ตลอด แปลว่าไม่มีประโยชน์ต่อการทำนาย)

cycles_per_unit = train.groupby("unit_number")["time_cycles"].max()
print("อายุเครื่องยนต์ (รอบ) - สถิติ:")
print(cycles_per_unit.describe())

plt.figure(figsize=(8, 4))
cycles_per_unit.sort_values().plot(kind="bar", width=0.9)
plt.xticks([])
plt.xlabel("เครื่องยนต์ (เรียงตามอายุ)")
plt.ylabel("จำนวนรอบก่อนเสีย")
plt.title("อายุการใช้งานของเครื่องยนต์แต่ละเครื่องใน train set")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_engine_life_distribution.png", dpi=120)
plt.close()

sensor_std = train[sensor_names].std()
flat_sensors = sensor_std[sensor_std < 1e-6].index.tolist()
print("เซ็นเซอร์ที่ค่าแทบไม่เปลี่ยน (ไม่มีประโยชน์):", flat_sensors)

# ดูเทรนด์ของเซ็นเซอร์ตัวอย่าง (s_2, s_3, s_4 มักจะเห็นแนวโน้มเสื่อมชัดเจน)
# สำหรับเครื่องยนต์ 3 เครื่องแรก
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)
for ax, sensor in zip(axes, ["s_2", "s_3", "s_4"]):
    for unit in [1, 2, 3]:
        sub = train[train["unit_number"] == unit]
        ax.plot(sub["time_cycles"], sub[sensor], label=f"engine {unit}")
    ax.set_title(f"เซ็นเซอร์ {sensor} เทียบกับรอบการทำงาน")
    ax.set_xlabel("cycle")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_sensor_trends.png", dpi=120)
plt.close()

# %% [markdown]
# ## 3) สร้าง Label: RUL (Remaining Useful Life)
#
# ข้อมูล train ไม่มีคอลัมน์ RUL มาให้ตรง ๆ เราต้องคำนวณเอง
# หลักการง่าย ๆ: ถ้าเครื่องยนต์เครื่องหนึ่งเสียตอนรอบที่ max_cycle
# แล้วตอนนี้อยู่ที่รอบ time_cycles ก็แปลว่า "เหลืออีก" max_cycle - time_cycles รอบ

max_cycle = train.groupby("unit_number")["time_cycles"].transform("max")
train["RUL"] = max_cycle - train["time_cycles"]

print(train[["unit_number", "time_cycles", "RUL"]].head(10))

# หมายเหตุ: ในความเป็นจริง ช่วงแรก ๆ เครื่องยนต์ยังไม่เสื่อมสภาพ RUL แบบเชิงเส้นตรง ๆ
# แบบนี้จะสูงเกินจริงในช่วงต้น เทคนิคที่ใช้กันบ่อยคือ "clip" ค่า RUL ไม่ให้เกินเพดาน
# (เช่น 125) เพราะช่วงที่เครื่องยังใหม่ ไม่ว่าจะรอบไหนก็ไม่ต่างกันมากในทางปฏิบัติ
RUL_CLIP = 125
train["RUL"] = train["RUL"].clip(upper=RUL_CLIP)

# %% [markdown]
# ## 4) เตรียมฟีเจอร์ (Features) สำหรับโมเดล
#
# เราจะตัด unit_number, time_cycles, RUL (label) ออกจากฟีเจอร์
# และตัดเซ็นเซอร์ที่ "นิ่ง" (flat_sensors) ออกด้วย เพราะไม่ช่วยให้โมเดลแยกแยะอะไรได้

drop_cols = ["unit_number", "time_cycles", "RUL"] + flat_sensors
feature_cols = [c for c in train.columns if c not in drop_cols]
print("จำนวนฟีเจอร์ที่ใช้:", len(feature_cols))
print(feature_cols)

X_train = train[feature_cols]
y_train = train["RUL"]

# สำหรับ test set: โจทย์จริงคือ "ทำนาย RUL ณ รอบสุดท้ายที่มีข้อมูล" ของแต่ละเครื่องยนต์
# (เพราะข้อมูล test ถูกตัดจบก่อนเครื่องเสียจริง เราจึงต้องทายว่าเหลืออีกกี่รอบ)
test_last = test.groupby("unit_number").last().reset_index()
X_test = test_last[feature_cols]
y_test = true_rul["RUL"].clip(upper=RUL_CLIP)

# %% [markdown]
# ## 5) โมเดลที่ 1: Linear Regression (baseline ง่ายที่สุด)
#
# เริ่มจากโมเดลง่ายสุดเสมอ เพื่อใช้เป็นเส้นฐาน (baseline) เทียบกับโมเดลที่ซับซ้อนขึ้น

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

rmse_lr = mean_squared_error(y_test, pred_lr) ** 0.5
mae_lr = mean_absolute_error(y_test, pred_lr)
print(f"[Linear Regression]  RMSE = {rmse_lr:.2f}  MAE = {mae_lr:.2f}")

# %% [markdown]
# ## 6) โมเดลที่ 2: Random Forest (โมเดลที่ซับซ้อนขึ้น จับ pattern แบบไม่เชิงเส้นได้)

rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

rmse_rf = mean_squared_error(y_test, pred_rf) ** 0.5
mae_rf = mean_absolute_error(y_test, pred_rf)
print(f"[Random Forest]      RMSE = {rmse_rf:.2f}  MAE = {mae_rf:.2f}")

# %% [markdown]
# ## 7) เปรียบเทียบผลลัพธ์ + ดูว่าเซ็นเซอร์ไหนสำคัญที่สุด
#
# RMSE (Root Mean Squared Error) และ MAE (Mean Absolute Error) คือ "โมเดลทายผิดไปกี่รอบ
# โดยเฉลี่ย" ยิ่งน้อยยิ่งดี (หน่วยเป็นจำนวนรอบการทำงาน)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_rf, alpha=0.6, label="Random Forest")
plt.scatter(y_test, pred_lr, alpha=0.4, label="Linear Regression")
lims = [0, RUL_CLIP]
plt.plot(lims, lims, "k--", label="ทายถูกเป๊ะ (ideal)")
plt.xlabel("RUL จริง")
plt.ylabel("RUL ที่โมเดลทาย")
plt.title("เปรียบเทียบค่าที่ทายได้ กับค่าจริง")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_pred_vs_actual.png", dpi=120)
plt.close()

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nเซ็นเซอร์/ฟีเจอร์ที่โมเดล Random Forest ให้ความสำคัญมากที่สุด:")
print(importances.head(10))

plt.figure(figsize=(7, 5))
importances.head(10).sort_values().plot(kind="barh")
plt.title("ความสำคัญของฟีเจอร์ (Random Forest)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_feature_importance.png", dpi=120)
plt.close()

print("\nสรุปผล:")
print(f"  Linear Regression  -> RMSE {rmse_lr:.2f}, MAE {mae_lr:.2f}")
print(f"  Random Forest      -> RMSE {rmse_rf:.2f}, MAE {mae_rf:.2f}")
print(f"\nรูปภาพทั้งหมดถูกบันทึกไว้ในโฟลเดอร์ '{OUT_DIR}/'")

# %% [markdown]
# ## 8) สร้างรายงานสรุปเป็นไฟล์ HTML
#
# ไฟล์นี้เปิดได้เองด้วยเบราว์เซอร์ทั่วไป (ดับเบิลคลิกได้เลย) ไม่ต้องพึ่งปุ่ม
# Export ของ VS Code ซึ่งบางทีอาจใช้งานไม่ได้ (เพราะกราฟที่ savefig() ไว้
# จะไม่ถูกแสดงใน Interactive Window ตั้งแต่แรก ปุ่ม Export เลย export ออกมา
# ไม่มีรูป) วิธีนี้ฝังรูปภาพไว้ในไฟล์ HTML เดียวเลย จึงส่งให้เพื่อนไฟล์เดียวได้

import base64

def _img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

report_images = [
    ("อายุการใช้งานของแต่ละเครื่องยนต์", "01_engine_life_distribution.png"),
    ("เทรนด์เซ็นเซอร์เทียบกับรอบการทำงาน", "02_sensor_trends.png"),
    ("ค่าที่โมเดลทาย เทียบกับค่าจริง", "03_pred_vs_actual.png"),
    ("ความสำคัญของฟีเจอร์ (Random Forest)", "04_feature_importance.png"),
]

img_html = ""
for title, filename in report_images:
    b64 = _img_to_base64(f"{OUT_DIR}/{filename}")
    img_html += f'<h3>{title}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%"><br>'

report_html = f"""<!doctype html>
<html lang="th"><head><meta charset="utf-8">
<title>RUL Prediction - รายงานผล</title>
<style>
body {{ font-family: "Leelawadee UI", Tahoma, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
td, th {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
</style></head>
<body>
<h1>รายงานผล: ทำนาย Remaining Useful Life (RUL)</h1>
<h2>สรุปความแม่นยำของโมเดล</h2>
<table>
<tr><th>โมเดล</th><th>RMSE</th><th>MAE</th></tr>
<tr><td>Linear Regression</td><td>{rmse_lr:.2f}</td><td>{mae_lr:.2f}</td></tr>
<tr><td>Random Forest</td><td>{rmse_rf:.2f}</td><td>{mae_rf:.2f}</td></tr>
</table>
<h2>กราฟ</h2>
{img_html}
</body></html>"""

with open("report.html", "w", encoding="utf-8") as f:
    f.write(report_html)

print("\nสร้างรายงานเสร็จแล้ว: เปิดไฟล์ report.html ด้วยเบราว์เซอร์เพื่อดูผลได้เลย")
