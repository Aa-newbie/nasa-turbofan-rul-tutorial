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
import time
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

# ใช้เช็คความคืบหน้าตอนรันทั้งไฟล์ (โดยเฉพาะหัวข้อ 12 ที่หนักและเงียบนานสุด)
# เรียก log("ข้อความ") ตรงไหนก็ได้ จะปริ้นเวลาที่ผ่านมาตั้งแต่เริ่มสคริปต์กำกับให้
_t0 = time.time()


def log(msg):
    print(f"[{time.time() - _t0:6.0f}s] {msg}")

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
log("หัวข้อ 1) โหลดข้อมูลเสร็จ")

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

# ลากเส้นค่าเฉลี่ยพาดไว้ เพื่อให้เห็นว่าเครื่องไหนอายุสั้น/ยาวกว่าค่ากลางแค่ไหน
mean_cycles = cycles_per_unit.mean()
mean_line = plt.axhline(mean_cycles, color="crimson", linestyle="--", linewidth=1.5,
                        label=f"ค่าเฉลี่ย {mean_cycles:.0f} รอบ")
# ระบุ handle เอง ไม่งั้น legend จะแถมชื่อคอลัมน์ "time_cycles" ของแท่งมาด้วย
plt.legend(handles=[mean_line])

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
log("หัวข้อ 5) Linear Regression baseline เสร็จ")

# %% [markdown]
# ## 6) โมเดลที่ 2: Random Forest (โมเดลที่ซับซ้อนขึ้น จับ pattern แบบไม่เชิงเส้นได้)

rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

rmse_rf = mean_squared_error(y_test, pred_rf) ** 0.5
mae_rf = mean_absolute_error(y_test, pred_rf)
print(f"[Random Forest]      RMSE = {rmse_rf:.2f}  MAE = {mae_rf:.2f}")
log("หัวข้อ 6) Random Forest baseline เสร็จ")

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
# ## 8) แบ่ง validation set ให้ถูกหลัก
#
# จนถึงตอนนี้เราวัดผลด้วย test set โดยตรง ซึ่ง "ใช้ดูผลครั้งเดียวตอนจบ" ได้ แต่ห้าม
# ใช้เลือกโมเดล เพราะถ้าลองสิบแบบแล้วหยิบตัวที่ได้เลขสวยที่สุด เท่ากับเราค่อย ๆ ปรับ
# ตัวเองให้เข้ากับข้อสอบ ตัวเลขในรายงานจะดูดีเกินความจริง
#
# วิธีที่ถูกคือกันเครื่องยนต์ส่วนหนึ่งจากชุด train ออกมาเป็น validation set
# ใช้ชุดนั้นตัดสินใจทุกอย่าง แล้วค่อยแตะ test set ครั้งเดียวตอนท้าย
#
# **จุดสำคัญ: ต้องแบ่งตาม "เครื่องยนต์" ไม่ใช่ตาม "แถว"** ถ้าสุ่มทีละแถว รอบที่ 100
# กับรอบที่ 101 ของเครื่องเดียวกันจะกระจายไปคนละฝั่ง ทั้งที่สองแถวนั้นแทบเหมือนกัน
# แต่เหตุผลที่หนักแน่นกว่าคือ มันไม่ตรงกับโจทย์จริง ที่เราต้องทำนายเครื่องยนต์
# "ตัวใหม่ที่ไม่เคยเห็นทั้งเครื่อง" ไม่ใช่เติมช่องว่างของเครื่องที่รู้จักอยู่แล้ว

from sklearn.ensemble import HistGradientBoostingRegressor

VAL_UNITS = 20
rng = np.random.default_rng(42)
shuffled_units = rng.permutation(train["unit_number"].unique())
val_units, fit_units = shuffled_units[:VAL_UNITS], shuffled_units[VAL_UNITS:]

fit_df = train[train["unit_number"].isin(fit_units)]
val_df = train[train["unit_number"].isin(val_units)]


# ชุด validation ต้องจำลองเงื่อนไขเดียวกับ test set คือ "ถูกตัดจบกลางคัน"
# ถ้าหยิบแถวสุดท้ายของแต่ละเครื่องมาตรง ๆ RUL จะเป็น 0 หมดทุกตัว (เพราะ train
# วิ่งจนพังจริง) กลายเป็นชุดวัดผลที่ไร้ประโยชน์ จึงต้องสุ่มจุดตัดเลียนแบบที่ NASA ทำ
#
# และต้องสุ่ม "หลายจุดต่อเครื่อง" ด้วย ถ้าเอาเครื่องละจุดจะได้ตัวอย่างแค่ 20 ตัว
# ซึ่งน้อยเกินกว่าจะวัดอะไรได้นิ่ง ๆ — ทดลองเปลี่ยนแค่ seed ของจุดตัดแล้ว RMSE
# แกว่งได้ถึง 5 หน่วย ซึ่งกว้างกว่าความต่างระหว่างโมเดลทุกตัวที่เราเทียบกันเสียอีก
# เท่ากับเครื่องมือวัดหยาบกว่าสิ่งที่จะวัด พอเพิ่มเป็น 30 จุดต่อเครื่อง ความแกว่ง
# เหลือราว 0.3 จึงเริ่มเชื่อถือได้
VAL_CUTS_PER_UNIT = 30


def cut_at_random(df, n_cuts=VAL_CUTS_PER_UNIT, seed=42):
    """สุ่มจุดตัดหลายจุดต่อเครื่อง เลียนแบบวิธีสร้างชุด test ของ NASA"""
    r = np.random.default_rng(seed)
    rows = []
    for _, g in df.groupby("unit_number"):
        picked = np.unique(r.integers(len(g) // 4, len(g), size=n_cuts))
        rows.append(g.iloc[picked])
    return pd.concat(rows)


val_cut = cut_at_random(val_df)
print(f"เทรนด้วย {len(fit_units)} เครื่อง / วัดผลด้วย {len(val_units)} เครื่องที่ไม่เคยเห็น")
print(f"ขนาด validation set: {len(val_cut)} ตัวอย่าง "
      f"(สุ่มจุดตัดเครื่องละไม่เกิน {VAL_CUTS_PER_UNIT} จุด)")
print("ช่วง RUL ของ validation set:", val_cut["RUL"].min(), "-", val_cut["RUL"].max())
log("หัวข้อ 8) แบ่ง validation set เสร็จ")

# %% [markdown]
# ## 9) โมเดลที่ 3: HistGradientBoosting
#
# Gradient Boosting ปลูกต้นไม้ทีละต้น โดยให้ต้นใหม่คอยแก้ความผิดพลาดที่ต้นก่อนหน้า
# ทำไว้ ต่างจาก Random Forest ที่ปลูกทุกต้นพร้อมกันแล้วเฉลี่ยผล การแก้ต่อกันเป็นทอด ๆ
# แบบนี้มักให้ความแม่นสูงกว่า และเวอร์ชัน "Hist" ของ sklearn เร็วเป็นพิเศษเพราะ
# จัดค่าตัวเลขเป็นช่วง ๆ (histogram) ก่อนคำนวณ แทนที่จะไล่ดูทุกค่า
#
# ตัวนี้ติดมากับ sklearn อยู่แล้ว ไม่ต้องติดตั้งไลบรารีเพิ่ม


def evaluate(model, X_fit, y_fit, X_eval, y_eval):
    """เทรนโมเดลแล้วคืนค่า (RMSE, MAE)"""
    model.fit(X_fit, y_fit)
    pred = model.predict(X_eval)
    return mean_squared_error(y_eval, pred) ** 0.5, mean_absolute_error(y_eval, pred)


def make_models():
    """สร้างโมเดลชุดใหม่ทุกครั้ง เพื่อไม่ให้ตัวที่เทรนไปแล้วปนกัน"""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8,
                                               random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }


print("\n--- วัดผลบน validation set (ฟีเจอร์ดิบ 18 ตัว) ---")
val_scores_base = {}
for name, model in make_models().items():
    rmse, mae = evaluate(model, fit_df[feature_cols], fit_df["RUL"],
                         val_cut[feature_cols], val_cut["RUL"])
    val_scores_base[name] = rmse
    print(f"  {name:<22} RMSE = {rmse:6.2f}  MAE = {mae:6.2f}")

# %% [markdown]
# ## 10) Feature Engineering: ให้โมเดลเห็น "ประวัติ" ไม่ใช่แค่ภาพนิ่ง
#
# นี่คือจุดที่ให้ผลตอบแทนสูงที่สุดในบทเรียนนี้ มากกว่าการเปลี่ยนอัลกอริทึมเสียอีก
#
# ปัญหาของทุกอย่างที่ทำมา: โมเดลเห็นข้อมูลแต่ละแถวแบบโดดเดี่ยว มันรู้ว่า "ตอนนี้
# s_11 = 47.5" แต่ไม่รู้ว่าเมื่อ 20 รอบก่อนเป็นเท่าไหร่ ทั้งที่ข้อมูลชุดนี้เป็น
# time-series และ **อัตราการเปลี่ยนแปลงคือสัญญาณการเสื่อมสภาพตัวจริง**
#
# ย้อนกลับไปดูกราฟเทรนด์เซ็นเซอร์ในหัวข้อที่ 2 จะเห็นว่าเส้นค่อย ๆ ไต่ขึ้นและแกว่ง
# มากขึ้นเมื่อใกล้พัง เราจึงสร้างฟีเจอร์ที่จับสองอย่างนี้ออกมาตรง ๆ:
#
# | ฟีเจอร์ | ความหมาย |
# |---|---|
# | `_mean20` | ค่าเฉลี่ย 20 รอบล่าสุด — กรอง noise ออก เห็นแนวโน้มชัดขึ้น |
# | `_std20`  | ความแกว่ง 20 รอบล่าสุด — เครื่องใกล้พังมักอ่านค่าไม่นิ่ง |
# | `_delta`  | ต่างจากตอนเครื่องใหม่เท่าไหร่ — วัดว่าเสื่อมไปไกลแค่ไหนแล้ว |
#
# ทั้งสามคำนวณจาก "อดีตของเครื่องนั้นเอง" เท่านั้น ไม่ได้แอบดูอนาคต จึงไม่เป็น
# data leakage และคำนวณกับชุด test ได้เหมือนกันทุกประการ

WINDOW = 20


def add_window_features(df, cols, window=WINDOW):
    """เพิ่มฟีเจอร์ที่สรุปประวัติย้อนหลังของเครื่องยนต์แต่ละเครื่อง"""
    df = df.copy()
    grp = df.groupby("unit_number")[cols]
    parts = [df]
    for stat in ["mean", "std"]:
        rolled = grp.rolling(window, min_periods=1).agg(stat)
        rolled = rolled.reset_index(level=0, drop=True)
        rolled.columns = [f"{c}_{stat}{window}" for c in cols]
        parts.append(rolled)
    delta = df[cols] - grp.transform("first")
    delta.columns = [f"{c}_delta" for c in cols]
    parts.append(delta)
    return pd.concat(parts, axis=1).fillna(0)


train_w = add_window_features(train, feature_cols)
test_w = add_window_features(test, feature_cols)
feature_cols_w = [c for c in train_w.columns if c not in drop_cols]
print(f"\nฟีเจอร์เพิ่มจาก {len(feature_cols)} ตัว เป็น {len(feature_cols_w)} ตัว")

# ใช้เครื่องยนต์ชุดเดิมในการแบ่ง เพื่อให้เทียบกับผลก่อนหน้าได้อย่างเป็นธรรม
fit_w = train_w[train_w["unit_number"].isin(fit_units)]
val_w_cut = cut_at_random(train_w[train_w["unit_number"].isin(val_units)])

print("\n--- วัดผลบน validation set (ฟีเจอร์ + window) ---")
val_scores_win = {}
for name, model in make_models().items():
    rmse, mae = evaluate(model, fit_w[feature_cols_w], fit_w["RUL"],
                         val_w_cut[feature_cols_w], val_w_cut["RUL"])
    val_scores_win[name] = rmse
    diff = rmse - val_scores_base[name]
    print(f"  {name:<22} RMSE = {rmse:6.2f}  MAE = {mae:6.2f}   ({diff:+.2f} จากเดิม)")

# %% [markdown]
# ## 11) ตัดสินใจครั้งเดียว แล้ววัดผลจริงบน test set
#
# ถึงตรงนี้เราเลือกทั้งโมเดลและชุดฟีเจอร์จาก validation set ล้วน ๆ โดยไม่เคยแตะ
# test set เลย ขั้นสุดท้ายคือ **เทรนใหม่ด้วยเครื่องยนต์ทั้ง 100 เครื่อง** (ยิ่งข้อมูล
# เยอะยิ่งดี และไม่ต้องกัน validation ไว้แล้วเพราะตัดสินใจเสร็จแล้ว) จากนั้นวัดกับ
# test set ครั้งเดียว
#
# ตัวเลขที่ได้ตรงนี้คือตัวเลขที่เอาไปรายงานได้อย่างซื่อสัตย์

best_name = min(val_scores_win, key=val_scores_win.get)
print(f"\nโมเดลที่ validation บอกว่าดีที่สุด: {best_name}")

test_w_last = test_w.groupby("unit_number").last().reset_index()
final_model = make_models()[best_name]
rmse_final, mae_final = evaluate(final_model, train_w[feature_cols_w], train_w["RUL"],
                                 test_w_last[feature_cols_w], y_test)

print("\n=== ผลสุดท้ายบน test set ===")
print(f"  Linear Regression (ฟีเจอร์ดิบ)   RMSE = {rmse_lr:6.2f}  MAE = {mae_lr:6.2f}")
print(f"  Random Forest (ฟีเจอร์ดิบ)       RMSE = {rmse_rf:6.2f}  MAE = {mae_rf:6.2f}")
print(f"  {best_name} + window  RMSE = {rmse_final:6.2f}  MAE = {mae_final:6.2f}")
log("หัวข้อ 11) เลือกแชมป์รอบแรก + วัด test เสร็จ (กำลังเข้า section 12 ซึ่งช้าสุด)")

# evaluate() เพิ่ง .fit() ให้ final_model ไปแล้วข้างบน (โมเดลถูกแก้ไข้ใน object เดิม
# เพราะไพทอนส่ง object ผ่านตัวแปรแบบ reference) เลยเรียก .predict() ซ้ำได้เลย
# โดยไม่ต้องเทรนใหม่ เอามาวาดกราฟเทียบค่าทายกับค่าจริงของโมเดลแชมป์ตัวจริง
pred_final = np.clip(final_model.predict(test_w_last[feature_cols_w]), 0, RUL_CLIP)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_final, alpha=0.6, color="#2f6f4e")
plt.plot([0, RUL_CLIP], [0, RUL_CLIP], "k--", label="ทายถูกเป๊ะ (ideal)")
plt.xlabel("RUL จริง")
plt.ylabel("RUL ที่โมเดลทาย")
plt.title(f"{best_name} + window บน test set (RMSE {rmse_final:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_final_pred_vs_actual.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 4))
bar_names = ["Linear\nRegression", "Random\nForest", f"{best_name}\n+ window"]
bar_values = [rmse_lr, rmse_rf, rmse_final]
bars = plt.bar(bar_names, bar_values, color=["#9aa5b1", "#6b8cae", "#2f6f4e"])
plt.bar_label(bars, fmt="%.2f", padding=3)
plt.ylabel("RMSE (ยิ่งน้อยยิ่งดี)")
plt.title("เปรียบเทียบความแม่นยำบน test set")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_model_comparison.png", dpi=120)
plt.close()

# %% [markdown]
# ## 12) รายงานผลให้ซื่อสัตย์: หลาย seed + ผสมโมเดล
#
# ตัวเลข RMSE ที่ได้จากการรันครั้งเดียวเชื่อไม่ได้เต็มร้อย เพราะโมเดลกลุ่มต้นไม้มี
# การสุ่มอยู่ข้างใน (สุ่มเลือกข้อมูลและฟีเจอร์ตอนปลูกแต่ละต้น) เปลี่ยนแค่
# `random_state` ผลก็ขยับแล้ว
#
# หัวข้อนี้ทำสองอย่างที่ควรทำก่อนเอาตัวเลขไปใส่รายงาน:
#
# 1. **รันหลาย seed แล้วรายงานค่าเฉลี่ย ± ส่วนเบี่ยงเบน** แทนเลขเดี่ยว
# 2. **ผสมโมเดล (ensemble)** เฉลี่ยคำทำนายของหลายโมเดล ซึ่งมักได้ผลดีกว่าและ
#    เสถียรกว่าโมเดลเดี่ยว เพราะแต่ละตัวผิดคนละแบบ ความผิดพลาดจึงหักล้างกันบางส่วน

from sklearn.ensemble import ExtraTreesRegressor

SEEDS = [0, 1, 2, 3, 4]


def models_for_seed(seed):
    return {
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8,
                                               random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, max_depth=8,
                                           random_state=seed, n_jobs=-1),
    }


def run_seeds(X_fit, y_fit, X_eval, y_eval, label=""):
    """เทรนทุกโมเดลด้วยหลาย seed แล้วคืน RMSE ของแต่ละตัว รวมทั้งแบบผสม

    เป็นจุดที่ใช้เวลานานที่สุดในสคริปต์ (เทรน 3 โมเดล x 5 seed = 15 ครั้งต่อการเรียก
    1 ครั้ง) เลยปริ้นความคืบหน้าทีละ seed กันเข้าใจผิดว่าโปรแกรมค้าง
    """
    scores = {name: [] for name in models_for_seed(0)}
    scores["Ensemble (เฉลี่ย 3 ตัว)"] = []
    for i, seed in enumerate(SEEDS, 1):
        preds = {}
        for name, model in models_for_seed(seed).items():
            model.fit(X_fit, y_fit)
            preds[name] = np.clip(model.predict(X_eval), 0, RUL_CLIP)
            scores[name].append(mean_squared_error(y_eval, preds[name]) ** 0.5)
        ens = np.mean(list(preds.values()), axis=0)
        scores["Ensemble (เฉลี่ย 3 ตัว)"].append(mean_squared_error(y_eval, ens) ** 0.5)
        log(f"{label} seed {i}/{len(SEEDS)} เสร็จแล้ว")
    return scores


def summarise(scores, title):
    print(f"\n{title}")
    print(f"  {'โมเดล':<26}{'เฉลี่ย':>9}{'± s.d.':>9}{'ต่ำสุด-สูงสุด':>18}")
    print("  " + "-" * 60)
    for name, v in scores.items():
        v = np.array(v)
        print(f"  {name:<26}{v.mean():>9.2f}{v.std():>9.2f}"
              f"{f'{v.min():.2f} - {v.max():.2f}':>18}")
    return {k: float(np.mean(v)) for k, v in scores.items()}


# --- ตัดสินใจบน validation ตามเดิม ---
log("หัวข้อ 12) เริ่มรอบ validation (5 seed x 3 โมเดล = 15 ครั้ง)")
val_scores = run_seeds(fit_w[feature_cols_w], fit_w["RUL"],
                       val_w_cut[feature_cols_w], val_w_cut["RUL"], label="[validation]")
val_mean = summarise(val_scores, f"--- validation set ({len(SEEDS)} seeds) ---")

# กฎการเลือก ประกาศไว้ก่อนดูผล test:
#   1. เอาตัวที่ RMSE เฉลี่ยต่ำสุดเป็นตัวตั้ง
#   2. ถ้ามีตัวอื่นห่างไม่เกิน TIE_MARGIN ถือว่า "เสมอกัน" ในทางสถิติ
#      แล้วเลือกตัวที่ส่วนเบี่ยงเบนต่ำที่สุดในกลุ่มที่เสมอกันแทน
#
# เหตุผล: เมื่อความแม่นพอ ๆ กัน โมเดลที่ผลไม่แกว่งตาม seed ย่อมน่าเชื่อถือกว่า
# ทำซ้ำได้จริงกว่า และเป็นเกณฑ์ที่ป้องกันไม่ให้เราไปเลือกตัวที่บังเอิญได้ seed ดี
TIE_MARGIN = 0.5

leader = min(val_mean, key=val_mean.get)
tied = [n for n, m in val_mean.items() if m - val_mean[leader] <= TIE_MARGIN]
champion = min(tied, key=lambda n: float(np.std(val_scores[n])))

if len(tied) > 1:
    print(f"\n  เสมอกันภายใน {TIE_MARGIN} RMSE: {', '.join(tied)}")
    print(f"  ตัดสินด้วยความเสถียร (s.d. ต่ำสุด)")
print(f"\n  >>> validation เลือก: {champion}")

# %% [markdown]
# ### วัดผลบน test set
#
# ตารางข้างล่างแสดงทุกโมเดลเพื่อการเรียนรู้ แต่ **ตัวที่เราเลือกถูกตัดสินจาก
# validation ไปแล้ว** การเห็นตัวเลข test ของตัวอื่นไม่ใช่ใบอนุญาตให้ย้อนกลับไป
# เปลี่ยนใจ — ถ้าทำแบบนั้นก็กลับไปเป็นการเลือกจากข้อสอบอีก
#
# สิ่งที่ตัวเลขชุดนี้บอกได้อย่างซื่อสัตย์คือ **ความไม่แน่นอนของผล** ซึ่งเป็นสิ่งที่
# ต้องรายงานคู่กับค่าเฉลี่ยเสมอ

log("หัวข้อ 12) เริ่มรอบ test (5 seed x 3 โมเดล = 15 ครั้ง, เทรนด้วย train เต็ม 100 เครื่อง ช้ากว่ารอบ validation)")
test_scores = run_seeds(train_w[feature_cols_w], train_w["RUL"],
                        test_w_last[feature_cols_w], y_test, label="[test]")
test_mean = summarise(test_scores, f"--- test set ({len(SEEDS)} seeds) ---")

champ = np.array(test_scores[champion])
print(f"\n  ผลที่ควรรายงานในเล่ม: {champion} -> RMSE {champ.mean():.2f} ± {champ.std():.2f}")
print(f"  (เทียบกับการรายงานเลขเดี่ยวจากการรันครั้งเดียว ซึ่งอาจได้ตั้งแต่ "
      f"{champ.min():.2f} ถึง {champ.max():.2f} แล้วแต่ดวง)")
log("หัวข้อ 12) เสร็จสมบูรณ์ (ส่วนที่ช้าสุดผ่านไปแล้ว ที่เหลือเร็ว)")

# %%
plt.figure(figsize=(8, 4.5))
names = list(test_scores)
means = [np.mean(test_scores[n]) for n in names]
stds = [np.std(test_scores[n]) for n in names]
colors = ["#2f6f4e" if n == champion else "#9aa5b1" for n in names]

plt.bar(range(len(names)), means, yerr=stds, capsize=6, color=colors)
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m + s + 0.15, f"{m:.2f}\n±{s:.2f}", ha="center", fontsize=9)
plt.xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=9)
plt.ylabel("RMSE (ยิ่งน้อยยิ่งดี)")
plt.ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25)
plt.title(f"ความแม่นยำบน test set — เฉลี่ยจาก {len(SEEDS)} seed พร้อมแถบความคลาดเคลื่อน")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_seed_variance.png", dpi=120)
plt.close()

# %% [markdown]
# ### ข้อควรรู้เรื่องความไม่แน่นอน
#
# นอกจากความสุ่มของโมเดลแล้ว ยังมีความไม่แน่นอนอีกชั้นที่ใหญ่กว่ามาก:
# **test set มีเครื่องยนต์แค่ 100 เครื่อง**
#
# ถ้าประมาณด้วยวิธี bootstrap (สุ่มเลือกเครื่องยนต์มาวัดซ้ำหลายพันรอบ) จะพบว่า
# ช่วงความเชื่อมั่น 95% ของ RMSE กว้างราว ±2.3 ซึ่งกว้างกว่าความต่างระหว่าง
# โมเดลทุกตัวในตารางข้างบน
#
# **สรุปที่ควรจำ: ถ้าโมเดลสองตัวต่างกันไม่ถึง 1 RMSE บนข้อมูลชุดนี้ ยังสรุปไม่ได้
# ว่าตัวไหนดีกว่า** — ในเล่มควรเขียนว่า "ให้ผลใกล้เคียงกัน" แทนที่จะประกาศผู้ชนะ
# เพราะถ้ากรรมการถามว่าต่างกันอย่างมีนัยสำคัญไหม จะได้ตอบได้

# %% [markdown]
# ## 13) สร้างรายงานสรุปเป็นไฟล์ HTML
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
    ("ค่าที่โมเดลทาย เทียบกับค่าจริง (baseline: Linear Regression / Random Forest)",
     "03_pred_vs_actual.png"),
    ("ความสำคัญของฟีเจอร์ (Random Forest)", "04_feature_importance.png"),
    ("เปรียบเทียบความแม่นยำของทุกโมเดล", "05_model_comparison.png"),
    (f"ค่าที่โมเดลแชมป์ทาย เทียบกับค่าจริง ({best_name} + window)",
     "08_final_pred_vs_actual.png"),
    ("ความไม่แน่นอนของผล เมื่อรันหลาย seed", "07_seed_variance.png"),
]

img_html = ""
for title, filename in report_images:
    b64 = _img_to_base64(f"{OUT_DIR}/{filename}")
    img_html += f'<h3>{title}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%"><br>'

# ตารางหลักของรายงาน: เอาโมเดลชุดเดียวกัน 4 ตัว (จากหัวข้อ 12) มาวางเทียบกัน
# 2 คอลัมน์ — วัดกับ validation (ใช้ตัดสินใจ) กับวัดกับ test (ยืนยันครั้งเดียว)
# เพื่อให้เห็นตรง ๆ ว่า "โมเดลชุดเดียวกัน ถูกวัดสองรอบ คนละจุดประสงค์" ไม่ใช่
# โมเดลคนละชุดกัน
def _combined_row(name):
    v_mean, v_std = np.mean(val_scores[name]), np.std(val_scores[name])
    t_mean, t_std = np.mean(test_scores[name]), np.std(test_scores[name])
    star, style = ("★ ", ' style="background:#eaf7ef;font-weight:700;"') if name == champion else ("", "")
    return (f"<tr{style}><td>{star}{name}</td>"
            f"<td>{v_mean:.2f} ± {v_std:.2f}</td>"
            f"<td>{t_mean:.2f} ± {t_std:.2f}</td></tr>")


main_table_html = "".join(_combined_row(name) for name in test_scores)

# ไฟล์ report.html จะถูกเขียนจริงท้ายสุดของสคริปต์ (หลังหัวข้อ 14) เพื่อให้รวมผล
# การทดลอง "คัดแล้ว vs ดิบทั้งหมด" เข้าไปในรายงานได้ด้วย ตรงนี้แค่เตรียมชิ้นส่วนไว้ก่อน

# %% [markdown]
# ## 14) ทดลองเพิ่มเติม: ฟีเจอร์ดิบทั้งหมด (ไม่คัดออกเลย) คุ้มไหม
#
# ย้อนกลับไปหัวข้อ 4 — เราตัดเซ็นเซอร์ 6 ตัวที่ "นิ่ง" ทิ้งไปตั้งแต่ต้น (ค่าแทบไม่
# เปลี่ยนเลยตลอดทั้งชุดข้อมูล) เหลือฟีเจอร์ใช้งานจริง 18 ตัว จาก 24 ตัวดิบ
# (3 operating settings + 21 เซ็นเซอร์) มาทดสอบจริงว่าการคัดทิ้งนั้นช่วยหรือเปล่า
# โดยลองโมเดล 2 แบบ x ชุดฟีเจอร์ 2 แบบ = 4 การทดลอง แล้ววัดผลด้วย validation
# set เหมือนเดิม (ไม่แตะ test — นี่แค่การทดลองเสริม ไม่ใช่การตัดสินใจเลือกแชมป์ใหม่)
#
# **Decision Tree** คือโมเดล "ต้นไม้เดี่ยว" ต้นเดียว (ก่อนจะเอาหลายร้อยต้นมารวมกัน
# เป็น Random Forest) เข้าใจง่าย: แบ่งข้อมูลเป็นกิ่งไปเรื่อย ๆ ตามเงื่อนไข "มากกว่า/
# น้อยกว่า" จนถึงใบ (leaf) ที่ให้คำตอบ

# %%
from sklearn.tree import DecisionTreeRegressor

raw_cols = setting_names + sensor_names   # ทุกคอลัมน์ดิบ ไม่คัดอะไรออกเลย
print(f"ฟีเจอร์ที่คัดแล้ว: {len(feature_cols)} ตัว | ฟีเจอร์ดิบทั้งหมด (ไม่คัด): {len(raw_cols)} ตัว")
print(f"เซ็นเซอร์ที่ถูกเพิ่มกลับเข้ามา: {flat_sensors}\n")

raw_experiments = {
    "Linear Regression (คัดแล้ว 18 ตัว)": (LinearRegression(), feature_cols),
    "Linear Regression (ดิบทั้งหมด 24 ตัว)": (LinearRegression(), raw_cols),
    "Decision Tree (คัดแล้ว 18 ตัว)": (DecisionTreeRegressor(max_depth=8, random_state=42), feature_cols),
    "Decision Tree (ดิบทั้งหมด 24 ตัว)": (DecisionTreeRegressor(max_depth=8, random_state=42), raw_cols),
}

raw_results = {}
print(f"{'การทดลอง':<38} RMSE    MAE")
for name, (mdl, cols) in raw_experiments.items():
    rmse, mae = evaluate(mdl, fit_df[cols], fit_df["RUL"], val_cut[cols], val_cut["RUL"])
    raw_results[name] = (rmse, mae)
    print(f"  {name:<36} {rmse:6.2f}  {mae:6.2f}")

# %% [markdown]
# ### อ่านผลยังไง
#
# ถ้า RMSE ของ "ดิบทั้งหมด" กับ "คัดแล้ว" ใกล้เคียงกันมาก (ต่างกันไม่ถึง ~1 RMSE
# ตามกฎที่คุยกันไปในหัวข้อ 12) แปลว่าเซ็นเซอร์ที่นิ่ง 6 ตัวนั้น **ไม่ได้ทำร้ายโมเดล
# แต่ก็ไม่ได้ช่วยอะไรเช่นกัน** — เหตุผลที่ควรตัดทิ้งจึงไม่ใช่เรื่อง "ความแม่นยำ" แต่
# เป็นเรื่อง **ความเรียบง่าย** (ฟีเจอร์น้อยกว่า อธิบายง่ายกว่า เทรนเร็วกว่าเล็กน้อย)
#
# แต่ถ้า RMSE ต่างกันชัดเจน (เกิน 1 RMSE) นั่นคือสัญญาณว่าโมเดลตัวนั้น "สับสน" กับ
# ฟีเจอร์ที่ไม่มีประโยชน์ — เป็นไปได้มากกับ Linear Regression เพราะโมเดลเชิงเส้น
# อ่อนไหวกับฟีเจอร์รบกวนมากกว่าโมเดลต้นไม้ (ต้นไม้เลือกเองได้ว่าจะใช้ฟีเจอร์ไหน
# ตัดจุดไหน ฟีเจอร์ที่ไม่มีประโยชน์จะแทบไม่ถูกเลือกไปใช้แบ่งกิ่งเลย)

# %% [markdown]
# ### ทำซ้ำการทดลองเดิม แต่ใช้ฟีเจอร์แบบ window (72/96 ตัว) แทนฟีเจอร์ดิบ
#
# ข้างบนลองกับฟีเจอร์ดิบ (18/24 ตัว) ไปแล้ว มาดูว่าข้อสรุปเดิมยังจริงอยู่ไหม เมื่อ
# เปลี่ยนไปใช้ฟีเจอร์แบบ window ที่มี mean/std/delta ของแต่ละเซ็นเซอร์ (หัวข้อ 10)
# — คราวนี้ "ดิบทั้งหมด" หมายถึงคำนวณ mean20/std20/delta จากเซ็นเซอร์ทั้ง 24 ตัว
# (รวมตัวนิ่ง 6 ตัวด้วย) ไม่ใช่แค่เอาคอลัมน์ดิบ 6 ตัวนั้นกลับมาเฉย ๆ

# %%
train_w_raw = add_window_features(train, raw_cols)
raw_cols_w = [c for c in train_w_raw.columns if c not in ["unit_number", "time_cycles", "RUL"]]
fit_w_raw = train_w_raw[train_w_raw["unit_number"].isin(fit_units)]
val_w_raw_cut = cut_at_random(train_w_raw[train_w_raw["unit_number"].isin(val_units)])

print(f"window ฟีเจอร์คัดแล้ว: {len(feature_cols_w)} ตัว | "
      f"window ฟีเจอร์ดิบทั้งหมด: {len(raw_cols_w)} ตัว\n")

window_experiments = {
    "Linear Regression (window, คัดแล้ว 72)": (LinearRegression(), fit_w, val_w_cut, feature_cols_w),
    "Linear Regression (window, ดิบทั้งหมด 96)": (LinearRegression(), fit_w_raw, val_w_raw_cut, raw_cols_w),
    "Decision Tree (window, คัดแล้ว 72)": (DecisionTreeRegressor(max_depth=8, random_state=42),
                                           fit_w, val_w_cut, feature_cols_w),
    "Decision Tree (window, ดิบทั้งหมด 96)": (DecisionTreeRegressor(max_depth=8, random_state=42),
                                              fit_w_raw, val_w_raw_cut, raw_cols_w),
}

window_results = {}
print(f"{'การทดลอง':<42} RMSE    MAE")
for name, (mdl, fit_src, val_src, cols) in window_experiments.items():
    rmse, mae = evaluate(mdl, fit_src[cols], fit_src["RUL"], val_src[cols], val_src["RUL"])
    window_results[name] = (rmse, mae)
    print(f"  {name:<40} {rmse:6.2f}  {mae:6.2f}")
log("หัวข้อ 14) ทดลองเสริมเสร็จ")

# %% [markdown]
# ### ทำไมผลรอบนี้อาจต่างจากรอบฟีเจอร์ดิบ
#
# กับฟีเจอร์ดิบ ทั้งสองโมเดลได้ผลแทบเท่ากันไม่ว่าจะคัดหรือไม่คัด เพราะเซ็นเซอร์นิ่ง
# ไม่มีข้อมูลอะไรให้ใช้เลย — แต่กับฟีเจอร์ window เซ็นเซอร์นิ่ง 6 ตัวนั้นก็ถูกนำไป
# คำนวณ mean20/std20/delta ด้วย ซึ่งถึงจะนิ่งในภาพรวมทั้งชุดข้อมูล แต่ในหน้าต่าง
# 20 รอบเล็ก ๆ อาจมีค่าแกว่งเล็กน้อยที่ไม่ใช่ศูนย์เป๊ะ ทำให้เกิดฟีเจอร์ใหม่ที่ไม่ได้
# "นิ่งสนิท" เหมือนต้นทาง — ถ้า Decision Tree เจอฟีเจอร์เหล่านี้แล้ว RMSE ขยับ
# (มากกว่า ~0.5 ซึ่งเป็น margin ที่ใช้ตัดสิน "เสมอกัน" ในหัวข้อ 12) ก็แปลว่าการคัด
# ฟีเจอร์ตั้งแต่ต้นมีผลจริงกับสายการทำงานแบบ window ไม่ใช่แค่ทฤษฎีลอย ๆ อีกต่อไป

# %% [markdown]
# ## 15) เขียนไฟล์ report.html (รวมทุกตาราง)
#
# เก็บชิ้นส่วน HTML ที่เตรียมไว้ตั้งแต่หัวข้อ 13 (`main_table_html`, `seed_rows`,
# `img_html`) มาประกอบกับตารางผลการทดลองของหัวข้อ 14 แล้วเขียนไฟล์จริงตรงนี้ —
# ต้องทำท้ายสุดของสคริปต์ เพราะเป็นจุดแรกที่ข้อมูลทุกส่วนพร้อมครบ

# %%
# คำอธิบายสั้น ๆ ของแต่ละ "ประเภท" โมเดล ใช้ในตาราง "โมเดลที่ใช้ในรายงานนี้คืออะไร"
model_glossary_html = """
<tr><td><b>Linear Regression</b></td><td>ลากเส้นตรง (หรือระนาบ) ให้ fit กับข้อมูลให้ดีที่สุด
ง่ายและตีความง่ายสุด แต่จับความสัมพันธ์ที่ไม่เป็นเส้นตรงไม่ได้</td></tr>
<tr><td><b>Decision Tree</b></td><td>ต้นไม้ตัดสินใจต้นเดียว แบ่งข้อมูลเป็นกิ่งไปเรื่อย ๆ
ตามเงื่อนไข "มากกว่า/น้อยกว่า" จนถึงคำตอบ</td></tr>
<tr><td><b>Random Forest</b></td><td>เอา Decision Tree หลายร้อยต้นมารวมกัน (แต่ละต้นเห็น
ข้อมูลคนละส่วน) แล้วเฉลี่ยคำตอบ แม่นกว่าต้นไม้เดี่ยว</td></tr>
<tr><td><b>Extra Trees</b></td><td>คล้าย Random Forest แต่สุ่มจุดตัดแบบเร็ว ๆ
ไม่พยายามหาจุดตัดที่ดีที่สุดเป๊ะ เทรนเร็วกว่า</td></tr>
<tr><td><b>HistGradientBoosting</b></td><td>ปลูกต้นไม้ทีละต้น โดยต้นใหม่คอยแก้ error
ที่ต้นก่อนหน้าทำพลาด (ต่างจาก Random Forest ที่ปลูกพร้อมกันทุกต้น) มักแม่นกว่า</td></tr>
<tr><td><b>Ensemble</b></td><td>เอาคำตอบจากหลายโมเดลข้างต้นมาเฉลี่ยรวมกัน มักแม่นและ
เสถียรกว่าโมเดลเดี่ยว เพราะแต่ละตัวผิดคนละแบบ ความผิดพลาดหักล้างกันบางส่วน</td></tr>
"""

report_html = f"""<!doctype html>
<html lang="th"><head><meta charset="utf-8">
<title>RUL Prediction - รายงานผล</title>
<style>
:root {{
  --bg: #f5f7fa; --card: #ffffff; --border: #e2e7ee; --text: #1f2430; --muted: #5b6472;
  --accent: #2f6f4e; --test-bg: #e3f5ea; --test-fg: #1f7a4d; --val-bg: #e5eefc; --val-fg: #2159a8;
}}
* {{ box-sizing: border-box; }}
body {{ font-family: "Leelawadee UI", Tahoma, sans-serif; max-width: 900px; margin: 40px auto;
       padding: 0 20px; background: var(--bg); color: var(--text); line-height: 1.65; }}
h1 {{ font-size: 1.7em; }}
h2 {{ font-size: 1.25em; margin-top: 0; }}
h3 {{ font-size: 1.02em; color: var(--accent); margin-bottom: 6px; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px;
        padding: 22px 26px; margin: 20px 0; box-shadow: 0 1px 3px rgba(20,30,40,0.05); }}
table {{ border-collapse: collapse; margin: 10px 0; width: 100%; }}
td, th {{ border: 1px solid var(--border); padding: 8px 12px; text-align: left; }}
th {{ background: #eef1f5; }}
tr:nth-child(even) td {{ background: #fafbfc; }}
.note {{ color: var(--muted); font-size: 0.92em; }}
.badge {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.76em;
         font-weight: 700; letter-spacing: .02em; vertical-align: middle; white-space: nowrap; }}
.badge-test {{ background: var(--test-bg); color: var(--test-fg); }}
.badge-val {{ background: var(--val-bg); color: var(--val-fg); }}
.hero {{ background: linear-gradient(135deg, #eaf7ef, #f4faf6); border: 2px solid var(--accent);
        border-radius: 16px; padding: 22px; text-align: center; margin: 22px 0; }}
.hero .label {{ font-size: 0.82em; color: var(--accent); font-weight: 700; text-transform: uppercase;
               letter-spacing: .06em; }}
.hero .model {{ font-size: 1.5em; font-weight: 800; color: #173d28; margin: 6px 0; }}
.hero .metric {{ font-size: 1.15em; color: #333; }}
.legend {{ background: #fffaf0; border: 1px solid #f0e0b8; border-radius: 10px; padding: 14px 18px;
          margin: 20px 0; font-size: 0.92em; }}
img {{ border-radius: 8px; border: 1px solid var(--border); }}
</style></head>
<body>
<h1>รายงานผล: ทำนาย Remaining Useful Life (RUL)</h1>
<p>โจทย์: จากข้อมูลเซ็นเซอร์ของเครื่องยนต์เจ็ท (NASA C-MAPSS, ชุด FD001) ทำนายว่า
เครื่องยนต์แต่ละเครื่อง "เหลืออีกกี่รอบการทำงานก่อนจะเสีย" (RUL) — เป็นตัวอย่าง
งาน Predictive Maintenance ที่ใช้จริงในอุตสาหกรรม ยิ่ง <b>RMSE/MAE ต่ำ ยิ่งแปลว่า
โมเดลทายแม่น</b> (หน่วยเป็นจำนวนรอบการทำงานที่ทายผิดไปโดยเฉลี่ย)</p>

<div class="hero">
<div class="label">คำตอบสุดท้าย (เฉลี่ยจาก {len(SEEDS)} seed) — วัดกับ Test set</div>
<div class="model">{champion} + window</div>
<div class="metric">RMSE {champ.mean():.2f} ± {champ.std():.2f}</div>
</div>

<div class="legend">
<span class="badge badge-test">Test set</span> = ชุดข้อมูลที่แตะแค่ <b>ครั้งเดียว</b>
หลังตัดสินใจทุกอย่างเสร็จแล้ว → ตัวเลขจากชุดนี้คือ<b>ผลจริงที่รายงานได้อย่างซื่อสัตย์</b>
<br><br>
<span class="badge badge-val">Validation set</span> = ชุดข้อมูลที่กันไว้ต่างหาก ใช้
<b>ทดลอง/เปรียบเทียบ</b>ระหว่างทาง เช็คซ้ำได้ไม่จำกัดครั้ง → ตัวเลขจากชุดนี้ใช้ดู
แนวโน้มได้ แต่<b>ไม่ใช่ผลสรุปสุดท้าย</b>
</div>

<div class="card">
<h2>โมเดลที่ใช้ในรายงานนี้คืออะไร</h2>
<table>
<tr><th>โมเดล</th><th>ทำงานยังไง</th></tr>
{model_glossary_html}
</table>
</div>

<div class="card">
<h2>จุดเริ่มต้นของบทเรียน <span class="badge badge-test">Test set</span></h2>
<p class="note">ก่อนจะรู้จัก window feature หรือ validation set ด้วยซ้ำ บทเรียนนี้
เริ่มจากโมเดลง่ายที่สุด 2 ตัว เทรนด้วย train ทั้งหมด วัดกับ test set ตรงๆ เพื่อตั้ง
เป็น "เส้นฐาน" (baseline) เทียบกับโมเดลที่พัฒนาต่อในตารางถัดไป</p>
<table>
<tr><th>โมเดล</th><th>RMSE</th><th>MAE</th></tr>
<tr><td>Linear Regression</td><td>{rmse_lr:.2f}</td><td>{mae_lr:.2f}</td></tr>
<tr><td>Random Forest</td><td>{rmse_rf:.2f}</td><td>{mae_rf:.2f}</td></tr>
</table>
<p class="note">ทั้งสองตัวนี้แพ้ทุกโมเดลในตารางถัดไป (ที่ต่ำสุดคือ ~13) เพราะยังไม่มี
window feature — เก็บไว้ให้เห็นว่าพัฒนาไปได้ไกลแค่ไหนจากจุดเริ่มต้นง่าย ๆ นี้</p>
</div>

<div class="card">
<h2>โมเดลไหนวัดกับอะไร — ตารางเดียวจบ</h2>
<p class="note">โมเดลทั้ง 4 ตัวด้านล่าง ถูกวัด <b>2 รอบ คนละจุดประสงค์</b>:
<span class="badge badge-val">Validation</span> วัดก่อน ใช้<b>เลือก</b>ว่าจะเอาตัวไหน
(รันซ้ำ {len(SEEDS)} seed ต่อโมเดล) ส่วน <span class="badge badge-test">Test</span>
วัดทีหลัง <b>ครั้งเดียว</b> หลังตัดสินใจเสร็จแล้ว เพื่อยืนยันผลจริง — แถวที่มี ★
และพื้นหลังเขียว คือตัวที่ validation เลือก แล้วพิสูจน์แล้วว่าดีที่สุดบน test จริงด้วย</p>
<table>
<tr><th>โมเดล</th><th>RMSE บน <span class="badge badge-val">Validation</span></th>
<th>RMSE บน <span class="badge badge-test">Test</span></th></tr>
{main_table_html}
</table>
</div>

<div class="card">
<h2>กราฟ</h2>
{img_html}
</div>
</body></html>"""

with open("report.html", "w", encoding="utf-8") as f:
    f.write(report_html)

print("\nสร้างรายงานเสร็จแล้ว: เปิดไฟล์ report.html ด้วยเบราว์เซอร์เพื่อดูผลได้เลย")
log("รันทั้งไฟล์เสร็จสมบูรณ์")
