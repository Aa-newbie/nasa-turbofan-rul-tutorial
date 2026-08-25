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


def run_seeds(X_fit, y_fit, X_eval, y_eval):
    """เทรนทุกโมเดลด้วยหลาย seed แล้วคืน RMSE ของแต่ละตัว รวมทั้งแบบผสม"""
    scores = {name: [] for name in models_for_seed(0)}
    scores["Ensemble (เฉลี่ย 3 ตัว)"] = []
    for seed in SEEDS:
        preds = {}
        for name, model in models_for_seed(seed).items():
            model.fit(X_fit, y_fit)
            preds[name] = np.clip(model.predict(X_eval), 0, RUL_CLIP)
            scores[name].append(mean_squared_error(y_eval, preds[name]) ** 0.5)
        ens = np.mean(list(preds.values()), axis=0)
        scores["Ensemble (เฉลี่ย 3 ตัว)"].append(mean_squared_error(y_eval, ens) ** 0.5)
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
val_scores = run_seeds(fit_w[feature_cols_w], fit_w["RUL"],
                       val_w_cut[feature_cols_w], val_w_cut["RUL"])
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

test_scores = run_seeds(train_w[feature_cols_w], train_w["RUL"],
                        test_w_last[feature_cols_w], y_test)
test_mean = summarise(test_scores, f"--- test set ({len(SEEDS)} seeds) ---")

champ = np.array(test_scores[champion])
print(f"\n  ผลที่ควรรายงานในเล่ม: {champion} -> RMSE {champ.mean():.2f} ± {champ.std():.2f}")
print(f"  (เทียบกับการรายงานเลขเดี่ยวจากการรันครั้งเดียว ซึ่งอาจได้ตั้งแต่ "
      f"{champ.min():.2f} ถึง {champ.max():.2f} แล้วแต่ดวง)")

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
    ("ค่าที่โมเดลทาย เทียบกับค่าจริง", "03_pred_vs_actual.png"),
    ("ความสำคัญของฟีเจอร์ (Random Forest)", "04_feature_importance.png"),
    ("เปรียบเทียบความแม่นยำของทุกโมเดล", "05_model_comparison.png"),
    ("ความไม่แน่นอนของผล เมื่อรันหลาย seed", "07_seed_variance.png"),
]

img_html = ""
for title, filename in report_images:
    b64 = _img_to_base64(f"{OUT_DIR}/{filename}")
    img_html += f'<h3>{title}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%"><br>'

seed_rows = "".join(
    f'<tr><td>{n}</td><td>{np.mean(v):.2f}</td><td>{np.std(v):.2f}</td></tr>'
    for n, v in test_scores.items()
)

# หมายเหตุ: "ฟีเจอร์ดิบ" ในตารางนี้แปลว่า "ยังไม่ผ่าน window feature engineering"
# (เทียบกับแถวสุดท้ายที่มี window) ไม่ใช่ "ไม่ได้คัดเซ็นเซอร์นิ่งออก" — โมเดล 2 ตัวนี้
# ใช้ feature_cols ที่คัดแล้ว (18 ตัว) เหมือนกัน ส่วนการทดลอง "คัดแล้ว vs ดิบทั้งหมด"
# แบบไม่คัดเซ็นเซอร์นิ่งเลย อยู่ในหัวข้อ 14 ท้ายไฟล์ (ตารางแยกต่างหากด้านล่าง)
main_table_html = f"""
<tr><th>โมเดล</th><th>RMSE</th><th>MAE</th></tr>
<tr><td>Linear Regression (ฟีเจอร์คัดแล้ว, ไม่มี window)</td><td>{rmse_lr:.2f}</td><td>{mae_lr:.2f}</td></tr>
<tr><td>Random Forest (ฟีเจอร์คัดแล้ว, ไม่มี window)</td><td>{rmse_rf:.2f}</td><td>{mae_rf:.2f}</td></tr>
<tr><td><b>{best_name} + window</b></td><td><b>{rmse_final:.2f}</b></td><td><b>{mae_final:.2f}</b></td></tr>
"""

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
def _rows_html(results):
    return "".join(
        f"<tr><td>{name}</td><td>{rmse:.2f}</td><td>{mae:.2f}</td></tr>"
        for name, (rmse, mae) in results.items()
    )


# คำอธิบายสั้น ๆ ของแต่ละ "ประเภท" โมเดล ใช้ครั้งเดียวตรงนี้ อ้างอิงได้จากทุกตาราง
# ด้านล่าง เพราะชื่อโมเดลในตารางต่าง ๆ ล้วนเป็นหนึ่งในประเภทเหล่านี้
# กรณีศึกษา Linear Regression: รวมผลทุกรอบที่เคยลองในสคริปต์นี้มาไว้ที่เดียว
# ดึงตัวเลขจริงจากตัวแปรที่คำนวณไว้แล้ว (ไม่ใช่พิมพ์ค่าคงที่) เพื่อให้ถูกต้องเสมอ
# ไม่ว่าจะรันสคริปต์กี่ครั้งก็ตาม
lr_raw_sel = raw_results["Linear Regression (คัดแล้ว 18 ตัว)"]
lr_raw_all = raw_results["Linear Regression (ดิบทั้งหมด 24 ตัว)"]
lr_win_sel = window_results["Linear Regression (window, คัดแล้ว 72)"]
lr_win_all = window_results["Linear Regression (window, ดิบทั้งหมด 96)"]

lr_case_study_html = f"""
<h2>กรณีศึกษา: Linear Regression ทำอะไร</h2>
<p class="note">Linear Regression พยายามหา "สมการเส้นตรง" ที่ทำนายคำตอบจากฟีเจอร์
ทั้งหมด รูปแบบคือเอาแต่ละฟีเจอร์คูณด้วย "น้ำหนัก" ของมัน แล้วบวกกันทั้งหมด — งาน
ของโมเดลคือหาน้ำหนักที่ทำให้ผลรวมนี้ใกล้เคียง RUL จริงที่สุด</p>
<p>ในสคริปต์นี้ Linear Regression ถูกทดลอง 5 รอบ ต่างกันที่ฟีเจอร์และชุดข้อมูล
ที่ใช้วัดผล:</p>
<table>
<tr><th>#</th><th>ฟีเจอร์</th><th>คัดเซ็นเซอร์นิ่งไหม</th><th>จำนวนฟีเจอร์</th>
<th>วัดกับอะไร</th><th>RMSE</th></tr>
<tr><td>1</td><td>ดิบ</td><td>คัดแล้ว</td><td>18</td><td>test set จริง (train เต็ม 100 เครื่อง)</td>
<td><b>{rmse_lr:.2f}</b></td></tr>
<tr><td>2</td><td>ดิบ</td><td>คัดแล้ว</td><td>18</td><td>validation (fit 80 เครื่อง)</td>
<td>{lr_raw_sel[0]:.2f}</td></tr>
<tr><td>3</td><td>ดิบ</td><td>ไม่คัด</td><td>24</td><td>validation</td>
<td>{lr_raw_all[0]:.2f}</td></tr>
<tr><td>4</td><td>window</td><td>คัดแล้ว</td><td>72</td><td>validation</td>
<td>{lr_win_sel[0]:.2f}</td></tr>
<tr><td>5</td><td>window</td><td>ไม่คัด</td><td>96</td><td>validation</td>
<td>{lr_win_all[0]:.2f}</td></tr>
</table>
<p class="note">สิ่งที่เห็นจากตารางนี้: Linear Regression <b>ไม่สนใจว่าจะคัดฟีเจอร์
นิ่งหรือไม่</b> (แถว 2-3 ค่าเท่ากัน, แถว 4-5 ค่าเท่ากัน) เพราะน้ำหนักของฟีเจอร์ที่ไม่
เปลี่ยนแปลงเลยจะถูกคำนวณออกมาใกล้ศูนย์อัตโนมัติ แต่<b>สนใจว่าจะใช้ window features
หรือไม่</b> (แถว 2→4 ดีขึ้นชัดเจน) — สรุปว่า feature engineering (window) สำคัญกว่า
การคัดฟีเจอร์นิ่งทิ้งสำหรับโมเดลนี้ แม้จะปรับฟีเจอร์ดีแค่ไหน Linear Regression ก็
ยังคงเป็นโมเดลที่แม่นน้อยที่สุดเมื่อเทียบกับโมเดลอื่นในรายงานนี้ เพราะข้อจำกัดคือ
"เส้นตรง" จับความสัมพันธ์ที่ไม่เป็นเส้นตรง (เช่น การเสื่อมสภาพที่เร่งขึ้นเรื่อย ๆ
ตอนใกล้พัง) ไม่ได้</p>
"""

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
body {{ font-family: "Leelawadee UI", Tahoma, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
td, th {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
.note {{ color: #555; font-size: 0.92em; }}
</style></head>
<body>
<h1>รายงานผล: ทำนาย Remaining Useful Life (RUL)</h1>
<p>โจทย์: จากข้อมูลเซ็นเซอร์ของเครื่องยนต์เจ็ท (NASA C-MAPSS, ชุด FD001) ทำนายว่า
เครื่องยนต์แต่ละเครื่อง "เหลืออีกกี่รอบการทำงานก่อนจะเสีย" (RUL) — เป็นตัวอย่าง
งาน Predictive Maintenance ที่ใช้จริงในอุตสาหกรรม ยิ่ง <b>RMSE/MAE ต่ำ ยิ่งแปลว่า
โมเดลทายแม่น</b> (หน่วยเป็นจำนวนรอบการทำงานที่ทายผิดไปโดยเฉลี่ย)</p>

<h2>โมเดลที่ใช้ในรายงานนี้คืออะไร</h2>
<table>
<tr><th>โมเดล</th><th>ทำงานยังไง</th></tr>
{model_glossary_html}
</table>

<h2>สรุปความแม่นยำของโมเดล</h2>
<p class="note">เทรนด้วยข้อมูล train ทั้งหมด วัดผลกับ test set (100 เครื่องยนต์ที่ไม่เคย
ใช้เทรนหรือเลือกโมเดลเลย) แถวสุดท้าย (ตัวหนา) คือโมเดลที่ validation set เลือกว่าดี
ที่สุด บวกกับฟีเจอร์แบบ window (ดูคำอธิบาย window feature ด้านล่าง)</p>
<table>
<tr><th>โมเดล</th><th>RMSE</th><th>MAE</th></tr>
{main_table_html}
</table>

<h2>ผลที่รายงานอย่างซื่อสัตย์ (เฉลี่ยจาก {len(SEEDS)} seed)</h2>
<p class="note">โมเดลกลุ่มต้นไม้มีการสุ่มอยู่ข้างใน รันครั้งเดียวได้ตัวเลขเดียวอาจ
เป็นเพราะบังเอิญได้ค่าสุ่มที่ดี ตารางนี้เลยรันซ้ำ {len(SEEDS)} รอบด้วยค่าสุ่มต่างกัน
แล้วรายงานค่าเฉลี่ย ± ส่วนเบี่ยงเบน (ยิ่งส่วนเบี่ยงเบนน้อย ยิ่งเชื่อถือได้ว่าไม่ใช่
แค่โชคช่วย)</p>
<table>
<tr><th>โมเดล</th><th>RMSE เฉลี่ย</th><th>± s.d.</th></tr>
{seed_rows}
</table>

<h2>ทดลองเสริม: ฟีเจอร์ที่คัดแล้ว vs ดิบทั้งหมด (ไม่คัดเลย)</h2>
<p class="note">ตอนต้นบทเรียนตัดเซ็นเซอร์ 6 ตัวที่ค่า "นิ่ง" (แทบไม่เปลี่ยนเลย)
ทิ้งไป ตารางนี้ทดสอบจริงว่าการตัดทิ้งนั้นช่วยความแม่นยำหรือแค่ช่วยให้เรียบง่ายขึ้น
โดยลองใส่เซ็นเซอร์นิ่งกลับเข้าไปเทียบกัน — วัดผลด้วย validation set (20 เครื่องยนต์
ที่กันไว้ต่างหาก) ไม่ใช่ test set เพราะเป็นแค่การทดลองเสริม ไม่ใช่การเลือกแชมป์ใหม่</p>
<h3>ฟีเจอร์ดิบ (18 ตัวคัดแล้ว / 24 ตัวไม่คัด)</h3>
<table>
<tr><th>การทดลอง</th><th>RMSE</th><th>MAE</th></tr>
{_rows_html(raw_results)}
</table>
<h3>ฟีเจอร์ window (72 ตัวคัดแล้ว / 96 ตัวไม่คัด)</h3>
<p class="note"><b>Window feature</b> คือฟีเจอร์ที่คำนวณจาก "ประวัติย้อนหลัง 20 รอบ"
ของแต่ละเซ็นเซอร์ (ค่าเฉลี่ย, ความแกว่ง, ระยะห่างจากค่าตอนเครื่องยังใหม่) แทนที่จะ
ใช้แค่ค่า ณ ปัจจุบันจุดเดียว ช่วยให้โมเดลเห็นแนวโน้มการเสื่อมสภาพ</p>
<table>
<tr><th>การทดลอง</th><th>RMSE</th><th>MAE</th></tr>
{_rows_html(window_results)}
</table>
{lr_case_study_html}
<h2>กราฟ</h2>
{img_html}
</body></html>"""

with open("report.html", "w", encoding="utf-8") as f:
    f.write(report_html)

print("\nสร้างรายงานเสร็จแล้ว (รวมผลหัวข้อ 14): เปิดไฟล์ report.html ด้วยเบราว์เซอร์เพื่อดูผลได้เลย")
