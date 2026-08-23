# %% [markdown]
# # บทเรียนต่อยอด: ทำนาย RUL ด้วย LSTM (Deep Learning)
#
# ไฟล์นี้แยกจาก `rul_tutorial.py` เพราะ **โครงสร้างข้อมูลคนละแบบกันโดยสิ้นเชิง**
#
# บทเรียนหลักมองข้อมูลเป็นตาราง 2 มิติ (แถว × ฟีเจอร์) แล้วให้โมเดลดูทีละแถว
# ส่วน LSTM กินข้อมูล 3 มิติ (ตัวอย่าง × เวลา × เซ็นเซอร์) คือดู "ช่วงเวลา"
# ทั้งช่วงพร้อมกัน แล้วอ่านแนวโน้มการเสื่อมสภาพออกมาเอง
#
# ในบทเรียนหลักเราต้องบอกโมเดลตรง ๆ ว่าให้ดูค่าเฉลี่ย 20 รอบและความแกว่ง 20 รอบ
# (feature engineering ด้วยมือ) แต่ LSTM ออกแบบมาให้ **หาเองว่าควรสนใจอะไรในลำดับ**
#
# ต้องติดตั้ง PyTorch ก่อน:
# ```
# pip install -r requirements-lstm.txt
# ```

# %%
import sys
import numpy as np
import pandas as pd
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams["font.family"] = "Leelawadee UI"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "plots"
SEQ_LEN = 30        # ดูข้อมูลย้อนหลังกี่รอบต่อหนึ่งตัวอย่าง
RUL_CLIP = 125      # เพดาน RUL เท่ากับบทเรียนหลัก เพื่อให้เทียบผลกันได้
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
print("PyTorch", torch.__version__, "| ใช้", torch.get_num_threads(), "threads")

# %% [markdown]
# ## 1) โหลดข้อมูลและสร้าง label (เหมือนบทเรียนหลักทุกประการ)

# %%
index_names = ["unit_number", "time_cycles"]
setting_names = ["setting_1", "setting_2", "setting_3"]
sensor_names = [f"s_{i}" for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

train = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)
test = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=col_names)
true_rul = pd.read_csv("data/RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])

flat_sensors = train[sensor_names].std()[lambda s: s < 1e-6].index.tolist()
feature_cols = [c for c in setting_names + sensor_names if c not in flat_sensors]

train["RUL"] = (train.groupby("unit_number")["time_cycles"].transform("max")
                - train["time_cycles"]).clip(upper=RUL_CLIP)

y_test = true_rul["RUL"].clip(upper=RUL_CLIP).values.astype(np.float32)
print(f"ฟีเจอร์ {len(feature_cols)} ตัว | train {train.shape} | test {test.shape}")

# %% [markdown]
# ## 2) Normalize — ขั้นตอนที่ neural network ขาดไม่ได้
#
# ต้นไม้อย่าง Random Forest ไม่สนใจสเกลของตัวเลข เพราะมันแค่หาจุดตัด "มากกว่า/น้อยกว่า"
# แต่ neural network คูณค่าด้วยน้ำหนักแล้วบวกกัน **ฟีเจอร์ที่มีค่าหลักพัน (s_3 ≈ 1590)
# จะกลบฟีเจอร์ที่มีค่าหลักหน่วย (setting_1 ≈ 0.0007) จนสนิท** ถ้าไม่ปรับสเกลก่อน
#
# สำคัญ: fit scaler กับชุด train เท่านั้น แล้วเอาไป transform ชุด test
# ถ้า fit รวมกันทั้งสองชุด = เอาข้อมูล test มาใช้ตั้งแต่ตอนเทรน (data leakage)

# %%
scaler = MinMaxScaler()
train_scaled = train.copy()
test_scaled = test.copy()
train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])
test_scaled[feature_cols] = scaler.transform(test[feature_cols])

print("ก่อน normalize — ช่วงค่าของ 3 ฟีเจอร์แรก:")
print(train[feature_cols[:3]].agg(["min", "max"]).round(3).to_string())
print("\nหลัง normalize:")
print(train_scaled[feature_cols[:3]].agg(["min", "max"]).round(3).to_string())

# %% [markdown]
# ## 3) แปลงตารางเป็นลำดับ 3 มิติ
#
# นี่คือหัวใจที่ต่างจากบทเรียนหลัก แทนที่ 1 แถว = 1 ตัวอย่าง เราใช้
# **หน้าต่างเลื่อน (sliding window)** ให้ 1 ตัวอย่าง = ข้อมูล 30 รอบติดกัน
#
# ```
# เครื่องที่ 1 มี 192 รอบ:
#   ตัวอย่างที่ 1 = รอบ 1-30   -> label คือ RUL ณ รอบที่ 30
#   ตัวอย่างที่ 2 = รอบ 2-31   -> label คือ RUL ณ รอบที่ 31
#   ...
#   ตัวอย่างสุดท้าย = รอบ 163-192
# ```
#
# หน้าต่างซ้อนทับกันได้ ทำให้ได้ตัวอย่างเยอะขึ้นมากจากข้อมูลชุดเดิม


# %%
def make_sequences(df, cols, seq_len=SEQ_LEN, with_label=True):
    """แปลงตารางเป็นก้อน 3 มิติ (ตัวอย่าง, เวลา, ฟีเจอร์)"""
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit_number"):
        arr = g[cols].to_numpy(dtype=np.float32)
        if len(arr) < seq_len:                       # เครื่องที่ข้อมูลสั้นกว่าหน้าต่าง
            pad = np.repeat(arr[:1], seq_len - len(arr), axis=0)
            arr = np.vstack([pad, arr])              # เติมด้วยแถวแรกซ้ำ ๆ ข้างหน้า
        rul = g["RUL"].to_numpy(dtype=np.float32) if with_label else None
        for i in range(len(arr) - seq_len + 1):
            xs.append(arr[i:i + seq_len])
            units.append(unit)
            if with_label:
                ys.append(rul[i + seq_len - 1])
    x = np.stack(xs)
    return (x, np.array(ys, dtype=np.float32), np.array(units)) if with_label else (x, np.array(units))


def last_sequence_per_unit(df, cols, seq_len=SEQ_LEN):
    """หยิบหน้าต่างสุดท้ายของแต่ละเครื่อง — ตรงกับโจทย์ของชุด test"""
    out = []
    for _, g in df.groupby("unit_number"):
        arr = g[cols].to_numpy(dtype=np.float32)
        if len(arr) < seq_len:
            pad = np.repeat(arr[:1], seq_len - len(arr), axis=0)
            arr = np.vstack([pad, arr])
        out.append(arr[-seq_len:])
    return np.stack(out)


# แบ่ง validation ตามเครื่องยนต์ เหมือนบทเรียนหลัก
rng = np.random.default_rng(SEED)
shuffled = rng.permutation(train["unit_number"].unique())
val_units, fit_units = shuffled[:20], shuffled[20:]

X_fit, y_fit, _ = make_sequences(train_scaled[train_scaled.unit_number.isin(fit_units)], feature_cols)
X_val, y_val, _ = make_sequences(train_scaled[train_scaled.unit_number.isin(val_units)], feature_cols)
X_test = last_sequence_per_unit(test_scaled, feature_cols)

print(f"X_fit  {X_fit.shape}   <- (ตัวอย่าง, เวลา, ฟีเจอร์)")
print(f"X_val  {X_val.shape}")
print(f"X_test {X_test.shape}  <- เครื่องละ 1 หน้าต่างสุดท้าย")

# %% [markdown]
# ## 4) สร้างโมเดล LSTM
#
# **LSTM (Long Short-Term Memory)** อ่านลำดับทีละก้าว พร้อมกับพก "ความจำ" ติดตัวไป
# ด้วย ที่แต่ละก้าวมันตัดสินใจว่าจะจำอะไรเพิ่ม ลืมอะไรทิ้ง และส่งอะไรออกไป
# ทำให้จับความสัมพันธ์ระยะยาวในลำดับได้ ต่างจาก RNN ธรรมดาที่ลืมของเก่าเร็วเกินไป
#
# โครงสร้างที่ใช้: LSTM 2 ชั้น (hidden 64) แล้วเอา **ผลลัพธ์ที่ก้าวสุดท้าย**
# (ซึ่งสรุปทั้ง 30 รอบไว้แล้ว) ส่งเข้าชั้น Linear เพื่อบีบเหลือตัวเลขเดียว = RUL


# %%
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq_len, hidden)
        return self.head(out[:, -1]).squeeze(-1)   # เอาเฉพาะก้าวสุดท้าย


model = LSTMRegressor(len(feature_cols))
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"\nพารามิเตอร์ที่ต้องเรียน: {n_params:,} ตัว")

# %% [markdown]
# ## 5) เทรน
#
# ต่างจาก sklearn ที่เรียก `.fit()` บรรทัดเดียวจบ — PyTorch ให้เราเขียนวงจรการเรียน
# เองทุกขั้น ซึ่งดูยุ่งกว่าแต่ควบคุมได้ละเอียดกว่ามาก แต่ละรอบ (epoch) ทำ 5 ขั้น:
#
# 1. `optimizer.zero_grad()` — ล้างค่าอนุพันธ์ของรอบก่อน
# 2. `model(xb)` — ทำนาย (forward pass)
# 3. `loss_fn(...)` — วัดว่าผิดไปเท่าไหร่
# 4. `loss.backward()` — คำนวณย้อนกลับว่าน้ำหนักแต่ละตัวมีส่วนผิดแค่ไหน
# 5. `optimizer.step()` — ขยับน้ำหนักไปในทางที่ผิดน้อยลง
#
# เราหาร label ด้วย 125 ให้อยู่ในช่วง 0-1 เพราะ neural network เรียนได้นิ่งกว่า
# เมื่อค่าเป้าหมายไม่ใหญ่เกินไป แล้วค่อยคูณกลับตอนวัดผล

# %%
EPOCHS = 40
BATCH = 256

loader = DataLoader(
    TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit / RUL_CLIP)),
    batch_size=BATCH, shuffle=True,
)
Xv = torch.from_numpy(X_val)
Xt = torch.from_numpy(X_test)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

history = {"train": [], "val": []}
best_rmse, best_state, best_epoch = float("inf"), None, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    train_loss = total / len(loader.dataset)

    model.eval()
    with torch.no_grad():
        pred_val = model(Xv).numpy() * RUL_CLIP
    val_rmse = mean_squared_error(y_val, pred_val) ** 0.5

    history["train"].append(train_loss ** 0.5 * RUL_CLIP)
    history["val"].append(val_rmse)

    # เก็บสถานะของ epoch ที่ validation ดีที่สุดไว้ (early stopping แบบง่าย)
    if val_rmse < best_rmse:
        best_rmse, best_epoch = val_rmse, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:>3}/{EPOCHS}  train RMSE {history['train'][-1]:6.2f}  "
              f"val RMSE {val_rmse:6.2f}")

print(f"\nepoch ที่ validation ดีที่สุด: {best_epoch} (RMSE {best_rmse:.2f})")
model.load_state_dict(best_state)

# %% [markdown]
# ## 6) วัดผลบน test set
#
# ใช้น้ำหนักจาก epoch ที่ validation ดีที่สุด ไม่ใช่ epoch สุดท้าย เพราะเส้น
# validation ไม่ได้ดีขึ้นเรื่อย ๆ แต่จะลงมาถึงจุดหนึ่งแล้ว **แกว่งขึ้นลง** เช่นรอบนี้
# epoch 40 (val RMSE 14.19) แย่กว่า epoch 35 (12.62) ถ้าเอาน้ำหนักของ epoch
# สุดท้ายมาใช้เฉย ๆ ก็เสียของฟรี ๆ
#
# หมายเหตุการอ่านกราฟ: เส้น validation จะอยู่**ต่ำกว่า**เส้น train ซึ่งดูขัดความรู้สึก
# แต่ไม่ใช่ความผิดปกติ — เพราะเส้น train วัดตอนเปิด dropout (สุ่มปิดเซลล์บางส่วน
# เพื่อกันการท่องจำ) ส่วนเส้น validation วัดตอนปิด dropout ใช้โมเดลเต็มกำลัง
# สองเส้นจึงวัดกันคนละเงื่อนไข เทียบขึ้นลงของแต่ละเส้นได้ แต่เทียบข้ามเส้นไม่ได้

# %%
model.eval()
with torch.no_grad():
    pred_test = model(Xt).numpy() * RUL_CLIP
pred_test = np.clip(pred_test, 0, RUL_CLIP)   # RUL ติดลบหรือเกินเพดานเป็นไปไม่ได้

rmse_lstm = mean_squared_error(y_test, pred_test) ** 0.5
mae_lstm = mean_absolute_error(y_test, pred_test)

print("=== ผลบน test set ===")
print(f"  LSTM                            RMSE = {rmse_lstm:6.2f}  MAE = {mae_lstm:6.2f}")
print("\nเทียบกับบทเรียนหลัก (rul_tutorial.py):")
print("  Linear Regression (ฟีเจอร์ดิบ)   RMSE =  20.83  MAE =  16.57")
print("  Random Forest (ฟีเจอร์ดิบ)       RMSE =  17.48  MAE =  12.48")
print("  HistGradientBoosting + window   RMSE =  13.48  MAE =   9.82")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(range(1, EPOCHS + 1), history["train"], label="train")
axes[0].plot(range(1, EPOCHS + 1), history["val"], label="validation")
axes[0].axvline(best_epoch, color="crimson", linestyle="--", linewidth=1,
                label=f"epoch ที่ดีที่สุด ({best_epoch})")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("RMSE")
axes[0].set_title("เส้นทางการเรียนรู้")
axes[0].legend()

axes[1].scatter(y_test, pred_test, alpha=0.6, color="#2f6f4e")
axes[1].plot([0, RUL_CLIP], [0, RUL_CLIP], "k--", label="ทายถูกเป๊ะ (ideal)")
axes[1].set_xlabel("RUL จริง")
axes[1].set_ylabel("RUL ที่ LSTM ทาย")
axes[1].set_title(f"LSTM บน test set (RMSE {rmse_lstm:.2f})")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_lstm.png", dpi=120)
plt.close()
print(f"\nบันทึกกราฟไว้ที่ {OUT_DIR}/06_lstm.png")

# %% [markdown]
# ## สรุป: LSTM คุ้มไหม
#
# ข้อสังเกตที่น่าสนใจสำหรับงานวิทยานิพนธ์ — เทียบกันตรง ๆ ระหว่างสองแนวทาง:
#
# | | HistGradientBoosting + window | LSTM |
# |---|---|---|
# | เวลาเทรน | ไม่กี่วินาที | หลักนาที |
# | ไลบรารีที่ต้องลง | มากับ sklearn | PyTorch (~200 MB) |
# | ต้องออกแบบฟีเจอร์เอง | ต้อง (mean/std/delta) | ไม่ต้อง หาเอง |
# | พารามิเตอร์ | หลักพัน | หลักหมื่น |
# | ต้อง normalize | ไม่ต้อง | ต้อง |
#
# ถ้าตัวเลขออกมาใกล้เคียงกัน นั่นคือข้อค้นพบที่มีน้ำหนักในตัวมันเอง: **โมเดลที่
# เรียบง่ายกว่าและอธิบายได้ง่ายกว่า ให้ผลทัดเทียมกับ deep learning บนข้อมูลชุดนี้**
# ซึ่งในงานซ่อมบำรุงจริงที่ต้องอธิบายให้วิศวกรเชื่อถือ ความเรียบง่ายมีค่ามาก
#
# ทางต่อยอดถ้าอยากดัน LSTM ให้ดีขึ้น: ลองปรับ `SEQ_LEN` (30 → 50), เพิ่ม hidden
# units, ใส่ bidirectional LSTM, หรือเปลี่ยนไปใช้ 1D CNN ซึ่งเร็วกว่าและมักได้ผล
# ใกล้เคียงกันบน dataset นี้
