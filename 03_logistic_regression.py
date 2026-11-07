"""
演算法 3：邏輯斯迴歸 (Logistic Regression)
業務情境：銀行根據客戶資料預測信用卡是否違約
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== 資料準備 ==========
# 模擬 500 位銀行信用卡持卡人資料
np.random.seed(42)
n = 500

age = np.random.randint(22, 65, n)                          # 年齡 (歲)
monthly_income = np.random.uniform(25000, 150000, n)         # 月收入 (元)
credit_limit = np.random.uniform(30000, 500000, n)           # 信用額度 (元)
credit_utilization = np.random.uniform(0.05, 1.0, n)         # 信用使用率 (0~1)
num_late_payments = np.random.choice([0, 1, 2, 3, 4, 5], n,
                                     p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05])  # 近一年遲繳次數

# 違約機率由邏輯函數決定
z = (
    -3.0
    - 0.02 * (age - 30)
    - 0.00003 * monthly_income
    + 2.0 * credit_utilization
    + 0.8 * num_late_payments
)
prob_default = 1 / (1 + np.exp(-z))
default = (np.random.rand(n) < prob_default).astype(int)  # 1=違約, 0=正常

df = pd.DataFrame({
    "年齡_歲": age,
    "月收入_元": np.round(monthly_income, 0),
    "信用額度_元": np.round(credit_limit, 0),
    "信用使用率": np.round(credit_utilization, 2),
    "近一年遲繳次數": num_late_payments,
    "是否違約": default  # 0=正常, 1=違約
})

print("=== 資料預覽（前 5 筆）===")
print(df.head())
print(f"\n資料筆數：{len(df)}")
print(f"違約比例：{default.mean():.2%}")

# ========== 特徵與目標 ==========
feature_cols = ["年齡_歲", "月收入_元", "信用額度_元", "信用使用率", "近一年遲繳次數"]
X = df[feature_cols]
y = df["是否違約"]

# ========== 資料分割 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 分層抽樣保持違約比例
)

# ========== 特徵標準化（邏輯斯迴歸對尺度敏感）==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 模型訓練 ==========
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# ========== 預測與評估 ==========
y_pred = model.predict(X_test_scaled)

print("\n=== 混淆矩陣 ===")
cm = confusion_matrix(y_test, y_pred)
print(f"                預測正常  預測違約")
print(f"  實際正常      {cm[0][0]:>6}    {cm[0][1]:>6}")
print(f"  實際違約      {cm[1][0]:>6}    {cm[1][1]:>6}")

print(f"\n準確率 (Accuracy)：{accuracy_score(y_test, y_pred):.4f}")

print("\n=== 分類報告 ===")
print(classification_report(y_test, y_pred, target_names=["正常", "違約"]))

# ========== 特徵係數解讀 ==========
print("=== 特徵係數（標準化後）===")
for name, coef in zip(feature_cols, model.coef_[0]):
    direction = "↑ 增加違約風險" if coef > 0 else "↓ 降低違約風險"
    print(f"  {name}: {coef:+.4f}  ({direction})")
