"""
演算法 1：簡單線性回歸 (Simple Linear Regression)
業務情境：便利商店根據當日最高氣溫預測冰品銷售量
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 資料準備 ==========
# 模擬某便利商店 30 天的氣溫與冰品銷售紀錄
np.random.seed(42)
temperature = np.round(np.random.uniform(18, 38, 30), 1)  # 當日最高氣溫 (°C)
# 冰品銷售量大致隨氣溫上升：基礎量 + 線性關係 + 隨機波動
ice_cream_sales = np.round(10 + 3.5 * temperature + np.random.normal(0, 8, 30), 0)

df = pd.DataFrame({
    "當日最高氣溫_攝氏": temperature,
    "冰品銷售量_份": ice_cream_sales
})

print("=== 資料預覽（前 5 筆）===")
print(df.head())
print(f"\n資料筆數：{len(df)}")

# ========== 特徵與目標 ==========
X = df[["當日最高氣溫_攝氏"]]  # 特徵：氣溫 (°C)
y = df["冰品銷售量_份"]         # 目標：冰品銷售量 (份)

# ========== 資料分割 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 模型訓練 ==========
model = LinearRegression()
model.fit(X_train, y_train)

# ========== 預測與評估 ==========
y_pred = model.predict(X_test)

print("\n=== 模型參數 ===")
print(f"截距 (Intercept)：{model.intercept_:.2f}")
print(f"斜率 (Coefficient)：{model.coef_[0]:.2f}")
print(f"  → 意義：氣溫每上升 1°C，冰品銷售量平均增加 {model.coef_[0]:.1f} 份")

print("\n=== 評估指標 ===")
print(f"MSE (均方誤差)：{mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE (平均絕對誤差)：{mean_absolute_error(y_test, y_pred):.2f}")
print(f"R-squared (決定係數)：{r2_score(y_test, y_pred):.4f}")

# ========== 新資料預測示範 ==========
new_temp = pd.DataFrame({"當日最高氣溫_攝氏": [25.0, 33.0, 37.5]})
predictions = model.predict(new_temp)
print("\n=== 新日期銷售預測 ===")
for t, p in zip(new_temp["當日最高氣溫_攝氏"], predictions):
    print(f"  氣溫 {t}°C → 預測冰品銷售量：{p:.0f} 份")
