"""
演算法 2：多元線性回歸 (Multiple Linear Regression)
業務情境：房仲公司根據房屋條件預測成交價格
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 資料準備 ==========
# 模擬 100 筆中古屋成交資料
np.random.seed(42)
n = 100

house_age = np.random.uniform(1, 40, n)          # 屋齡 (年)
floor_area = np.random.uniform(15, 60, n)         # 室內坪數 (坪)
dist_to_mrt = np.random.uniform(50, 2000, n)      # 距最近捷運站距離 (公尺)
num_convenience = np.random.randint(0, 15, n)      # 周邊便利商店數量 (家)

# 房價 = 基礎價 + 各特徵線性關係 + 隨機波動（單位：萬元）
price = (
    300
    - 5 * house_age            # 屋齡越高，價格越低
    + 20 * floor_area          # 坪數越大，價格越高
    - 0.08 * dist_to_mrt       # 離捷運越遠，價格越低
    + 8 * num_convenience      # 周邊便利商店越多（生活機能好），價格越高
    + np.random.normal(0, 50, n)
)
price = np.round(np.maximum(price, 100), 1)  # 確保價格合理

df = pd.DataFrame({
    "屋齡_年": np.round(house_age, 1),
    "室內坪數_坪": np.round(floor_area, 1),
    "距捷運站_公尺": np.round(dist_to_mrt, 0),
    "周邊便利商店數_家": num_convenience,
    "成交價_萬元": price
})

print("=== 資料預覽（前 5 筆）===")
print(df.head())
print(f"\n資料筆數：{len(df)}")

# ========== 特徵與目標 ==========
feature_cols = ["屋齡_年", "室內坪數_坪", "距捷運站_公尺", "周邊便利商店數_家"]
X = df[feature_cols]
y = df["成交價_萬元"]

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
print(f"截距：{model.intercept_:.2f} 萬元")
for name, coef in zip(feature_cols, model.coef_):
    print(f"  {name} 的係數：{coef:.4f}")

print("\n=== 評估指標 ===")
mse = mean_squared_error(y_test, y_pred)
print(f"MSE：{mse:.2f}")
print(f"RMSE：{np.sqrt(mse):.2f} 萬元")
print(f"MAE：{mean_absolute_error(y_test, y_pred):.2f} 萬元")
print(f"R-squared：{r2_score(y_test, y_pred):.4f}")

# ========== 新資料預測示範 ==========
new_house = pd.DataFrame({
    "屋齡_年": [5, 20, 35],
    "室內坪數_坪": [30, 25, 40],
    "距捷運站_公尺": [200, 800, 1500],
    "周邊便利商店數_家": [10, 5, 2]
})
predictions = model.predict(new_house)
print("\n=== 新物件價格預測 ===")
for i, row in new_house.iterrows():
    print(f"  屋齡{row['屋齡_年']}年 / {row['室內坪數_坪']}坪 / "
          f"距捷運{row['距捷運站_公尺']}m / 便利商店{row['周邊便利商店數_家']}家 "
          f"→ 預測成交價：{predictions[i]:.1f} 萬元")
