"""
演算法 5：決策樹 (Decision Tree)
業務情境：電信公司根據客戶行為預測是否流失 (Churn)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== 資料準備 ==========
# 模擬 600 位電信用戶的合約與使用行為
np.random.seed(42)
n = 600

contract_months = np.random.randint(1, 72, n)           # 合約月數
monthly_charge = np.random.uniform(299, 1999, n)         # 月租費 (元)
data_usage_gb = np.random.uniform(0.5, 50, n)            # 月數據用量 (GB)
customer_service_calls = np.random.choice(
    range(0, 8), n, p=[0.3, 0.25, 0.2, 0.1, 0.07, 0.04, 0.02, 0.02]
)  # 近三個月客服電話次數
has_paperless_billing = np.random.choice([0, 1], n, p=[0.4, 0.6])  # 是否電子帳單

# 流失邏輯：合約短、月租高、客服電話多 → 較易流失
churn_prob = 1 / (1 + np.exp(-(
    -2.0
    - 0.03 * contract_months
    + 0.001 * monthly_charge
    + 0.5 * customer_service_calls
    - 0.01 * data_usage_gb
)))
churn = (np.random.rand(n) < churn_prob).astype(int)

df = pd.DataFrame({
    "合約月數": contract_months,
    "月租費_元": np.round(monthly_charge, 0).astype(int),
    "月數據用量_GB": np.round(data_usage_gb, 1),
    "近三月客服電話次數": customer_service_calls,
    "電子帳單": has_paperless_billing,
    "是否流失": churn  # 0=留存, 1=流失
})

print("=== 資料預覽（前 5 筆）===")
print(df.head())
print(f"\n資料筆數：{len(df)}")
print(f"流失率：{churn.mean():.2%}")

# ========== 特徵與目標 ==========
feature_cols = ["合約月數", "月租費_元", "月數據用量_GB", "近三月客服電話次數", "電子帳單"]
X = df[feature_cols]
y = df["是否流失"]

# ========== 資料分割 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 模型訓練（限制深度避免過擬合）==========
model = DecisionTreeClassifier(
    max_depth=4,           # 限制樹深度
    min_samples_leaf=10,   # 葉節點最少樣本數
    random_state=42
)
model.fit(X_train, y_train)

# ========== 預測與評估 ==========
y_pred = model.predict(X_test)

print("\n=== 混淆矩陣 ===")
cm = confusion_matrix(y_test, y_pred)
print(f"                預測留存  預測流失")
print(f"  實際留存      {cm[0][0]:>6}    {cm[0][1]:>6}")
print(f"  實際流失      {cm[1][0]:>6}    {cm[1][1]:>6}")

print(f"\n準確率：{accuracy_score(y_test, y_pred):.4f}")

print("\n=== 分類報告 ===")
print(classification_report(y_test, y_pred, target_names=["留存", "流失"]))

# ========== 特徵重要度 ==========
print("=== 特徵重要度 ===")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {name:<20s} {imp:.4f} {bar}")

# ========== 決策規則（文字版）==========
print("\n=== 決策樹規則（前 3 層）===")
tree_rules = export_text(model, feature_names=feature_cols, max_depth=3)
print(tree_rules)
