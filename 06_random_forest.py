"""
演算法 6：隨機森林 (Random Forest)
業務情境：人資部門根據員工資料預測離職風險
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== 資料準備 ==========
# 模擬 800 位企業員工的 HR 數據
np.random.seed(42)
n = 800

satisfaction = np.random.uniform(0.1, 1.0, n)           # 工作滿意度 (0~1)
years_at_company = np.random.randint(0, 20, n)           # 年資 (年)
monthly_hours = np.random.uniform(140, 310, n)           # 月平均工時 (小時)
num_projects = np.random.randint(2, 8, n)                # 參與專案數
last_evaluation = np.random.uniform(0.3, 1.0, n)        # 最近績效考核 (0~1)
salary_level = np.random.choice([0, 1, 2], n, p=[0.35, 0.45, 0.20])  # 薪資等級 0=低,1=中,2=高

# 離職邏輯：滿意度低、工時過高、薪資低 → 容易離職
resign_prob = 1 / (1 + np.exp(-(
    1.5
    - 4.0 * satisfaction
    + 0.01 * monthly_hours
    - 0.05 * years_at_company
    + 0.3 * num_projects
    - 0.5 * salary_level
)))
resigned = (np.random.rand(n) < resign_prob).astype(int)

df = pd.DataFrame({
    "工作滿意度": np.round(satisfaction, 2),
    "年資_年": years_at_company,
    "月平均工時_小時": np.round(monthly_hours, 0).astype(int),
    "參與專案數": num_projects,
    "績效考核分數": np.round(last_evaluation, 2),
    "薪資等級": salary_level,       # 0=低, 1=中, 2=高
    "是否離職": resigned            # 0=留任, 1=離職
})

print("=== 資料預覽（前 5 筆）===")
print(df.head())
print(f"\n資料筆數：{len(df)}")
print(f"離職率：{resigned.mean():.2%}")

# ========== 特徵與目標 ==========
feature_cols = ["工作滿意度", "年資_年", "月平均工時_小時", "參與專案數", "績效考核分數", "薪資等級"]
X = df[feature_cols]
y = df["是否離職"]

# ========== 資料分割 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 模型訓練 ==========
model = RandomForestClassifier(
    n_estimators=100,       # 100 棵決策樹
    max_depth=6,            # 每棵樹最大深度
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1               # 使用所有 CPU 核心平行訓練
)
model.fit(X_train, y_train)

# ========== 預測與評估 ==========
y_pred = model.predict(X_test)

print("\n=== 混淆矩陣 ===")
cm = confusion_matrix(y_test, y_pred)
print(f"                預測留任  預測離職")
print(f"  實際留任      {cm[0][0]:>6}    {cm[0][1]:>6}")
print(f"  實際離職      {cm[1][0]:>6}    {cm[1][1]:>6}")

print(f"\n準確率：{accuracy_score(y_test, y_pred):.4f}")

print("\n=== 分類報告 ===")
print(classification_report(y_test, y_pred, target_names=["留任", "離職"]))

# ========== 特徵重要度（隨機森林的核心優勢）==========
print("=== 特徵重要度排序 ===")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:<16s} {imp:.4f} {bar}")

print("\n→ HR 可優先針對重要度最高的因素制定留才策略")
