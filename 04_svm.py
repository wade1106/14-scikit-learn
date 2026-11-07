"""
演算法 4：支援向量機 (Support Vector Machine, SVM)
業務情境：醫院根據腫瘤細胞特徵判斷良性 vs 惡性
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== 資料準備（使用 sklearn 內建乳癌診斷資料集）==========
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["診斷結果"] = data.target  # 0=惡性, 1=良性

# 選取 4 個臨床上最具代表性的特徵
selected_features = [
    "mean radius",          # 腫瘤平均半徑
    "mean texture",         # 腫瘤平均紋理（灰階標準差）
    "mean perimeter",       # 腫瘤平均周長
    "mean concavity"        # 腫瘤平均凹度
]

print("=== 資料預覽（前 5 筆）===")
print(df[selected_features + ["診斷結果"]].head())
print(f"\n資料筆數：{len(df)}")
print(f"類別分布 → 良性(1)：{(df['診斷結果']==1).sum()} / 惡性(0)：{(df['診斷結果']==0).sum()}")

# ========== 特徵與目標 ==========
X = df[selected_features]
y = df["診斷結果"]

# ========== 資料分割 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 特徵標準化（SVM 對特徵尺度極為敏感）==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 模型訓練 ==========
# 使用 RBF 核函數，適合非線性可分資料
model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
model.fit(X_train_scaled, y_train)

# ========== 預測與評估 ==========
y_pred = model.predict(X_test_scaled)

print("\n=== 混淆矩陣 ===")
cm = confusion_matrix(y_test, y_pred)
print(f"                預測惡性  預測良性")
print(f"  實際惡性      {cm[0][0]:>6}    {cm[0][1]:>6}")
print(f"  實際良性      {cm[1][0]:>6}    {cm[1][1]:>6}")

print(f"\n準確率：{accuracy_score(y_test, y_pred):.4f}")

print("\n=== 分類報告 ===")
print(classification_report(y_test, y_pred, target_names=["惡性", "良性"]))

print(f"支援向量數量：{model.n_support_}")
print(f"  → 惡性類別：{model.n_support_[0]} 個支援向量")
print(f"  → 良性類別：{model.n_support_[1]} 個支援向量")
