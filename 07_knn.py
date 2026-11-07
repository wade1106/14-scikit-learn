"""
演算法 7：K-最近鄰演算法 (K-Nearest Neighbors, KNN)
業務情境：葡萄酒莊根據化學成分分析判斷葡萄酒等級
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== 資料準備（使用 sklearn 內建葡萄酒資料集）==========
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["葡萄酒等級"] = data.target  # 0, 1, 2 三個等級

# 選取 5 個主要化學指標作為特徵
selected_features = [
    "alcohol",             # 酒精濃度 (%)
    "malic_acid",          # 蘋果酸含量
    "flavanoids",          # 類黃酮含量
    "color_intensity",     # 色澤強度
    "proline"              # 脯胺酸含量
]

print("=== 資料預覽（前 5 筆）===")
print(df[selected_features + ["葡萄酒等級"]].head())
print(f"\n資料筆數：{len(df)}")
print(f"各等級分布：\n{df['葡萄酒等級'].value_counts().sort_index()}")

# ========== 特徵與目標 ==========
X = df[selected_features]
y = df["葡萄酒等級"]

# ========== 資料分割 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 特徵標準化（KNN 基於距離計算，必須標準化）==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 嘗試不同 K 值找最佳 ==========
print("\n=== 不同 K 值的準確率 ===")
best_k, best_acc = 1, 0
for k in range(1, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = knn.score(X_test_scaled, y_test)
    marker = ""
    if acc > best_acc:
        best_k, best_acc = k, acc
        marker = " ← 目前最佳"
    print(f"  K={k:>2d}  準確率={acc:.4f}{marker}")

# ========== 使用最佳 K 值訓練最終模型 ==========
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"\n=== 最終模型（K={best_k}）===")
print(f"準確率：{accuracy_score(y_test, y_pred):.4f}")

print("\n=== 混淆矩陣 ===")
cm = confusion_matrix(y_test, y_pred)
labels = ["等級0", "等級1", "等級2"]
print(f"{'':>12s}  預測{labels[0]}  預測{labels[1]}  預測{labels[2]}")
for i, label in enumerate(labels):
    print(f"  實際{label}  {'    '.join(f'{cm[i][j]:>4}' for j in range(3))}")

print("\n=== 分類報告 ===")
print(classification_report(y_test, y_pred, target_names=labels))
