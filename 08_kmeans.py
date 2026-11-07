"""
演算法 8：K-Means 分群演算法 (K-Means Clustering)
業務情境：電商平台根據消費行為將會員分群，實現精準行銷
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ========== 資料準備 ==========
# 模擬 500 位電商會員的 RFM 消費行為資料
np.random.seed(42)
n = 500

# 高價值客戶群（約 100 人）
n1 = 100
recency_1 = np.random.randint(1, 15, n1)              # 最近一次消費距今天數（短）
frequency_1 = np.random.randint(15, 40, n1)            # 近半年消費次數（高）
monetary_1 = np.random.uniform(8000, 30000, n1)        # 近半年消費金額（高）

# 一般活躍客戶群（約 200 人）
n2 = 200
recency_2 = np.random.randint(10, 45, n2)
frequency_2 = np.random.randint(5, 18, n2)
monetary_2 = np.random.uniform(2000, 10000, n2)

# 沉睡客戶群（約 200 人）
n3 = 200
recency_3 = np.random.randint(40, 120, n3)
frequency_3 = np.random.randint(1, 6, n3)
monetary_3 = np.random.uniform(300, 3000, n3)

recency = np.concatenate([recency_1, recency_2, recency_3])
frequency = np.concatenate([frequency_1, frequency_2, frequency_3])
monetary = np.concatenate([monetary_1, monetary_2, monetary_3])

df = pd.DataFrame({
    "最近消費天數_R": recency,
    "消費次數_F": frequency,
    "消費金額_M": np.round(monetary, 0).astype(int)
})

# 打亂順序（模擬真實情境，不知道事先分群）
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("=== 資料預覽（前 5 筆）===")
print(df.head())
print(f"\n資料筆數：{len(df)}")
print(f"\n=== 基本統計 ===")
print(df.describe().round(1))

# ========== 特徵標準化 ==========
feature_cols = ["最近消費天數_R", "消費次數_F", "消費金額_M"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

# ========== 用 Elbow Method 選擇最佳 K ==========
print("\n=== Elbow Method：不同 K 值的慣性 (Inertia) ===")
inertias = []
sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    sil_scores.append(sil)
    print(f"  K={k}  Inertia={km.inertia_:>8.1f}  Silhouette={sil:.4f}")

# ========== 使用 K=3 進行分群（符合業務假設）==========
best_k = 3
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["群組"] = model.fit_predict(X_scaled)

print(f"\n=== 分群結果（K={best_k}）===")
print(f"Silhouette Score：{silhouette_score(X_scaled, df['群組']):.4f}")

# ========== 各群組特徵分析 ==========
print("\n=== 各群組平均特徵 ===")
group_summary = df.groupby("群組")[feature_cols].mean().round(1)
print(group_summary)

print("\n=== 各群組人數 ===")
print(df["群組"].value_counts().sort_index())

# ========== 業務標籤命名 ==========
print("\n=== 群組業務解讀 ===")
for cluster_id in sorted(df["群組"].unique()):
    subset = df[df["群組"] == cluster_id]
    avg_r = subset["最近消費天數_R"].mean()
    avg_f = subset["消費次數_F"].mean()
    avg_m = subset["消費金額_M"].mean()

    # 根據 RFM 特徵自動命名
    if avg_r < 20 and avg_f > 12:
        label = "高價值忠誠客戶 (VIP)"
    elif avg_r < 50 and avg_f > 5:
        label = "一般活躍客戶"
    else:
        label = "沉睡/流失風險客戶"

    print(f"\n  群組 {cluster_id}：{label}")
    print(f"    人數：{len(subset)}")
    print(f"    平均最近消費：{avg_r:.0f} 天前")
    print(f"    平均消費次數：{avg_f:.0f} 次/半年")
    print(f"    平均消費金額：{avg_m:,.0f} 元/半年")

    if "VIP" in label:
        print(f"    → 行銷建議：推薦高單價新品、給予專屬折扣維繫忠誠度")
    elif "活躍" in label:
        print(f"    → 行銷建議：推播促銷活動、交叉銷售提升客單價")
    else:
        print(f"    → 行銷建議：發送喚醒優惠券、限時折扣刺激回購")
