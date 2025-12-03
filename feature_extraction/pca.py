# pca.py
"""
对 full_wing_features.csv 做 PCA：
- 自动选择数值特征（几何 + EFD + skeleton graph）
- 画 PCA(PC1, PC2) 按 sex 着色
- 画 PCA(PC1, PC2) 按 gene 着色
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ====================== 路径设置 ======================
CSV_PATH = r"full_wing_features.csv"   # 改成你的 full_wing_features.csv 路径
OUT_DIR  = r"pca_plots"                # 存放图像的文件夹

os.makedirs(OUT_DIR, exist_ok=True)


# ====================== 读入数据 ======================
df = pd.read_csv(CSV_PATH)
print("Loaded CSV:", CSV_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ====================== 选择特征列 ======================
# 元信息列（非数值特征）
meta_cols = ["file_name", "gene", "sex", "side", "magnification", "img_id"]

# 剔除元信息，保留数值型列
candidate_cols = [c for c in df.columns if c not in meta_cols]
feature_cols = [c for c in candidate_cols if np.issubdtype(df[c].dtype, np.number)]

print(f"Using {len(feature_cols)} numeric features for PCA.")
print("Example features:", feature_cols[:10])

# 去掉全 NaN 的列（以防少数特征全是 NaN）
feature_cols = [c for c in feature_cols if not df[c].isna().all()]
print(f"After dropping all-NaN features: {len(feature_cols)} features remain.")

X = df[feature_cols].values

# 如有 NaN，用列均值简单填一下（避免 PCA 崩）
col_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_means, inds[1])


# ====================== 标准化 + PCA ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
PC = pca.fit_transform(X_scaled)

df["PC1"] = PC[:, 0]
df["PC2"] = PC[:, 1]

print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)


# ====================== 画 PCA：按 Sex 上色 ======================
plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=df,
    x="PC1", y="PC2",
    hue="sex",
    s=50,
    palette="Set1"
)
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.title("PCA of Wing Features (colored by Sex)")
plt.tight_layout()

sex_fig_path = os.path.join(OUT_DIR, "pca_sex.png")
plt.savefig(sex_fig_path, dpi=300)
print("Saved:", sex_fig_path)

# 如果你在交互环境下想看图，也可以加上：
# plt.show()
plt.close()


# ====================== 画 PCA：按 Gene 上色 ======================
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="PC1", y="PC2",
    hue="gene",
    s=50,
    palette="Set2"
)
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.title("PCA of Wing Features (colored by Gene)")
plt.tight_layout()

gene_fig_path = os.path.join(OUT_DIR, "pca_gene.png")
plt.savefig(gene_fig_path, dpi=300)
print("Saved:", gene_fig_path)

# plt.show()
plt.close()
