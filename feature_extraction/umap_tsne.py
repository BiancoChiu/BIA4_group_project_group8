# umap_tsne.py
"""
UMAP / t-SNE on full wing morphology features
- 自动筛选所有数值特征（geom + EFD + skeleton graph）
- 输出 gene clustering 可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP


# ============ 路径 =============
CSV_PATH = r"full_wing_features.csv"
OUT_DIR  = r"umap_tsne_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 读 CSV ============
df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH, df.shape)

# ============ 特征选择（自动） ============
meta_cols = ["file_name", "gene", "sex", "side", "magnification", "img_id"]

feature_cols = [
    c for c in df.columns
    if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)
]

# 去掉全 NaN 列
feature_cols = [
    c for c in feature_cols if not df[c].isna().all()
]

X = df[feature_cols].values
print("Using", len(feature_cols), "features.")

# NaN → 列均值
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# UMAP
# ============================================================
um = UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0)
embedding_umap = um.fit_transform(X_scaled)

df["UMAP1"] = embedding_umap[:, 0]
df["UMAP2"] = embedding_umap[:, 1]

plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df,
    x="UMAP1", y="UMAP2",
    hue="gene",
    s=50,
    palette="Set2"
)
plt.title("UMAP of Wing Features (Gene Clustering)")
plt.tight_layout()

umap_fig = os.path.join(OUT_DIR, "umap_gene.png")
plt.savefig(umap_fig, dpi=300)
plt.close()
print("Saved:", umap_fig)


# ============================================================
# t-SNE
# ============================================================
try:
    # 老版本 sklearn (<=1.1) 使用 n_iter
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=0
    )
except TypeError:
    # 新版本 sklearn (>=1.2) 使用 n_iter_without_progress
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter_without_progress=300,
        random_state=0
    )

embedding_tsne = tsne.fit_transform(X_scaled)

df["TSNE1"] = embedding_tsne[:, 0]
df["TSNE2"] = embedding_tsne[:, 1]

plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df,
    x="TSNE1", y="TSNE2",
    hue="gene",
    s=50,
    palette="Set2"
)
plt.title("t-SNE of Wing Features (Gene Clustering)")
plt.tight_layout()

tsne_fig = os.path.join(OUT_DIR, "tsne_gene.png")
plt.savefig(tsne_fig, dpi=300)
plt.close()
print("Saved:", tsne_fig)
