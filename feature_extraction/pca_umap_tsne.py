"""
pca_umap_tsne.py

对 full wing morphology features（geom + EFD + skeleton graph + intervein）
进行 PCA / UMAP / t-SNE 降维，并按 gene 上色绘图。

完全自动选择数值特征，无需手动修改列名。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


# ====================== 路径 ======================
CSV_PATH = r"full_wing_features.csv"   # 改成你的路径
OUT_DIR  = r"dimreduce_plots"
os.makedirs(OUT_DIR, exist_ok=True)


# ====================== 读数据 ======================
df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH, df.shape)
print("Columns:", len(df.columns))


# ====================== 自动选择数值特征 ======================
meta_cols = ["file_name", "gene", "sex", "side", "magnification", "img_id"]

feature_cols = [
    c for c in df.columns
    if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)
]

# 去掉全 NaN 的列
feature_cols = [c for c in feature_cols if not df[c].isna().all()]

X = df[feature_cols].values

# 数值填补
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ====================== PCA ======================
pca = PCA(n_components=2)
PC = pca.fit_transform(X_scaled)
df["PC1"], df["PC2"] = PC[:, 0], PC[:, 1]

plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="gene", s=45, palette="Set2")
plt.title("PCA of Wing Features (Including Intervein)")
plt.axhline(0, color="gray", lw=0.4)
plt.axvline(0, color="gray", lw=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_gene.png"), dpi=300)
plt.close()

print("PCA explained variance:", pca.explained_variance_ratio_[:2])
# ====================== PCA (sex) ======================
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="sex", s=45, palette="Set1")
plt.title("PCA of Wing Features (Sex)")
plt.axhline(0, color="gray", lw=0.4)
plt.axvline(0, color="gray", lw=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_sex.png"), dpi=300)
plt.close()

# ====================== UMAP ======================
um = UMAP(
    n_neighbors=12,
    min_dist=0.05,
    metric="euclidean",
    random_state=0
)
UM = um.fit_transform(X_scaled)
df["UMAP1"], df["UMAP2"] = UM[:, 0], UM[:, 1]

plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="gene", s=45, palette="Set2")
plt.title("UMAP of Wing Features (Including Intervein)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "umap_gene.png"), dpi=300)
plt.close()

# ====================== UMAP (sex) ======================
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="sex", s=45, palette="Set1")
plt.title("UMAP of Wing Features (Sex)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "umap_sex.png"), dpi=300)
plt.close()
# ====================== t-SNE ======================
try:
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=0
    )
except TypeError:
    # sklearn >=1.2 fallback
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter_without_progress=300,
        random_state=0
    )

TS = tsne.fit_transform(X_scaled)
df["TSNE1"], df["TSNE2"] = TS[:, 0], TS[:, 1]

plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="TSNE1", y="TSNE2", hue="gene", s=45, palette="Set2")
plt.title("t-SNE of Wing Features (Including Intervein)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsne_gene.png"), dpi=300)
plt.close()


# ====================== t-SNE (sex) ======================
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="TSNE1", y="TSNE2", hue="sex", s=45, palette="Set1")
plt.title("t-SNE of Wing Features (Sex)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsne_sex.png"), dpi=300)
plt.close()

print("Saved all plots to:", OUT_DIR)