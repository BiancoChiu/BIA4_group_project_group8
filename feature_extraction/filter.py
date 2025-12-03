import pandas as pd

df = pd.read_csv("wing_features_40X.csv")
# 假设有列：sex, Type, wing_area, aspect_ratio, skeleton_density, ...

df.describe().T

import seaborn as sns
import matplotlib.pyplot as plt

for col in ["wing_area", "aspect_ratio", "skeleton_density"]:
    plt.figure()
    sns.boxplot(data=df, x="sex", y=col)
    plt.title(col)
    plt.show()

for col in ["wing_area", "aspect_ratio", "skeleton_density"]:
    plt.figure()
    sns.boxplot(data=df, x="gene", y=col)
    plt.title(col)
    plt.show()