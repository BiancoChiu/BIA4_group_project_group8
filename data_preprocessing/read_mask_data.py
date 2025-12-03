import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

mask_path = "/home/clab/Downloads/jiaxin_temporal/Droso/inference_pred_masks/egfr_F_R_oly_4X_86_cropped.png"

# === 读取 mask ===
mask = np.array(Image.open(mask_path))

print("Mask shape:", mask.shape)
print("Unique labels:", np.unique(mask))
print("Label counts:", np.bincount(mask.flatten()))

# === 建一个离散 colormap（你可以自己换颜色） ===
colors = [
    (0.0, 0.0, 0.0),     # label 0 → black (or unused)
    (1.0, 0.0, 0.0),     # label 1 → red
    (0.0, 1.0, 0.0),     # label 2 → green
    (0.0, 0.0, 1.0),     # label 3 → blue
    (1.0, 1.0, 0.0),     # label 4 → yellow
    (1.0, 2.0, 1.0),     # label 5 → magenta
]

cmap = ListedColormap(colors)

# === 绘图 ===
plt.figure(figsize=(10, 8))
plt.imshow(mask, cmap=cmap, vmin=0, vmax=5)
plt.colorbar(ticks=[0, 1, 2, 3, 4,5])
plt.title("Mask Visualization")
plt.tight_layout()
plt.show()
plt.savefig("mask_visualization.png", dpi=300)
