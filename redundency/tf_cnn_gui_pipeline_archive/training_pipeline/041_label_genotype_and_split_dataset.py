import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ================== 0. Basic path settings ==================
# train layer of paths
base_dir = r"/Users/yanyanru/desktop/yanru/genotype/train3"

# Output directory: Save the npy and tags
out_dir = os.path.join(base_dir, "genodataset")
os.makedirs(out_dir, exist_ok=True)

# Desired uniform image size (width, height)
IMG_W, IMG_H = 320, 256

# ================== 1. Find all *-binary category folders ==================
binary_folders = []
for name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, name)
    if os.path.isdir(folder_path) and name.endswith("-binary"):
        binary_folders.append(name)

binary_folders.sort()  # eg: ['egfr-binary', 'mam-binary', ...]
print("Discover the category folders：", binary_folders)

# Establish a mapping of the Category Name -> Integer Label
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(binary_folders)}
print("Category mapping：", class_to_idx)

# ================== 2. Read all images, heap them into a large array ==================
X_list = []
y_list = []

# Supported image file extensions
exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]

for cls_name in binary_folders:
    cls_idx = class_to_idx[cls_name]
    folder_path = os.path.join(base_dir, cls_name)

    # All image paths under the current category
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    print(f"{cls_name}：find {len(img_paths)} images")

    for p in img_paths:
        img = Image.open(p).convert("L")                   # Grayscale
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)   # Unified dimensions 480x360
        arr = np.array(img, dtype=np.uint8)                # [H, W]

        # If normalize, change to float32 /255.0, and keep 0~255 here first
        arr = np.expand_dims(arr, axis=-1)                 # [H, W, 1]
        X_list.append(arr)
        y_list.append(cls_idx)

# Heap into a numpy array
X = np.stack(X_list, axis=0)            # [N, H, W, 1]
y_int = np.array(y_list, dtype=np.int64)  # [N,] Integer label

num_classes = len(binary_folders)
y_onehot = np.eye(num_classes, dtype=np.float32)[y_int]   # [N, C]

print("X shape:", X.shape)
print("y_int shape:", y_int.shape)
print("y_onehot shape:", y_onehot.shape)

# Save the entire dataset
np.save(os.path.join(out_dir, "X_all.npy"), X)
np.save(os.path.join(out_dir, "y_int_all.npy"), y_int)
np.save(os.path.join(out_dir, "y_onehot_all.npy"), y_onehot)

with open(os.path.join(out_dir, "class_indices.txt"), "w", encoding="utf-8") as f:
    for cls_name, idx in class_to_idx.items():
        f.write(f"{idx}\t{cls_name}\n")

# ================== 3. Divide Training Set/Test Set（8:2） ==================
# Use y_int stratify to ensure that the ratio of each class is similar
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=0.2,        # 2:8 ratio -> 20% test, 80% training
    stratify=y_int,
    random_state=42
)

print("training sets:", X_train.shape, y_train.shape)
print("test sets:", X_test.shape, y_test.shape)

np.save(os.path.join(out_dir, "X_train.npy"), X_train)
np.save(os.path.join(out_dir, "y_train.npy"), y_train)
np.save(os.path.join(out_dir, "X_test.npy"), X_test)
np.save(os.path.join(out_dir, "y_test.npy"), y_test)

print("All done, data saved to：", out_dir)
