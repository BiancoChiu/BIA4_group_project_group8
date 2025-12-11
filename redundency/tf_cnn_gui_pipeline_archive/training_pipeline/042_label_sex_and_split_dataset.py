import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ================== 0. Basic settings ==================
# Here are the items containing egfr-binary, mam-binary, ... The layer of the table of contents
base_dir = r"/Users/yanyanru/desktop/genotype/train3"

# Output npy save path
out_dir = os.path.join(base_dir, "sexdata_trainingandtest")
os.makedirs(out_dir, exist_ok=True)

# Uniform image size (width, height)
IMG_W, IMG_H = 320, 256

# ================== 1. Locate all *-binary folders ==================
binary_folders = []
for name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, name)
    if os.path.isdir(folder_path) and name.endswith("-binary"):
        binary_folders.append(name)

binary_folders.sort()
print("The genotype folder being used：", binary_folders)
# like ['egfr-binary', 'mam-binary', 'samw-binary', 'star-binary', 'tkv-binary']

# ================== 2. Go through all the images, read and gender tag them according to the file name ==================
X_list = []
y_list = []   # 0 = F, 1 = M

exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]

for folder_name in binary_folders:
    folder_path = os.path.join(base_dir, folder_name)

    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    print(f"{folder_name} found {len(img_paths)} images")

    for p in img_paths:
        fname = os.path.basename(p)

        # --------- Resolve gender labels from file names ----------
        # Assume that there is a naming convention involved “…_F_…” or “…_M_…”
        if "_F_" in fname or "_f_" in fname:
            label = 0     # Female
        elif "_M_" in fname or "_m_" in fname:
            label = 1     # Male
        else:
            print("⚠ Gender cannot be determined;skipped.：", fname)
            continue

        # --------- Reading images, gray-scale conversion, scaling ----------
        img = Image.open(p).convert("L")                    # Grayscale
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)    # Unified dimensions
        arr = np.array(img, dtype=np.uint8)                # [H, W]
        arr = np.expand_dims(arr, axis=-1)                 # [H, W, 1]

        X_list.append(arr)
        y_list.append(label)

# Heap into a numpy array
X = np.stack(X_list, axis=0)                    # [N, H, W, 1]
y_int = np.array(y_list, dtype=np.int64)        # [N,], 0/1

print("Total number of images:", X.shape[0])
print("Female counts:", np.sum(y_int == 0))
print("Male   counts:", np.sum(y_int == 1))

# ================== 3. Generate one-hot tags ==================
num_classes = 2
y_onehot = np.eye(num_classes, dtype=np.float32)[y_int]   # [N, 2]

# ================== 4. Divide Training Set/Test Set（8:2） ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=0.2,
    stratify=y_int,     
    random_state=42
)

print("training sets:", X_train.shape, y_train.shape)
print("test sets:", X_test.shape, y_test.shape)

# ================== 5. Save as .npy for later use ==================
np.save(os.path.join(out_dir, "X_all.npy"), X)
np.save(os.path.join(out_dir, "y_int_all.npy"), y_int)
np.save(os.path.join(out_dir, "y_onehot_all.npy"), y_onehot)

np.save(os.path.join(out_dir, "X_train.npy"), X_train)
np.save(os.path.join(out_dir, "y_train.npy"), y_train)
np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

# additional explanatory document
with open(os.path.join(out_dir, "label_meaning.txt"), "w", encoding="utf-8") as f:
    f.write("0\tF (female)\n")
    f.write("1\tM (male)\n")

print("✅ Gender dichotomous data have been prepared and stored at：", out_dir)
