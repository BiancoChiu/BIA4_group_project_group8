# utils/dataset_wing.py

import os
import glob
import re
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad as tv_pad
from collections import Counter
from ast import literal_eval
import ast

FNAME_PATTERN = re.compile(
    r'(?P<type>[A-Za-z0-9]+)_'      # egfr
    r'(?P<sex>[FM])_'               # F / M
    r'(?P<side>[LR])_'              # L / R
    r'(?P<scope>[A-Za-z]+)_'        # oly / lei
    r'(?P<mag>\d+X)_'               # 4X / 10X
    r'(?P<idx>\d+)'                 # 1,2,3...
    r'(?:_cropped)?'                # 可选的 '_cropped'
    r'\.png$',                      # 后缀改成 .png
    re.IGNORECASE
)

def parse_wing_filename(fname):
    m = FNAME_PATTERN.match(os.path.basename(fname))
    if m is None:
        raise ValueError(f"Filename {fname} does not match required pattern")
    return m.groupdict()


class WingImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if not self.paths:
            raise RuntimeError(f"No tif files found in {root_dir}")
        
        self.metadata = [parse_wing_filename(p) for p in self.paths]
        self.transform = transform if transform else T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        meta = self.metadata[idx]

        img = Image.open(path).convert("L")
        img = self.transform(img)

        return img, meta

class WingClsDataset(Dataset):
    def __init__(self, root_dir, label_key="type", transform=None):
        """
        dataloader with label
        """
        self.base = WingImageDataset(root_dir, transform=transform)
        self.label_key = label_key

        # 从 base 里收集所有 label，做 string -> int 映射
        all_labels = [meta[label_key] for _, meta in self.base]
        self.label2idx = {lab: i for i, lab in enumerate(sorted(set(all_labels)))}
        self.idx2label = {i: lab for lab, i in self.label2idx.items()}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, meta = self.base[idx]  # img: [1,H,W], meta: dict
        label = self.label2idx[meta[self.label_key]]
        return img, label, meta
    
class WingCSVFeatureDataset(Dataset):
    """
    读取 full_wing_features*.csv：
      - feature 列：字符串形式的 list，如 "[0.1, 2.3, nan, ...]"
      - label_col：比如 "gene"，作为分类标签
    在 __init__ 中会做：
      1) 解析字符串 -> list[float]
      2) NaN / inf -> 0
      3) 按列 z-score 标准化 (全数据统计 mean/std)
    """

    def __init__(self, csv_path, label_col="gene"):
        df = pd.read_csv(csv_path)
        self.label_col = label_col

        # ---------- 解析 feature 列 ----------
        raw_feature_strs = df["feature"].astype(str).tolist()
        feat_list = []

        for s in raw_feature_strs:
            s = s.strip()
            # 去掉 list 的中括号
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]

            # 彻底空的行
            if s.strip() == "":
                feat_list.append([])
                continue

            vals = []
            for tok in s.split(","):
                tok = tok.strip()
                if tok in ("nan", "NaN", "None", ""):
                    vals.append(np.nan)
                else:
                    vals.append(float(tok))
            feat_list.append(vals)

        feat_array = np.array(feat_list, dtype=np.float32)  # [N, D]

        # =============== NaN 检查 ===============
        nan_mask = np.isnan(feat_array).any(axis=1)
        nan_indices = np.where(nan_mask)[0]
        if len(nan_indices) > 0:
            print("⚠️ [WARN] Samples with NaN detected:", nan_indices.tolist())
            print("⚠️ Total:", len(nan_indices))
            # 如果想看具体是哪几张图：
            cols_to_show = [c for c in ["file_name","gene","sex","side","img_id"] if c in df.columns]
            print(df.loc[nan_indices, cols_to_show])
        else:
            print("✔️ No NaN detected in features.")

        # =============== NaN / inf → 0 填充 ===============
        feat_array = np.nan_to_num(
            feat_array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # =============== 按列做 z-score 标准化 ===============
        col_mean = feat_array.mean(axis=0, keepdims=True)  # [1, D]
        col_std  = feat_array.std(axis=0, keepdims=True)   # [1, D]
        col_std[col_std < 1e-6] = 1.0   # 防止除 0

        feat_array = (feat_array - col_mean) / col_std

        # 保存下来，之后如果要做 inference，可以重用
        self.col_mean = col_mean
        self.col_std  = col_std

        self.features = torch.tensor(feat_array, dtype=torch.float32)  # [N, D]

        # ---------- labels ----------
        labels_raw = df[label_col].tolist()
        self.label2idx = {lab: i for i, lab in enumerate(sorted(set(labels_raw)))}
        self.idx2label = {i: lab for lab, i in self.label2idx.items()}
        labels_idx = [self.label2idx[x] for x in labels_raw]
        self.labels = torch.tensor(labels_idx, dtype=torch.long)

        # ---------- meta ----------
        meta_cols = ["file_name", "gene", "sex", "side", "magnification", "img_id"]
        meta_cols = [c for c in meta_cols if c in df.columns]
        self.metas = df[meta_cols].to_dict(orient="records")

        print(f"[INFO] Loaded CSV: {csv_path}")
        print(f"[INFO] Num samples: {len(self.features)}, feature dim: {self.features.shape[1]}")
        print(f"[INFO] label2idx: {self.label2idx}")

        # 简单 sanity check：
        print("[DEBUG] any NaN in features after norm:", np.isnan(feat_array).any())

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx]     # [D]
        y = self.labels[idx]       # scalar
        meta = self.metas[idx]     # dict
        return x, y, meta
    
class SegPairSameDirDataset(Dataset):
    """
    读取同一目录中的 (image, mask) 对：
        image: id1.png
        mask:  id1_mask.png

    要求：
        - root_dir 下已经是 padding 好的 PNG
        - 所有原图命名为:  idX.png
        - 对应 mask 命名:  idX_mask.png
    """

    def __init__(
        self,
        root_dir,
        img_suffix=".png",
        mask_suffix="_mask.png",
        img_transform=None,
        mask_transform=None,
        as_gray=True,
    ):
        self.root_dir = root_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.as_gray = as_gray

        self.img_transform = img_transform or T.ToTensor()
        self.mask_transform = mask_transform or T.ToTensor()

        # 找到所有 png
        all_png = sorted(glob.glob(os.path.join(root_dir, "*" + img_suffix)))

        self.pairs = []
        for img_path in all_png:
            fname = os.path.basename(img_path)

            # 跳过 *_mask.png，本身是 mask，不是原图
            if fname.endswith(self.mask_suffix):
                continue

            stem = os.path.splitext(fname)[0]          # 例如 "id1"
            mask_name = stem + self.mask_suffix        # "id1_mask.png"
            mask_path = os.path.join(root_dir, mask_name)

            if os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))
            else:
                print(f"[WARN] Mask not found for {img_path} -> {mask_path}")

        if not self.pairs:
            raise RuntimeError(f"No (image, mask) pairs found in {root_dir}")

        print(f"[INFO] Found {len(self.pairs)} image-mask pairs in {root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # ---- image ----
        mode = "L" if self.as_gray else "RGB"
        img = Image.open(img_path).convert(mode)
        img = self.img_transform(img)              # 可以 ToTensor

        # ---- mask（这里不能 ToTensor）----
        mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = torch.from_numpy(mask_np).long()    # [H,W] 保持整数 0-255

        info = {"img_path": img_path, "mask_path": mask_path}
        return img, mask, info

class FusionWingDataset(Dataset):
    """
      - image tensor: [1, H, W]
      - fature: [F]
      - label: int
      - meta: dict csv
    """

    def __init__(self,
                 image_dir: str,
                 csv_path: str,
                 label_col: str = "gene",
                 transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.label_col = label_col

        # 1) 读 CSV
        df = pd.read_csv(csv_path)
        if "feature" not in df.columns:
            raise ValueError("CSV must contain a 'feature' column (list of floats).")

        # 把 'feature' 字符串 -> list[float]
        def _parse_feat(s):
            if isinstance(s, list):
                return np.array(s, dtype=np.float32)
            return np.array(ast.literal_eval(s), dtype=np.float32)

        df["feature"] = df["feature"].apply(_parse_feat)

        # 2) 建立 file_name -> row 的映射
        self.row_by_fname = {}
        for _, row in df.iterrows():
            fname = row["file_name"]
            self.row_by_fname[fname] = row

        # 3) 遍历 image_dir 中所有 png，保证能在 CSV 中找到
        self.img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not self.img_paths:
            raise RuntimeError(f"No .png files found in {image_dir}")

        missing = []
        for p in self.img_paths:
            fname = os.path.basename(p)
            if fname not in self.row_by_fname:
                missing.append(fname)
        if missing:
            print(f"[WARN] {len(missing)} images not found in CSV (show first 5):", missing[:5])

        # 只保留 CSV 中确实有记录的图片
        self.img_paths = [p for p in self.img_paths
                          if os.path.basename(p) in self.row_by_fname]

        # 4) 构建 label 映射（string -> int）
        all_labels = []
        for p in self.img_paths:
            row = self.row_by_fname[os.path.basename(p)]
            all_labels.append(row[label_col])

        uniq_labels = sorted(set(all_labels))
        self.label2idx = {lab: i for i, lab in enumerate(uniq_labels)}
        self.idx2label = {i: lab for lab, i in self.label2idx.items()}

        # 5) 整体 feature matrix（用于标准化）
        feat_list = []
        for p in self.img_paths:
            row = self.row_by_fname[os.path.basename(p)]
            feat_list.append(row["feature"])
        feats = np.stack(feat_list, axis=0)            # (N, F)

        self.feat_mean = feats.mean(axis=0)
        self.feat_std = feats.std(axis=0)
        self.feat_std[self.feat_std < 1e-6] = 1.0      # 避免除 0

        print(f"[INFO] Dataset size: {len(self.img_paths)}, feature dim: {feats.shape[1]}")
        print(f"[INFO] label2idx: {self.label2idx}")
        print(f"[DEBUG] first 5-dim feature mean:", self.feat_mean[:5])
        print(f"[DEBUG] first 5-dim feature std :", self.feat_std[:5])

        self.transform = transform if transform is not None else T.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        fname = os.path.basename(img_path)
        row = self.row_by_fname[fname]

        # label
        label_str = row[self.label_col]
        label = self.label2idx[label_str]

        # feature
        feat = row["feature"].astype(np.float32)
        feat = (feat - self.feat_mean) / self.feat_std   # 标准化
        feat = torch.from_numpy(feat)                    # (F,)

        # image
        from PIL import Image
        img = Image.open(img_path).convert("L")
        img = self.transform(img)                        # [1, H, W]

        meta = dict(row)
        return img, feat, label, meta
        

if __name__ == "__main__":
    ds = SegPairSameDirDataset(
        root_dir="/home/clab/Downloads/jiaxin_temporal/Droso/data/unet_segdata_padded",
        as_gray=True,
    )

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    for imgs, masks, info in loader:
        print(masks)
        print(Counter(masks.flatten().tolist()))
        print(imgs.shape, masks.shape)
        print(info["img_path"][0])
        break