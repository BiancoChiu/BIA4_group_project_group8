# utils/dataset_wing.py

import os
import glob
import re
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad as tv_pad
from collections import Counter

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