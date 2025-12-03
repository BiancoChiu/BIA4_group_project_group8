import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np

# 如果你的 UNet 定义在 unet_seg_withDice.py 里，就这样 import：
from models import UNet   # 如果名字不一样，改这里

# 类别 ID（和你训练时保持一致）
CLASS_IDS = [1, 2, 3, 4, 5]  
NUM_CLASSES = len(CLASS_IDS)


def load_model(ckpt_path, device=None):
    """
    加载训练好的 UNet 模型权重，返回 model.eval()
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, num_classes=NUM_CLASSES)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path} on {device}")
    return model, device


def infer_single_image(
    model,
    device,
    img_path,
    save_path=None,
    as_color=False,
):
    """
    对单张灰度 wing 图做 segmentation 推理。
    - model: 已经 .eval() 的 UNet
    - img_path: 原始 png 路径
    - save_path: 如果不为 None，则把预测 mask 保存成 png
    - as_color: True 时用简单调色板上色（方便肉眼看）
    """
    # 1. 读图 & 预处理
    img = Image.open(img_path).convert("L")   # 灰度
    transform = T.ToTensor()                  # -> [1,H,W], [0,1]
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1,1,H,W]

    # 2. 前向
    with torch.no_grad():
        logits = model(img_tensor)           # [1,C,H,W]
        probs = F.softmax(logits, dim=1)     # [1,C,H,W]
        pred = probs.argmax(dim=1)[0]        # [H,W]，值∈ {0,1,2,3(映射到4)}

    pred_np = pred.cpu().numpy().astype(np.uint8)

    # 注意：如果你的最后一层已经把“第3通道”对齐 class4，则这里 pred 值应该就是 {0,1,2,3}，
    # 你想保持 mask 里的 4，就可以：
    #   0 -> 0, 1 -> 1, 2 -> 2, 3 -> 4
    pred_remap = pred_np.copy()
    pred_remap[pred_np == 3] = 4

    # 3. 可选保存
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_img = Image.fromarray(pred_remap)

        if as_color:
            # 简单给每个 label 一个颜色: 0 黑,1 红,2 绿,4 蓝
            palette = [0,0,0,  255,0,0,  0,255,0,  0,0,0,  0,0,255] + [0,0,0]*251
            out_img = out_img.convert("P")
            out_img.putpalette(palette)

        out_img.save(save_path)
        print(f"[INFO] Saved prediction to {save_path}")

    return pred_remap


class InferenceImageDataset(Dataset):
    """
    用于 batch 推理的 Dataset：只读取单通道 png 图片。
    假设目录下是 idX.png（原图），你会为它生成 idX_pred.png。
    """
    def __init__(self, root_dir, img_suffix=".png", as_gray=True):
        self.root_dir = root_dir
        self.img_suffix = img_suffix
        self.as_gray = as_gray

        self.img_paths = []
        for fn in sorted(os.listdir(root_dir)):
            if fn.endswith(img_suffix) and ("_mask" not in fn) and ("_pred" not in fn):
                self.img_paths.append(os.path.join(root_dir, fn))

        if not self.img_paths:
            raise RuntimeError(f"No images found in {root_dir} with suffix {img_suffix}")

        print(f"[INFO] Inference dataset: found {len(self.img_paths)} images in {root_dir}")

        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mode = "L" if self.as_gray else "RGB"
        img = Image.open(img_path).convert(mode)
        img_tensor = self.transform(img)   # [1,H,W]

        return img_tensor, img_path


def infer_directory(
    model,
    device,
    src_dir,
    out_dir,
    batch_size=2,
    as_color=False,
):
    """
    对 src_dir 下所有 png 做 batch 推理，并把预测 mask 保存到 out_dir。

    - src_dir: 输入图片目录（只读原图，不读 *_mask）
    - out_dir: 输出目录（自动创建）
    """
    os.makedirs(out_dir, exist_ok=True)

    dataset = InferenceImageDataset(src_dir, img_suffix=".png", as_gray=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        for imgs, paths in loader:
            imgs = imgs.to(device)    # [B,1,H,W]
            logits = model(imgs)      # [B,C,H,W]
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # [B,H,W]

            preds_np = preds.cpu().numpy().astype(np.uint8)

            # 映射 0,1,2,3 -> 0,1,2,4
            preds_remap = preds_np.copy()
            preds_remap[preds_np == 3] = 4

            for mask_arr, img_path in zip(preds_remap, paths):
                base = os.path.basename(img_path)
                stem, _ = os.path.splitext(base)
                out_path = os.path.join(out_dir, stem + "_pred.png")

                out_img = Image.fromarray(mask_arr)

                if as_color:
                    palette = [0,0,0,  255,0,0,  0,255,0,  0,0,0,  0,0,255] + [0,0,0]*251
                    out_img = out_img.convert("P")
                    out_img.putpalette(palette)

                out_img.save(out_path)
                print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    ckpt = "/home/clab/Downloads/jiaxin_temporal/Droso/outputdice2/unet_seg_epoch18.pth"
    model, device = load_model(ckpt)

    src_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_sampled_png/egfr_F_R_oly_4X_86_cropped.png"
    out_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/inference_pred_masks/egfr_F_R_oly_4X_86_cropped.png"
    infer_single_image(model, device, src_dir, out_dir)
"""
    infer_directory(
        model,
        device,
        src_dir=src_dir,
        out_dir=out_dir,
        batch_size=2,
        as_color=True,
    )
"""
