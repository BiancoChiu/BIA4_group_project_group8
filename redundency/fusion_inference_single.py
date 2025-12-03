import os
import ast
import json
import tempfile
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn

from PIL import Image
import torchvision.transforms as T

from feature_extraction import extract_wing_features_featurelist

from models import DeeperWingCNN2, FeatureEncoder

def load_train_feature_stats(train_csv_path):
    """
    从训练特征 CSV 里读取 'feature' 列，计算 mean/std
    """
    df = pd.read_csv(train_csv_path)
    if "feature" not in df.columns:
        raise RuntimeError(f"CSV {train_csv_path} 必须包含 'feature' 列")

    def _parse_feat(s):
        if isinstance(s, list):
            return np.array(s, dtype=np.float32)
        return np.array(ast.literal_eval(s), dtype=np.float32)

    df["feature"] = df["feature"].apply(_parse_feat)
    feats_all = np.stack(df["feature"].to_list(), axis=0)

    feat_mean = feats_all.mean(axis=0)
    feat_std = feats_all.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0  # 防止除 0

    print("[INFO] train features shape:", feats_all.shape)
    print("[DEBUG] first 5 mean:", feat_mean[:5])
    print("[DEBUG] first 5 std :", feat_std[:5])

    return feat_mean.astype(np.float32), feat_std.astype(np.float32)


def extract_feature_for_single_image(img_path,
                                     efd_order=10,
                                     downsample=2,
                                     max_regions=6,
                                     vein_radius=3):
    """
    调用你封装好的 extract_wing_features，但只针对一张图片。

    做法：
      1) 创建临时目录
      2) 把该图片复制进去
      3) 调用 extract_wing_features(临时目录, csv_out=tmp_csv)
      4) 读 tmp_csv，返回 feature 向量 (np.array)
    """
    tmp_dir = tempfile.mkdtemp(prefix="wing_single_")
    tmp_csv = os.path.join(tmp_dir, "single_features.csv")

    try:
        # 复制图片
        basename = os.path.basename(img_path)
        tmp_img_path = os.path.join(tmp_dir, basename)
        shutil.copy2(img_path, tmp_img_path)

        # 调用你封装好的函数
        extract_wing_features_featurelist(
            img_dir=tmp_dir,
            csv_out=tmp_csv,
            efd_order=10,
            downsample=2,
            max_regions=6,
            verbose=True,
        )

        # 读 CSV
        df = pd.read_csv(tmp_csv)
        print(df.head())
        if "feature" not in df.columns:
            raise RuntimeError("single_features.csv 中缺少 'feature' 列，请确认 extract_wing_features 已写入合并后的 feature 列。")

        # 一般只有一行；保险起见，只取第 0 行
        feat_str = df.loc[0, "feature"]

        if isinstance(feat_str, list):
            feat_vec = np.array(feat_str, dtype=np.float32)
        else:
            feat_vec = np.array(ast.literal_eval(feat_str), dtype=np.float32)

        return feat_vec
    finally:
        # 用完删掉临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

class FusionClassifier(nn.Module):
    def __init__(self,
                 cnn_encoder: DeeperWingCNN2,
                 feat_encoder: FeatureEncoder,
                 num_classes: int,
                 fusion_hidden: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.feat_encoder = feat_encoder

        cnn_emb_dim = cnn_encoder.emb_dim      # e.g. 256
        feat_emb_dim = feat_encoder.emb_dim    # e.g. 64
        fusion_in_dim = cnn_emb_dim + feat_emb_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, img, feat):
        """
        img:  [B, 1, H, W]
        feat: [B, F] (74)
        """
        img_emb = self.cnn_encoder.encode(img)   # (B, 256)
        feat_emb = self.feat_encoder(feat)       # (B, 64)

        x = torch.cat([img_emb, feat_emb], dim=1)  # (B, 320)
        logits = self.fusion_head(x)
        return logits

def build_fusion_model(
    num_classes,
    feat_dim,
    ckpt_path,
    device
):
    """
    构建 DeeperWingCNN2 + FeatureEncoder + FusionClassifier，并加载权重
    """
    # CNN encoder
    cnn = DeeperWingCNN2(in_channels=1, num_classes=num_classes).to(device)

    # MLP encoder for handcrafted features
    feat_encoder = FeatureEncoder(
        in_dim=feat_dim,
        hidden_dim=128,
        out_dim=64,
        dropout=0.1,
    ).to(device)

    # Fusion model
    fusion_model = FusionClassifier(
        cnn_encoder=cnn,
        feat_encoder=feat_encoder,
        num_classes=num_classes,
        fusion_hidden=256,
        dropout=0.2,
    ).to(device)

    # 加载权重
    state = torch.load(ckpt_path, map_location=device)
    fusion_model.load_state_dict(state, strict=True)
    fusion_model.eval()
    print(f"[INFO] Loaded fusion model from {ckpt_path}")

    return fusion_model


def main():
    refer_csv_path = "/home/clab/Downloads/jiaxin_temporal/Droso/data/reference_feature.csv"
    model_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/fusion_cnn_mlp_sex_model_try_best"
    ckpt_path = os.path.join(model_dir, "fusion_best.pth")
    label_map_path = os.path.join(model_dir, "label2idx.json")

    img_path = "/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_sampled_png/egfr_M_R_oly_4X_3_cropped.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    feat_mean, feat_std = load_train_feature_stats(refer_csv_path)
    feat_dim = feat_mean.shape[0]

    with open(label_map_path, "r") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}
    num_classes = len(label2idx)
    print("[INFO] label2idx:", label2idx)

    fusion_model = build_fusion_model(
        num_classes=num_classes,
        feat_dim=feat_dim,
        ckpt_path=ckpt_path,
        device=device,
    )

    feat_vec = extract_feature_for_single_image(
        img_path,
        efd_order=10,
        downsample=2,
        max_regions=6,
        
    )
    print("[DEBUG] raw feature shape:", feat_vec.shape)


    feat_vec = np.nan_to_num(feat_vec, nan=0.0)

    feat_norm = (feat_vec - feat_mean) / feat_std
    feat_tensor = torch.from_numpy(feat_norm).float().unsqueeze(0).to(device)  # [1, F]

    pil_img = Image.open(img_path).convert("L")
    img_tensor = T.ToTensor()(pil_img).unsqueeze(0).to(device)  # [1,1,H,W]

    #
    with torch.no_grad():
        logits = fusion_model(img_tensor, feat_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = idx2label[pred_idx]

    print("\n========== PREDICTION RESULT ==========")
    print(f"[IMAGE] {os.path.basename(img_path)}")
    print(f"[PRED ] gene = {pred_label} (idx={pred_idx})")
    print("[PROBS]")
    for i in range(num_classes):
        print(f"  {idx2label[i]:>5s}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
