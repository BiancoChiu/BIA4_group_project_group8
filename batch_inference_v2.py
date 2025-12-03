import os
import ast
import json
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn

from PIL import Image
import torchvision.transforms as T

from feature_extraction import extract_wing_features_featurelist
from models import DeeperWingCNN2, FeatureEncoder


# =========================================================
# 0. parser（简单）
# =========================================================
def get_parser():
    parser = argparse.ArgumentParser(description="Batch inference for gene + sex models")

    parser.add_argument(
        "--img_dir",
        type=str,
        default="./data/crop_testing",
        help="Path to input image folder"
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        default="./output/batch_inference_gene_sex.csv",
        help="Path to output CSV"
    )

    return parser.parse_args()


# =========================================================
# 1. 自动设定 PROJECT_ROOT & 切换工作区
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


# =========================================================
# 工具函数：Mean/Std
# =========================================================
def load_train_feature_stats(train_csv_path):
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
    feat_std[feat_std < 1e-6] = 1.0

    return feat_mean.astype(np.float32), feat_std.astype(np.float32)


# =========================================================
# 模型结构
# =========================================================
class FusionClassifier(nn.Module):
    def __init__(self, cnn_encoder, feat_encoder, num_classes, fusion_hidden=256, dropout=0.2):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.feat_encoder = feat_encoder

        fusion_in_dim = cnn_encoder.emb_dim + feat_encoder.emb_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, img, feat):
        img_emb = self.cnn_encoder.encode(img)
        feat_emb = self.feat_encoder(feat)
        x = torch.cat([img_emb, feat_emb], dim=1)
        return self.fusion_head(x)


def build_fusion_model(num_classes, feat_dim, ckpt_path, device):
    cnn = DeeperWingCNN2(in_channels=1, num_classes=num_classes).to(device)
    feat_encoder = FeatureEncoder(
        in_dim=feat_dim, hidden_dim=128, out_dim=64, dropout=0.1
    ).to(device)

    model = FusionClassifier(
        cnn_encoder=cnn,
        feat_encoder=feat_encoder,
        num_classes=num_classes,
        fusion_hidden=256,
        dropout=0.2,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")
    return model


# =========================================================
# Batch inference （双模型）
# =========================================================
def batch_inference_gene_sex(img_dir, out_csv):
    REFER_CSV_PATH = "./data/reference_feature.csv"
    GENE_MODEL_DIR = "./fusion_cnn_mlp_model"
    SEX_MODEL_DIR  = "./fusion_cnn_mlp_sex_model_try_best"

    GENE_CKPT_PATH = os.path.join(GENE_MODEL_DIR, "fusion_best.pth")
    GENE_LABEL_MAP = os.path.join(GENE_MODEL_DIR, "label2idx.json")

    SEX_CKPT_PATH  = os.path.join(SEX_MODEL_DIR, "fusion_best.pth")
    SEX_LABEL_MAP  = os.path.join(SEX_MODEL_DIR, "label2idx.json")

    # ====== device ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== feature stats ======
    feat_mean, feat_std = load_train_feature_stats(REFER_CSV_PATH)
    feat_dim = feat_mean.shape[0]

    # ====== gene label2idx ======
    with open(GENE_LABEL_MAP, "r") as f:
        gene_label2idx = json.load(f)
    idx2gene = {int(v): k for k, v in gene_label2idx.items()}

    # ====== sex label2idx ======
    with open(SEX_LABEL_MAP, "r") as f:
        sex_label2idx = json.load(f)
    idx2sex = {int(v): k for k, v in sex_label2idx.items()}

    # ====== build two models ======
    fusion_gene = build_fusion_model(len(idx2gene), feat_dim, GENE_CKPT_PATH, device)
    fusion_sex  = build_fusion_model(len(idx2sex),  feat_dim, SEX_CKPT_PATH,  device)

    # ====== feature extraction ======
    print("[INFO] Extracting handcrafted features...")
    df_feat = extract_wing_features_featurelist(
        img_dir=img_dir,
        csv_out=None,
        efd_order=10,
        downsample=2,
        max_regions=6,
        verbose=True,
    )

    all_records = []

    # ====== loop inference ======
    with torch.no_grad():
        for _, row in df_feat.iterrows():
            fname = row["file_name"]
            feat_val = row["feature"]

            # feature
            if isinstance(feat_val, list):
                feat_vec = np.array(feat_val, dtype=np.float32)
            else:
                feat_vec = np.array(ast.literal_eval(feat_val), dtype=np.float32)

            feat_vec = np.nan_to_num(feat_vec, nan=0.0)
            feat_norm = (feat_vec - feat_mean) / feat_std
            feat_tensor = torch.tensor(feat_norm).float().unsqueeze(0).to(device)

            # image
            img_path = os.path.join(img_dir, fname)
            pil_img = Image.open(img_path).convert("L")
            img_tensor = T.ToTensor()(pil_img).unsqueeze(0).to(device)

            # gene
            probs_gene = F.softmax(fusion_gene(img_tensor, feat_tensor), dim=1)[0].cpu().numpy()
            gene_idx = int(probs_gene.argmax())
            gene_label = idx2gene[gene_idx]

            # sex
            probs_sex = F.softmax(fusion_sex(img_tensor, feat_tensor), dim=1)[0].cpu().numpy()
            sex_idx = int(probs_sex.argmax())
            sex_label = idx2sex[sex_idx]

            all_records.append({
                "file_name": fname,
                "gene_pred": gene_label,
                "sex_pred": sex_label,
                "gene_probs": probs_gene.tolist(),
                "sex_probs": probs_sex.tolist(),
            })

    # ====== ensure output dir exists ======
    out_dir = os.path.dirname(out_csv)
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame(all_records)
    df_out.to_csv(out_csv, index=False)

    print("[INFO] Saved:", out_csv)
    print("[INFO] Final shape:", df_out.shape)


# =========================================================
# entry point
# =========================================================
if __name__ == "__main__":
    args = get_parser()
    batch_inference_gene_sex(
        img_dir=args.img_dir,
        out_csv=args.out_csv
    )
