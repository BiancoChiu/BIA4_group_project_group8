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
import gradio as gr

from feature_extraction import extract_wing_features_featurelist
from models import DeeperWingCNN2, FeatureEncoder


# ===================== 1. 工具函数：读取训练特征的 mean/std =====================
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
    feat_std[feat_std < 1e-6] = 1.0  # 防止除 0

    print("[INFO] train features shape:", feats_all.shape)
    print("[DEBUG] first 5 mean:", feat_mean[:5])
    print("[DEBUG] first 5 std :", feat_std[:5])

    return feat_mean.astype(np.float32), feat_std.astype(np.float32)


# ===================== 2. 单张图像特征提取 =====================
def extract_feature_for_single_image(
    img_path,
    efd_order=10,
    downsample=2,
    max_regions=6,
    vein_radius=3,
):
   
    tmp_dir = tempfile.mkdtemp(prefix="wing_single_")
    tmp_csv = os.path.join(tmp_dir, "single_features.csv")

    try:
        basename = os.path.basename(img_path)
        tmp_img_path = os.path.join(tmp_dir, basename)
        shutil.copy2(img_path, tmp_img_path)


        extract_wing_features_featurelist(
            img_dir=tmp_dir,
            csv_out=tmp_csv,
            efd_order=efd_order,
            downsample=downsample,
            max_regions=max_regions,
            verbose=False,
        )

        df = pd.read_csv(tmp_csv)
        if "feature" not in df.columns:
            raise RuntimeError(
                "single_features.csv lack 'feature' col, plz extract_wing_features to feature col."
            )

        feat_str = df.loc[0, "feature"]

        if isinstance(feat_str, list):
            feat_vec = np.array(feat_str, dtype=np.float32)
        else:
            feat_vec = np.array(ast.literal_eval(feat_str), dtype=np.float32)

        return feat_vec
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ===================== 3. 模型部分 =====================
class FusionClassifier(nn.Module):
    def __init__(
        self,
        cnn_encoder: DeeperWingCNN2,
        feat_encoder: FeatureEncoder,
        num_classes: int,
        fusion_hidden: int = 256,
        dropout: float = 0.2,
    ):
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
        feat: [B, F]
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
    device,
):

    cnn = DeeperWingCNN2(in_channels=1, num_classes=num_classes).to(device)

    feat_encoder = FeatureEncoder(
        in_dim=feat_dim,
        hidden_dim=128,
        out_dim=64,
        dropout=0.1,
    ).to(device)

    fusion_model = FusionClassifier(
        cnn_encoder=cnn,
        feat_encoder=feat_encoder,
        num_classes=num_classes,
        fusion_hidden=256,
        dropout=0.2,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    fusion_model.load_state_dict(state, strict=True)
    fusion_model.eval()
    print(f"[INFO] Loaded fusion model from {ckpt_path}")

    return fusion_model



REFER_CSV_PATH = "/home/clab/Downloads/jiaxin_temporal/Droso/full_wing_features4mlpclassifier.csv"
MODEL_DIR      = "/home/clab/Downloads/jiaxin_temporal/Droso/results_fusion_cnn_mlp_trybest"
CKPT_PATH      = os.path.join(MODEL_DIR, "fusion_best.pth")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label2idx.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

# 加载特征 mean/std
feat_mean, feat_std = load_train_feature_stats(REFER_CSV_PATH)
feat_dim = feat_mean.shape[0]

# 加载 label 映射
with open(LABEL_MAP_PATH, "r") as f:
    label2idx = json.load(f)
idx2label = {int(v): k for k, v in label2idx.items()}
num_classes = len(label2idx)
print("[INFO] label2idx:", label2idx)

# 构建融合模型
fusion_model = build_fusion_model(
    num_classes=num_classes,
    feat_dim=feat_dim,
    ckpt_path=CKPT_PATH,
    device=device,
)


# ===================== 5. Gradio 回调函数 =====================
def gradio_predict(image_path):

    if image_path is None:
        return {}

    # 1) 提取 hand-crafted feature
    feat_vec = extract_feature_for_single_image(
        image_path,
        efd_order=10,
        downsample=2,
        max_regions=6,
    )
    feat_vec = np.nan_to_num(feat_vec, nan=0.0)

    feat_norm = (feat_vec - feat_mean) / feat_std
    feat_tensor = torch.from_numpy(feat_norm).float().unsqueeze(0).to(device)  # [1, F]

    # 2) 图像预处理（和训练一致，这里用灰度 + ToTensor）
    pil_img = Image.open(image_path).convert("L")
    img_tensor = T.ToTensor()(pil_img).unsqueeze(0).to(device)  # [1,1,H,W]

    # 3) 前向推理
    with torch.no_grad():
        logits = fusion_model(img_tensor, feat_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    prob_dict = {idx2label[i]: float(probs[i]) for i in range(num_classes)}
    return prob_dict


# ===================== 6. 启动 Gradio App =====================
def main():
    title = "Drosophila Wing Gene Classifier (Fusion CNN + Geometric Features)"
    description = (
        "Plz upload an image of wing, the model would extract both the geometric features and"
        "computer vision features, and then classify the gene of the wing."
    )

    iface = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Image(type="filepath", label="Upload wing image"),
        outputs=gr.Label(num_top_classes=5, label="Predicted gene (with probabilities)"),
        title=title,
        description=description,
    )

    iface.launch()


if __name__ == "__main__":
    main()
