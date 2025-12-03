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


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)   # 保证相对路径以 Droso/ 为根

# 相对路径配置
REFER_CSV_PATH = "./data/reference_feature.csv"

GENE_MODEL_DIR = "./fusion_cnn_mlp_model"
SEX_MODEL_DIR  = "./fusion_cnn_mlp_sex_model_try_best"

GENE_CKPT_PATH = os.path.join(GENE_MODEL_DIR, "fusion_best.pth")
GENE_LABEL_MAP = os.path.join(GENE_MODEL_DIR, "label2idx.json")

SEX_CKPT_PATH  = os.path.join(SEX_MODEL_DIR, "fusion_best.pth")
SEX_LABEL_MAP  = os.path.join(SEX_MODEL_DIR, "label2idx.json")



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
    device
):

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


    state = torch.load(ckpt_path, map_location=device)
    fusion_model.load_state_dict(state, strict=True)
    fusion_model.eval()
    print(f"[INFO] Loaded fusion model from {ckpt_path}")

    return fusion_model


def extract_feature_from_pil_image(
    pil_img,
    efd_order=10,
    downsample=2,
    max_regions=6,
):

    tmp_dir = tempfile.mkdtemp(prefix="wing_gradio_")
    try:
        img_path = os.path.join(tmp_dir, "input.png")
        pil_img.save(img_path)

        df = extract_wing_features_featurelist(
            img_dir=tmp_dir,
            csv_out=None,
            efd_order=efd_order,
            downsample=downsample,
            max_regions=max_regions,
            verbose=False,
        )

        if "feature" not in df.columns:
            raise RuntimeError("extract_wing_features_featurelist 输出 df 中缺少 'feature' 列")

        feat_val = df.loc[0, "feature"]

        if isinstance(feat_val, list):
            feat_vec = np.array(feat_val, dtype=np.float32)
        else:
            feat_vec = np.array(ast.literal_eval(feat_val), dtype=np.float32)

        return feat_vec

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

feat_mean, feat_std = load_train_feature_stats(REFER_CSV_PATH)
feat_dim = feat_mean.shape[0]

# gene label2idx
with open(GENE_LABEL_MAP, "r") as f:
    gene_label2idx = json.load(f)
idx2gene = {int(v): k for k, v in gene_label2idx.items()}
num_gene_classes = len(idx2gene)
print("[INFO] gene label2idx:", gene_label2idx)

# sex label2idx
with open(SEX_LABEL_MAP, "r") as f:
    sex_label2idx = json.load(f)
idx2sex = {int(v): k for k, v in sex_label2idx.items()}
num_sex_classes = len(idx2sex)
print("[INFO] sex label2idx:", sex_label2idx)

# 两个模型
fusion_gene = build_fusion_model(
    num_classes=num_gene_classes,
    feat_dim=feat_dim,
    ckpt_path=GENE_CKPT_PATH,
    device=device,
)
fusion_sex = build_fusion_model(
    num_classes=num_sex_classes,
    feat_dim=feat_dim,
    ckpt_path=SEX_CKPT_PATH,
    device=device,
)

to_tensor = T.ToTensor()

def predict_gene_and_sex(pil_img):
    if pil_img is None:
        return "Please upload an image."


    pil_gray = pil_img.convert("L")

    # 1) handcrafted features
    feat_vec = extract_feature_from_pil_image(
        pil_gray,
        efd_order=10,
        downsample=2,
        max_regions=6,
    )

    feat_vec = np.nan_to_num(feat_vec, nan=0.0)
    feat_norm = (feat_vec - feat_mean) / feat_std
    feat_tensor = torch.from_numpy(feat_norm).float().unsqueeze(0).to(device)  # [1, F]

    # 2) image tensor
    img_tensor = to_tensor(pil_gray).unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        # gene
        logits_gene = fusion_gene(img_tensor, feat_tensor)
        probs_gene = F.softmax(logits_gene, dim=1)[0].cpu().numpy()
        gene_idx = int(probs_gene.argmax())
        gene_label = idx2gene[gene_idx]

        # sex
        logits_sex = fusion_sex(img_tensor, feat_tensor)
        probs_sex = F.softmax(logits_sex, dim=1)[0].cpu().numpy()
        sex_idx = int(probs_sex.argmax())
        sex_label = idx2sex[sex_idx]


    lines = []
    lines.append(f"### Prediction")
    lines.append("")
    lines.append(f"- **Gene**: **{gene_label}** (idx = {gene_idx})")
    lines.append(f"- **Sex** : **{sex_label}** (idx = {sex_idx})")
    lines.append("")
    lines.append("#### Gene probabilities")
    for i, p in enumerate(probs_gene):
        lines.append(f"- {idx2gene[i]}: {p:.4f}")
    lines.append("")
    lines.append("#### Sex probabilities")
    for i, p in enumerate(probs_sex):
        lines.append(f"- {idx2sex[i]}: {p:.4f}")

    return "\n".join(lines)

demo = gr.Interface(
    fn=predict_gene_and_sex,
    inputs=gr.Image(type="pil", label="Wing image"),
    outputs=gr.Markdown(label="Predictions"),
    title="Droso Wing Classifier (Gene + Sex)",
    description="Upload a wing image (cropped), the model will predict both gene and sex.",
)

if __name__ == "__main__":
    demo.launch()
