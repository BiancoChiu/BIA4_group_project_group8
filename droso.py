import os
import json
import glob
import ast
import numpy as np
import pandas as pd

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from utils import FusionWingDataset
import torchvision.transforms as T
from sklearn.metrics import f1_score
from collections import Counter

from models import DeeperWingCNN2, FeatureEncoder
 
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

def main():
    # ---------- 配置 ----------
    image_dir   = "/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_magnification_cropped_png"
    csv_path    = "/home/clab/Downloads/jiaxin_temporal/Droso/data/reference_feature.csv"
    results_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/fusion_cnn_mlp_sex_model_try_best"

    os.makedirs(results_dir, exist_ok=True)
    CNN_CKPT = None
    # CNN_CKPT = "/path/to/your/deeper_cnn_best.pth"

    # 是否先冻结 CNN，只训练 feature + fusion
    FREEZE_CNN = False

    batch_size = 8
    num_epochs = 60
    lr = 3e-4
    weight_decay = 5e-4
    val_ratio = 0.2
    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------- Dataset & DataLoader ----------
    dataset = FusionWingDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        label_col="sex",
        transform=T.ToTensor(),
    )

    # 保存 label2idx
    with open(os.path.join(results_dir, "label2idx.json"), "w") as f:
        json.dump(dataset.label2idx, f, indent=2)
    print(f"[INFO] Saved label2idx.json to {results_dir}")

    n = len(dataset)
    indices = np.random.permutation(n)
    split = int((1 - val_ratio) * n)
    train_idx = indices[:split]
    val_idx   = indices[split:]

    # 看看类分布
    train_labels = [dataset[i][2] for i in train_idx]
    val_labels   = [dataset[i][2] for i in val_idx]
    print("Train class distribution (idx):", Counter(train_labels))
    print("Val   class distribution (idx):", Counter(val_labels))

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    num_classes = len(dataset.label2idx)
    feat_dim = len(dataset.row_by_fname[next(iter(dataset.row_by_fname))]["feature"])

    # ---------- 模型 ----------
    cnn = DeeperWingCNN2(in_channels=1, num_classes=num_classes)

    if CNN_CKPT is not None:
        state = torch.load(CNN_CKPT, map_location="cpu")
        cnn.load_state_dict(state, strict=True)
        print(f"[INFO] Loaded CNN checkpoint from {CNN_CKPT}")

    if FREEZE_CNN:
        for p in cnn.parameters():
            p.requires_grad = False
        print("[INFO] CNN encoder is FROZEN.")

    cnn = cnn.to(device)

    feat_encoder = FeatureEncoder(in_dim=feat_dim, hidden_dim=128, out_dim=64, dropout=0.1).to(device)
    fusion_model = FusionClassifier(
        cnn_encoder=cnn,
        feat_encoder=feat_encoder,
        num_classes=num_classes,
        fusion_hidden=256,
        dropout=0.2,
    ).to(device)

    # 只有需要更新梯度的参数会被优化
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, fusion_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(num_epochs):
        # ---- Train ----
        fusion_model.train()
        total_correct, total_samples = 0, 0
        total_loss = 0.0

        for imgs, feats, labels, metas in train_loader:
            imgs = imgs.to(device)           # [B,1,H,W]
            feats = feats.to(device)         # [B,F]
            labels = labels.to(device)       # [B]

            optimizer.zero_grad()
            logits = fusion_model(imgs, feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += imgs.size(0)
            total_loss += loss.item() * imgs.size(0)

        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples

        # ---- Validation ----
        fusion_model.eval()
        val_correct, val_samples = 0, 0
        val_loss_sum = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, feats, labels, metas in val_loader:
                imgs = imgs.to(device)
                feats = feats.to(device)
                labels = labels.to(device)

                logits = fusion_model(imgs, feats)
                loss = criterion(logits, labels)

                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_samples += imgs.size(0)
                val_loss_sum += loss.item() * imgs.size(0)

                all_preds.append(pred.cpu())
                all_labels.append(labels.cpu())

        val_acc = val_correct / val_samples
        val_loss = val_loss_sum / val_samples

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"train_acc={train_acc:.3f}, train_loss={train_loss:.3f}, "
            f"val_acc={val_acc:.3f}, val_loss={val_loss:.3f}, "
            f"val_macro_f1={macro_f1:.3f}"
        )

        # 保存当前轮的预测
        

        # 更新 best model（按 macro-F1）
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = fusion_model.state_dict()
            torch.save(best_state, os.path.join(results_dir, "fusion_best.pth"))
            print(f"[INFO] New best model saved (epoch {epoch+1}, macro_f1={macro_f1:.3f})")
            np.save(os.path.join(results_dir, f"val_preds_epoch{epoch+1}.npy"), all_preds)
            np.save(os.path.join(results_dir, f"val_labels_epoch{epoch+1}.npy"), all_labels)
            print(f"[INFO] Saved predictions to {results_dir}")

    print(f"[INFO] Training done. Best macro-F1 = {best_macro_f1:.3f}")


if __name__ == "__main__":
    main()
