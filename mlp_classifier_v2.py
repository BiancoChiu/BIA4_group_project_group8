import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from ast import literal_eval
from utils import WingCSVFeatureDataset
from visualization import plot_confusion_matrix, compute_f1_macro

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================
#  3. 训练脚本主体
# ==========================================================
if __name__ == "__main__":
    csv_path    = "/home/clab/Downloads/jiaxin_temporal/Droso/data/full_wing_featuresv3fortrain.csv"
    results_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/results_mlp_classifier_test2"

    os.makedirs(results_dir, exist_ok=True)

    # 固定一下随机种子（可选）
    np.random.seed(42)
    torch.manual_seed(42)

    # ---------- 载入 Dataset ----------
    dataset = WingCSVFeatureDataset(csv_path, label_col="gene")
    N, D = dataset.features.shape
    num_classes = len(dataset.label2idx)

    # 保存 label2idx
    with open(os.path.join(results_dir, "label2idx.json"), "w") as f:
        json.dump(dataset.label2idx, f, indent=2)
    print("[INFO] Saved label2idx.json to", results_dir)

    # ---------- Train / Val 划分 ----------
    indices = np.random.permutation(N)
    split = int(0.8 * N)
    train_idx = indices[:split]
    val_idx   = indices[split:]

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=0)

    # ---------- Model / Optimizer ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = MLPClassifier(in_dim=D, num_classes=num_classes, hidden_dim=128, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] MLP in_dim={D}, num_classes={num_classes}")

    num_epochs = 20

    for epoch in range(num_epochs):

        # ----------------- Train -----------------
        mlp.train()
        total_correct, total_samples = 0, 0
        total_loss = 0.0

        for feats, labels, metas in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = mlp(feats)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += feats.size(0)
            total_loss += loss.item() * feats.size(0)

        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples

        # ----------------- Validation -----------------
        mlp.eval()
        val_correct, val_samples = 0, 0
        val_loss_sum = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for feats, labels, metas in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)

                logits = mlp(feats)
                loss = F.cross_entropy(logits, labels)

                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_samples += feats.size(0)
                val_loss_sum += loss.item() * feats.size(0)

                all_preds.append(pred.cpu())
                all_labels.append(labels.cpu())

        val_acc = val_correct / val_samples
        val_loss = val_loss_sum / val_samples

        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels).numpy()
        macro_f1, per_class_f1 = compute_f1_macro(all_labels_np, all_preds_np, num_classes)

        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"train_acc={train_acc:.3f}, train_loss={train_loss:.3f}, "
            f"val_acc={val_acc:.3f}, val_loss={val_loss:.3f}"
            f"val_macro_f1={macro_f1:.3f}"
        )

        # 保存当前 epoch 的预测
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        np.save(os.path.join(results_dir, f"val_preds_epoch{epoch+1}.npy"), all_preds)
        np.save(os.path.join(results_dir, f"val_labels_epoch{epoch+1}.npy"), all_labels)
        print("[INFO] Saved predictions to", results_dir)

        # 可选：每个 epoch 保存一次模型
    torch.save(mlp.state_dict(), os.path.join(results_dir, f"mlp_epoch{epoch+1}.pth"))
    print("[INFO] Saved model to", os.path.join(results_dir, f"mlp_epoch{epoch+1}.pth"))
