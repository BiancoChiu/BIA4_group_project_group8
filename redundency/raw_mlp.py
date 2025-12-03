import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import json
import time

from utils import WingClsDataset           
from models import (            
    WingFeatureExtractor,
    MLPClassifier,
)

if __name__ == "__main__":

    data_dir    = r"/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_magnification_cropped_png"
    results_dir = "results_mlp"

    # ==============================
    # 1. Dataset & label 映射
    # ==============================
    dataset = WingClsDataset(
        data_dir,
        label_key="type",
    )

    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Results directory: {results_dir}")

    print("[INFO] Class mapping:", dataset.label2idx)
    img0, label0, meta0 = dataset[0]
    print("[INFO] Example sample -> Label:", label0, "Meta:", meta0)

    label2idx = dataset.label2idx                      # e.g. {"egfr":0, "tkv":1, ...}
    idx2label = {v: k for k, v in label2idx.items()}   # e.g. {0:"egfr", 1:"tkv", ...}

    with open(os.path.join(results_dir, "label2idx.json"), "w") as f:
        json.dump(label2idx, f, indent=2)
    print("[INFO] Saved label2idx.json")

    # ==============================
    # 2. Train / Val 划分
    # ==============================
    n = len(dataset)
    indices = np.random.permutation(n)
    split = int(0.8 * n)

    train_idx = indices[:split]
    val_idx   = indices[split:]

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    # 注意：这里 DataLoader 还是返回 (imgs, labels, metas)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=0)

    # ==============================
    # 3. 构建特征提取器 + MLP
    # ==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label2idx)

    # 形态学特征提取器（在 CPU 上跑 skimage / skan）
    extractor = WingFeatureExtractor(
        efd_order=10,
        max_regions=6,
        vein_radius=3,
        downsample=2,
        nan_fill=0.0,
    )
    feature_dim = extractor.feature_dim
    print(f"[INFO] Feature dim = {feature_dim}")

    # 小 MLP 分类器
    mlp = MLPClassifier(in_dim=feature_dim, num_classes=num_classes, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    # ==============================
    # 4. 简单看一眼一个 batch 的特征
    # ==============================
    imgs_dbg, labels_dbg, metas_dbg = next(iter(train_loader))
    start = time.time()
    feats_dbg = extractor.extract_batch(imgs_dbg)   # [B, D] on CPU
    end = time.time()
    
    print("[TEST] time for one image:", end - start, "seconds")

    #print("[DEBUG] One batch images:", imgs_dbg.shape)
    #print("[DEBUG] One batch features:", feats_dbg.shape)
    #print("[DEBUG] First sample first 10 features:", feats_dbg[0, :10])

    # ==============================
    # 5. 训练若干 epoch
    # ==============================
    for epoch in range(5):
        print('start training')
        # ----------------- Train -----------------
        mlp.train()
        total_correct, total_samples = 0, 0
        total_loss = 0.0

        for imgs, labels, metas in train_loader:
            # imgs: [B,1,H,W] on CPU
            # 先提特征（CPU），再把特征搬到 GPU
            with torch.no_grad():
                feats = extractor.extract_batch(imgs)   # [B, D], CPU
            
            print(feats.shape)

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
            for imgs, labels, metas in val_loader:
                # 提特征
                feats = extractor.extract_batch(imgs)   # [B, D], CPU
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

        print(
            f"Epoch {epoch+1}: "
            f"train_acc={train_acc:.3f}, train_loss={train_loss:.3f}, "
            f"val_acc={val_acc:.3f}, val_loss={val_loss:.3f}"
        )

        # ----------------- Save preds/labels -----------------
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        np.save(os.path.join(results_dir, f"val_preds_epoch{epoch+1}.npy"), all_preds)
        np.save(os.path.join(results_dir, f"val_labels_epoch{epoch+1}.npy"), all_labels)

        print("[INFO] Saved predictions to", results_dir)
