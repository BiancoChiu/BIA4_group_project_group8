import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
import os
import json

from utils import WingClsDataset   # dataset
from models import SimpleWingCNN, DeeperWingCNN

if __name__ == "__main__":

    data_dir = r"/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_magnification_cropped_png"
    results_dir = "results"

    dataset = WingClsDataset(
        data_dir,
        label_key="type",
    )

    
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Results directory: {results_dir}")

    print("[INFO] Class mapping:", dataset.label2idx)
    img0, label0, meta0 = dataset[0]
    print("[INFO] Example sample -> Label:", label0, "Meta:", meta0)

    label2idx = dataset.label2idx                      # e.g. {"typeA":0, "typeB":1, ...}
    idx2label = {v: k for k, v in label2idx.items()}   # e.g. {0:"typeA", 1:"typeB", ...}

    with open(os.path.join(results_dir, "label2idx.json"), "w") as f:
        json.dump(label2idx, f, indent=2)
    print("[INFO] Saved label2idx.json")

    n = len(dataset)
    indices = np.random.permutation(n)
    split = int(0.8 * n)

    train_idx = indices[:split]
    val_idx   = indices[split:]

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False, num_workers=0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label2idx)

    cnn = DeeperWingCNN(in_channels=1, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)

    for epoch in range(5):

        # ----------------- Train -----------------
        cnn.train()
        total_correct, total_samples = 0, 0
        total_loss = 0.0

        for imgs, labels, metas in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = cnn(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += imgs.size(0)
            total_loss += loss.item() * imgs.size(0)

        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples


        # ----------------- Validation -----------------
        cnn.eval()
        val_correct, val_samples = 0, 0
        val_loss_sum = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels, metas in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = cnn(imgs)
                loss = F.cross_entropy(logits, labels)

                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_samples += imgs.size(0)
                val_loss_sum += loss.item() * imgs.size(0)

                # store results
                all_preds.append(pred.cpu())
                all_labels.append(labels.cpu())

        val_acc = val_correct / val_samples
        val_loss = val_loss_sum / val_samples

        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, train_loss={train_loss:.3f}, val_acc={val_acc:.3f}, val_loss={val_loss:.3f}")

        # ----------------- Save preds/labels -----------------
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        np.save(os.path.join(results_dir, "val_preds.npy"), all_preds)
        np.save(os.path.join(results_dir, "val_labels.npy"), all_labels)

        print("[INFO] Saved predictions to results/")
