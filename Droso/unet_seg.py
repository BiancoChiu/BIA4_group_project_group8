import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

from utils.dataloader import SegPairSameDirDataset
from models.unet import UNet
from sklearn.metrics import recall_score
from visualization import plot_confusion_matrix

def main():
    # ----------------- 数据路径 & 参数 -----------------
    root_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/data/unet_segdata_padded"
    output_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/output3/"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 2
    num_epochs = 10
    lr = 1e-3

    # 如果你的 mask 里：
    # - 类别为 0,1,2,3
    # - padding 像素值 = 4
    # 可以设置 num_classes=4，ignore_index=4；
    # 或设置 num_classes=5，忽略第 4 类，只用于 padding。
    NUM_CLASSES = 5
    IGNORE_INDEX = 5

    # ----------------- Dataset & Dataloader -----------------
    dataset = SegPairSameDirDataset(
        root_dir=root_dir,
        img_suffix=".png",
        mask_suffix="_mask.png",
        as_gray=True,
    )

    n = len(dataset)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    print(f"[INFO] Train size: {len(train_set)}, Val size: {len(val_set)}")

    # ----------------- 模型 & loss & 优化器 -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # CrossEntropyLoss: 输入 [B,C,H,W]，target [B,H,W]
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ----------------- 训练循环 -----------------
    for epoch in range(1, num_epochs + 1):
        # ======== Train ========
        model.train()
        train_loss_sum = 0.0

        for imgs, masks, info in train_loader:
            imgs = imgs.to(device)  # [B,1,H,W]
            # masks: [B,1,H,W] float / int -> [B,H,W] long
            masks = masks.squeeze(1).long().to(device)

            # 如果你的 mask 是 0/255，可以这样转成 0/1：
            # masks = (masks > 0).long()

            optimizer.zero_grad()
            logits = model(imgs)   
            #print("logits shape:", logits.shape) 
            
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / len(train_set)

        model.eval()
        val_loss_sum = 0.0
        total_correct = 0
        total_valid = 0

        # 用来保存像素级 label
        all_true = []
        all_pred = []

        with torch.no_grad():
            for imgs, masks, info in val_loader:
                imgs = imgs.to(device)
                masks = masks.squeeze(1).long().to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)

                preds = logits.argmax(dim=1)       # [B,H,W]
                unique, counts = torch.unique(preds, return_counts=True)
                #print("Predicted class distribution:", dict(zip(unique.tolist(), counts.tolist())))

                # 有效区域（不是 padding=4）
                valid = masks != IGNORE_INDEX

                # accuracy
                total_correct += ((preds == masks) & valid).sum().item()
                total_valid += valid.sum().item()

                # append pixel labels for confusion matrix
                all_true.append(masks[valid].cpu().numpy())
                all_pred.append(preds[valid].cpu().numpy())
                gt_class1 = (masks == 1)

                pred_class1 = (preds == 1)

                true_positive = (gt_class1 & pred_class1).sum().item()

                total_gt_class1 = gt_class1.sum().item()

                recall_class1 = true_positive / total_gt_class1 if total_gt_class1 > 0 else 0.0

        val_loss = val_loss_sum / len(val_set)
        val_acc = total_correct / total_valid if total_valid > 0 else 0.0

        # 整合 pixel-level 标签
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)

        # ------- recall -------
        val_recall = recall_score(all_true, all_pred, average="macro")

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_pixel_acc={val_acc:.4f} | "
            f"val_recall={val_recall:.4f} |"
            f"recell_class1={recall_class1:.4f}"
        )

    # ------- 保存 confusion matrix -------
        class_names = ["1", "2", "3", "4"]  # ← 你需要替换名字
        plot_confusion_matrix(
            y_true=all_true,
            y_pred=all_pred,
            class_names=class_names,
            normalize=True,
            save_path=os.path.join(output_dir, f"confusion_matrix_epoch{epoch}.png")
        )

    # 训练结束后保存模型
    #os.makedirs("checkpoints", exist_ok=True)
        if epoch // 4 == 0:
            ckpt_path = os.path.join(output_dir, f"unet_seg_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print("[INFO] Saved model to", ckpt_path)


if __name__ == "__main__":
    main()