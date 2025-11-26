import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import recall_score

from utils.dataloader import SegPairSameDirDataset
from models.unet import UNet
from visualization import plot_confusion_matrix


# =========================
# 一些小工具函数
# =========================
def map_raw_mask_to_train_ids(raw_masks, ignore_index=255):
    """
    raw_masks: [B, H, W]，像素值是 1,2,3,4,5
        - 1,2,3,4: 真正的语义类（4 是 bg）
        - 5: padding 区域，训练和评估都要忽略

    返回:
        train_masks: [B, H, W]
            - 0,1,2,3  对应原始 1,2,3,4
            - ignore_index 对应原始 5
    """
    # 记录 padding 区域
    padding_mask = (raw_masks == 5)

    # 所有标签先减 1：1→0, 2→1, 3→2, 4→3, 5→4
    train_masks = raw_masks - 1

    # 再把 padding 的位置改成 ignore_index
    train_masks[padding_mask] = ignore_index

    return train_masks


def main():
    # ----------------- 数据路径 & 参数 -----------------
    root_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/data/unet_segdata_padded"
    output_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/output/"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 2
    num_epochs = 20
    lr = 1e-4

    # 真实语义类只有 4 个（原始 1,2,3,4），5 是 padding
    NUM_CLASSES = 4
    IGNORE_INDEX = 255  # 任意不在 [0, NUM_CLASSES-1] 的值即可

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

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

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

            # 原始 masks: [B,1,H,W] -> [B,H,W]，像素值 1,2,3,4,5
            raw_masks = masks.squeeze(1).long().to(device)

            # 映射到训练用的类别 index：0,1,2,3,IGNORE_INDEX
            train_masks = map_raw_mask_to_train_ids(
                raw_masks, ignore_index=IGNORE_INDEX
            )

            optimizer.zero_grad()
            logits = model(imgs)   # [B,4,H,W]

            loss = criterion(logits, train_masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / len(train_set)

        # ======== Validation ========
        model.eval()
        val_loss_sum = 0.0
        total_correct = 0
        total_valid = 0

        # 用来保存像素级 label（内部 index 0–3）
        all_true = []
        all_pred = []

        # 记录原始标签 1 的 recall（内部 index 0）
        recall_class1 = 0.0

        with torch.no_grad():
            for imgs, masks, info in val_loader:
                imgs = imgs.to(device)
                raw_masks = masks.squeeze(1).long().to(device)

                val_masks = map_raw_mask_to_train_ids(
                    raw_masks, ignore_index=IGNORE_INDEX
                )

                logits = model(imgs)
                loss = criterion(logits, val_masks)
                val_loss_sum += loss.item() * imgs.size(0)

                preds = logits.argmax(dim=1)       # [B,H,W]，值在 0,1,2,3

                # ---------- 打印预测的类别分布 ----------
                unique, counts = torch.unique(preds, return_counts=True)
                internal_dist = dict(zip(unique.tolist(), counts.tolist()))
                # 映射回原始 label（+1：0→1, 1→2,...,3→4）
                orig_labels = [u + 1 for u in unique.tolist()]
                orig_dist = dict(zip(orig_labels, counts.tolist()))
                print(f"Epoch {epoch:02d} Val Pred class (index 0-3): {internal_dist}")
                print(f"Epoch {epoch:02d} Val Pred class (orig 1-4): {orig_dist}")

                # ---------- 有效区域（不是 padding） ----------
                valid = val_masks != IGNORE_INDEX

                # accuracy
                total_correct += ((preds == val_masks) & valid).sum().item()
                total_valid += valid.sum().item()

                # append pixel labels for confusion matrix / recall
                all_true.append(val_masks[valid].cpu().numpy())
                all_pred.append(preds[valid].cpu().numpy())

                # 原始 label=1 → 内部 index=0
                gt_class0 = (val_masks == 0) & valid
                pred_class0 = (preds == 0) & valid

                true_positive = (gt_class0 & pred_class0).sum().item()
                total_gt_class0 = gt_class0.sum().item()

                if total_gt_class0 > 0:
                    recall_class1 = true_positive / total_gt_class0
                else:
                    recall_class1 = 0.0

        val_loss = val_loss_sum / len(val_set)
        val_acc = total_valid and (total_correct / total_valid) or 0.0

        # 整合 pixel-level 标签（内部 index 0–3）
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)

        # ------- macro recall (对 4 个类平均) -------
        val_recall = recall_score(all_true, all_pred, average="macro")

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_pixel_acc={val_acc:.4f} | "
            f"val_macro_recall={val_recall:.4f} | "
            f"recall_class1(orig_label=1)={recall_class1:.4f}"
        )

        # ------- 保存 confusion matrix -------
        # class_names 用原始标签名字，对应内部 index 0,1,2,3
        class_names = ["1", "2", "3", "4(bg)"]
        plot_confusion_matrix(
            y_true=all_true,
            y_pred=all_pred,
            class_names=class_names,
            normalize=True,
            save_path=os.path.join(output_dir, f"confusion_matrix_epoch{epoch}.png")
        )

    # 训练结束后保存模型
    ckpt_path = os.path.join(output_dir, "unet_seg_final.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("[INFO] Saved model to", ckpt_path)


if __name__ == "__main__":
    main()
