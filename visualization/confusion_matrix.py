import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import os

def compute_f1_macro(y_true, y_pred, num_classes):
    """
    y_true, y_pred: 1D numpy array, int labels in [0, num_classes-1]
    返回:
      macro_f1: float
      per_class_f1: list[num_classes]
    """
    f1_list = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        if tp == 0 and fp == 0 and fn == 0:
            # 这个类在 y_true 和 y_pred 里都没出现，跳过
            f1 = np.nan
        else:
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_list.append(f1)

    # macro-F1: 忽略 NaN 的类
    valid_f1 = [f for f in f1_list if not np.isnan(f)]
    macro_f1 = float(np.mean(valid_f1)) if valid_f1 else np.nan
    return macro_f1, f1_list

def plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        normalize=False,
        cmap="Blues",
        save_path="confusion_matrix.png"
    ):
    """
    Draw and save confusion matrix (pure matplotlib).
    """

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # ticks / labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(class_names, fontsize=11)

    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=16)

    # values inside grid
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            txt = f"{v:.2f}" if normalize else f"{v}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if v > (cm.max() * 0.6) else "black",
                fontsize=12
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {save_path}")


if __name__ == "__main__":

    # ========== PATHS ==========
    base_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/fusion_cnn_mlp_sex_model_try_best"
    true_path = os.path.join(base_dir, "val_labels_epoch28.npy")
    pred_path = os.path.join(base_dir, "val_preds_epoch28.npy")
    label_map_path = os.path.join(base_dir, "label2idx.json")   # added

    # ========== LOAD ARRAYS ==========
    y_true = np.load(true_path).reshape(-1)
    y_pred = np.load(pred_path).reshape(-1)
    print("[INFO] Loaded shapes:", y_true.shape, y_pred.shape)

    # ========== LOAD LABEL MAPPING ==========
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            label2idx = json.load(f)

        # ensure sorted by index
        idx2label = {int(v): k for k, v in label2idx.items()}
        class_names = [idx2label[i] for i in range(len(idx2label))]
        print("[INFO] Loaded class names:", class_names)
    else:
        print("[WARNING] label2idx.json not found! Using default numeric class names.")
        num_classes = len(np.unique(y_true))
        class_names = [f"Class {i}" for i in range(num_classes)]

    # ========== SAVE PLOTS ==========
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        normalize=False,
        save_path=os.path.join(base_dir, "confusion_matrix_raw.png")
    )

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        normalize=True,
        save_path=os.path.join(base_dir, "confusion_matrix_normalized.png")
    )
