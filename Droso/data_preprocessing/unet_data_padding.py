import os
import glob
import json
from PIL import Image, ImageOps

SRC_DIR = "/home/clab/Downloads/jiaxin_temporal/Droso/data/unet_segdata"
OUT_DIR = "/home/clab/Downloads/jiaxin_temporal/Droso/data/unet_segdata_padded"

os.makedirs(OUT_DIR, exist_ok=True)

#padding target size
TARGET_W = 2048
TARGET_H = 1536


def pad_to_target(img, target_w, target_h, fill_value):
    """右下 padding 到统一尺寸"""
    w, h = img.size
    if w > target_w or h > target_h:
        raise ValueError(
            f"Image larger than target size: got {w}x{h}, target {target_w}x{target_h}"
        )
    pad_w = target_w - w
    pad_h = target_h - h
    return ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=fill_value)


def main():
    # 找到所有原图
    tif_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.tif")))
    print(f"[INFO] Found {len(tif_paths)} tif images")

    id_counter = 1
    index_map = {}

    for tif_path in tif_paths:
        base = os.path.basename(tif_path)
        stem = os.path.splitext(base)[0]  # 原始名字（任何格式：1 / 8-1 / Sample 1）

        # 对应 mask
        mask_name = stem + "_Simple Segmentation.png"
        mask_path = os.path.join(SRC_DIR, mask_name)

        if not os.path.exists(mask_path):
            print(f"[WARN] Missing mask for {tif_path}, skip.")
            continue

        # 读取图像
        img = Image.open(tif_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # padding：img=0, mask=4
        img_pad = pad_to_target(img, TARGET_W, TARGET_H, fill_value=220)
        mask_pad = pad_to_target(mask, TARGET_W, TARGET_H, fill_value=5)

        # 生成 id
        new_id = f"id{id_counter}"

        # 保存文件名
        img_out = os.path.join(OUT_DIR, f"{new_id}.png")
        mask_out = os.path.join(OUT_DIR, f"{new_id}_mask.png")

        img_pad.save(img_out)
        mask_pad.save(mask_out)

        # 记录映射
        index_map[new_id] = {
            "orig_image": tif_path,
            "orig_mask": mask_path,
            "saved_image": img_out,
            "saved_mask": mask_out,
        }

        print(f"[OK] {new_id}: {tif_path} → {img_out}")

        id_counter += 1

    # 保存索引文件，便于追踪
    index_file = os.path.join(OUT_DIR, "index.json")
    with open(index_file, "w") as f:
        json.dump(index_map, f, indent=4)

    print(f"[DONE] saved mapping file to {index_file}")


if __name__ == "__main__":
    main()
