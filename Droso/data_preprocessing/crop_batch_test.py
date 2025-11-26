import os
from glob import glob

import cv2
from skimage import io
import numpy as np


def crop_bottom(image, bottom_ratio=0.2):
    """
    只从底部裁掉一部分图像。

    参数
    ----
    image : np.ndarray
        输入图像，可以是灰度 (H, W) 或彩色 (H, W, C)。
    bottom_ratio : float
        要切掉的底部高度比例，例如 0.2 表示切掉底部 20% 高度。
    """
    h = image.shape[0]
    cut_h = int(h * bottom_ratio)

    if cut_h <= 0:
        # 不裁剪
        return image

    # 保留上方 [0, h - cut_h) 区域，宽度与通道保持不变
    cropped = image[: h - cut_h, ...]
    return cropped


def crop_wing_from_image(image, output_path="", bottom_ratio=0.2):
    """
    按照“只切掉图像下面一部分”的需求封装的裁剪函数。
    目前只是简单调用 crop_bottom，并负责保存。
    """
    cropped = crop_bottom(image, bottom_ratio=bottom_ratio)

    if output_path:
        # 灰度 / 彩色都可以直接用 cv2.imwrite
        cv2.imwrite(output_path, cropped)

    return cropped


def batch_crop_tif_folder(input_dir, output_dir=None, bottom_ratio=0.2):
    """
    批量处理一个文件夹下的所有 .tif 文件：
    - 读入灰度（或彩色）tif
    - 裁掉底部 bottom_ratio 高度
    - 保存为 .png 到输出目录
    """
    tif_files = glob(os.path.join(input_dir, "*.tif"))
    print(f"Found {len(tif_files)} files in {input_dir}")

    if output_dir is None:
        output_dir = os.path.join(input_dir, "cropped_png")
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(tif_files, 1):
        # 读取 tif：通常为灰度 (H, W)，skimage.io 会自动处理
        img = io.imread(path)

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, base + "_cropped.png")

        cropped = crop_wing_from_image(
            img,
            output_path=out_path,
            bottom_ratio=bottom_ratio,
        )

        print(f"[{i}/{len(tif_files)}] Saved: {out_path}")


if __name__ == "__main__":
    # ======= 在这里改路径和参数 ========
    input_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_magnification"
    output_dir = "/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_magnification_cropped_png"

    # 切掉底部 20%，可以改成 0.1 / 0.3 等
    bottom_ratio = 0.115

    batch_crop_tif_folder(input_dir, output_dir=output_dir, bottom_ratio=bottom_ratio)
