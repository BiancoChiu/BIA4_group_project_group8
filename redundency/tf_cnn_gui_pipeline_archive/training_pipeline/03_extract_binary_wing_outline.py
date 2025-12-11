import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import gaussian, meijering, threshold_otsu
from skimage import exposure, morphology
from skimage.morphology import binary_closing, disk

# ========== 1. Path setting ==========
input_folder  = r"/Users/yanyanru/desktop/genotype/train/egrf3"  # input folder
output_folder = r"/Users/yanyanru/desktop/genotype/train/egrf-binary"  # output binary folder

os.makedirs(output_folder, exist_ok=True)

# ========== 2. Batch processing ==========
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        continue

    in_path = os.path.join(input_folder, filename)
    print(f"deal with: {in_path}")

    # 1) Reading and downsampling
    img = imread(in_path)[::2, ::2]

    # 2) Turn grayscale & normalize to [0,1]
    if img.ndim == 3:
        gray = rgb2gray(img)
    else:
        gray = img.astype("float32")
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    # 3) Slight smoothing
    smooth = gaussian(gray, sigma=1.0)

    # 4) Meijering Ridge filtering to enhance linear structures
    vein_resp = meijering(
        smooth,
        sigmas=range(2, 8),
        black_ridges=True
    )

    vein_resp = exposure.rescale_intensity(vein_resp, out_range=(0, 1))

    # 5) Threshold + morphology processing to get clean binary map
    th_otsu = threshold_otsu(vein_resp)
    factor = 0.1         
    binary = vein_resp > (th_otsu * factor)

    binary = binary_closing(binary, disk(2))
    clean = morphology.remove_small_objects(binary, min_size=200)

    # 6) Save the black-and-white binary image（0/255）
    bin_uint8 = (clean.astype(np.uint8) * 255)
    out_name = os.path.splitext(filename)[0] + "_binary.png"
    out_path = os.path.join(output_folder, out_name)

    imsave(out_path, bin_uint8)
    print(f"Save the binary image: {out_path}")

print("All processing is complete！")
