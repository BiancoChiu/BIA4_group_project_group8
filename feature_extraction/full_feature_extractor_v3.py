"""
final_feature_extractor.py

输出 DataFrame columns:
    file_name, gene, sex, side, magnification, img_id, feature_list
"""

import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu, meijering
from skimage.morphology import (
    remove_small_objects, remove_small_holes,
    binary_closing, binary_dilation, disk, skeletonize
)
from skimage.measure import label, regionprops, find_contours
from skimage import exposure, morphology

from pyefd import elliptic_fourier_descriptors
from skan import Skeleton, summarize


# ---------------------- 0. 文件名解析 ----------------------
def parse_filename(fname):
    name = os.path.basename(fname)
    name_no_ext = os.path.splitext(name)[0]

    parts = name_no_ext.split("_")

    gene = parts[0] if len(parts) > 0 else None
    sex = parts[1] if len(parts) > 1 else None
    side = parts[2] if len(parts) > 2 else None

    magnification = None
    for p in parts:
        if p.endswith("X"):
            magnification = p

    idx = None
    for p in parts:
        if p.isdigit():
            idx = int(p)

    return {
        "file_name": fname,
        "gene": gene,
        "sex": sex,
        "side": side,
        "magnification": magnification,
        "img_id": idx,
    }


# ---------------------- 1. Mask ----------------------
def get_wing_mask(img_path, downsample=2, sigma=2.0, min_size=8000):
    img = imread(img_path)
    if img.ndim == 3:
        gray = rgb2gray(img)
    else:
        gray = img.astype("float32")
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    if downsample > 1:
        gray = gray[::downsample, ::downsample]

    blur = gaussian(gray, sigma=sigma)
    th = threshold_otsu(blur)
    bw = blur < th  # 翅膀更暗

    bw = remove_small_objects(bw, min_size=min_size)
    bw = binary_closing(bw, disk(3))
    bw = remove_small_holes(bw, area_threshold=2000)

    lab = label(bw)
    props = regionprops(lab)
    if len(props) == 0:
        return None
    region = max(props, key=lambda r: r.area)
    return (lab == region.label).astype(np.uint8)


def get_vein_mask_meijering(img_path, downsample=2, factor=0.1, min_size=200):
    img = imread(img_path)

    if downsample > 1:
        img = img[::downsample, ::downsample]

    if img.ndim == 3:
        gray = rgb2gray(img)
    else:
        gray = img.astype("float32")
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    smooth = gaussian(gray, sigma=1.0)
    vein_resp = meijering(smooth, sigmas=range(2, 8), black_ridges=True)
    vein_resp = exposure.rescale_intensity(vein_resp, out_range=(0, 1))

    th_otsu = threshold_otsu(vein_resp)
    binary = vein_resp > (th_otsu * factor)

    binary = binary_closing(binary, disk(2))
    clean = morphology.remove_small_objects(binary, min_size=min_size)

    return clean.astype(bool)


# ---------------------- 2. 几何特征 ----------------------
def extract_geom_features(mask):
    lab = label(mask)
    props = regionprops(lab)
    if len(props) == 0:
        return [np.nan] * 9

    region = max(props, key=lambda r: r.area)
    area = region.area
    perimeter = region.perimeter
    major = region.major_axis_length
    minor = region.minor_axis_length
    aspect = major / minor if minor > 0 else np.nan

    eccentricity = region.eccentricity
    solidity = region.solidity

    skel = skeletonize(mask > 0)
    sk_len = np.count_nonzero(skel)
    sk_density = sk_len / area if area > 0 else np.nan

    return [
        float(area), float(perimeter), float(major), float(minor),
        float(aspect), float(eccentricity), float(solidity),
        float(sk_len), float(sk_density)
    ]


# ---------------------- 3. EFD ----------------------
def extract_efd(mask, order=10):
    contours = find_contours(mask, 0.5)
    if len(contours) == 0:
        return [np.nan] * (4 * order)

    cont = max(contours, key=lambda x: x.shape[0])
    cont = np.fliplr(cont)

    if cont.shape[0] < 20:
        return [np.nan] * (4 * order)

    coeffs = elliptic_fourier_descriptors(cont, order=order, normalize=True)
    return list(coeffs.flatten().astype(float))


# ---------------------- 4. Skeleton graph ----------------------
def extract_skel_graph(mask):
    skel = skeletonize(mask > 0)
    sk = Skeleton(skel)
    df = summarize(sk)

    if df.shape[0] == 0:
        return [0, 0, 0, 0, 0, 0, 0]

    if "branch-distance" in df.columns:
        dist_col = "branch-distance"
    else:
        dist_col = "distance"

    lengths = df[dist_col].values

    total = float(lengths.sum())
    mean = float(lengths.mean())
    maxl = float(lengths.max())
    std = float(lengths.std())
    count = len(lengths)

    if "branch-type" in df.columns:
        endpoint = int((df["branch-type"] == 1).sum())
        junction = int((df["branch-type"] == 3).sum())
    else:
        endpoint = junction = 0

    return [count, endpoint, junction, total, mean, maxl, std]


# ---------------------- 5. Intervein 特征 ----------------------
def extract_intervein(mask, vein_mask, max_regions=6):
    from skimage.measure import label, regionprops

    wing_mask = mask.astype(bool)
    vein_mask = vein_mask.astype(bool)

    inter = wing_mask & (~vein_mask)

    lab = label(inter)
    props = regionprops(lab)

    wing_area = float(wing_mask.sum())
    if wing_area == 0 or len(props) == 0:
        return [np.nan] * (max_regions * 3)

    props = sorted(props, key=lambda r: r.area, reverse=True)

    vals = []
    for k in range(max_regions):
        if k < len(props):
            r = props[k]
            vals += [
                float(r.area) / wing_area,
                float(r.eccentricity),
                float(r.solidity),
            ]
        else:
            vals += [np.nan, np.nan, np.nan]

    return vals


# ---------------------- 主函数：输出 feature_list ----------------------
def extract_wing_features_featurelist(
    img_dir,
    csv_out=None,
    efd_order=10,
    downsample=2,
    max_regions=6,
    verbose=True,
):
    file_list = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
    ]

    if verbose:
        print(f"[INFO] Found {len(file_list)} images.")

    rows = []

    for i, fname in enumerate(file_list, 1):
        if verbose and i % 10 == 0:
            print(f"[INFO] Processing {i}/{len(file_list)}")

        meta = parse_filename(fname)
        img_path = os.path.join(img_dir, fname)

        mask = get_wing_mask(img_path, downsample=downsample)

        if mask is None:
            # 全 NaN
            feature_list = (
                [np.nan]*9 + [np.nan]*(4*efd_order) +
                [np.nan]*7 + [np.nan]*(max_regions*3)
            )
        else:
            geom = extract_geom_features(mask)
            efd = extract_efd(mask, order=efd_order)
            skel = extract_skel_graph(mask)

            vein_mask = get_vein_mask_meijering(img_path, downsample=downsample)
            vein_mask &= mask.astype(bool)

            inter = extract_intervein(mask, vein_mask, max_regions=max_regions)

            # ------- 合并成 feature_list -------
            feature_list = geom + efd + skel + inter

        record = meta.copy()
        record["feature"] = feature_list
        rows.append(record)

    df = pd.DataFrame(rows)

    if csv_out:
        df.to_csv(csv_out, index=False)
        if verbose:
            print(f"[INFO] Saved to {csv_out}")

    return df


# ---------------------- CLI 调用 ----------------------
if __name__ == "__main__":
    extract_wing_features_featurelist(
        img_dir="/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_magnification_cropped_png",
        csv_out="/home/clab/Downloads/jiaxin_temporal/Droso/data/full_wing_featuresv3fortrain.csv",
    )
