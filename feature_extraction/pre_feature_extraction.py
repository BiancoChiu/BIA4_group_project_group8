"""
full_feature_extractor_from_folder.py

从翅膀图片文件夹自动提取：
1) 文件名解析的 metadata（gene / sex / side / id ...）
2) 几何特征
3) EFD 轮廓特征
4) skeleton graph 特征
5) intervein 区域特征

输出 CSV：full_wing_features.csv

列结构大致为：
[file_name, gene, sex, side, magnification, img_id, feature]

其中 feature 是一个 list，包含所有数值特征：
[geom..., efd_..., skeleton..., intervein...]
"""

import os
import numpy as np
import pandas as pd

from skimage.morphology import binary_dilation
from scipy.ndimage import distance_transform_edt  # 如果不用可以删掉

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    remove_small_objects, binary_closing, disk, skeletonize
)
from skimage.measure import label, regionprops, find_contours

from pyefd import elliptic_fourier_descriptors
from skan import Skeleton, summarize


# ==========================================================
#                 路径配置（改这 2 个）
# ==========================================================
IMG_DIR  = r"/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_magnification_cropped_png"
CSV_OUT  = r"/home/clab/Downloads/jiaxin_temporal/Droso/data/full_wing_features4mlpclassifier.csv"


# ==========================================================
#       0. 文件名解析（gene / sex / side / id）
# ==========================================================
def parse_filename(fname):
    """
    假设格式类似：
        egfr_F_L_oly_4X_8_cropped.png
        tkv_M_R_oly_4X_63_cropped.jpg
    自动提取 gene / sex / side / magnification / index
    """
    name = os.path.basename(fname)
    name_no_ext = os.path.splitext(name)[0]

    parts = name_no_ext.split("_")
    # egfr_F_L_oly_4X_8_cropped
    #  0    1 2   3   4   5    6

    gene = parts[0]
    sex = parts[1]
    side = parts[2]

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
        "img_id": idx
    }


# ==========================================================
#       1. 原图 → mask
# ==========================================================
def get_wing_mask(img_path, downsample=2):
    img = imread(img_path)
    if img.ndim == 3:
        img = rgb2gray(img)

    if downsample > 1:
        img = img[::downsample, ::downsample]

    # 平滑 + 阈值
    blur = gaussian(img, sigma=1.0)
    th = threshold_otsu(blur)
    bw = blur > th

    # 去噪 + 闭操作
    bw = remove_small_objects(bw, min_size=200)
    bw = binary_closing(bw, disk(3))

    # 最大连通域
    lab = label(bw)
    props = regionprops(lab)
    if len(props) == 0:
        return None

    region = max(props, key=lambda r: r.area)
    return (lab == region.label).astype(np.uint8)


# ==========================================================
#       2. 几何特征
# ==========================================================
def extract_geom_features(mask):
    lab = label(mask)
    props = regionprops(lab)
    if len(props) == 0:
        return {k: np.nan for k in [
            "wing_area","perimeter","major_axis","minor_axis",
            "aspect_ratio","eccentricity","solidity",
            "skeleton_len","skeleton_density"
        ]}

    region = max(props, key=lambda r: r.area)

    area = region.area
    perimeter = region.perimeter
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    aspect_ratio = major_axis / minor_axis if minor_axis>0 else np.nan
    eccentricity = region.eccentricity
    solidity = region.solidity

    # skeleton
    skel = skeletonize(mask>0)
    sk_len = np.count_nonzero(skel)
    sk_density = sk_len / area if area>0 else np.nan

    return {
        "wing_area": float(area),
        "perimeter": float(perimeter),
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "aspect_ratio": float(aspect_ratio),
        "eccentricity": float(eccentricity),
        "solidity": float(solidity),
        "skeleton_len": float(sk_len),
        "skeleton_density": float(sk_density)
    }


# ==========================================================
#       3. EFD 特征
# ==========================================================
def get_longest_contour(mask):
    contours = find_contours(mask, 0.5)
    if len(contours) == 0:
        return None
    cont = max(contours, key=lambda x: x.shape[0])
    # (row,col)->(x,y)
    return np.fliplr(cont)

def extract_efd_features(contour, order=10):
    if contour is None or contour.shape[0] < 20:
        return np.full(4*order, np.nan)

    coeffs = elliptic_fourier_descriptors(
        contour, order=order, normalize=True
    )
    return coeffs.flatten()


# ==========================================================
#       4. skeleton graph 特征
# ==========================================================
def extract_skeleton_graph_features(mask):
    skel = skeletonize(mask>0)
    sk = Skeleton(skel)
    df_s = summarize(sk)

    if df_s.shape[0] == 0:
        return {
            "branch_count":0, "endpoint_count":0, "junction_count":0,
            "branch_total_len":0, "branch_mean_len":0,
            "branch_max_len":0, "branch_len_std":0
        }

    # 找距离列
    if "branch-distance" in df_s.columns:
        dist_col="branch-distance"
    elif "distance" in df_s.columns:
        dist_col="distance"
    else:
        dist_col=[c for c in df_s.columns if "dist" in c][0]

    lengths = df_s[dist_col].values

    total = float(lengths.sum())
    mean  = float(lengths.mean())
    maxl  = float(lengths.max())
    std   = float(lengths.std())
    count = len(lengths)

    # branch-type: 1=endpoint,3=junction
    if "branch-type" in df_s.columns:
        endpoint = int((df_s["branch-type"]==1).sum())
        junction = int((df_s["branch-type"]==3).sum())
    else:
        endpoint = 0
        junction = 0

    return {
        "branch_count":count,
        "endpoint_count":endpoint,
        "junction_count":junction,
        "branch_total_len":total,
        "branch_mean_len":mean,
        "branch_max_len":maxl,
        "branch_len_std":std
    }


# ===================== intervein 区域特征 =====================
def extract_intervein_features(wing_mask, skel, max_regions=6, vein_radius=3):
    """
    由翅膀 mask + skeleton 近似得到 intervein 区域，并提取前 max_regions 个区域的形态特征。
    返回 keys 形如：
      iv1_area_frac, iv1_ecc, iv1_sol, ..., iv6_area_frac, iv6_ecc, iv6_sol
    """
    from skimage.measure import label, regionprops

    wing_mask = wing_mask.astype(bool)
    skel = skel.astype(bool)

    vein_mask = binary_dilation(skel, disk(vein_radius))
    intervein_mask = wing_mask & (~vein_mask)

    lab = label(intervein_mask)
    props = regionprops(lab)

    wing_area = float(wing_mask.sum()) if wing_mask is not None else 0.0

    feats = {}
    for k in range(1, max_regions + 1):
        feats[f"iv{k}_area_frac"] = np.nan
        feats[f"iv{k}_ecc"] = np.nan
        feats[f"iv{k}_sol"] = np.nan

    if len(props) == 0 or wing_area == 0:
        return feats

    props_sorted = sorted(props, key=lambda r: r.area, reverse=True)

    for idx, region in enumerate(props_sorted[:max_regions]):
        k = idx + 1
        area_frac = float(region.area) / wing_area
        ecc = float(region.eccentricity)
        sol = float(region.solidity)

        feats[f"iv{k}_area_frac"] = area_frac
        feats[f"iv{k}_ecc"] = ecc
        feats[f"iv{k}_sol"] = sol

    return feats


# ==========================================================
#                   主流程
# ==========================================================
def main():
    file_list = [f for f in os.listdir(IMG_DIR)
                 if f.lower().endswith((".png",".jpg",".jpeg",".tif",".bmp"))]

    print("Found", len(file_list), "images.")

    out_records = []
    EFD_ORDER = 10
    MAX_IV_REGIONS = 6

    # 提前定义 feature 顺序，保证每一行一致
    geom_keys = [
        "wing_area", "perimeter", "major_axis", "minor_axis",
        "aspect_ratio", "eccentricity", "solidity",
        "skeleton_len", "skeleton_density"
    ]
    efd_keys = [f"efd_{i}" for i in range(4 * EFD_ORDER)]
    skel_keys = [
        "branch_count", "endpoint_count", "junction_count",
        "branch_total_len", "branch_mean_len",
        "branch_max_len", "branch_len_std"
    ]
    iv_keys = [
        f"iv{k}_{suffix}"
        for k in range(1, MAX_IV_REGIONS + 1)
        for suffix in ["area_frac", "ecc", "sol"]
    ]
    feature_keys = geom_keys + efd_keys + skel_keys + iv_keys

    for fname in file_list:
        img_path = os.path.join(IMG_DIR, fname)

        # ---- 解析文件名 ----
        meta = parse_filename(fname)

        # ---- 生成 mask ----
        mask = get_wing_mask(img_path)
        if mask is None:
            print(f"[WARN] no wing detected in: {img_path}")
            geom = {k: np.nan for k in geom_keys}
            efd_vec = np.full(4 * EFD_ORDER, np.nan)
            skel_feats = {k: np.nan for k in skel_keys}
            intervein_feats = {k: np.nan for k in iv_keys}
        else:
            skel = skeletonize(mask > 0)

            geom = extract_geom_features(mask)

            cont = get_longest_contour(mask)
            efd_vec = extract_efd_features(cont, order=EFD_ORDER)

            skel_feats = extract_skeleton_graph_features(mask)

            intervein_feats = extract_intervein_features(
                mask, skel,
                max_regions=MAX_IV_REGIONS,
                vein_radius=3
            )

        # 组装一行 dict
        rec = dict(meta)

        # 暂存——方便后面统一拼成 feature list
        for k in geom_keys:
            rec[k] = geom.get(k, np.nan)

        for i, k in enumerate(efd_keys):
            rec[k] = float(efd_vec[i]) if i < len(efd_vec) else np.nan

        for k in skel_keys:
            rec[k] = skel_feats.get(k, np.nan)

        for k in iv_keys:
            rec[k] = intervein_feats.get(k, np.nan)

        # ==========================================
        #  单一 feature 列：list[float]，顺序由 feature_keys 确定
        # ==========================================
        feature_vector = [rec[k] for k in feature_keys]
        rec["feature"] = feature_vector

        # 如果你只想在 CSV 里保留 feature 这一列，
        # 那就把原来的数值特征列删掉：
        for k in feature_keys:
            del rec[k]

        out_records.append(rec)

    df = pd.DataFrame(out_records)
    df.to_csv(CSV_OUT, index=False)
    print("Saved:", CSV_OUT)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()
