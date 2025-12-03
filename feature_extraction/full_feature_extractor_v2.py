"""
full_feature_extractor_from_folder.py

从翅膀图片文件夹自动提取：
1) 文件名解析的 metadata（gene / sex / side / id ...）
2) 几何特征
3) EFD 轮廓特征
4) skeleton graph 特征
5) intervein 区域特征

核心函数：
    extract_wing_features(
        img_dir,
        csv_out=None,
        efd_order=10,
        downsample=2,
        max_regions=6,
        vein_radius=3,
        verbose=True,
    ) -> pandas.DataFrame
"""

import os
import numpy as np
import pandas as pd

from skimage.morphology import binary_dilation
from scipy.ndimage import distance_transform_edt  # 可选，不用也行

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    remove_small_objects, binary_closing, disk, skeletonize
)
from skimage.measure import label, regionprops, find_contours

from pyefd import elliptic_fourier_descriptors
from skan import Skeleton, summarize

def parse_filename(fname):
    """
    假设格式类似：
        egfr_F_L_oly_4X_8_cropped.png
        tkv_M_R_oly_4X_63_cropped.jpg
    自动提取 gene / sex / side / magnification / index
    """
    name = os.path.basename(fname)
    name_no_ext = os.path.splitext(name)[0]

    # 用下划线分割
    parts = name_no_ext.split("_")
    # egfr_F_L_oly_4X_8_cropped
    #  0    1 2   3   4   5    6

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
        "file_name": name,   
        "gene": gene,
        "sex": sex,
        "side": side,
        "magnification": magnification,
        "img_id": idx
    }


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

    bw = remove_small_objects(bw, min_size=200)
    bw = binary_closing(bw, disk(3))

    lab = label(bw)
    props = regionprops(lab)
    if len(props) == 0:
        return None

    region = max(props, key=lambda r: r.area)
    return (lab == region.label).astype(np.uint8)

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


def extract_intervein_features(wing_mask, skel, max_regions=6, vein_radius=3):
    """
    由翅膀 mask + skeleton 近似得到 intervein 区域，并提取前 max_regions 个区域的形态特征。
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

def extract_wing_features(
    img_dir,
    csv_out=None,
    efd_order=10,
    downsample=2,
    max_regions=6,
    vein_radius=3,
    verbose=True,
):
    """
    返回的 DataFrame 列：
        file_name, gene, sex, side, magnification, img_id, feature_list

    feature_list 内部顺序固定为：
        [geom(9), efd(4*efd_order), skeleton(7), intervein(max_regions*3)]
    """
    file_list = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
    ]

    if verbose:
        print(f"[INFO] Found {len(file_list)} images in {img_dir}")

    out_records = []

    # 固定顺序的 key 列表，后面拼 list 要用
    geom_keys = [
        "wing_area", "perimeter", "major_axis", "minor_axis",
        "aspect_ratio", "eccentricity", "solidity",
        "skeleton_len", "skeleton_density",
    ]
    skel_keys = [
        "branch_count", "endpoint_count", "junction_count",
        "branch_total_len", "branch_mean_len",
        "branch_max_len", "branch_len_std",
    ]

    for i, fname in enumerate(file_list, 1):
        img_path = os.path.join(img_dir, fname)

        if verbose and i % 10 == 0:
            print(f"[INFO] Processing {i}/{len(file_list)}: {fname}")

        # ---------- meta ----------
        meta = parse_filename(fname)

        # ---------- mask ----------
        mask = get_wing_mask(img_path, downsample=downsample)

        if mask is None:
            if verbose:
                print(f"[WARN] no wing detected in: {img_path}")

            # 填 NaN，占位，保证长度一致
            geom = {k: np.nan for k in geom_keys}
            efd_vec = np.full(4 * efd_order, np.nan)
            skel_feats = {k: np.nan for k in skel_keys}
            intervein_feats = {
                f"iv{k}_{suffix}": np.nan
                for k in range(1, max_regions + 1)
                for suffix in ["area_frac", "ecc", "sol"]
            }
        else:
            skel = skeletonize(mask > 0)

            # 几何
            geom = extract_geom_features(mask)

            # EFD
            cont = get_longest_contour(mask)
            efd_vec = extract_efd_features(cont, order=efd_order)

            # skeleton graph
            skel_feats = extract_skeleton_graph_features(mask)

            # intervein
            intervein_feats = extract_intervein_features(
                mask, skel,
                max_regions=max_regions,
                vein_radius=vein_radius
            )

        # ---------- 把所有 feature concat 成一个 list ----------
        feature_list = []

        # 1) geom 按 geom_keys 顺序
        for k in geom_keys:
            feature_list.append(float(geom[k]))

        # 2) EFD（已经是扁平向量）
        feature_list.extend([float(x) for x in efd_vec])

        # 3) skeleton graph 按 skel_keys 顺序
        for k in skel_keys:
            feature_list.append(float(skel_feats[k]))

        # 4) intervein，按 iv1..ivN × [area_frac, ecc, sol] 的顺序
        for region_idx in range(1, max_regions + 1):
            for suffix in ["area_frac", "ecc", "sol"]:
                key = f"iv{region_idx}_{suffix}"
                feature_list.append(float(intervein_feats[key]))

        # ---------- 最终只保留 meta + feature_list ----------
        rec = meta.copy()
        rec["feature"] = feature_list

        out_records.append(rec)

    df = pd.DataFrame(out_records)

    if csv_out is not None:
        df.to_csv(csv_out, index=False)
        if verbose:
            print("[INFO] Saved:", csv_out)
            print("[INFO] Final shape:", df.shape)

    return df



if __name__ == "__main__":

    IMG_DIR  = r"/home/clab/Downloads/jiaxin_temporal/Droso/data/crop_testing"
    CSV_OUT  = r"/home/clab/Downloads/jiaxin_temporal/Droso/data/full_wing_featuresv2test.csv"

    extract_wing_features(
        IMG_DIR,
        csv_out=CSV_OUT,
        efd_order=10,
        downsample=2,
        max_regions=6,
        vein_radius=3,
        verbose=True,
    )
