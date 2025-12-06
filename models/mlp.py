
import numpy as np
import torch
import torch.nn as nn

from skimage.morphology import binary_dilation, remove_small_objects, binary_closing, disk, skeletonize
from skimage.filters import gaussian, threshold_otsu
from skimage.color import rgb2gray
from skimage.measure import label, regionprops, find_contours

from pyefd import elliptic_fourier_descriptors
from skan import Skeleton, summarize

def get_wing_mask_from_array(img_np, downsample: int = 2):
    """
    img_np: 2D numpy array, float or uint8, 单通道
    """
    # 保证是 2D
    if img_np.ndim == 3:
        # 如果不小心传了 [C,H,W] 进来，取第一个 channel
        img_np = img_np[0]

    # 下采样
    if downsample > 1:
        img_np = img_np[::downsample, ::downsample]

    # skimage 的后续操作需要 float
    img_np = img_np.astype(np.float32)

    blur = gaussian(img_np, sigma=1.0)
    th = threshold_otsu(blur)
    bw = blur > th

    bw = remove_small_objects(bw, min_size=200)
    bw = binary_closing(bw, disk(3))

    lab = label(bw)
    props = regionprops(lab)
    if len(props) == 0:
        return None

    region = max(props, key=lambda r: r.area)
    mask = (lab == region.label).astype(np.uint8)
    return mask


def extract_geom_features_from_mask(mask):
    lab = label(mask)
    props = regionprops(lab)
    if len(props) == 0:
        return {
            k: np.nan
            for k in [
                "wing_area",
                "perimeter",
                "major_axis",
                "minor_axis",
                "aspect_ratio",
                "eccentricity",
                "solidity",
                "skeleton_len",
                "skeleton_density",
            ]
        }

    region = max(props, key=lambda r: r.area)

    area = region.area
    perimeter = region.perimeter
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else np.nan
    eccentricity = region.eccentricity
    solidity = region.solidity

    # skeleton
    skel = skeletonize(mask > 0)
    sk_len = np.count_nonzero(skel)
    sk_density = sk_len / area if area > 0 else np.nan

    return {
        "wing_area": float(area),
        "perimeter": float(perimeter),
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "aspect_ratio": float(aspect_ratio),
        "eccentricity": float(eccentricity),
        "solidity": float(solidity),
        "skeleton_len": float(sk_len),
        "skeleton_density": float(sk_density),
    }


def get_longest_contour(mask):
    contours = find_contours(mask, 0.5)
    if len(contours) == 0:
        return None
    cont = max(contours, key=lambda x: x.shape[0])
    # (row,col)->(x,y)
    return np.fliplr(cont)


def extract_efd_features(contour, order: int = 10):
    if contour is None or contour.shape[0] < 20:
        return np.full(4 * order, np.nan, dtype=np.float32)

    coeffs = elliptic_fourier_descriptors(contour, order=order, normalize=True)
    return coeffs.flatten().astype(np.float32)


def extract_skeleton_graph_features_from_mask(mask):
    skel = skeletonize(mask > 0)
    sk = Skeleton(skel)
    df_s = summarize(sk)

    if df_s.shape[0] == 0:
        return {
            "branch_count": 0,
            "endpoint_count": 0,
            "junction_count": 0,
            "branch_total_len": 0.0,
            "branch_mean_len": 0.0,
            "branch_max_len": 0.0,
            "branch_len_std": 0.0,
        }

    # 找距离列
    if "branch-distance" in df_s.columns:
        dist_col = "branch-distance"
    elif "distance" in df_s.columns:
        dist_col = "distance"
    else:
        dist_col = [c for c in df_s.columns if "dist" in c][0]

    lengths = df_s[dist_col].values

    total = float(lengths.sum())
    mean = float(lengths.mean())
    maxl = float(lengths.max())
    std = float(lengths.std())
    count = len(lengths)

    # branch-type: 1=endpoint,3=junction
    if "branch-type" in df_s.columns:
        endpoint = int((df_s["branch-type"] == 1).sum())
        junction = int((df_s["branch-type"] == 3).sum())
    else:
        endpoint = 0
        junction = 0

    return {
        "branch_count": count,
        "endpoint_count": endpoint,
        "junction_count": junction,
        "branch_total_len": total,
        "branch_mean_len": mean,
        "branch_max_len": maxl,
        "branch_len_std": std,
    }


def extract_intervein_features(wing_mask, skel, max_regions: int = 6, vein_radius: int = 3):
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


#   2. feature_keys & dict -> tensor
def build_feature_keys(efd_order: int = 10, max_regions: int = 6):
    keys: list[str] = []

    # 几何特征
    keys.extend(
        [
            "wing_area",
            "perimeter",
            "major_axis",
            "minor_axis",
            "aspect_ratio",
            "eccentricity",
            "solidity",
            "skeleton_len",
            "skeleton_density",
        ]
    )

    # EFD
    for i in range(4 * efd_order):
        keys.append(f"efd_{i}")

    # skeleton graph
    keys.extend(
        [
            "branch_count",
            "endpoint_count",
            "junction_count",
            "branch_total_len",
            "branch_mean_len",
            "branch_max_len",
            "branch_len_std",
        ]
    )

    # intervein
    for k in range(1, max_regions + 1):
        for suffix in ["area_frac", "ecc", "sol"]:
            keys.append(f"iv{k}_{suffix}")

    return keys


def feature_dict_to_tensor(feat_dict: dict, feature_keys: list[str], nan_fill: float = 0.0) -> torch.Tensor:
    """
    按照 feature_keys 的顺序，把 dict 变成 [D] 的 torch.FloatTensor。
    NaN / inf 用 nan_fill（默认 0）填。
    """
    vals = []
    for k in feature_keys:
        v = feat_dict.get(k, np.nan)
        vals.append(v)

    arr = np.array(vals, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=nan_fill, posinf=nan_fill, neginf=nan_fill)
    return torch.from_numpy(arr)  # shape: [D]

#   3. WingFeatureExtractor 类：img_tensor -> feature_tensor
class WingFeatureExtractor:
    """
    用法示例（配合你的 DataLoader）:

        extractor = WingFeatureExtractor()
        for img, label, meta in dataloader:
            # img: [B,1,H,W]
            feat = extractor.extract_batch(img)  # [B, D]

    或者对单张：

        feat_1d = extractor.extract_one(img_single)  # [D]
    """

    def __init__(
        self,
        efd_order: int = 10,
        max_regions: int = 6,
        vein_radius: int = 3,
        downsample: int = 2,
        nan_fill: float = 0.0,
    ):
        self.efd_order = efd_order
        self.max_regions = max_regions
        self.vein_radius = vein_radius
        self.downsample = downsample
        self.nan_fill = nan_fill

        self.feature_keys = build_feature_keys(efd_order=efd_order, max_regions=max_regions)
        self.feature_dim = len(self.feature_keys)

    # --------- 核心：单张 tensor -> feature_dict ---------
    def _compute_feature_dict_from_tensor(self, img_tensor: torch.Tensor) -> dict:
        """
        img_tensor: [1,H,W] 或 [H,W]
        """
        if img_tensor.ndim == 3:  # [C,H,W]
            img_np = img_tensor[0].detach().cpu().numpy()
        else:
            img_np = img_tensor.detach().cpu().numpy()

        mask = get_wing_mask_from_array(img_np, downsample=self.downsample)
        EFD_DIM = 4 * self.efd_order

        if mask is None:
            geom = {
                k: np.nan
                for k in [
                    "wing_area",
                    "perimeter",
                    "major_axis",
                    "minor_axis",
                    "aspect_ratio",
                    "eccentricity",
                    "solidity",
                    "skeleton_len",
                    "skeleton_density",
                ]
            }
            efd_vec = np.full(EFD_DIM, np.nan, dtype=np.float32)
            skel_feats = {
                k: np.nan
                for k in [
                    "branch_count",
                    "endpoint_count",
                    "junction_count",
                    "branch_total_len",
                    "branch_mean_len",
                    "branch_max_len",
                    "branch_len_std",
                ]
            }
            intervein_feats = {
                f"iv{k}_{suffix}": np.nan
                for k in range(1, self.max_regions + 1)
                for suffix in ["area_frac", "ecc", "sol"]
            }
        else:
            skel = skeletonize(mask > 0)
            geom = extract_geom_features_from_mask(mask)
            cont = get_longest_contour(mask)
            efd_vec = extract_efd_features(cont, order=self.efd_order)
            skel_feats = extract_skeleton_graph_features_from_mask(mask)
            intervein_feats = extract_intervein_features(
                mask,
                skel,
                max_regions=self.max_regions,
                vein_radius=self.vein_radius,
            )

        feats = {}
        feats.update(geom)
        feats.update({f"efd_{i}": float(efd_vec[i]) for i in range(len(efd_vec))})
        feats.update(skel_feats)
        feats.update(intervein_feats)

        return feats

    # --------- 对外接口：单张 -> [D] ---------
    def extract_one(self, img_tensor: torch.Tensor) -> torch.Tensor:
        feat_dict = self._compute_feature_dict_from_tensor(img_tensor)
        feat_tensor = feature_dict_to_tensor(
            feat_dict,
            feature_keys=self.feature_keys,
            nan_fill=self.nan_fill,
        )
        return feat_tensor  # [D]

    # --------- 对外接口：batch -> [B,D] ---------
    def extract_batch(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        img_batch: [B,1,H,W] 或 [B,H,W]
        返回: [B, D] 的 feature tensor
        """
        feats_list = []
        for i in range(img_batch.shape[0]):
            feats_list.append(self.extract_one(img_batch[i]))
        return torch.stack(feats_list, dim=0)  # [B,D]

class MLPClassifier(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    # 随机造一张假图测试一下 feature pipeline 是否跑通
    dummy_img = torch.rand(1, 512, 512)  # [1,H,W]

    extractor = WingFeatureExtractor()
    feat = extractor.extract_one(dummy_img)
    print("Feature dim:", feat.shape)

    # 假设有 5 个基因类别
    model = MLPClassifier(in_dim=extractor.feature_dim, num_classes=5)
    logits = model(feat.unsqueeze(0))  # [1,D] -> [1,num_classes]
    print("Logits shape:", logits.shape)
