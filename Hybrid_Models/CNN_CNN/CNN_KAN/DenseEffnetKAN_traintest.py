"""
Hybrid 3D DenseNet121 + EfficientNet-B0 with KAN Head
for Lung Nodule Classification (LUNA16)

Architecture:
    ┌─────────────────────┐    ┌──────────────────────────┐
    │  3D DenseNet121      │    │  3D EfficientNet-B0       │
    │  Feature Extractor  │    │  Feature Extractor        │
    │  → 1024-d           │    │  → 1280-d                 │
    └────────┬────────────┘    └─────────────┬────────────┘
             └──────────────┬─────────────────┘
                     Concat → 2304-d
                            │
                   KAN Classification Head
        KANLinear(2304→512) → LayerNorm → Dropout(0.5)
        → KANLinear(512→128) → LayerNorm → Dropout(0.3)
        → KANLinear(128→1)

KAN Layer (EfficientKAN, Kolmogorov-Arnold Network):
    Each connection is parameterised as:
        φ(x) = scale_base * base_activation(x) * W_base
              + scale_spline * Σ c_i * B_i(x)    [B-spline basis]
    This replaces fixed activation functions with learnable
    univariate spline functions on every weight edge.

Reference:
    Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756
    Blealtan's EfficientKAN (github.com/Blealtan/efficient-kan)
"""

import os
import json
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, confusion_matrix, recall_score)
from sklearn.calibration import calibration_curve
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────── Configuration ───────────────────────────────
SEED = 42
DATA_DIR = r"D:\Projects\Thesis Lung Malignancy Classification ML & DL\LUNA16 Dataset Preprocessed"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 5115.0 / 822.0

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

DENSENET_FEATURE_DIM     = 1024
EFFICIENTNET_FEATURE_DIM = 1280
HYBRID_FEATURE_DIM       = DENSENET_FEATURE_DIM + EFFICIENTNET_FEATURE_DIM  # 2304

# KAN hyperparameters
KAN_GRID_SIZE   = 5     # number of grid intervals for B-spline
KAN_SPLINE_ORDER = 3    # cubic B-splines (order k=3)
KAN_SCALE_NOISE = 0.1   # noise std for spline weight init
KAN_SCALE_BASE  = 1.0
KAN_SCALE_SPLINE = 1.0
KAN_GRID_RANGE  = (-1.0, 1.0)

# ─────────────────────────────── Reproducibility ─────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────── Dataset ─────────────────────────────────────
class NodulePatchDataset(Dataset):
    def __init__(self, metadata_df, data_dir, transforms=None):
        self.metadata   = metadata_df.reset_index(drop=True)
        self.data_dir   = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row       = self.metadata.iloc[idx]
        filename  = os.path.basename(row["filepath"])
        split     = row["split"]
        label     = int(row["label"])
        subfolder = "pos" if label == 1 else "neg"

        local_path = os.path.join(self.data_dir, split, subfolder, filename)
        patch = np.load(local_path).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)

        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)

        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ─────────────────────────────── KAN Layer ───────────────────────────────────

class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network linear layer (EfficientKAN formulation).

    Each scalar input x_i to output y_j is modelled as:
        φ_{i,j}(x_i) = scale_base * SiLU(x_i) * w_base_{i,j}
                       + scale_spline * Σ_k c_{i,j,k} * B_k(x_i)

    where B_k are B-spline basis functions of order `spline_order`
    on a uniform grid of `grid_size` intervals over `grid_range`.

    Parameters
    ----------
    in_features   : int
    out_features  : int
    grid_size     : int   – number of grid intervals
    spline_order  : int   – B-spline order (3 = cubic)
    scale_noise   : float – std for spline weight initialisation
    scale_base    : float – multiplier for the SiLU residual branch
    scale_spline  : float – multiplier for the spline branch
    grid_range    : tuple – (low, high) for the initial grid
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int   = KAN_GRID_SIZE,
        spline_order: int = KAN_SPLINE_ORDER,
        scale_noise: float = KAN_SCALE_NOISE,
        scale_base: float  = KAN_SCALE_BASE,
        scale_spline: float = KAN_SCALE_SPLINE,
        grid_range: tuple  = KAN_GRID_RANGE,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.scale_base   = scale_base
        self.scale_spline = scale_spline

        # Build uniform grid extended by spline_order knots on each side
        # Shape: (grid_size + 2*spline_order + 1,)
        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1,
                             dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", grid)

        # Learnable parameters
        # base_weight: standard linear weights (used with SiLU residual)
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features))

        # spline_weight: coefficients c_{i,j,k}
        # shape: (out_features, in_features, grid_size + spline_order)
        num_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_basis))

        self.base_activation = nn.SiLU()
        self._reset_parameters(scale_noise)

    def _reset_parameters(self, scale_noise: float):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0,
                        std=scale_noise / math.sqrt(self.in_features))

    # ── B-spline basis computation ────────────────────────────────────────
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions for each input feature.

        Parameters
        ----------
        x : (B, in_features)

        Returns
        -------
        bases : (B, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features, \
            f"Expected (B, {self.in_features}), got {x.shape}"

        x = x.unsqueeze(-1)         # (B, in_features, 1)
        grid = self.grid            # (n_knots,)

        # Order-0 indicator basis: 1 if grid[k] <= x < grid[k+1]
        # grid[:-1]: (n_knots-1,)  grid[1:]: (n_knots-1,)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        # bases: (B, in_features, n_knots-1)

        # Cox–de Boor recursion up to order `spline_order`
        for k in range(1, self.spline_order + 1):
            # Left numerator: x - grid[:-k-1]  (drop last k+1 knots)
            left_num  = x - grid[:-(k + 1)]
            left_den  = grid[k:-1]   - grid[:-(k + 1)]

            # Right numerator: grid[k+1:] - x
            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:-k]

            # Safe division (avoid 0/0 when knots coincide)
            left  = torch.where(left_den  != 0,
                                left_num  / left_den,
                                torch.zeros_like(left_num))
            right = torch.where(right_den != 0,
                                right_num / right_den,
                                torch.zeros_like(right_num))

            bases = left * bases[..., :-1] + right * bases[..., 1:]

        # Final shape: (B, in_features, grid_size + spline_order)
        assert bases.size(-1) == self.grid_size + self.spline_order
        return bases

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_features)
        → (B, out_features)
        """
        # Residual (base) branch: SiLU + linear
        base_out = F.linear(
            self.base_activation(x),
            self.base_weight * self.scale_base
        )   # (B, out_features)

        # Spline branch: sum over basis functions
        spline_basis = self.b_splines(x)
        # spline_basis : (B, in_features, num_basis)
        # spline_weight: (out_features, in_features, num_basis)
        spline_out = torch.einsum(
            'big,oig->bo',
            spline_basis,
            self.spline_weight * self.scale_spline
        )   # (B, out_features)

        return base_out + spline_out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"grid={self.grid_size}, order={self.spline_order}")


# ─────────────── 3D EfficientNet-B0 Implementation ───────────────────────────

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation3D(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(in_channels, se_channels), SwishActivation(),
            nn.Linear(se_channels, in_channels), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1, 1)


class MBConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_skip          = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        mid_channels = _make_divisible(in_channels * expand_ratio, 8)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv3d(in_channels, mid_channels, 1, bias=False),
                       nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3),
                       SwishActivation()]
        pad = (kernel_size - 1) // 2
        layers += [nn.Conv3d(mid_channels, mid_channels, kernel_size,
                             stride=stride, padding=pad,
                             groups=mid_channels, bias=False),
                   nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3),
                   SwishActivation()]
        layers.append(SqueezeExcitation3D(mid_channels, se_ratio))
        layers += [nn.Conv3d(mid_channels, out_channels, 1, bias=False),
                   nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-3)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            if self.training and self.drop_connect_rate > 0:
                keep_prob   = 1 - self.drop_connect_rate
                rand_tensor = torch.rand(x.size(0), 1, 1, 1, 1, device=x.device)
                rand_tensor = torch.floor(rand_tensor + keep_prob)
                out = out / keep_prob * rand_tensor
            out = out + x
        return out


class EfficientNet3D_B0(nn.Module):
    """3D EfficientNet-B0 backbone → 1280-d feature vector."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=0.01, eps=1e-3), SwishActivation()
        )
        stage_configs = [
            (32,  16,  3, 1, 1, 1), (16,  24,  3, 2, 6, 2),
            (24,  40,  5, 2, 6, 2), (40,  80,  3, 2, 6, 3),
            (80,  112, 5, 1, 6, 3), (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1),
        ]
        total_blocks = sum(cfg[5] for cfg in stage_configs)
        block_idx = 0
        stages = []
        for (in_ch, out_ch, k, s, expand, n_layers) in stage_configs:
            stage = []
            for i in range(n_layers):
                stride    = s if i == 0 else 1
                inch      = in_ch if i == 0 else out_ch
                drop_rate = 0.2 * block_idx / total_blocks
                stage.append(MBConv3D(inch, out_ch, k, stride, expand,
                                      drop_connect_rate=drop_rate))
                block_idx += 1
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)
        self.head_conv = nn.Sequential(
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, 1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.01, eps=1e-3),
            SwishActivation()
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        return self.global_pool(x).flatten(1)   # (B, 1280)


# ─────────────── DenseNet121 Feature Extractor ───────────────────────────────

class DenseNet121FeatureExtractor(nn.Module):
    """
    MONAI DenseNet121 with classification head removed.
    Returns 1024-d feature vector.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        _backbone    = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1)
        self.features = _backbone.features   # 1024 output channels

    def forward(self, x):
        x = self.features(x)                          # (B, 1024, d, h, w)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool3d(x, output_size=1)   # (B, 1024, 1, 1, 1)
        return x.flatten(1)                           # (B, 1024)


# ─────────────── Hybrid Model with KAN Head ──────────────────────────────────

class HybridKANModel(nn.Module):
    """
    Dual-backbone hybrid model:
        DenseNet121 (1024-d) + EfficientNet-B0 (1280-d)
        → Concat (2304-d)
        → KAN Head → 1 logit

    KAN Head:
        KANLinear(2304→512) → LayerNorm → Dropout(0.5)
        → KANLinear(512→128) → LayerNorm → Dropout(0.3)
        → KANLinear(128→1)

    LayerNorm is applied after each KAN layer (except the last) for
    training stability; the spline grid is defined on [-1, 1], so
    LayerNorm-normalised inputs stay well within the active grid range.
    """
    def __init__(self, in_channels=1, dropout_1=0.5, dropout_2=0.3):
        super().__init__()
        self.densenet_extractor     = DenseNet121FeatureExtractor(in_channels)
        self.efficientnet_extractor = EfficientNet3D_B0(in_channels)

        # KAN classification head
        self.kan_layer1 = KANLinear(HYBRID_FEATURE_DIM, 512)
        self.norm1      = nn.LayerNorm(512)
        self.drop1      = nn.Dropout(dropout_1)

        self.kan_layer2 = KANLinear(512, 128)
        self.norm2      = nn.LayerNorm(128)
        self.drop2      = nn.Dropout(dropout_2)

        self.kan_out    = KANLinear(128, 1)

    def forward(self, x):
        f_dense  = self.densenet_extractor(x)           # (B, 1024)
        f_eff    = self.efficientnet_extractor(x)       # (B, 1280)
        f        = torch.cat([f_dense, f_eff], dim=1)  # (B, 2304)

        f = self.drop1(self.norm1(self.kan_layer1(f)))  # (B, 512)
        f = self.drop2(self.norm2(self.kan_layer2(f)))  # (B, 128)
        out = self.kan_out(f)                           # (B, 1)
        return out.squeeze(1)                           # (B,)


# ─────────────── Evaluation Metrics ──────────────────────────────────────────

def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    total_positives    = (all_labels == 1).sum()
    descending_indices = np.argsort(all_probs)[::-1]
    sorted_labels      = all_labels[descending_indices]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans
    sensitivity  = tps / total_positives
    froc_scores  = {}
    for target in FROC_THRESHOLDS:
        valid_idx = np.where(fps_per_scan <= target)[0]
        froc_scores[f"{target} FP/scan"] = (
            sensitivity[valid_idx[-1]] if len(valid_idx) > 0 else 0.0)
    return froc_scores


def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
    bootstrapped_scores = []
    rng = np.random.RandomState(SEED)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        bootstrapped_scores.append(
            roc_auc_score(np.array(y_true)[indices],
                          np.array(y_probs)[indices]))
    sorted_scores = np.sort(np.array(bootstrapped_scores))
    return np.percentile(sorted_scores, 2.5), np.percentile(sorted_scores, 97.5)


def evaluate_model(loader, model, device, desc="Validating", return_preds=False):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion    = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(device))

    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc=desc):
            patches, labels = patches.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(patches)
                loss   = criterion(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs, labels = [float(probs)], [float(labels.cpu())]
            else:
                probs, labels = probs.tolist(), labels.cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_uids.extend(uids)
            running_loss += loss.item() * patches.size(0)

    # Patient-level aggregation
    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict[uid]['prob']  = max(patient_dict[uid]['prob'],  prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)

    y_true_patient  = [v['label'] for v in patient_dict.values()]
    y_probs_patient = [v['prob']  for v in patient_dict.values()]
    y_pred_patient  = [1 if p >= 0.5 else 0 for p in y_probs_patient]
    total_scans     = len(patient_dict)

    avg_loss = running_loss / len(loader.dataset)
    auc  = roc_auc_score(y_true_patient, y_probs_patient) \
           if len(np.unique(y_true_patient)) > 1 else 0.5
    auprc = average_precision_score(y_true_patient, y_probs_patient)
    f1    = f1_score(y_true_patient, y_pred_patient)
    sens  = recall_score(y_true_patient, y_pred_patient)

    cm = confusion_matrix(y_true_patient, y_pred_patient, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    froc = calculate_candidate_froc(all_labels, all_probs, total_scans)

    if return_preds:
        ci_lower, ci_upper = calculate_95_ci(y_true_patient, y_probs_patient)
        print(f"AUROC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        frac_pos, mean_pred = calibration_curve(
            y_true_patient, y_probs_patient, n_bins=10)
        pd.DataFrame({'mean_predicted_probability': mean_pred,
                      'fraction_of_positives': frac_pos}
                    ).to_csv("hybrid_kan_calibration_curve.csv", index=False)
        print("[Saved] Calibration data → 'hybrid_kan_calibration_curve.csv'")

        pred_df = pd.DataFrame({'seriesuid':   all_uids,
                                'label':       all_labels,
                                'probability': all_probs})
        return avg_loss, auc, auprc, f1, sens, spec, froc, pred_df

    return avg_loss, auc, auprc, f1, sens, spec, froc


# ─────────────── Main ────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load metadata ──────────────────────────────────────────────────────
    print("Loading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict    = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)

    # ── Transforms ────────────────────────────────────────────────────────
    train_transforms = Compose([
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandGaussianNoise(prob=0.2, std=0.01)
    ])

    train_dataset = NodulePatchDataset(train_meta, DATA_DIR, transforms=train_transforms)
    val_dataset   = NodulePatchDataset(val_meta,   DATA_DIR, transforms=None)
    test_dataset  = NodulePatchDataset(test_meta,  DATA_DIR, transforms=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)

    # ── Model ─────────────────────────────────────────────────────────────
    model = HybridKANModel().to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    densenet_params  = sum(p.numel() for p in model.densenet_extractor.parameters())
    effnet_params    = sum(p.numel() for p in model.efficientnet_extractor.parameters())
    kan_params       = (sum(p.numel() for p in model.kan_layer1.parameters()) +
                        sum(p.numel() for p in model.kan_layer2.parameters()) +
                        sum(p.numel() for p in model.kan_out.parameters()))

    print("\n--- Hybrid KAN Model Summary ---")
    print(f"  Backbone 1  : 3D DenseNet121         → {DENSENET_FEATURE_DIM}-d  ({densenet_params:,} params)")
    print(f"  Backbone 2  : 3D EfficientNet-B0     → {EFFICIENTNET_FEATURE_DIM}-d ({effnet_params:,} params)")
    print(f"  Fused Dim   : {HYBRID_FEATURE_DIM}-d")
    print(f"  Head        : KAN (grid={KAN_GRID_SIZE}, order={KAN_SPLINE_ORDER})  ({kan_params:,} params)")
    print(f"  Total       : {total_params:,} params")
    print(f"  Trainable   : {trainable_params:,} params")
    print("─" * 42)

    # ── Training setup ────────────────────────────────────────────────────
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs    = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler        = SequentialLR(optimizer,
                                    schedulers=[warmup_scheduler, cosine_scheduler],
                                    milestones=[warmup_epochs])
    scaler = GradScaler('cuda')

    best_val_auc     = 0.0
    patience_counter = 0
    history          = []

    # ── Training loop ─────────────────────────────────────────────────────
    print("\n--- Starting Training ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for patches, labels, _ in progress_bar:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(patches)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * patches.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader.dataset)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc = \
            evaluate_model(val_loader, model, device, desc="Validating")

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": avg_train_loss, "val_loss": val_loss,
            "val_auc": val_auc, "val_auprc": val_auprc,
            "val_f1": val_f1, "val_sensitivity": val_sens,
            "val_specificity": val_spec
        })

        print(f"\nEpoch {epoch:3d} | LR: {current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val AUROC: {val_auc:.4f} | AUPRC: {val_auprc:.4f} | "
              f"F1: {val_f1:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f}")

        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_hybrid_kan.pth")
            print(f"  ──▶ [Saved] New Best Val AUROC: {best_val_auc:.4f}")
            with open("best_metrics_hybrid_kan.json", "w") as f:
                json.dump({"epoch": epoch, "auc": val_auc, "auprc": val_auprc,
                           "f1": val_f1, "sensitivity": val_sens,
                           "specificity": val_spec}, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping after {epoch} epochs ***")
                break

    pd.DataFrame(history).to_csv("hybrid_kan_training_log.csv", index=False)
    print("\n[Saved] Training history → 'hybrid_kan_training_log.csv'")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("=" * 55)
    model.load_state_dict(torch.load("best_model_hybrid_kan.pth",
                                     map_location=device, weights_only=True))

    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, \
        test_froc, pred_df = evaluate_model(
            test_loader, model, device, desc="Testing", return_preds=True)

    pred_df.to_csv("hybrid_kan_test_predictions.csv", index=False)
    print("[Saved] Test predictions → 'hybrid_kan_test_predictions.csv'")

    print("\n--- FINAL TEST RESULTS (PATIENT-LEVEL) ---")
    print(f"  AUROC (Primary):        {test_auc:.4f}")
    print(f"  AUPRC:                  {test_auprc:.4f}")
    print(f"  F1 Score:               {test_f1:.4f}")
    print(f"  Sensitivity (Recall):   {test_sens:.4f}")
    print(f"  Specificity:            {test_spec:.4f}")

    print(f"\n--- FROC (Candidate-level) ---")
    for fp_rate, sensitivity in test_froc.items():
        print(f"  {fp_rate:12s} : {sensitivity:.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()