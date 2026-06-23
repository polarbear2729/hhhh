"""
3D Hybrid CNN with KAN Head for Lung Nodule Classification
===========================================================
Combines three parallel 3D CNN feature extractors whose outputs are
concatenated and fused by a KAN (Kolmogorov-Arnold Network) classification head.

Feature Extractors & Dimensions:
    ┌─────────────────────────────────────────────┐
    │  Branch          │ Backbone      │ Feature-d │
    ├─────────────────────────────────────────────┤
    │  Branch 1        │ ResNet18      │   512     │
    │  Branch 2        │ DenseNet121   │  1024     │
    │  Branch 3        │ EfficientNet  │  1280     │
    │                  │   -B0 (3D)    │           │
    ├─────────────────────────────────────────────┤
    │  Concatenated    │               │  2816     │
    └─────────────────────────────────────────────┘

KAN Fusion Head:
    KANLinear(2816→512) → KANLinear(512→256) → KANLinear(256→1)

    Each KANLinear layer uses B-spline basis functions (grid_size=5, spline_order=3)
    with residual SiLU activations, replacing all learned affine MLP transformations
    with learnable univariate spline functions on each input-output pair.

Output files produced:
    best_model_hybrid.pth         — best checkpoint (by val AUROC)
    best_metrics_hybrid.json      — metrics at best epoch
    hybrid_training_log.csv       — epoch-by-epoch history
    hybrid_test_predictions.csv   — candidate-level test predictions
    hybrid_calibration_curve.csv  — calibration data for reliability plot

KAN Reference:
    Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756 (2024).
    Implementation follows efficient-kan: github.com/Blealtan/efficient-kan
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, recall_score,
)
from sklearn.calibration import calibration_curve
from monai.networks.nets import DenseNet121, ResNet
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise, RandAffine,
)
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
SEED = 42
DATA_DIR           = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH      = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE         = 8
NUM_WORKERS        = 4
PIN_MEMORY         = True
PERSISTENT_WORKERS = True

MAX_EPOCHS               = 100
EARLY_STOPPING_PATIENCE  = 15
LR                       = 1e-4
WEIGHT_DECAY             = 1e-4
POS_WEIGHT               = 5115.0 / 822.0   # neg:pos ratio for BCEWithLogitsLoss

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Feature dimensions — fixed by each backbone's architecture; do NOT modify.
RESNET18_FEATURE_DIM     = 512
DENSENET121_FEATURE_DIM  = 1024
EFFICIENTNET_FEATURE_DIM = 1280
TOTAL_FEATURE_DIM        = (RESNET18_FEATURE_DIM +
                             DENSENET121_FEATURE_DIM +
                             EFFICIENTNET_FEATURE_DIM)  # 2816

# KAN hyperparameters
KAN_GRID_SIZE    = 5     # number of B-spline grid intervals
KAN_SPLINE_ORDER = 3     # cubic B-splines (degree)
KAN_SCALE_NOISE  = 0.1   # std of noise used to initialise spline weights
KAN_SCALE_BASE   = 1.0   # scale for residual SiLU base weight init
KAN_SCALE_SPLINE = 1.0   # scale for spline weight init
KAN_GRID_EPS     = 0.02  # fraction of range added as padding when updating grid
KAN_GRID_RANGE   = [-1.0, 1.0]  # initial uniform grid range


# ═══════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═══════════════════════════════════════════════════════════════════════
#  KAN Layer
# ═══════════════════════════════════════════════════════════════════════
class KANLinear(nn.Module):
    """
    One layer of a Kolmogorov-Arnold Network (efficient-kan implementation).

    Each scalar input x_i is mapped to each scalar output y_j by a
    learnable univariate function φ_{ij}:

        y_j = Σ_i  φ_{ij}(x_i)
        φ_{ij}(x) = w_b · silu(x)  +  w_s · spline(x)

    where spline(x) is a sum of B-spline basis functions evaluated on a
    uniform grid of `grid_size` intervals with spline degree `spline_order`.

    The grid can be adaptively updated after each forward pass via
    `update_grid(x)` to better cover the actual input distribution.

    Args:
        in_features   : input dimension
        out_features  : output dimension
        grid_size     : number of B-spline grid intervals
        spline_order  : degree of B-spline basis (3 = cubic)
        scale_noise   : std of noise for initial spline weight perturbation
        scale_base    : init scale for residual SiLU weight
        scale_spline  : init scale for spline weight
        enable_standalone_scale_spline: if True, learns a per-output scale
                         applied to the spline output (recommended)
        base_activation: activation for residual path (default: SiLU)
        grid_eps      : fraction of data range added as grid padding
        grid_range    : initial [min, max] for the uniform grid
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        grid_size:    int   = KAN_GRID_SIZE,
        spline_order: int   = KAN_SPLINE_ORDER,
        scale_noise:  float = KAN_SCALE_NOISE,
        scale_base:   float = KAN_SCALE_BASE,
        scale_spline: float = KAN_SCALE_SPLINE,
        enable_standalone_scale_spline: bool = True,
        base_activation = nn.SiLU,
        grid_eps:     float = KAN_GRID_EPS,
        grid_range:   list  = KAN_GRID_RANGE,
    ):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.grid_size     = grid_size
        self.spline_order  = spline_order
        self.scale_noise   = scale_noise
        self.scale_base    = scale_base
        self.scale_spline  = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps      = grid_eps

        # Build the initial uniform grid: shape (in_features, grid_size + 2*spline_order + 1)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
            * h + grid_range[0]
        )  # (grid_size + 2*spline_order + 1,)
        grid = grid.unsqueeze(0).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Learnable weights
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.empty(out_features, in_features)
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                self.scale_spline *
                self._curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order],  # (grid_size+1, in)
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.constant_(self.spline_scaler, self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values for input x.

        Args:
            x : (batch, in_features)
        Returns:
            bases : (batch, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid  # (in_features, grid_size + 2*spline_order + 1)

        x = x.unsqueeze(-1)  # (B, in, 1)
        # Zeroth-order basis: 1 inside each interval, 0 outside
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Cox-de-Boor recursion
        for k in range(1, self.spline_order + 1):
            left_denom  = grid[:, k:-1]      - grid[:, :-(k + 1)]
            right_denom = grid[:, k + 1:]    - grid[:, 1:(-k)]

            left_denom  = torch.where(left_denom  == 0, torch.ones_like(left_denom),  left_denom)
            right_denom = torch.where(right_denom == 0, torch.ones_like(right_denom), right_denom)

            left  = (x - grid[:, :-(k + 1)]) / left_denom  * bases[:, :, :-1]
            right = (grid[:, k + 1:] - x)    / right_denom * bases[:, :,  1:]
            bases = left + right

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute spline coefficients that interpolate (x, y).

        Args:
            x : (grid_size+1, in_features)  — evaluation points (knots)
            y : (grid_size+1, in_features, out_features)
        Returns:
            coeffs : (out_features, in_features, grid_size + spline_order)
        """
        # x: (n_pts, in) → batch of 1-D points per input
        A = self.b_splines(x.T).transpose(0, 1)  # (in, n_pts, grid+order)
        B = y.transpose(0, 1)                     # (in, n_pts, out)
        # Solve A @ coeffs = B  (least-squares per input feature)
        solution = torch.linalg.lstsq(A, B).solution  # (in, grid+order, out)
        result   = solution.permute(2, 0, 1)           # (out, in, grid+order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, in_features)
        Returns:
            y : (batch, out_features)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Residual (SiLU) path
        base_out = F.linear(self.base_activation(x), self.base_weight)

        # Spline path: (B, in, grid+order) → (B, in*(grid+order)) → (B, out)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Adaptively update the B-spline grid to cover the distribution of x.
        Call once per batch (or periodically) during training for best results.

        Args:
            x      : (batch, in_features)  — activations from the previous layer
            margin : extra fractional margin added on each side of the data range
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # Current spline output on old grid (used to refit coefficients)
        splines = self.b_splines(x)         # (B, in, grid+order)
        splines = splines.permute(1, 0, 2)  # (in, B, grid+order)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)  # (in, grid+order, out)
        unscaled   = torch.bmm(splines, orig_coeff)              # (in, B, out)

        # Build new uniform grid from actual data quantiles
        x_sorted = x.sort(dim=0).values.T  # (in, B)
        adaptive_grid = x_sorted[
            :,
            [int(batch * i / self.grid_size) for i in range(self.grid_size)]
            + [-1],
        ]  # (in, grid_size + 1)

        uniform_step = (x_sorted[:, -1] - x_sorted[:, 0]) / self.grid_size
        uniform_grid = (
            torch.arange(self.grid_size + 1, device=x.device).unsqueeze(0)
            * uniform_step.unsqueeze(1)
            + x_sorted[:, :1]
        )  # (in, grid_size + 1)

        grid = self.grid_eps * uniform_grid + (1 - self.grid_eps) * adaptive_grid

        # Extend with repeated boundary knots for the B-spline
        pre  = grid[:, :1].expand(-1, self.spline_order)
        post = grid[:, -1:].expand(-1, self.spline_order)
        grid = torch.cat([pre, grid, post], dim=1)  # (in, total_knots)
        self.grid.copy_(grid)

        # Refit spline coefficients on the new grid
        x_eval    = x.T[..., None].expand(-1, -1, self.out_features)  # (in, B, out)
        self.spline_weight.data.copy_(
            self._curve2coeff(
                x.T,          # (in, B) → used as (n_pts, in) inside _curve2coeff
                unscaled.permute(1, 0, 2),  # (B, in, out)
            )
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  KAN Head
# ═══════════════════════════════════════════════════════════════════════
class KANHead(nn.Module):
    """
    KAN fusion head replacing the original MLP head.

    Architecture:
        KANLinear(2816 → 512)
        KANLinear(512  → 256)
        KANLinear(256  →   1)

    Each KANLinear layer parameterises its activation functions as
    learnable B-splines (grid_size=5, cubic, per-connection), plus a
    residual SiLU path. No explicit Dropout or LayerNorm is used —
    the spline functions act as implicit regularisers; add Dropout(0.3)
    wrappers below if overfitting is observed on small datasets.

    Args:
        in_features  : input dimension (default: TOTAL_FEATURE_DIM = 2816)
        hidden1      : first hidden dimension (default: 512)
        hidden2      : second hidden dimension (default: 256)
        out_features : output logit dimension (default: 1)
        grid_size    : B-spline grid intervals (default: KAN_GRID_SIZE = 5)
        spline_order : B-spline degree (default: KAN_SPLINE_ORDER = 3)
    """

    def __init__(
        self,
        in_features:  int = TOTAL_FEATURE_DIM,
        hidden1:      int = 512,
        hidden2:      int = 256,
        out_features: int = 1,
        grid_size:    int = KAN_GRID_SIZE,
        spline_order: int = KAN_SPLINE_ORDER,
    ):
        super().__init__()
        self.layer1 = KANLinear(in_features, hidden1,
                                grid_size=grid_size, spline_order=spline_order)
        self.layer2 = KANLinear(hidden1, hidden2,
                                grid_size=grid_size, spline_order=spline_order)
        self.layer3 = KANLinear(hidden2, out_features,
                                grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def update_grids(self, x: torch.Tensor):
        """
        Update the B-spline grids of all layers given an input batch x.
        Call during training (e.g. after each epoch or every N steps)
        to keep the grids aligned with the actual feature distributions.
        """
        x = self.layer1(x)    # pass through to get layer-2 input distribution
        self.layer2.update_grid(x)
        x = self.layer2(x)
        self.layer3.update_grid(x)


# ═══════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════
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
        patch = np.expand_dims(patch, axis=0)       # (1, D, H, W)

        if self.transforms is not None:
            patch = self.transforms(patch)

        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)

        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]


# ═══════════════════════════════════════════════════════════════════════
#  3D EfficientNet-B0
# ═══════════════════════════════════════════════════════════════════════

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
        se_ch = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, se_ch),
            SwishActivation(),
            nn.Linear(se_ch, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1, 1)
        return x * scale


class MBConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_skip          = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate

        mid_ch = _make_divisible(in_channels * expand_ratio, 8)
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv3d(in_channels, mid_ch, 1, bias=False),
                nn.BatchNorm3d(mid_ch, momentum=0.01, eps=1e-3),
                SwishActivation(),
            ]
        pad = (kernel_size - 1) // 2
        layers += [
            nn.Conv3d(mid_ch, mid_ch, kernel_size,
                      stride=stride, padding=pad, groups=mid_ch, bias=False),
            nn.BatchNorm3d(mid_ch, momentum=0.01, eps=1e-3),
            SwishActivation(),
            SqueezeExcitation3D(mid_ch, se_ratio),
            nn.Conv3d(mid_ch, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-3),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            if self.training and self.drop_connect_rate > 0:
                keep_prob   = 1 - self.drop_connect_rate
                rand_tensor = torch.rand(x.size(0), 1, 1, 1, 1, device=x.device)
                rand_tensor = torch.floor(rand_tensor + keep_prob)
                out         = out / keep_prob * rand_tensor
            out = out + x
        return out


class EfficientNet3D_B0(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=0.01, eps=1e-3),
            SwishActivation(),
        )
        stage_configs = [
            (32,  16,  3, 1, 1, 1),
            (16,  24,  3, 2, 6, 2),
            (24,  40,  5, 2, 6, 2),
            (40,  80,  3, 2, 6, 3),
            (80,  112, 5, 1, 6, 3),
            (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1),
        ]
        total_blocks = sum(cfg[5] for cfg in stage_configs)
        block_idx    = 0
        stages       = []
        for in_ch, out_ch, k, s, expand, n_layers in stage_configs:
            stage = []
            for i in range(n_layers):
                stride    = s    if i == 0 else 1
                inch      = in_ch if i == 0 else out_ch
                drop_rate = 0.2 * block_idx / total_blocks
                stage.append(MBConv3D(inch, out_ch, k, stride, expand,
                                      drop_connect_rate=drop_rate))
                block_idx += 1
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)
        self.head_conv = nn.Sequential(
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, kernel_size=1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.01, eps=1e-3),
            SwishActivation(),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.global_pool(x)
        return x.flatten(1)


# ═══════════════════════════════════════════════════════════════════════
#  Hybrid Model  (KAN head)
# ═══════════════════════════════════════════════════════════════════════
class HybridCNNWithKANHead(nn.Module):
    """
    3D Hybrid CNN: three parallel feature extractors + KAN fusion head.

                Input  (B, 1, D, H, W)
                   │
          ┌────────┼─────────────┐
          ▼        ▼             ▼
      ResNet18  DenseNet121  EfficientNet-B0
      [512-d]   [1024-d]      [1280-d]
          │        │             │
          └────────┴─────────────┘
                   │  torch.cat  →  (B, 2816)
                   ▼
              KAN Fusion Head
           ─────────────────────
           KANLinear(2816 → 512)   ← learnable B-spline activations
           KANLinear(512  → 256)     per input-output pair + SiLU residual
           KANLinear(256  →   1)
                   │
               Logit (B,)   ──►  BCEWithLogitsLoss
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1,
                 grid_size: int = KAN_GRID_SIZE,
                 spline_order: int = KAN_SPLINE_ORDER):
        super().__init__()

        # ── Branch 1: ResNet18 → 512-d ──────────────────────────────────
        self.resnet18 = ResNet(
            block            = ResNetBlock,
            layers           = [2, 2, 2, 2],
            block_inplanes   = [64, 128, 256, 512],
            spatial_dims     = 3,
            n_input_channels = in_channels,
            num_classes      = RESNET18_FEATURE_DIM,
        )
        assert hasattr(self.resnet18, 'fc'), \
            "MONAI ResNet structure changed: 'fc' attribute missing."
        self.resnet18.fc = nn.Identity()

        # ── Branch 2: DenseNet121 → 1024-d ──────────────────────────────
        self.densenet121 = DenseNet121(
            spatial_dims = 3,
            in_channels  = in_channels,
            out_channels = 1,
        )
        assert hasattr(self.densenet121, 'class_layers'), \
            "MONAI DenseNet structure changed: 'class_layers' missing."
        assert hasattr(self.densenet121.class_layers, 'out'), \
            "MONAI DenseNet structure changed: 'out' missing in class_layers."
        self.densenet121.class_layers.out = nn.Identity()

        # ── Branch 3: EfficientNet-B0 → 1280-d ──────────────────────────
        self.efficientnet_b0 = EfficientNet3D_B0(in_channels=in_channels)

        # ── KAN Fusion Head: 2816-d → 1 ─────────────────────────────────
        self.kan_head = KANHead(
            in_features  = TOTAL_FEATURE_DIM,
            hidden1      = 512,
            hidden2      = 256,
            out_features = num_classes,
            grid_size    = grid_size,
            spline_order = spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_r = self.resnet18(x)           # (B, 512)
        f_d = self.densenet121(x)        # (B, 1024)
        f_e = self.efficientnet_b0(x)    # (B, 1280)

        combined = torch.cat([f_r, f_d, f_e], dim=1)  # (B, 2816)
        out      = self.kan_head(combined)             # (B, 1)
        return out.squeeze(1)                          # (B,)

    def update_kan_grids(self, x: torch.Tensor):
        """
        Pass a batch through the backbone and update KAN grids on the
        concatenated feature vector. Recommended: call once per epoch
        at the start of training (before the gradient update).

        Usage in training loop:
            with torch.no_grad():
                model.update_kan_grids(first_batch_patches)
        """
        with torch.no_grad():
            f_r = self.resnet18(x)
            f_d = self.densenet121(x)
            f_e = self.efficientnet_b0(x)
            combined = torch.cat([f_r, f_d, f_e], dim=1)
            # Update layer 1 grid directly, then propagate
            self.kan_head.layer1.update_grid(combined)
            h1 = self.kan_head.layer1(combined)
            self.kan_head.layer2.update_grid(h1)
            h2 = self.kan_head.layer2(h1)
            self.kan_head.layer3.update_grid(h2)


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation Utilities
# ═══════════════════════════════════════════════════════════════════════

def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    total_positives = (all_labels == 1).sum()
    desc_idx        = np.argsort(all_probs)[::-1]
    sorted_labels   = all_labels[desc_idx]

    tps          = np.cumsum(sorted_labels == 1)
    fps          = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans
    sensitivity  = tps / total_positives

    froc_scores = {}
    for target in FROC_THRESHOLDS:
        valid_idx = np.where(fps_per_scan <= target)[0]
        sens = sensitivity[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
        froc_scores[f"{target} FP/scan"] = sens

    return froc_scores


def calculate_95_ci(y_true, y_probs, n_bootstraps: int = 1000):
    rng    = np.random.RandomState(SEED)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[idx])) < 2:
            continue
        scores.append(
            roc_auc_score(np.array(y_true)[idx], np.array(y_probs)[idx])
        )
    scores = np.sort(np.array(scores))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def evaluate_model(loader, model, device,
                   desc: str = "Validating",
                   return_preds: bool = False):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(device)
    )

    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc=desc):
            patches, labels = patches.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(patches)
                loss   = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs  = [float(probs)]
                labels = [float(labels.cpu())]
            else:
                probs  = probs.tolist()
                labels = labels.cpu().tolist()

            all_probs.extend(probs)
            all_labels.extend(labels)
            all_uids.extend(uids)
            running_loss += loss.item() * patches.size(0)

    # Scan-level aggregation
    patient_dict: dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict[uid]['prob']  = max(patient_dict[uid]['prob'],  prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)

    y_true_pat  = [v['label'] for v in patient_dict.values()]
    y_probs_pat = [v['prob']  for v in patient_dict.values()]
    y_pred_pat  = [1 if p >= 0.5 else 0 for p in y_probs_pat]
    total_scans = len(patient_dict)

    avg_loss = running_loss / len(loader.dataset)

    auc  = roc_auc_score(y_true_pat, y_probs_pat) \
           if len(np.unique(y_true_pat)) > 1 else 0.5
    auprc = average_precision_score(y_true_pat, y_probs_pat)
    f1    = f1_score(y_true_pat, y_pred_pat)
    sens  = recall_score(y_true_pat, y_pred_pat)

    cm = confusion_matrix(y_true_pat, y_pred_pat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    froc = calculate_candidate_froc(all_labels, all_probs, total_scans)

    if return_preds:
        ci_lo, ci_hi = calculate_95_ci(y_true_pat, y_probs_pat)
        print(f"  AUROC 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

        frac_pos, mean_pred = calibration_curve(
            y_true_pat, y_probs_pat, n_bins=10
        )
        pd.DataFrame({
            'mean_predicted_probability': mean_pred,
            'fraction_of_positives':      frac_pos,
        }).to_csv("hybrid_calibration_curve.csv", index=False)
        print("  [Saved] hybrid_calibration_curve.csv")

        pred_df = pd.DataFrame({
            'seriesuid':   all_uids,
            'label':       all_labels,
            'probability': all_probs,
        })
        return avg_loss, auc, auprc, f1, sens, spec, froc, pred_df

    return avg_loss, auc, auprc, f1, sens, spec, froc


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df   = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)

    print(f"  Train: {len(train_meta):,}  |  Val: {len(val_meta):,}"
          f"  |  Test: {len(test_meta):,}")

    train_transforms = Compose([
        RandAffine(
            prob            = 0.8,
            translate_range = (5, 5, 5),
            padding_mode    = 'zeros',
            spatial_size    = None,
        ),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandGaussianNoise(prob=0.2, std=0.01),
    ])

    train_ds = NodulePatchDataset(train_meta, DATA_DIR, transforms=train_transforms)
    val_ds   = NodulePatchDataset(val_meta,   DATA_DIR, transforms=None)
    test_ds  = NodulePatchDataset(test_meta,  DATA_DIR, transforms=None)

    loader_kwargs = dict(
        batch_size         = BATCH_SIZE,
        num_workers        = NUM_WORKERS,
        pin_memory         = PIN_MEMORY,
        persistent_workers = PERSISTENT_WORKERS,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False,                   **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False,                   **loader_kwargs)

    model = HybridCNNWithKANHead().to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rn18_p      = sum(p.numel() for p in model.resnet18.parameters())
    dn121_p     = sum(p.numel() for p in model.densenet121.parameters())
    enb0_p      = sum(p.numel() for p in model.efficientnet_b0.parameters())
    kan_p       = sum(p.numel() for p in model.kan_head.parameters())

    # KAN head parameter breakdown
    l1_p = sum(p.numel() for p in model.kan_head.layer1.parameters())
    l2_p = sum(p.numel() for p in model.kan_head.layer2.parameters())
    l3_p = sum(p.numel() for p in model.kan_head.layer3.parameters())

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         Hybrid 3D CNN + KAN — Model Summary         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Branch 1  ResNet18        {rn18_p:>12,} params  → {RESNET18_FEATURE_DIM}-d  ║")
    print(f"║  Branch 2  DenseNet121     {dn121_p:>12,} params  → {DENSENET121_FEATURE_DIM}-d ║")
    print(f"║  Branch 3  EfficientNet-B0 {enb0_p:>12,} params  → {EFFICIENTNET_FEATURE_DIM}-d ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  KAN Head  (2816→512→256→1){kan_p:>12,} params         ║")
    print(f"║    Layer 1  KANLinear(2816→512) {l1_p:>10,} params     ║")
    print(f"║    Layer 2  KANLinear(512→256)  {l2_p:>10,} params     ║")
    print(f"║    Layer 3  KANLinear(256→1)    {l3_p:>10,} params     ║")
    print(f"║    Grid size: {KAN_GRID_SIZE}   Spline order: {KAN_SPLINE_ORDER}                   ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Concat feature dim        {TOTAL_FEATURE_DIM:>12,}               ║")
    print(f"║  Total params              {total_p:>12,}               ║")
    print(f"║  Trainable params          {trainable_p:>12,}               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs    = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler        = SequentialLR(
        optimizer,
        schedulers = [warmup_scheduler, cosine_scheduler],
        milestones = [warmup_epochs],
    )
    scaler = GradScaler('cuda')

    best_val_auc     = 0.0
    patience_counter = 0
    history          = []

    print("─" * 58)
    print("  Starting Training")
    print("─" * 58)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        # ── KAN grid update (once per epoch, using first batch) ──────────
        # Keeps B-spline knots aligned with the evolving feature distribution.
        first_batch = next(iter(train_loader))
        first_patches = first_batch[0].to(device)
        model.update_kan_grids(first_patches)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for patches, labels, _ in pbar:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(patches)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * patches.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader.dataset)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc = \
            evaluate_model(val_loader, model, device, desc="Validating")

        history.append({
            "epoch":           epoch,
            "lr":              current_lr,
            "train_loss":      avg_train_loss,
            "val_loss":        val_loss,
            "val_auc":         val_auc,
            "val_auprc":       val_auprc,
            "val_f1":          val_f1,
            "val_sensitivity": val_sens,
            "val_specificity": val_spec,
        })

        print(f"\nEpoch {epoch:3d} │ LR {current_lr:.2e} │ "
              f"Train Loss {avg_train_loss:.4f} │ Val Loss {val_loss:.4f}")
        print(f"         │ AUROC {val_auc:.4f} │ AUPRC {val_auprc:.4f} │ "
              f"F1 {val_f1:.4f} │ Sens {val_sens:.4f} │ Spec {val_spec:.4f}")

        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_hybrid.pth")
            print(f"  ✓ [Saved] New best validation AUROC: {best_val_auc:.4f}")

            with open("best_metrics_hybrid.json", "w") as f:
                json.dump({
                    "epoch":       epoch,
                    "auc":         val_auc,
                    "auprc":       val_auprc,
                    "f1":          val_f1,
                    "sensitivity": val_sens,
                    "specificity": val_spec,
                }, f, indent=4)
        else:
            patience_counter += 1
            print(f"  · No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping triggered after epoch {epoch} ***")
                break

    pd.DataFrame(history).to_csv("hybrid_training_log.csv", index=False)
    print("\n[Saved] hybrid_training_log.csv")

    print("\n" + "═" * 58)
    print("  Loading best model for final test evaluation")
    print("═" * 58)

    model.load_state_dict(
        torch.load("best_model_hybrid.pth", map_location=device, weights_only=True)
    )

    (test_loss, test_auc, test_auprc, test_f1,
     test_sens, test_spec, test_froc, pred_df) = evaluate_model(
        test_loader, model, device, desc="Testing", return_preds=True
    )

    pred_df.to_csv("hybrid_test_predictions.csv", index=False)
    print("  [Saved] hybrid_test_predictions.csv")

    print("\n─── FINAL TEST RESULTS (Scan-Level) ─────────────────")
    print(f"  AUROC (Primary)       : {test_auc:.4f}")
    print(f"  AUPRC                 : {test_auprc:.4f}")
    print(f"  F1 Score              : {test_f1:.4f}")
    print(f"  Sensitivity (Recall)  : {test_sens:.4f}")
    print(f"  Specificity           : {test_spec:.4f}")

    print("\n─── FROC (Candidate-Level: FP/Scan → Sensitivity) ───")
    for fp_rate, sensitivity in test_froc.items():
        print(f"  {fp_rate:14s} : {sensitivity:.4f}")
    print("═" * 58 + "\n")


if __name__ == "__main__":
    main()