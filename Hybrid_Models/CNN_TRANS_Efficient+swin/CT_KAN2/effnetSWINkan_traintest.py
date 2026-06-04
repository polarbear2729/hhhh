"""
Hybrid 3D EfficientNet-B0 + 3D Swin-Tiny with KAN Head – Lung Nodule Classification
Dual-backbone feature extractor: EfficientNet-B0 (local CNN / SE features, 1280-d) fused
with SwinTransformer3D (global attention features, 384-d), followed by a KAN head.

Replaces the original MLP head with a Kolmogorov-Arnold Network (KAN) head:
  - KAN uses learnable B-spline activation functions on EACH connection weight,
    rather than fixed activations on layer outputs (as in MLP).
  - Each KANLinear layer learns: y_i = Σ_j φ_ij(x_j)  where φ_ij is a spline + residual.
  - More expressive per-parameter than MLP for tabular/feature-level tasks.

Architecture:
  EfficientNet-B0 (GAP after head conv)  → 1280-d
  SwinTransformer3D (embed=48, 4 stages) →  384-d
  Concatenated                           → 1664-d
  KAN head                               → 1664 → 512 → 256 → 1

KAN implementation: Efficient B-spline KAN (Liu et al., 2024)
  - grid_size   : number of spline control points (default 5)
  - spline_order: B-spline degree (default 3, i.e. cubic)
  - residual_std: scale for SiLU residual branch init
  - base_activation: SiLU (default) applied to residual path
  - grid_range  : input domain for splines (default [-1, 1]; inputs are LayerNorm'd)

All other training components remain identical to the MLP variant:
  - NormalizeIntensity, Label smoothing, Warmup + Cosine LR scheduling
  - FROC / CPM / AUROC / AUPRC metrics, Youden threshold at test time
  - Patient-level max-aggregation
  - Gradient clipping + AMP + skip on non-finite gradients
  - pos_weight computed dynamically from training split
"""

import os, json, random, math
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
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, recall_score, roc_curve
)
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise, NormalizeIntensity
)
from sklearn.calibration import calibration_curve
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------------------ Configuration ------------------------------
SEED = 42
DATA_DIR = r"D:\Projects\Thesis Lung Malignancy Classification ML & DL\LUNA16 Dataset Preprocessed"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE = 4          # reduced: dual-backbone ~2× memory
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR = 5e-5               # conservative; transformer-sensitive
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# EfficientNet-B0 final feature dimension (identical to 2D version)
EFFICIENTNET_FEATURE_DIM = 1280

# Swin pretrained toggle
USE_PRETRAINED_SWIN = False
PRETRAINED_SWIN_PATH = "swin3d_tiny_pretrained.pth"

# KAN head hyperparameters
KAN_GRID_SIZE    = 5      # number of B-spline grid intervals; more = richer but slower
KAN_SPLINE_ORDER = 3      # cubic B-splines (degree 3 is standard)
KAN_GRID_RANGE   = [-1, 1]  # spline domain; inputs are LayerNorm'd before each KAN layer
KAN_RESIDUAL_STD = 0.1    # std for residual (SiLU) weight init


# =====================================================================
# Self-contained 3D EfficientNet-B0 (identical to MLP hybrid script)
# =====================================================================

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
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, se_channels),
            SwishActivation(),
            nn.Linear(se_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1, 1)
        return x * scale


class MBConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_skip = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        mid_channels = _make_divisible(in_channels * expand_ratio, 8)
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv3d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3),
                SwishActivation()
            ]
        pad = (kernel_size - 1) // 2
        layers += [
            nn.Conv3d(mid_channels, mid_channels, kernel_size,
                      stride=stride, padding=pad, groups=mid_channels, bias=False),
            nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3),
            SwishActivation()
        ]
        layers.append(SqueezeExcitation3D(mid_channels, se_ratio))
        layers += [
            nn.Conv3d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-3)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            if self.training and self.drop_connect_rate > 0:
                keep_prob = 1 - self.drop_connect_rate
                rand_tensor = torch.rand(x.size(0), 1, 1, 1, 1, device=x.device)
                rand_tensor = torch.floor(rand_tensor + keep_prob)
                out = out / keep_prob * rand_tensor
            out = out + x
        return out


class EfficientNet3D_B0(nn.Module):
    """3D EfficientNet-B0 backbone — returns (B, 1280) feature vector."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=0.01, eps=1e-3),
            SwishActivation()
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
        block_idx = 0
        stages = []
        for (in_ch, out_ch, k, s, expand, n_layers) in stage_configs:
            stage = []
            for i in range(n_layers):
                stride = s if i == 0 else 1
                inch = in_ch if i == 0 else out_ch
                drop_rate = 0.2 * block_idx / total_blocks
                stage.append(MBConv3D(inch, out_ch, k, stride, expand,
                                      drop_connect_rate=drop_rate))
                block_idx += 1
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)
        self.head_conv = nn.Sequential(
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, kernel_size=1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.01, eps=1e-3),
            SwishActivation()
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.global_pool(x)
        return x.flatten(1)   # (B, 1280)


# =====================================================================
# Self-contained 3D Swin-Tiny (identical to MLP hybrid script)
# =====================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class _SwinMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B,
               D // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2], C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, window_size[0] * window_size[1] * window_size[2], C)


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B,
                     D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _SwinMlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        D_p, H_p, W_p = x.shape[1], x.shape[2], x.shape[3]
        if any(self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        x = window_reverse(self.attn(window_partition(x, self.window_size)), self.window_size, B, D_p, H_p, W_p)
        if any(self.shift_size):
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        x = x[:, :D, :H, :W, :]
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        xs = [x[:, d::2, h::2, w::2, :] for d in range(2) for h in range(2) for w in range(2)]
        return self.reduction(self.norm(torch.cat(xs, -1)))


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            ) for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.downsample(x) if self.downsample else x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 4, 1))


class SwinTransformer3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_chans=1, embed_dim=48,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=(4, 4, 4), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim, norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=int(embed_dim * 2 ** i), depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging if (i < len(depths) - 1) else None,
                norm_layer=norm_layer
            ) for i in range(len(depths))
        ])
        self.norm = norm_layer(int(embed_dim * 2 ** (len(depths) - 1)))

    def forward(self, x):
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).mean(dim=(1, 2, 3))   # (B, 384)


# =====================================================================
# KAN (Kolmogorov-Arnold Network) — B-spline implementation
# Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
#            arXiv:2404.19756
# =====================================================================

class KANLinear(nn.Module):
    """
    Single KAN layer: in_features → out_features.

    Each output neuron i aggregates learnable univariate spline functions φ_ij
    applied to each input neuron j:
        y_i = Σ_j [ w_b_ij * b(x_j) + w_s_ij * spline_ij(x_j) ]
    where:
        b(x)        = SiLU(x)   (residual / base activation)
        spline_ij(x) = Σ_k  c_ijk * B_k(x)   (B-spline with learnable coefficients)

    Parameters
    ----------
    in_features   : int
    out_features  : int
    grid_size     : int   — number of grid intervals (more = finer spline)
    spline_order  : int   — B-spline degree (3 = cubic)
    base_activation : nn.Module class  — applied to residual path (default SiLU)
    residual_std  : float — init std for base (residual) weight
    grid_range    : [float, float] — domain of spline grid
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation=nn.SiLU,
        residual_std: float = 0.1,
        grid_range: list = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]

        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        # Build uniform extended grid of size (grid_size + 2*spline_order + 1)
        # so that B-splines of order `spline_order` are well-defined over [grid_range]
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
        grid = grid * h + grid_range[0]   # shape: (grid_size + 2*spline_order + 1,)
        self.register_buffer("grid", grid)

        # Number of B-spline basis functions per input = grid_size + spline_order
        n_basis = grid_size + spline_order

        # Spline coefficients: (out_features, in_features, n_basis)
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, n_basis)
        )
        nn.init.normal_(self.spline_weight, 0.0, 0.1)

        # Residual (base) weights: (out_features, in_features)
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.normal_(self.base_weight, 0.0, residual_std)

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.base_activation = base_activation()

    # ------------------------------------------------------------------
    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all B-spline basis functions at x.

        Args:
            x : (B, in_features) — already clamped to grid range

        Returns:
            bases : (B, in_features, n_basis)
                    where n_basis = grid_size + spline_order
        """
        # x: (B, in_features) → (B, in_features, 1)
        x = x.unsqueeze(-1)
        grid = self.grid  # (n_grid_pts,)

        # Degree-0 (indicator) basis
        # bases shape: (B, in_features, n_grid_pts - 1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # De Boor recursion: build up to degree `spline_order`
        for k in range(1, self.spline_order + 1):
            # Left term: (x - t_i) / (t_{i+k} - t_i)
            left_num  = x - grid[:-(k + 1)]          # (B, F, n)
            left_den  = grid[k:-1] - grid[:-(k + 1)]  # (n,)
            left_den  = torch.where(left_den.abs() < 1e-8,
                                    torch.ones_like(left_den), left_den)
            left = (left_num / left_den) * bases[:, :, :-1]

            # Right term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1})
            right_num = grid[(k + 1):] - x             # (B, F, n)
            right_den = grid[(k + 1):] - grid[1:-k]    # (n,)
            right_den = torch.where(right_den.abs() < 1e-8,
                                    torch.ones_like(right_den), right_den)
            right = (right_num / right_den) * bases[:, :, 1:]

            bases = left + right   # (B, in_features, n_grid_pts - k - 1)

        # Final shape: (B, in_features, grid_size + spline_order)
        return bases.contiguous()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_features)
        Returns:
            y : (B, out_features)
        """
        # Clamp inputs to spline domain to prevent out-of-range extrapolation
        x_clamped = x.clamp(self.grid[0].item(), self.grid[-1].item())

        # ── Residual (base) path ──────────────────────────────────────
        # base_out: (B, out_features)
        base_out = F.linear(self.base_activation(x), self.base_weight)

        # ── Spline path ───────────────────────────────────────────────
        # bases: (B, in_features, n_basis)
        bases = self._b_splines(x_clamped)

        # spline_weight: (out_features, in_features, n_basis)
        # Combine: for each (out, in) pair, dot bases with its spline_weight vector
        # Result: (B, out_features)
        # Efficient einsum: 'bin, oin -> bo'
        spline_out = torch.einsum('bin,oin->bo', bases, self.spline_weight)

        return base_out + spline_out + self.bias


# =====================================================================
# KAN Head: stacks KANLinear layers with LayerNorm + Dropout between them
# =====================================================================

class KANHead(nn.Module):
    """
    Multi-layer KAN head for classification.

    Architecture (for dims=[1664, 512, 256, 1]):
        Input (B, 1664)
        → LayerNorm(1664)          # normalise into spline domain [-1,1]-ish
        → KANLinear(1664 → 512)
        → LayerNorm(512)
        → Dropout(p=dropout)
        → KANLinear(512 → 256)
        → LayerNorm(256)
        → Dropout(p=dropout/2)
        → KANLinear(256 → 1)
        Output (B, 1)  (raw logit)

    LayerNorm before each KANLinear is critical: it keeps the input in the
    spline's learned domain, preventing the spline from operating in an
    extrapolation regime.

    Parameters
    ----------
    dims         : list[int]  — layer widths including input and output
    grid_size    : int
    spline_order : int
    dropout      : float
    grid_range   : list[float, float]
    residual_std : float
    """

    def __init__(
        self,
        dims: list,
        grid_size: int = KAN_GRID_SIZE,
        spline_order: int = KAN_SPLINE_ORDER,
        dropout: float = 0.5,
        grid_range: list = None,
        residual_std: float = KAN_RESIDUAL_STD,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = KAN_GRID_RANGE

        layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = (i == len(dims) - 2)
            layers.append(nn.LayerNorm(d_in))
            layers.append(
                KANLinear(d_in, d_out,
                          grid_size=grid_size,
                          spline_order=spline_order,
                          residual_std=residual_std,
                          grid_range=grid_range)
            )
            if not is_last:
                drop_p = dropout if i == 0 else dropout / 2
                layers.append(nn.Dropout(p=drop_p))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# Hybrid Model: EfficientNet-B0 + Swin3D-Tiny → KAN Head
# =====================================================================

class HybridEfficientNetSwinKANHead(nn.Module):
    """
    Dual-backbone hybrid with KAN fusion head:
      1. EfficientNet-B0 backbone  (local CNN + SE features, 1280-d)
      2. SwinTransformer3D         (global attention features,  384-d)
      3. Concatenation             (1664-d)
      4. KAN head: 1664 → 512 → 256 → 1
         Each connection uses learned B-spline univariate functions
         rather than a fixed non-linearity (as in the MLP variant).

    KAN vs MLP head:
      - MLP: y = σ(Wx + b)            — fixed activation σ, learned W
      - KAN: y = Σ_j φ_j(x_j)         — learned φ_j (spline + residual)
      KAN can model more complex feature interactions per parameter,
      especially beneficial when fused features span very different scales
      (CNN texture features vs. transformer global context).
    """

    EFFICIENTNET_DIM = 1280
    SWIN_DIM         = 384
    FUSED_DIM        = EFFICIENTNET_DIM + SWIN_DIM   # 1664

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        dropout: float = 0.5,
        swin_pretrained_path: str = None,
        kan_grid_size: int = KAN_GRID_SIZE,
        kan_spline_order: int = KAN_SPLINE_ORDER,
        kan_grid_range: list = None,
        kan_residual_std: float = KAN_RESIDUAL_STD,
    ):
        super().__init__()
        if kan_grid_range is None:
            kan_grid_range = KAN_GRID_RANGE

        # ── EfficientNet-B0 backbone ──────────────────────────────────
        self.efficientnet = EfficientNet3D_B0(in_channels=in_channels)

        # ── Swin-Tiny backbone ────────────────────────────────────────
        self.swin = SwinTransformer3D(
            in_chans=in_channels, embed_dim=48,
            depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
            window_size=(4, 4, 4), patch_size=(4, 4, 4),
            mlp_ratio=4.0, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )

        # ── Optional Swin pretrained weights ─────────────────────────
        if swin_pretrained_path is not None and os.path.exists(swin_pretrained_path):
            print(f"[Pretrained] Loading Swin weights from {swin_pretrained_path}")
            state = torch.load(swin_pretrained_path, map_location='cpu')
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            swin_dict = self.swin.state_dict()
            matched = {k: v for k, v in state.items() if k in swin_dict}
            swin_dict.update(matched)
            self.swin.load_state_dict(swin_dict, strict=False)
            print(f"[Pretrained] Loaded {len(matched)} matching Swin parameters.")
        elif swin_pretrained_path is not None:
            print(f"[Pretrained] WARNING: '{swin_pretrained_path}' not found. Swin trains from scratch.")

        # ── KAN fusion head ───────────────────────────────────────────
        # Dims: 1664 → 512 → 256 → 1  (mirrors MLP topology for fair comparison)
        self.head = KANHead(
            dims=[self.FUSED_DIM, 512, 256, num_classes],
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            dropout=dropout,
            grid_range=kan_grid_range,
            residual_std=kan_residual_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e_feat = self.efficientnet(x)            # (B, 1280)
        s_feat = self.swin(x)                    # (B,  384)
        fused  = torch.cat([e_feat, s_feat], 1)  # (B, 1664)
        return self.head(fused).squeeze(1)        # (B,)


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Dataset
# =====================================================================

class NodulePatchDataset(Dataset):
    def __init__(self, metadata_df, data_dir, transforms=None):
        self.metadata  = metadata_df.reset_index(drop=True)
        self.data_dir  = data_dir
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


# =====================================================================
# Transforms
# =====================================================================

def get_transforms(mode: str = "train"):
    if mode == "train":
        return Compose([
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandGaussianNoise(prob=0.2, std=0.01),
            NormalizeIntensity(nonzero=False, channel_wise=True),
        ])
    else:
        return Compose([
            NormalizeIntensity(nonzero=False, channel_wise=True),
        ])


# =====================================================================
# Sanity check
# =====================================================================

def run_forward_sanity_check(model: nn.Module, device: torch.device):
    print("\n[SANITY CHECK] Running forward pass on dummy input (2, 1, 64, 64, 64)...")
    model.eval()
    dummy = torch.randn(2, 1, 64, 64, 64).to(device)
    with torch.no_grad():
        logits = model(dummy)
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        print(f"  logit shape : {logits.shape}  (expected (2,))")
        print(f"  NaN: {has_nan}  Inf: {has_inf}  (must both be False)")
        print(f"  logit range : [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"  prob  range : [{torch.sigmoid(logits).min().item():.4f}, {torch.sigmoid(logits).max().item():.4f}]")
        if logits.abs().max().item() > 5.0:
            print("  [WARN] Large initial logits – check head initialisation")
        else:
            print("  Initial logit magnitude OK (< 5.0)")
    if has_nan or has_inf:
        raise RuntimeError("[SANITY CHECK FAILED] NaN/Inf in forward pass at init.")
    print("[SANITY CHECK] Passed.\n")
    model.train()


# =====================================================================
# FROC / evaluation helpers
# =====================================================================

def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    idx  = np.argsort(all_probs)[::-1]
    tps  = np.cumsum(all_labels[idx] == 1)
    fps  = np.cumsum(all_labels[idx] == 0)
    fps_per_scan = fps / total_scans
    sensitivity  = tps / total_positives
    froc_scores = {}
    for target in FROC_THRESHOLDS:
        valid_idx = np.where(fps_per_scan <= target)[0]
        froc_scores[f"{target} FP/scan"] = float(sensitivity[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0
    return froc_scores


def calculate_95_ci(y_true, y_probs, n_bootstraps: int = 1000):
    rng = np.random.RandomState(SEED)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[idx])) < 2:
            continue
        scores.append(roc_auc_score(np.array(y_true)[idx], np.array(y_probs)[idx]))
    s = np.sort(scores)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def get_youden_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate_model(loader, model, device, pos_weight_tensor,
                   desc="Validating", return_preds=False,
                   threshold=0.5, label_smoothing=0.0):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc=desc):
            patches, labels = patches.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(patches)
                if label_smoothing > 0:
                    smooth_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
                    loss = F.binary_cross_entropy_with_logits(
                        logits, smooth_labels, pos_weight=pos_weight_tensor)
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits, labels, pos_weight=pos_weight_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = [float(probs)]; labels_list = [float(labels.cpu())]
            else:
                probs = probs.tolist(); labels_list = labels.cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels_list)
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

    y_true_p  = [v['label'] for v in patient_dict.values()]
    y_probs_p = [v['prob']  for v in patient_dict.values()]
    y_pred_p  = [1 if p >= threshold else 0 for p in y_probs_p]
    total_scans = len(patient_dict)

    avg_loss = running_loss / len(loader.dataset)
    auc   = roc_auc_score(y_true_p, y_probs_p) if len(np.unique(y_true_p)) > 1 else 0.5
    auprc = average_precision_score(y_true_p, y_probs_p)
    f1    = f1_score(y_true_p, y_pred_p, zero_division=0)
    sens  = recall_score(y_true_p, y_pred_p, zero_division=0)
    cm    = confusion_matrix(y_true_p, y_pred_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    froc  = calculate_candidate_froc(all_labels, all_probs, total_scans)
    cpm   = float(np.mean(list(froc.values())))

    pos_probs = [p for p, l in zip(all_probs, all_labels) if l == 1]
    neg_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
    print(f"  [Diag] pos: mean={np.mean(pos_probs):.4f} std={np.std(pos_probs):.4f} | "
          f"neg: mean={np.mean(neg_probs):.4f} std={np.std(neg_probs):.4f}")
    print(f"  [Diag] pred_pos={sum(y_pred_p)} pred_neg={len(y_pred_p)-sum(y_pred_p)} | "
          f"true_pos={sum(int(l) for l in y_true_p)} true_neg={sum(1-int(l) for l in y_true_p)}")

    if return_preds:
        ci_lower, ci_upper = calculate_95_ci(y_true_p, y_probs_p)
        print(f"  AUROC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        frac_pos, mean_pred = calibration_curve(y_true_p, y_probs_p, n_bins=10)
        pd.DataFrame({'mean_predicted_probability': mean_pred,
                      'fraction_of_positives': frac_pos}
                     ).to_csv("hybrid_effnet_kan_calibration_curve.csv", index=False)
        pred_df = pd.DataFrame({'seriesuid': all_uids,
                                'label': all_labels,
                                'probability': all_probs})
        return avg_loss, auc, auprc, f1, sens, spec, froc, cpm, pred_df

    return avg_loss, auc, auprc, f1, sens, spec, froc, cpm


# =====================================================================
# Main
# =====================================================================

def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device   : {device}")
    print(f"LR             : {LR}  (warmup {WARMUP_EPOCHS} ep)")
    print(f"Label smooth   : {LABEL_SMOOTHING}")
    print(f"Batch size     : {BATCH_SIZE}  (reduced for dual-backbone memory)")
    print(f"KAN grid_size  : {KAN_GRID_SIZE}  spline_order: {KAN_SPLINE_ORDER}")

    # ── Metadata ────────────────────────────────────────────────────
    print("\nLoading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)
    print(f"  Train: {len(train_meta)} | Val: {len(val_meta)} | Test: {len(test_meta)}")
    print(f"  Train pos: {(train_meta['label']==1).sum()} | neg: {(train_meta['label']==0).sum()}")

    # pos_weight computed dynamically from training split
    neg_count = (train_meta["label"] == 0).sum()
    pos_count = (train_meta["label"] == 1).sum()
    pos_weight_value = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"Class balance: neg={neg_count}, pos={pos_count}, pos_weight={pos_weight_value:.2f}")

    # ── DataLoaders ──────────────────────────────────────────────────
    train_ds = NodulePatchDataset(train_meta, DATA_DIR, transforms=get_transforms("train"))
    val_ds   = NodulePatchDataset(val_meta,   DATA_DIR, transforms=get_transforms("val"))
    test_ds  = NodulePatchDataset(test_meta,  DATA_DIR, transforms=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)

    # ── Model ────────────────────────────────────────────────────────
    swin_pretrained = PRETRAINED_SWIN_PATH if USE_PRETRAINED_SWIN else None
    model = HybridEfficientNetSwinKANHead(swin_pretrained_path=swin_pretrained).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Model Summary ---")
    print(f"Model          : Hybrid EfficientNet-B0 + Swin3D-Tiny + KAN Head")
    print(f"Total params   : {total_params:,}")
    print(f"Trainable      : {trainable:,}")
    print(f"Fused feat dim : {HybridEfficientNetSwinKANHead.FUSED_DIM}  "
          f"(EfficientNet {HybridEfficientNetSwinKANHead.EFFICIENTNET_DIM}"
          f"  +  Swin {HybridEfficientNetSwinKANHead.SWIN_DIM})")
    print(f"KAN topology   : {HybridEfficientNetSwinKANHead.FUSED_DIM} → 512 → 256 → 1")
    print(f"KAN grid_size  : {KAN_GRID_SIZE},  spline_order: {KAN_SPLINE_ORDER}")
    print("---------------------")

    run_forward_sanity_check(model, device)

    # ── Optimizer / Scheduler ────────────────────────────────────────
    optimizer    = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - WARMUP_EPOCHS))
    scheduler    = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                                milestones=[WARMUP_EPOCHS])

    scaler = GradScaler('cuda', init_scale=256)
    best_val_auc = 0.0
    patience_counter = 0
    history = []
    total_skipped = 0

    # ── Training loop ────────────────────────────────────────────────
    print("--- Starting Training ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        grad_norms = []
        epoch_skips = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for patches, labels, _ in pbar:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(patches)
                if LABEL_SMOOTHING > 0:
                    smooth_labels = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
                    loss = F.binary_cross_entropy_with_logits(logits, smooth_labels,
                                                              pos_weight=pos_weight)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, labels,
                                                              pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if not torch.isfinite(total_norm):
                optimizer.zero_grad(); scaler.update()
                epoch_skips += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'skip': epoch_skips})
                continue
            grad_norms.append(total_norm.item())
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item() * patches.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        total_skipped += epoch_skips
        eff_samples    = max(len(train_loader.dataset) - epoch_skips * BATCH_SIZE, 1)
        avg_train_loss = train_loss / eff_samples
        avg_grad_norm  = np.mean(grad_norms) if grad_norms else float('nan')
        max_grad_norm  = np.max(grad_norms)  if grad_norms else float('nan')
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        skip_pct = epoch_skips / len(train_loader) * 100
        skip_str = f"skipped={epoch_skips}/{len(train_loader)} ({skip_pct:.1f}%)"
        if skip_pct > 10:
            skip_str += "  *** HIGH SKIP RATE ***"
        print(f"\n  [Grad] avg={avg_grad_norm:.4f}  max={max_grad_norm:.4f}  "
              f"scale={scaler.get_scale():.0f}  {skip_str}")

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc, val_cpm = \
            evaluate_model(val_loader, model, device, pos_weight,
                           desc="Validating", label_smoothing=0.0)

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": avg_train_loss, "val_loss": val_loss,
            "val_auc": val_auc, "val_auprc": val_auprc,
            "val_f1": val_f1, "val_sensitivity": val_sens,
            "val_specificity": val_spec, "val_cpm": val_cpm,
            "avg_grad_norm": avg_grad_norm, "skipped_steps": epoch_skips,
        })

        print(f"\nEpoch {epoch:3d} | LR: {current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val AUROC: {val_auc:.4f} | AUPRC: {val_auprc:.4f} | "
              f"F1: {val_f1:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | CPM: {val_cpm:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_hybrid_effnet_kan.pth")
            print(f"  --> [Saved] New Best Val AUROC: {best_val_auc:.4f}")
            with open("best_metrics_hybrid_effnet_kan.json", "w") as f:
                json.dump({"epoch": epoch, "auc": val_auc, "auprc": val_auprc,
                           "f1": val_f1, "sensitivity": val_sens,
                           "specificity": val_spec, "cpm": val_cpm}, f, indent=4)
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping at epoch {epoch} ***")
                break

    print(f"\nTotal skipped gradient steps: {total_skipped}")
    pd.DataFrame(history).to_csv("hybrid_effnet_kan_training_log.csv", index=False)
    print("[Saved] Training log -> 'hybrid_effnet_kan_training_log.csv'")

    # ── Test evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("=" * 55)
    model.load_state_dict(torch.load("best_model_hybrid_effnet_kan.pth",
                                     map_location=device, weights_only=True))

    print("\nComputing Youden threshold on validation set...")
    model.eval()
    val_probs_all, val_labels_all = [], []
    with torch.no_grad():
        for patches, labels, _ in tqdm(val_loader, desc="Val (Youden)"):
            patches = patches.to(device)
            with autocast('cuda'):
                logits = model(patches)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs_all.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])
            val_labels_all.extend(labels.tolist())
    youden_threshold = (get_youden_threshold(val_labels_all, val_probs_all)
                        if len(np.unique(val_labels_all)) > 1 else 0.5)
    print(f"  Youden threshold (val): {youden_threshold:.4f}")

    (test_loss, test_auc, test_auprc, test_f1_05, test_sens_05, test_spec_05,
     test_froc, test_cpm, pred_df) = evaluate_model(
        test_loader, model, device, pos_weight,
        desc="Testing (t=0.50)", return_preds=True, threshold=0.5, label_smoothing=0.0)

    pred_df.to_csv("hybrid_effnet_kan_test_predictions.csv", index=False)
    print("[Saved] Test predictions -> 'hybrid_effnet_kan_test_predictions.csv'")

    # Youden threshold metrics on test set
    patient_dict_test = {}
    for _, row in pred_df.iterrows():
        uid = row["seriesuid"]
        if uid not in patient_dict_test:
            patient_dict_test[uid] = {'prob': row["probability"], 'label': row["label"]}
        else:
            patient_dict_test[uid]['prob']  = max(patient_dict_test[uid]['prob'],  row["probability"])
            patient_dict_test[uid]['label'] = max(patient_dict_test[uid]['label'], row["label"])
    y_true_t  = [v['label'] for v in patient_dict_test.values()]
    y_probs_t = [v['prob']  for v in patient_dict_test.values()]
    y_pred_y  = [1 if p >= youden_threshold else 0 for p in y_probs_t]
    cm_y = confusion_matrix(y_true_t, y_pred_y, labels=[0, 1])
    tn_y, fp_y, fn_y, tp_y = cm_y.ravel()
    test_f1_y   = f1_score(y_true_t, y_pred_y, zero_division=0)
    test_sens_y = recall_score(y_true_t, y_pred_y, zero_division=0)
    test_spec_y = tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else 0.0

    print("\n--- FINAL TEST RESULTS (SCAN-LEVEL) ---")
    print(f"AUROC           : {test_auc:.4f}")
    print(f"AUPRC           : {test_auprc:.4f}")
    print(f"CPM             : {test_cpm:.4f}")
    print(f"\n  threshold=0.50  → F1:{test_f1_05:.4f} Sens:{test_sens_05:.4f} Spec:{test_spec_05:.4f}")
    print(f"  threshold={youden_threshold:.4f} → F1:{test_f1_y:.4f} Sens:{test_sens_y:.4f} Spec:{test_spec_y:.4f} (Youden)")
    print(f"\n--- FROC ---")
    for fp_rate, sens_val in test_froc.items():
        print(f"  {fp_rate:12s} : {sens_val:.4f}")
    print(f"  CPM (mean)    : {test_cpm:.4f}")
    print("=" * 55)

    with open("final_test_metrics_hybrid_effnet_kan.json", "w") as f:
        json.dump({
            "auroc": test_auc, "auprc": test_auprc, "cpm": test_cpm,
            "threshold_05":     {"f1": test_f1_05, "sensitivity": test_sens_05, "specificity": test_spec_05},
            "threshold_youden": {"value": youden_threshold, "f1": test_f1_y,
                                 "sensitivity": test_sens_y, "specificity": test_spec_y},
            "froc": test_froc,
        }, f, indent=4)
    print("[Saved] Final metrics -> 'final_test_metrics_hybrid_effnet_kan.json'")


if __name__ == "__main__":
    main()