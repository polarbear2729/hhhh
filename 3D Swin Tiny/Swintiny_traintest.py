# swin3d_tinyvit.py  (verbose v1)
"""
3D Swin Transformer Tiny for lung nodule classification.
Trained from scratch on 64x64x64 patches — mirrors the TinyViT pipeline exactly:
  - Same dataset class   (NodulePatchDataset3D)
  - Same training loop   (BCEWithLogitsLoss, AdamW, CosineAnnealingLR)
  - Same evaluation      (AUROC, AUPRC, F1, Sensitivity, Specificity, FROC)
  - Same GradCAM++ code  (adapted for 3D feature maps)
  - Same verbose logging
"""

import os
import math
import warnings
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from monai.transforms import Compose, Resized, RandFlipd, RandRotate90d, NormalizeIntensityd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings("ignore")


# ===========================================================================
# 0.  CONFIGURATION  (mirrors TinyViT config exactly)
# ===========================================================================
DATA_DIR          = r"D:\Projects\Thesis Lung Malignancy Classification ML & DL\LUNA16 Dataset Preprocessed"
METADATA_PATH     = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH= os.path.join(DATA_DIR, "patient_split.csv")
MODEL_SAVE_PATH   = "best_model_swin3d.pth"

PATCH_SIZE        = 64          # cubic patch: 64 x 64 x 64
IN_CHANNELS       = 1           # single-channel CT volume (3D)
NUM_CLASSES       = 1           # binary: malignant vs benign
BATCH_SIZE        = 8          # 3D volumes are heavier; reduce if OOM
NUM_WORKERS       = 4
NUM_EPOCHS        = 200
LR                = 1e-4
WEIGHT_DECAY      = 1e-2
DROPOUT           = 0.5
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Swin-Tiny 3D architecture hyperparameters
EMBED_DIM         = 48          # C in the paper; Tiny uses 48
DEPTHS            = [2, 2, 6, 2]        # transformer blocks per stage
NUM_HEADS         = [3, 6, 12, 24]      # attention heads per stage
WINDOW_SIZE       = (4, 4, 4)           # 3D window; 64/4=16 windows per dim
MLP_RATIO         = 4.0
QKV_BIAS          = True
DROP_PATH_RATE    = 0.2                 # stochastic depth


# ===========================================================================
# 1.  DATASET  — 3D version (replaces 2.5D stacking)
# ===========================================================================
class NodulePatchDataset3D(torch.utils.data.Dataset):
    """
    Loads a 64x64x64 .npy volume.
    Returns shape: (1, 64, 64, 64)  — single channel 3D tensor.
    Compare with TinyViT which returned (3, 224, 224).
    """
    def __init__(self, metadata_df, data_dir, transforms=None):
        self.metadata   = metadata_df.reset_index(drop=True)
        self.data_dir   = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row      = self.metadata.iloc[idx]
        filename = os.path.basename(row["filepath"])
        split    = row["split"]
        label    = int(row["label"])
        subfolder= "pos" if label == 1 else "neg"

        local_path = os.path.join(self.data_dir, split, subfolder, filename)
        vol = np.load(local_path).astype(np.float32)  # (D, H, W)

        # Add channel dim → (1, D, H, W)
        vol = vol[np.newaxis, ...]
        data_dict = {"img": vol}

        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        img = data_dict["img"]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)

        return img, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]


# ===========================================================================
# 2.  3D SWIN TRANSFORMER — built from scratch, no pretraining
# ===========================================================================

# --------------- helpers ---------------
def window_partition_3d(x, window_size):
    """
    Partition 3D feature map into non-overlapping windows.
    x: (B, D, H, W, C)
    window_size: (Wd, Wh, Ww)
    Returns: (num_windows*B, Wd, Wh, Ww, C)
    """
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size
    x = x.view(B,
                D // Wd, Wd,
                H // Wh, Wh,
                W // Ww, Ww,
                C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, Wd, Wh, Ww, C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    """
    Reverse window partition.
    windows: (num_windows*B, Wd, Wh, Ww, C)
    Returns: (B, D, H, W, C)
    """
    Wd, Wh, Ww = window_size
    B = int(windows.shape[0] / (D * H * W / Wd / Wh / Ww))
    x = windows.view(B,
                     D // Wd, H // Wh, W // Ww,
                     Wd, Wh, Ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


def get_relative_position_index_3d(Wd, Wh, Ww):
    """Pre-compute relative position indices for 3D window attention."""
    coords_d = torch.arange(Wd)
    coords_h = torch.arange(Wh)
    coords_w = torch.arange(Ww)
    # (3, Wd, Wh, Ww)
    coords   = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
    coords_flat = coords.flatten(1)                        # (3, N)
    relative    = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, N, N)
    relative    = relative.permute(1, 2, 0).contiguous()             # (N, N, 3)
    relative[:, :, 0] += Wd - 1
    relative[:, :, 1] += Wh - 1
    relative[:, :, 2] += Ww - 1
    relative[:, :, 0] *= (2 * Wh - 1) * (2 * Ww - 1)
    relative[:, :, 1] *= (2 * Ww - 1)
    return relative.sum(-1)                                # (N, N)


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self-Attention with relative position bias.
    Supports both regular and shifted windows.
    """
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size          # (Wd, Wh, Ww)
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        Wd, Wh, Ww = window_size
        # relative position bias table: (2Wd-1) * (2Wh-1) * (2Ww-1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*Wd-1) * (2*Wh-1) * (2*Ww-1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        rel_idx = get_relative_position_index_3d(Wd, Wh, Ww)
        self.register_buffer("relative_position_index", rel_idx)

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop= nn.Dropout(attn_drop)
        self.proj     = nn.Linear(dim, dim)
        self.proj_drop= nn.Dropout(proj_drop)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """x: (num_windows*B, N, C)   N = Wd*Wh*Ww"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q   = q * self.scale
        attn= q @ k.transpose(-2, -1)

        # relative position bias
        Wd, Wh, Ww = self.window_size
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(Wd*Wh*Ww, Wd*Wh*Ww, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()   # (num_heads, N, N)
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """
    One Swin Transformer block: W-MSA (or SW-MSA) + FFN.
    shift_size = (0,0,0) → regular window attention
    shift_size = window_size//2 → shifted window attention
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=(4,4,4),
                 shift_size=(0,0,0), mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution   # (D, H, W) in tokens
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size

        # clamp shift if resolution < window
        self.shift_size = tuple(
            0 if input_resolution[i] <= window_size[i] else shift_size[i]
            for i in range(3)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # pre-compute attention mask for shifted windows
        if any(s > 0 for s in self.shift_size):
            self.register_buffer("attn_mask",
                                 self._compute_attn_mask(input_resolution))
        else:
            self.attn_mask = None

    def _compute_attn_mask(self, resolution):
        D, H, W = resolution
        img_mask = torch.zeros(1, D, H, W, 1)
        Wd, Wh, Ww = self.window_size
        Sd, Sh, Sw = self.shift_size

        d_slices = (slice(0, -Wd), slice(-Wd, -Sd), slice(-Sd, None))
        h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None))
        w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None))

        cnt = 0
        for d_s, h_s, w_s in itertools.product(d_slices, h_slices, w_slices):
            img_mask[:, d_s, h_s, w_s, :] = cnt
            cnt += 1

        windows = window_partition_3d(img_mask, self.window_size)  # (nW, Wd, Wh, Ww, 1)
        windows = windows.view(-1, Wd * Wh * Ww)
        attn_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        """x: (B, D*H*W, C)"""
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # cyclic shift
        if any(s > 0 for s in self.shift_size):
            Sd, Sh, Sw = self.shift_size
            x = torch.roll(x, shifts=(-Sd, -Sh, -Sw), dims=(1, 2, 3))

        windows = window_partition_3d(x, self.window_size)
        Wd, Wh, Ww = self.window_size
        windows = windows.view(-1, Wd * Wh * Ww, C)

        attn_out = self.attn(windows, mask=self.attn_mask)
        attn_out = attn_out.view(-1, Wd, Wh, Ww, C)
        x = window_reverse_3d(attn_out, self.window_size, D, H, W)

        # reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            Sd, Sh, Sw = self.shift_size
            x = torch.roll(x, shifts=(Sd, Sh, Sw), dims=(1, 2, 3))

        x = x.view(B, D * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    """
    Downsample by 2x in each spatial dim (factor 8 total).
    Input : (B, D*H*W, C)
    Output: (B, D/2 * H/2 * W/2, 2C)
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim    = dim
        self.norm   = nn.LayerNorm(8 * dim)
        self.reduce = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)

        # collect 2x2x2 neighbors
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)  # (B, D/2, H/2, W/2, 8C)
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduce(x)
        return x


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding: (B, 1, D, H, W) → (B, N, embed_dim)
    Uses a Conv3d with stride=patch_size (non-overlapping patches).
    Patch size fixed at 2 to give 32x32x32 tokens from 64x64x64 input.
    """
    def __init__(self, patch_size=2, in_channels=1, embed_dim=48):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)                           # (B, embed_dim, D/p, H/p, W/p)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, D*H*W, C)
        x = self.norm(x)
        return x, (D, H, W)


class DropPath(nn.Module):
    """Stochastic depth (drop entire sample residual path)."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand  = torch.floor(rand + keep)
        return x / keep * rand


class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer Tiny backbone (from scratch, no pretraining).

    Input : (B, 1, 64, 64, 64)
    Output: 1D feature vector of size embed_dim * 8 = 384

    Stage resolutions (tokens after patch embed with patch_size=2):
      Stage 0: 32x32x32  (embed_dim=48)
      Stage 1: 16x16x16  (embed_dim=96)   after PatchMerging
      Stage 2:  8x 8x 8  (embed_dim=192)  after PatchMerging
      Stage 3:  4x 4x 4  (embed_dim=384)  after PatchMerging
    """
    def __init__(self, in_channels=1, embed_dim=48, depths=(2,2,6,2),
                 num_heads=(3,6,12,24), window_size=(4,4,4),
                 mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 patch_size=2):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim  = embed_dim

        self.patch_embed = PatchEmbed3D(patch_size=patch_size,
                                        in_channels=in_channels,
                                        embed_dim=embed_dim)
        self.pos_drop    = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule (linear)
        total_blocks = sum(depths)
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, total_blocks)]

        # build stages
        self.layers = nn.ModuleList()
        block_idx   = 0

        # resolution after patch embed
        init_res = (64 // patch_size,) * 3   # (32, 32, 32)

        for i_layer in range(self.num_layers):
            # resolution at this stage
            res = tuple(r // (2 ** i_layer) for r in init_res)
            dim = embed_dim * (2 ** i_layer)

            blocks = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                # alternate regular / shifted windows
                shift = tuple(w // 2 for w in window_size) \
                        if i_block % 2 == 1 else (0, 0, 0)
                blocks.append(SwinTransformerBlock3D(
                    dim=dim,
                    input_resolution=res,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx + i_block],
                ))
            block_idx += depths[i_layer]

            # PatchMerging after every stage except the last
            downsample = PatchMerging3D(res, dim) \
                         if i_layer < self.num_layers - 1 else None

            self.layers.append(nn.ModuleDict({
                    'blocks': blocks,
                    'downsample': downsample if downsample is not None else nn.Identity()
}))

        self.norm     = nn.LayerNorm(embed_dim * (2 ** (self.num_layers - 1)))
        self.avgpool  = nn.AdaptiveAvgPool1d(1)
        self.num_features = embed_dim * (2 ** (self.num_layers - 1))  # 384

    def forward_features(self, x):
        x, (D, H, W) = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            for block in layer['blocks']:
                x = block(x)
            
            x = layer['downsample'](x)

        x = self.norm(x)                   # (B, N, C)
        x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
        x = x.flatten(1)                   # (B, C)
        return x

    def forward(self, x):
        return self.forward_features(x)


# ===========================================================================
# 3.  FULL MODEL with MLP HEAD  (identical structure to TinyViTWithMLPHead)
# ===========================================================================
class Swin3DWithMLPHead(nn.Module):
    """
    3D Swin Tiny backbone + MLP classifier head.
    Mirrors TinyViTWithMLPHead exactly — same head architecture.
    """
    def __init__(self, num_classes=1, dropout=0.5):
        super().__init__()
        self.backbone = SwinTransformer3D(
            in_channels     = IN_CHANNELS,
            embed_dim       = EMBED_DIM,
            depths          = DEPTHS,
            num_heads       = NUM_HEADS,
            window_size     = WINDOW_SIZE,
            mlp_ratio       = MLP_RATIO,
            qkv_bias        = QKV_BIAS,
            drop_rate       = 0.0,
            attn_drop_rate  = 0.0,
            drop_path_rate  = DROP_PATH_RATE,
            patch_size      = 2,
        )
        feat_dim = self.backbone.num_features   # 384

        # Identical MLP head as TinyViT
        self.mlp_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.mlp_head(features)
        return out.squeeze(1)


# ===========================================================================
# 4.  GRADCAM++ (3D-adapted, mirrors TinyViT GradCAM code)
# ===========================================================================
def find_target_layer_3d(model):
    """
    Same logic as TinyViT find_target_layer but looks for 5D tensors (3D conv).
    Falls back to the backbone norm layer (1D token output).
    """
    model.eval()
    dummy = torch.randn(1, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE).to(
        next(model.parameters()).device)

    target_name = None
    best_size   = 0

    def hook_fn(name):
        def hook(module, input, output):
            nonlocal target_name, best_size
            if isinstance(output, torch.Tensor) and output.dim() == 5:
                d, h, w = output.shape[2], output.shape[3], output.shape[4]
                if d > 1 and d * h * w > best_size:
                    best_size   = d * h * w
                    target_name = name
        return hook

    handles = []
    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(hook_fn(name)))
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()

    if target_name is not None:
        print(f"✔ Chosen target layer (Conv3d): {target_name} "
              f"(spatial volume {best_size})")
        return target_name
    else:
        print("⚠ No Conv3d layer found, using backbone.norm (token layer)")
        return "backbone.norm"


def reshape_token_to_grid_3d(tensor, resolution=None):
    """
    If tensor is (B, N, C), reshape to (B, C, D, H, W).
    resolution: (D, H, W) — inferred as cube root of N if not given.
    Mirrors TinyViT's reshape_token_to_grid for 3D.
    """
    if tensor.dim() == 3:
        B, N, C = tensor.shape
        if resolution is not None:
            D, H, W = resolution
        else:
            cbrt = round(N ** (1/3))
            assert cbrt ** 3 == N, f"Cannot reshape {N} tokens to a cube."
            D = H = W = cbrt
        return tensor.transpose(1, 2).view(B, C, D, H, W)
    return tensor


def grad_cam_plusplus_3d(fmap, grad):
    """
    GradCAM++ for 3D feature maps.
    fmap, grad: (B, C, D, H, W)
    Returns heatmap: (B, D, H, W)
    Mirrors TinyViT grad_cam_plusplus exactly — only dim change.
    """
    grad_2 = grad.pow(2)
    grad_3 = grad_2 * grad
    sum_activations = fmap.sum(dim=(2, 3, 4), keepdim=True)   # 3D sum

    numerator   = grad_2 * fmap
    denominator = 2 * grad_2 + sum_activations * grad_3
    denominator = torch.where(denominator != 0,
                              denominator,
                              torch.ones_like(denominator))

    alpha   = numerator / denominator
    alpha   = alpha.sum(dim=(2, 3, 4), keepdim=True)           # 3D sum
    weights = F.relu(grad) * alpha
    heatmap = (weights * fmap).sum(dim=1)                      # (B, D, H, W)
    heatmap = F.relu(heatmap)
    return heatmap


def grad_cam_vanilla_3d(fmap, grad):
    """Standard GradCAM — 3D version. Fallback when GradCAM++ is zero."""
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    heatmap = (weights * fmap).sum(dim=1)
    return F.relu(heatmap)


def produce_heatmap_3d(model, input_tensor, target_layer_name, class_idx=1):
    """
    Full 3D GradCAM++ pipeline — verbose, mirrors TinyViT produce_heatmap.
    Returns:
        heatmap_centre_slice (H, W)  — central depth slice of 3D heatmap
        superimposed        (BGR)    — overlay on centre CT slice
    """
    model.eval()
    features  = {}
    gradients = {}

    def forward_hook(module, input, output):
        features['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    target_layer = dict(model.named_modules())[target_layer_name]
    fwd_handle   = target_layer.register_forward_hook(forward_hook)
    bwd_handle   = target_layer.register_backward_hook(backward_hook)

    logits = model(input_tensor)
    gradient_sign = 1.0 if class_idx == 1 else -1.0
    model.zero_grad()
    logits.backward(gradient=torch.full_like(logits, gradient_sign),
                    retain_graph=True)

    fmap = features['value'].detach()
    grad = gradients['value'].detach()
    fwd_handle.remove()
    bwd_handle.remove()

    # Reshape tokens → 5D if needed
    fmap_5d = reshape_token_to_grid_3d(fmap)
    grad_5d = reshape_token_to_grid_3d(grad)

    print(f"\n--- 3D Heatmap Diagnostic for class {class_idx} ---")
    print(f"Target layer      : {target_layer_name}")
    print(f"Logit             : {logits.item():.4f}  →  "
          f"prob {torch.sigmoid(logits).item():.4f}")
    print(f"fmap shape        : {fmap.shape} → 5D: {fmap_5d.shape}")
    print(f"Grad  min/mean/max: {grad.min().item():.4f} / "
          f"{grad.mean().item():.4f} / {grad.max().item():.4f}")
    print(f"Fmap  min/mean/max: {fmap.min().item():.4f} / "
          f"{fmap.mean().item():.4f} / {fmap.max().item():.4f}")

    # GradCAM++
    heatmap_pp = grad_cam_plusplus_3d(fmap_5d, grad_5d)
    max_pp     = heatmap_pp.max().item()
    print(f"GradCAM++ raw 3D heatmap min/max: "
          f"{heatmap_pp.min().item():.6f} / {max_pp:.6f}")

    if max_pp < 1e-6:
        print("→ GradCAM++ heatmap is ZERO. Falling back to standard GradCAM.")
        heatmap_3d = grad_cam_vanilla_3d(fmap_5d, grad_5d)
        print(f"GradCAM raw 3D min/max: "
              f"{heatmap_3d.min().item():.6f} / {heatmap_3d.max().item():.6f}")
    else:
        heatmap_3d = heatmap_pp

    # Take centre depth slice for 2D visualisation
    heatmap_3d_np = heatmap_3d.squeeze(0).cpu().numpy()   # (D, H, W)
    centre_d      = heatmap_3d_np.shape[0] // 2
    heatmap_2d    = heatmap_3d_np[centre_d]                # (H, W)

    hm_min, hm_max = heatmap_2d.min(), heatmap_2d.max()
    print(f"Centre-slice heatmap range: [{hm_min:.6f}, {hm_max:.6f}]")

    if hm_max - hm_min > 1e-8:
        heatmap_norm = (heatmap_2d - hm_min) / (hm_max - hm_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_2d)
        print("⚠ Heatmap is constant – normalised to all zeros.")

    heatmap_resized = cv2.resize(heatmap_norm, (224, 224))

    # Overlay on centre CT slice
    vol_np    = input_tensor.squeeze(0).squeeze(0).cpu().numpy()  # (D, H, W)
    ct_slice  = vol_np[vol_np.shape[0] // 2]                      # (H, W)
    ct_disp   = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
    ct_3ch    = np.stack([ct_disp] * 3, axis=-1)
    ct_224    = cv2.resize(ct_3ch, (224, 224))

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized),
                                      cv2.COLORMAP_JET)
    superimposed  = cv2.addWeighted(np.uint8(255 * ct_224), 0.5,
                                    heatmap_color, 0.5, 0)

    return heatmap_resized, superimposed


# ===========================================================================
# 5.  FROC UTILITY  (identical to TinyViT version)
# ===========================================================================
def compute_froc_score(df, fp_target=1.0):
    """
    Simplified FROC: sensitivity at ≤ fp_target false positives per scan.
    Mirrors the evaluation approach used for TinyViT.
    """
    thresholds = np.linspace(0, 1, 200)
    fps_per_scan_list, sens_list = [], []
    n_scans = df['seriesuid'].nunique()

    for thr in thresholds:
        preds = (df['prob'] >= thr).astype(int)
        tp = ((preds == 1) & (df['label'] == 1)).sum()
        fp = ((preds == 1) & (df['label'] == 0)).sum()
        fn = ((preds == 0) & (df['label'] == 1)).sum()
        sensitivity = tp / (tp + fn + 1e-8)
        fps_per_scan = fp / n_scans
        fps_per_scan_list.append(fps_per_scan)
        sens_list.append(sensitivity)

    fps_arr  = np.array(fps_per_scan_list)
    sens_arr = np.array(sens_list)
    idx      = np.argmin(np.abs(fps_arr - fp_target))
    return float(sens_arr[idx])


# ===========================================================================
# 6.  TRAINING  (mirrors TinyViT training loop exactly)
# ===========================================================================
def train_model():
    print(f"Using device: {DEVICE}")

    # ---- load metadata (same as TinyViT) ----
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict    = dict(zip(patient_split["seriesuid"],
                                 patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)

    print(f"Splits — Train: {len(train_meta)}  Val: {len(val_meta)}  "
          f"Test: {len(test_meta)}")

    # ---- transforms (3D) ----
    train_transforms = Compose([
        Resized(keys=["img"], spatial_size=(PATCH_SIZE,)*3, mode="trilinear"),
        RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["img"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["img"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["img"], prob=0.5, max_k=3),
        NormalizeIntensityd(keys=["img"], nonzero=True),
    ])
    val_transforms = Compose([
        Resized(keys=["img"], spatial_size=(PATCH_SIZE,)*3, mode="trilinear"),
        NormalizeIntensityd(keys=["img"], nonzero=True),
    ])

    train_dataset = NodulePatchDataset3D(train_meta, DATA_DIR, train_transforms)
    val_dataset   = NodulePatchDataset3D(val_meta,   DATA_DIR, val_transforms)
    test_dataset  = NodulePatchDataset3D(test_meta,  DATA_DIR, val_transforms)

    # ---- weighted sampler for class imbalance (same as TinyViT) ----
    labels        = train_meta["label"].values
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights= class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=NUM_WORKERS)

    # ---- model ----
    model = Swin3DWithMLPHead(dropout=DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Swin3D-Tiny parameters: {total_params/1e6:.2f}M\n")

    # ---- loss, optimizer, scheduler (identical to TinyViT) ----
    # pos_weight for BCEWithLogitsLoss
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ---- training loop ----
    best_auroc   = 0.0
    history      = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for patches, labels_t, _, _ in tqdm(train_loader,
                                            desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]",
                                            leave=False):
            patches  = patches.to(DEVICE)
            labels_t = labels_t.to(DEVICE)
            optimizer.zero_grad()
            logits = model(patches)
            loss   = criterion(logits, labels_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * patches.size(0)

        train_loss /= len(train_dataset)
        scheduler.step()

        # --- validate ---
        model.eval()
        val_loss = 0.0
        val_probs, val_labels = [], []
        with torch.no_grad():
            for patches, labels_v, _, _ in tqdm(val_loader,
                                                desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]",
                                                leave=False):
                patches  = patches.to(DEVICE)
                labels_v = labels_v.to(DEVICE)
                logits   = model(patches)
                loss     = criterion(logits, labels_v)
                val_loss += loss.item() * patches.size(0)
                probs     = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels_v.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auroc = roc_auc_score(val_labels, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUROC: {val_auroc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✔ Saved best model (AUROC={best_auroc:.4f})")

    print(f"\nTraining complete. Best Val AUROC: {best_auroc:.4f}")

    # ---- test evaluation (mirrors TinyViT) ----
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    test_probs, test_labels, test_uids = [], [], []
    with torch.no_grad():
        for patches, labels_t, uids, _ in tqdm(test_loader, desc="Testing"):
            patches = patches.to(DEVICE)
            logits  = model(patches)
            probs   = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            test_labels.extend(labels_t.numpy())
            test_uids.extend(uids)

    df_test = pd.DataFrame({
        'seriesuid': test_uids,
        'label'    : test_labels,
        'prob'     : test_probs,
    })

    # patient-level aggregation (max prob per patient — same as TinyViT)
    patient_pred = df_test.groupby('seriesuid').agg(
        prob =('prob',  'max'),
        label=('label', 'max')
    ).reset_index()
    patient_pred['pred'] = (patient_pred['prob'] >= 0.5).astype(int)

    auroc = roc_auc_score(patient_pred['label'], patient_pred['prob'])
    auprc = average_precision_score(patient_pred['label'], patient_pred['prob'])
    f1    = f1_score(patient_pred['label'], patient_pred['pred'])
    sens  = (((patient_pred['pred']==1) & (patient_pred['label']==1)).sum() /
              (patient_pred['label']==1).sum())
    spec  = (((patient_pred['pred']==0) & (patient_pred['label']==0)).sum() /
              (patient_pred['label']==0).sum())
    froc  = compute_froc_score(df_test, fp_target=1.0)

    print("\n" + "="*55)
    print("  TEST RESULTS — 3D Swin-Tiny")
    print("="*55)
    print(f"  AUROC       : {auroc:.4f}")
    print(f"  AUPRC       : {auprc:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  Sensitivity : {sens:.4f}")
    print(f"  Specificity : {spec:.4f}")
    print(f"  FROC @1FP   : {froc:.4f}")
    print("="*55)

    return model, test_dataset, test_meta, df_test


# ===========================================================================
# 7.  GRADCAM++ VISUALISATION  (mirrors TinyViT main exactly)
# ===========================================================================
def run_gradcam(model, test_dataset, test_meta, df_test):
    target_layer_name = find_target_layer_3d(model)

    # patient-level results
    patient_pred = df_test.groupby('seriesuid').agg(
        prob =('prob',  'max'),
        label=('label', 'max')
    ).reset_index()
    patient_pred['pred'] = (patient_pred['prob'] >= 0.5).astype(int)

    tp = patient_pred[(patient_pred['label']==1) & (patient_pred['pred']==1)]
    fn = patient_pred[(patient_pred['label']==1) & (patient_pred['pred']==0)]
    fp = patient_pred[(patient_pred['label']==0) & (patient_pred['pred']==1)]
    tn = patient_pred[(patient_pred['label']==0) & (patient_pred['pred']==0)]
    print(f"Available – TP:{len(tp)}  FN:{len(fn)}  FP:{len(fp)}  TN:{len(tn)}")

    # Select cases — same limits as TinyViT
    selected = []
    for i in range(min(4, len(tp))): selected.append(('TP', i+1, tp.iloc[i]))
    for i in range(min(3, len(fn))): selected.append(('FN', i+1, fn.iloc[i]))
    for i in range(min(3, len(fp))): selected.append(('FP', i+1, fp.iloc[i]))
    for i in range(min(3, len(tn))): selected.append(('TN', i+1, tn.iloc[i]))

    heatmaps, titles = [], []
    for case_type, case_num, row in selected:
        uid        = row['seriesuid']
        case_name  = f"{case_type}-{case_num}"
        best       = df_test[df_test['seriesuid'] == uid].loc[
                        df_test[df_test['seriesuid']==uid]['prob'].idxmax()]
        idx        = test_meta[test_meta['filepath']==best['filepath']].index[0]
        img_tensor, label, _, _ = test_dataset[idx]
        input_t    = img_tensor.unsqueeze(0).to(DEVICE)

        cidx = 1 if row['label'] == 1 else 0
        print(f"\n{'='*60}")
        print(f"Case: {case_name} | UID: {uid[:12]}... | "
              f"Prob: {row['prob']:.4f} | True label: {int(row['label'])}")
        print(f"{'='*60}")

        heatmap, overlay = produce_heatmap_3d(
            model, input_t, target_layer_name, class_idx=cidx)
        heatmaps.append((img_tensor, heatmap, overlay))
        titles.append(f"{case_name} | UID:{uid[:8]}... | "
                      f"Prob:{row['prob']:.3f} | Label:{int(row['label'])}")

    n = len(heatmaps)
    if n == 0:
        print("No cases to display.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, max(12, 4*n)))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (img_t, hm, ov) in enumerate(heatmaps):
        # show centre depth slice of the 3D volume
        vol_np   = img_t.squeeze(0).cpu().numpy()          # (D, H, W)
        ct_slice = vol_np[vol_np.shape[0] // 2]
        ct_disp  = (ct_slice - ct_slice.min()) / \
                   (ct_slice.max() - ct_slice.min() + 1e-8)

        axes[i, 0].imshow(ct_disp, cmap='gray')
        axes[i, 0].set_title("Original patch (centre slice)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(hm, cmap='jet')
        axes[i, 1].set_title("GradCAM++ heatmap (centre slice)")
        axes[i, 1].axis('off')

        ov_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)
        axes[i, 2].imshow(ov_rgb)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')

        fig.text(0.5, 0.02 + (n - i - 1) / n * 0.9,
                 titles[i], ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("gradcampp_swin3d.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: gradcampp_swin3d.png")


# ===========================================================================
# 8.  ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    model, test_dataset, test_meta, df_test = train_model()
    run_gradcam(model, test_dataset, test_meta, df_test)