# swin3d_tinyvit_training_only.py
"""
3D Swin Transformer Tiny for lung nodule classification.
Trained from scratch on 64x64x64 patches — training + test evaluation only (no GradCAM).
"""

import os
import warnings
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from monai.transforms import Compose, Resized, RandFlipd, RandRotate90d, NormalizeIntensityd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings("ignore")


# ===========================================================================
# 0.  CONFIGURATION
# ===========================================================================
DATA_DIR          = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH     = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH= os.path.join(DATA_DIR, "patient_split.csv")
MODEL_SAVE_PATH   = "best_model_swin3d.pth"

PATCH_SIZE        = 64          # cubic patch: 64 x 64 x 64
IN_CHANNELS       = 1           # single-channel CT volume (3D)
NUM_CLASSES       = 1           # binary: malignant vs benign
BATCH_SIZE        = 8
NUM_WORKERS       = 12
NUM_EPOCHS        = 200
LR                = 1e-4
WEIGHT_DECAY      = 1e-2
DROPOUT           = 0.5
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Swin-Tiny 3D architecture hyperparameters
EMBED_DIM         = 48
DEPTHS            = [2, 2, 6, 2]
NUM_HEADS         = [3, 6, 12, 24]
WINDOW_SIZE       = (4, 4, 4)
MLP_RATIO         = 4.0
QKV_BIAS          = True
DROP_PATH_RATE    = 0.2


# ===========================================================================
# 1.  DATASET  — 3D version
# ===========================================================================
class NodulePatchDataset3D(torch.utils.data.Dataset):
    """
    Loads a 64x64x64 .npy volume.
    Returns shape: (1, 64, 64, 64)
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

        return img, torch.tensor(label, dtype=torch.float32), row["seriesuid"]


# ===========================================================================
# 2.  3D SWIN TRANSFORMER (from scratch)
# ===========================================================================

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
    """
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        Wd, Wh, Ww = window_size
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
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q   = q * self.scale
        attn= q @ k.transpose(-2, -1)

        Wd, Wh, Ww = self.window_size
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(Wd*Wh*Ww, Wd*Wh*Ww, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
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
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=(4,4,4),
                 shift_size=(0,0,0), mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size

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

        windows = window_partition_3d(img_mask, self.window_size)
        windows = windows.view(-1, Wd * Wh * Ww)
        attn_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        if any(s > 0 for s in self.shift_size):
            Sd, Sh, Sw = self.shift_size
            x = torch.roll(x, shifts=(-Sd, -Sh, -Sw), dims=(1, 2, 3))

        windows = window_partition_3d(x, self.window_size)
        Wd, Wh, Ww = self.window_size
        windows = windows.view(-1, Wd * Wh * Ww, C)

        attn_out = self.attn(windows, mask=self.attn_mask)
        attn_out = attn_out.view(-1, Wd, Wh, Ww, C)
        x = window_reverse_3d(attn_out, self.window_size, D, H, W)

        if any(s > 0 for s in self.shift_size):
            Sd, Sh, Sw = self.shift_size
            x = torch.roll(x, shifts=(Sd, Sh, Sw), dims=(1, 2, 3))

        x = x.view(B, D * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    """Downsample by 2x in each spatial dim."""
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

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduce(x)
        return x


class PatchEmbed3D(nn.Module):
    """Non-overlapping 3D patch embedding with stride=patch_size."""
    def __init__(self, patch_size=2, in_channels=1, embed_dim=48):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (D, H, W)


class DropPath(nn.Module):
    """Stochastic depth."""
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
    3D Swin Transformer Tiny backbone.
    Input : (B, 1, 64, 64, 64)
    Output: 1D feature vector of size 384
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

        total_blocks = sum(depths)
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, total_blocks)]

        self.layers = nn.ModuleList()
        block_idx   = 0
        init_res = (64 // patch_size,) * 3

        for i_layer in range(self.num_layers):
            res = tuple(r // (2 ** i_layer) for r in init_res)
            dim = embed_dim * (2 ** i_layer)

            blocks = nn.ModuleList()
            for i_block in range(depths[i_layer]):
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

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = x.flatten(1)
        return x

    def forward(self, x):
        return self.forward_features(x)


class Swin3DWithMLPHead(nn.Module):
    """3D Swin Tiny backbone + MLP classifier head."""
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
# 3.  FROC UTILITY
# ===========================================================================
def compute_froc_score(df, fp_target=1.0):
    """Simplified FROC: sensitivity at ≤ fp_target false positives per scan."""
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
# 4.  TRAINING + TEST EVALUATION
# ===========================================================================
def main():
    print(f"Using device: {DEVICE}")

    # ---- load metadata ----
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

    # ---- weighted sampler for class imbalance ----
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

    # ---- loss, optimizer, scheduler ----
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
        for patches, labels_t, _ in tqdm(train_loader,
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
            for patches, labels_v, _ in tqdm(val_loader,
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

    # After you have val_probs and val_labels from the best model
    from sklearn.metrics import f1_score, roc_curve

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

# Find threshold that maximizes F1
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(val_labels, (val_probs >= t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"Optimal threshold from validation: {best_thresh:.4f}")

# Alternatively, use Youden's index from ROC curve
    fpr, tpr, roc_thresh = roc_curve(val_labels, val_probs)
    youden = tpr - fpr
    best_thresh_youden = roc_thresh[np.argmax(youden)]
    print(f"Youden threshold: {best_thresh_youden:.4f}")

    # ---- test evaluation ----
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    test_probs, test_labels, test_uids = [], [], []
    with torch.no_grad():
        for patches, labels_t, uids in tqdm(test_loader, desc="Testing"):
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

    # patient-level aggregation (max prob per patient)
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

    # Save test predictions to CSV
    df_test.to_csv("swin3d_test_predictions.csv", index=False)
    print("\nTest predictions saved to 'swin3d_test_predictions.csv'")


if __name__ == "__main__":
    main()