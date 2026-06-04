"""
3D Swin‑Tiny (self‑contained) with MLP Head – Lung Nodule Classification
Fully aligned with DenseNet121 / EfficientNet‑B0 / TinyViT scripts.
Includes input normalization, label smoothing, and optional pretrained loading.
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
    Compose, RandRotate90, RandFlip, RandGaussianNoise,
    NormalizeIntensityd                           # new: intensity normalisation
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

BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR = 5e-5                     # reduced from 1e‑4 – gentler for transformers
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10            # extended from 5 – more stable ramp‑up
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1         # new – prevents over‑confidence on negatives

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Pretrained model toggle
USE_PRETRAINED = True                    # set to True if you have a checkpoint
PRETRAINED_PATH = "swin3d_tiny_pretrained.pth"   # path to weights (state_dict)
USE_MONAI_PRETRAINED = True          # Set to True to load Swin‑UNETR backbone weights
MONAI_MMAR_ITEM = "wholeBody_ct_segmentation"   # MONAI model zoo item
MMAR_DIR = "./mmar_models"           # Directory where the pretrained model will be downloaded
# ================== Self‑contained 3D Swin‑Tiny ==================
# (DropPath, Mlp, WindowAttention, SwinTransformerBlock, PatchMerging,
#  BasicLayer, PatchEmbed3D, SwinTransformer3D – unchanged from previous fix)
# … included below for brevity – identical to the working version …

# ------------------------------ Fixed SwinTransformerBlock ------------------
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0],
                H // window_size[1], window_size[1],
                W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0]*window_size[1]*window_size[2], C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Pad to multiples of window size
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        D_p, H_p, W_p = x.shape[1], x.shape[2], x.shape[3]

        if any(self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))

        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, B, D_p, H_p, W_p)

        if any(self.shift_size):
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))

        x = x[:, :D, :H, :W, :]   # crop back to original
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)
    def forward(self, x):
        B, D, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0,0,0) if (i % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2)
            self.blocks.append(
                SwinTransformerBlock(dim=dim, num_heads=num_heads,
                                     window_size=window_size, shift_size=shift,
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            )
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)
    def forward(self, x):
        x = self.proj(x)                # (B, C, D, H, W)
        x = x.permute(0, 2, 3, 4, 1)   # (B, D, H, W, C)
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformer3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=48,
                 depths=(2,2,2,2), num_heads=(3,6,12,24),
                 window_size=(4,4,4), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim, norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                downsample=PatchMerging if (i_layer < len(depths)-1) else None,
                norm_layer=norm_layer
            )
            self.layers.append(layer)
        self.norm = norm_layer(int(embed_dim * 2 ** (len(depths)-1)))

    def forward(self, x):
        x = self.patch_embed(x)          # (B, D, H, W, C)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                 # (B, D, H, W, C)
        x = x.mean(dim=(1,2,3))          # global average pool -> (B, C)
        return x

# ------------------------------ Model --------------------------------------
class SwinTiny3DWithMLPHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5, pretrained_path=None):
        super().__init__()
        self.backbone = SwinTransformer3D(
            in_chans=in_channels,
            embed_dim=48,
            depths=(2,2,2,2),
            num_heads=(3,6,12,24),
            window_size=(4,4,4),
            patch_size=(4,4,4),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"[Pretrained] Loading weights from {pretrained_path}")
            state = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            # Handle possible missing keys in the head
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"[Pretrained] Loaded {len(pretrained_dict)} matching parameters.")
        elif pretrained_path is not None:
            print(f"[Pretrained] WARNING: Path '{pretrained_path}' not found. Training from scratch.")

    def forward(self, x):
        feat = self.backbone(x)          # (B, 384)
        logit = self.head(feat)          # (B, 1)
        return logit.squeeze(1)          # (B,)

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------ Dataset (unchanged) ------------------------------
# ------------------------------ Dataset ------------------------------------
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
        patch = np.load(local_path).astype(np.float32)      # (64,64,64)
        patch = np.expand_dims(patch, axis=0)               # (1,64,64,64)

        if self.transforms is not None:
            patch = self.transforms(patch)                  # apply array-level transforms

        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ------------------------------ Transforms (array‑level) --------------------
from monai.transforms import NormalizeIntensity   # array‑level normalisation

def get_transforms(mode="train"):
    if mode == "train":
        return Compose([
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandGaussianNoise(prob=0.2, std=0.01),
            NormalizeIntensity(nonzero=False, channel_wise=True),   # per‑image norm
        ])
    else:
        return Compose([
            NormalizeIntensity(nonzero=False, channel_wise=True),
        ])

# ------------------------------ Sanity check --------------------------------
def run_forward_sanity_check(model, device):
    print("\n[SANITY CHECK] Running forward pass on dummy input...")
    model.eval()
    # dummy must be normalized similarly; we use torch.randn (already ~N(0,1))
    dummy = torch.randn(2, 1, 64, 64, 64).to(device)
    with torch.no_grad():
        raw_out = model.backbone(dummy)
        print(f"  backbone output shape : {raw_out.shape}  (expected (2, 384))")
        has_nan = torch.isnan(raw_out).any().item()
        has_inf = torch.isinf(raw_out).any().item()
        print(f"  NaN in backbone out   : {has_nan}  (must be False)")
        print(f"  Inf in backbone out   : {has_inf}  (must be False)")
        logits = model(dummy)
        print(f"  final logit shape     : {logits.shape}  (expected (2,))")
        print(f"  logit range           : [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        probs = torch.sigmoid(logits)
        print(f"  sigmoid prob range    : [{probs.min().item():.4f}, {probs.max().item():.4f}]")
        if logits.abs().max().item() > 5.0:
            print("  [WARN] Large initial logits — check head initialisation")
        else:
            print("  Initial logit magnitude OK (< 5.0)")
    if has_nan or has_inf:
        raise RuntimeError("[SANITY CHECK FAILED] NaN/Inf in backbone output at init.")
    print("[SANITY CHECK] Passed.\n")
    model.train()

# ------------------------------ Evaluation (unchanged) ---------------------
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    descending_indices = np.argsort(all_probs)[::-1]
    sorted_labels = all_labels[descending_indices]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans
    sensitivity = tps / total_positives
    froc_scores = {}
    for target in FROC_THRESHOLDS:
        valid_idx = np.where(fps_per_scan <= target)[0]
        froc_scores[f"{target} FP/scan"] = float(sensitivity[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0
    return froc_scores

def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
    bootstrapped_scores = []
    rng = np.random.RandomState(SEED)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        bootstrapped_scores.append(roc_auc_score(np.array(y_true)[indices], np.array(y_probs)[indices]))
    s = np.sort(np.array(bootstrapped_scores))
    return np.percentile(s, 2.5), np.percentile(s, 97.5)

def get_youden_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j = np.argmax(tpr - fpr)
    return float(thresholds[j])

def evaluate_model(loader, model, device, pos_weight_tensor,
                   desc="Validating", return_preds=False, threshold=0.5,
                   label_smoothing=0.0):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    # Use standard BCEWithLogitsLoss, optionally with label smoothing
    if label_smoothing > 0:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        # We'll implement label smoothing manually in the loop
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc=desc):
            patches, labels = patches.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(patches)
                if label_smoothing > 0:
                    # smooth labels: 1 -> 1-smoothing, 0 -> smoothing
                    smooth_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
                    loss = F.binary_cross_entropy_with_logits(logits, smooth_labels,
                                                              pos_weight=pos_weight_tensor)
                else:
                    loss = criterion(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = [float(probs)]
                labels = [float(labels.cpu())]
            else:
                probs = probs.tolist()
                labels = labels.cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_uids.extend(uids)
            running_loss += loss.item() * patches.size(0)

    # Patient‑level aggregation
    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict[uid]['prob']  = max(patient_dict[uid]['prob'], prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)

    y_true_p  = [v['label'] for v in patient_dict.values()]
    y_probs_p = [v['prob']  for v in patient_dict.values()]
    y_pred_p  = [1 if p >= threshold else 0 for p in y_probs_p]
    total_scans = len(patient_dict)
    avg_loss = running_loss / len(loader.dataset)
    auc = roc_auc_score(y_true_p, y_probs_p) if len(np.unique(y_true_p)) > 1 else 0.5
    auprc = average_precision_score(y_true_p, y_probs_p)
    f1 = f1_score(y_true_p, y_pred_p, zero_division=0)
    sens = recall_score(y_true_p, y_pred_p, zero_division=0)
    cm = confusion_matrix(y_true_p, y_pred_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    froc = calculate_candidate_froc(all_labels, all_probs, total_scans)
    cpm = float(np.mean(list(froc.values())))

    pos_probs = [p for p, l in zip(all_probs, all_labels) if l == 1]
    neg_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
    print(f"  [Diag] pos: mean={np.mean(pos_probs):.4f} std={np.std(pos_probs):.4f} | "
          f"neg: mean={np.mean(neg_probs):.4f} std={np.std(neg_probs):.4f}")
    print(f"  [Diag] pred_pos={sum(y_pred_p)} pred_neg={len(y_pred_p)-sum(y_pred_p)} | "
          f"true_pos={sum(int(l) for l in y_true_p)} true_neg={sum(1-int(l) for l in y_true_p)}")

    if return_preds:
        ci_lower, ci_upper = calculate_95_ci(y_true_p, y_probs_p)
        print(f"  AUROC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_p, y_probs_p, n_bins=10
        )
        pd.DataFrame({
            'mean_predicted_probability': mean_predicted_value,
            'fraction_of_positives':      fraction_of_positives
        }).to_csv("swin3d_tiny_calibration_curve.csv", index=False)
        pred_df = pd.DataFrame({
            'seriesuid':   all_uids,
            'label':       all_labels,
            'probability': all_probs
        })
        return avg_loss, auc, auprc, f1, sens, spec, froc, cpm, pred_df
    return avg_loss, auc, auprc, f1, sens, spec, froc, cpm

# ------------------------------ Main ---------------------------------------
def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    print(f"Pretrained   : {USE_PRETRAINED}")
    print(f"LR           : {LR} (warmup {WARMUP_EPOCHS} ep)")
    print(f"Label smooth : {LABEL_SMOOTHING}")
    print(f"Input norm   : per‑image NormalizeIntensityd")

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

    # pos_weight
    train_df = metadata[metadata["split"] == "train"]
    neg_count = (train_df["label"] == 0).sum()
    pos_count = (train_df["label"] == 1).sum()
    pos_weight_value = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"Class balance (train): neg={neg_count}, pos={pos_count}, pos_weight={pos_weight_value:.2f}")

    train_transforms = get_transforms("train")
    val_transforms   = get_transforms("val")

    train_dataset = NodulePatchDataset(train_meta, DATA_DIR, transforms=train_transforms)
    val_dataset   = NodulePatchDataset(val_meta,   DATA_DIR, transforms=val_transforms)
    test_dataset  = NodulePatchDataset(test_meta,  DATA_DIR, transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)

    # Model (load pretrained if enabled)
    pretrained = PRETRAINED_PATH if USE_PRETRAINED else None
    model = SwinTiny3DWithMLPHead(pretrained_path=pretrained).to(device)

    if USE_MONAI_PRETRAINED:
        print("\n[Pretrained] Loading MONAI Swin‑UNETR backbone weights...")
        from monai.apps import load_from_mmar
        try:
            pretrained_model = load_from_mmar(
                item=MONAI_MMAR_ITEM,
                mmar_dir=MMAR_DIR,
                map_location='cpu',
                pretrained=True,
            )
        # The pretrained model is a SwinUNETR; extract its encoder (SwinTransformer)
            backbone_state = pretrained_model.swinViT.state_dict()
        # Load into our backbone, ignoring the missing head
            missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
            print(f"  Loaded backbone weights. Missing keys: {len(missing)} (expected head-related), "
              f"Unexpected keys: {len(unexpected)} (should be 0)")
            print("[Pretrained] Backbone initialized from Swin‑UNETR (BTCV CT segmentation).")
        except Exception as e:
            print(f"[Pretrained] Failed to load MONAI pretrained model: {e}")
            print("[Pretrained] Training will continue from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Model Summary ---")
    print(f"Model : 3D Swin‑Tiny (self‑contained) + MLP Head")
    print(f"Total params: {total_params:,}")
    print("---------------------")

    run_forward_sanity_check(model, device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # base loss; smoothing applied in evaluate and training loop
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - WARMUP_EPOCHS))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[WARMUP_EPOCHS])

    scaler = GradScaler('cuda', init_scale=256)
    best_val_auc = 0.0
    patience_counter = 0
    history = []
    total_skipped = 0

    print("--- Starting Training ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        grad_norms = []
        epoch_skips = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for patches, labels, _ in progress_bar:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(patches)
                if LABEL_SMOOTHING > 0:
                    smooth_labels = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
                    loss = F.binary_cross_entropy_with_logits(logits, smooth_labels,
                                                              pos_weight=pos_weight)
                else:
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if not torch.isfinite(total_norm):
                optimizer.zero_grad()
                scaler.update()
                epoch_skips += 1
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'skip': epoch_skips})
                continue
            grad_norms.append(total_norm.item())
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * patches.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        total_skipped += epoch_skips
        avg_train_loss = train_loss / max(len(train_loader.dataset) - epoch_skips * BATCH_SIZE, 1)
        avg_grad_norm = np.mean(grad_norms) if grad_norms else float('nan')
        max_grad_norm = np.max(grad_norms) if grad_norms else float('nan')
        current_scale = scaler.get_scale()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        skip_pct = epoch_skips / len(train_loader) * 100
        skip_str = f"skipped={epoch_skips}/{len(train_loader)} ({skip_pct:.1f}%)"
        if skip_pct > 10:
            skip_str += "  *** HIGH SKIP RATE ***"
        print(f"\n  [Grad] avg={avg_grad_norm:.4f}  max={max_grad_norm:.4f}  "
              f"scale={current_scale:.0f}  {skip_str}")

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc, val_cpm = \
            evaluate_model(val_loader, model, device, pos_weight, desc="Validating",
                           label_smoothing=0.0)   # no smoothing on validation

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
            torch.save(model.state_dict(), "best_model_swin3d_tiny.pth")
            print(f"  --> [Saved] New Best Val AUROC: {best_val_auc:.4f}")
            with open("best_metrics_swin3d_tiny.json", "w") as f:
                json.dump({
                    "epoch": epoch, "auc": val_auc, "auprc": val_auprc,
                    "f1": val_f1, "sensitivity": val_sens, "specificity": val_spec,
                    "cpm": val_cpm
                }, f, indent=4)
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping at epoch {epoch} ***")
                break

    print(f"\nTotal skipped gradient steps: {total_skipped}")
    pd.DataFrame(history).to_csv("swin3d_tiny_training_log.csv", index=False)
    print("[Saved] Training log -> 'swin3d_tiny_training_log.csv'")

    print("\n" + "="*55)
    print("  LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("="*55)
    model.load_state_dict(torch.load("best_model_swin3d_tiny.pth", map_location=device, weights_only=True))

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
    youden_threshold = get_youden_threshold(val_labels_all, val_probs_all) if len(np.unique(val_labels_all)) > 1 else 0.5
    print(f"  Youden threshold (val): {youden_threshold:.4f}")

    test_loss, test_auc, test_auprc, test_f1_05, test_sens_05, test_spec_05, \
        test_froc, test_cpm, pred_df = \
        evaluate_model(test_loader, model, device, pos_weight,
                       desc="Testing (t=0.50)", return_preds=True, threshold=0.5,
                       label_smoothing=0.0)

    pred_df.to_csv("swin3d_tiny_test_predictions.csv", index=False)
    print("[Saved] Test predictions -> 'swin3d_tiny_test_predictions.csv'")

    # Recompute at Youden threshold
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
    for fp_rate, sens in test_froc.items():
        print(f"  {fp_rate:12s} : {sens:.4f}")
    print(f"  CPM (mean)    : {test_cpm:.4f}")
    print("="*55)

    with open("final_test_metrics_swin3d_tiny.json", "w") as f:
        json.dump({
            "auroc": test_auc, "auprc": test_auprc, "cpm": test_cpm,
            "threshold_05":     {"f1": test_f1_05, "sensitivity": test_sens_05, "specificity": test_spec_05},
            "threshold_youden": {"value": youden_threshold, "f1": test_f1_y,
                                 "sensitivity": test_sens_y, "specificity": test_spec_y},
            "froc": test_froc,
        }, f, indent=4)
    print("[Saved] Final metrics -> 'final_test_metrics_swin3d_tiny.json'")

if __name__ == "__main__":
    main()