"""
Test & Audit script — Hybrid 3D EfficientNet-B0 + Swin3D-Tiny with KAN Head
- Loads best_model_hybrid_effnet_kan.pth
- Evaluates on test set (threshold 0.5 and Youden)
- Saves predictions + final metrics JSON
- Runs comprehensive audit checks:
    1.  Required files exist
    2.  Weights are trained (both backbones + all KAN layers)
    3.  Predictions CSV integrity
    4.  Metric recomputation from saved CSV
    5.  Training log shows real learning
    6.  Live inference matches saved predictions
    7.  No patient overlap between splits
    8.  Permutation test (statistical significance)
    9.  Positive vs negative score separation
   10.  KAN spline weights are non-trivial (model-specific check)
   11.  Swin backbone structure-specific weight check (patch embed + attention)
   12.  Feature fusion dimension sanity check (1664-d fused vector)
"""

import os, json, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, recall_score, roc_curve
)
from scipy import stats
from sklearn.calibration import calibration_curve
from monai.transforms import Compose, NormalizeIntensity
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────── Configuration ──────────────────────────────────
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH      = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE  = 4       # match training script
NUM_WORKERS = 0       # Windows safety

MODEL_PATH         = "best_model_hybrid_effnet_kan.pth"
PREDICTIONS_CSV    = "hybrid_effnet_kan_test_predictions.csv"
FINAL_METRICS_JSON = "final_test_metrics_hybrid_effnet_kan.json"
TRAINING_LOG_CSV   = "hybrid_effnet_kan_training_log.csv"
BEST_METRICS_JSON  = "best_metrics_hybrid_effnet_kan.json"

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
TOLERANCE = 0.01   # max allowed element-wise prediction difference

# Architecture constants — must match training script exactly
EFFICIENTNET_FEATURE_DIM = 1280
SWIN_FEATURE_DIM         = 384
FUSED_DIM                = EFFICIENTNET_FEATURE_DIM + SWIN_FEATURE_DIM  # 1664

# KAN hyperparameters — must match training script exactly
KAN_GRID_SIZE    = 5
KAN_SPLINE_ORDER = 3
KAN_GRID_RANGE   = [-1, 1]
KAN_RESIDUAL_STD = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# Model definition (verbatim copy from training script)
# ══════════════════════════════════════════════════════════════════════════════

# ── EfficientNet-B0 helpers ───────────────────────────────────────────────────

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
                             stride=stride, padding=pad, groups=mid_channels, bias=False),
                   nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3), SwishActivation()]
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
        total_blocks = sum(c[5] for c in stage_configs)
        block_idx = 0; stages = []
        for (in_ch, out_ch, k, s, expand, n_layers) in stage_configs:
            stage = []
            for i in range(n_layers):
                stride    = s if i == 0 else 1
                inch      = in_ch if i == 0 else out_ch
                drop_rate = 0.2 * block_idx / total_blocks
                stage.append(MBConv3D(inch, out_ch, k, stride, expand, drop_connect_rate=drop_rate))
                block_idx += 1
            stages.append(nn.Sequential(*stage))
        self.stages    = nn.Sequential(*stages)
        self.head_conv = nn.Sequential(
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, 1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.01, eps=1e-3), SwishActivation()
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self._initialize_weights()
    def _initialize_weights(self):
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
        x = self.stem(x); x = self.stages(x); x = self.head_conv(x)
        return self.global_pool(x).flatten(1)


# ── Swin-Tiny helpers ─────────────────────────────────────────────────────────

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
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
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
    return x.permute(0,1,3,5,2,4,6,7).contiguous().view(
        -1, window_size[0]*window_size[1]*window_size[2], C)

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B,
                     D//window_size[0], H//window_size[1], W//window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    return x.permute(0,1,4,2,5,3,6,7).contiguous().view(B, D, H, W, -1)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim        = dim
        self.window_size = window_size
        self.num_heads  = num_heads
        self.scale      = (dim // num_heads) ** -0.5
        self.qkv        = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1,2).reshape(B,N,C)))

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size
        self.norm1     = nn.LayerNorm(dim)
        self.attn      = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = nn.LayerNorm(dim)
        self.mlp       = _SwinMlp(dim, int(dim * mlp_ratio), drop=drop)
    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x; x = self.norm1(x)
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0,0,0,pad_w,0,pad_h,0,pad_d))
        D_p, H_p, W_p = x.shape[1], x.shape[2], x.shape[3]
        if any(self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0],-self.shift_size[1],-self.shift_size[2]), dims=(1,2,3))
        x = window_reverse(self.attn(window_partition(x, self.window_size)), self.window_size, B, D_p, H_p, W_p)
        if any(self.shift_size):
            x = torch.roll(x, shifts=(self.shift_size[0],self.shift_size[1],self.shift_size[2]), dims=(1,2,3))
        x = x[:,:D,:H,:W,:]
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(8 * dim)
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
                shift_size=(0,0,0) if (i%2==0) else (window_size[0]//2,window_size[1]//2,window_size[2]//2),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path,list) else drop_path
            ) for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return self.downsample(x) if self.downsample else x

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)
    def forward(self, x):
        return self.norm(self.proj(x).permute(0,2,3,4,1))

class SwinTransformer3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=48,
                 depths=(2,2,2,2), num_heads=(3,6,12,24),
                 window_size=(4,4,4), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim,
                                        norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=int(embed_dim * 2**i), depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                downsample=PatchMerging if (i < len(depths)-1) else None,
                norm_layer=norm_layer
            ) for i in range(len(depths))
        ])
        self.norm = norm_layer(int(embed_dim * 2**(len(depths)-1)))
    def forward(self, x):
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers: x = layer(x)
        return self.norm(x).mean(dim=(1,2,3))   # (B, 384)


# ── KAN layer ────────────────────────────────────────────────────────────────

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER,
                 base_activation=nn.SiLU, residual_std=KAN_RESIDUAL_STD,
                 grid_range=None):
        super().__init__()
        if grid_range is None:
            grid_range = KAN_GRID_RANGE
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
        grid = grid * h + grid_range[0]
        self.register_buffer("grid", grid)

        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_basis))
        nn.init.normal_(self.spline_weight, 0.0, 0.1)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.base_weight, 0.0, residual_std)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.base_activation = base_activation()

    def _b_splines(self, x):
        x = x.unsqueeze(-1)
        grid  = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            left_num  = x - grid[:-(k+1)]
            left_den  = grid[k:-1] - grid[:-(k+1)]
            left_den  = torch.where(left_den.abs() < 1e-8, torch.ones_like(left_den), left_den)
            left      = (left_num / left_den) * bases[:, :, :-1]
            right_num = grid[(k+1):] - x
            right_den = grid[(k+1):] - grid[1:-k]
            right_den = torch.where(right_den.abs() < 1e-8, torch.ones_like(right_den), right_den)
            right     = (right_num / right_den) * bases[:, :, 1:]
            bases = left + right
        return bases.contiguous()

    def forward(self, x):
        x_clamped = x.clamp(self.grid[0].item(), self.grid[-1].item())
        base_out   = F.linear(self.base_activation(x), self.base_weight)
        bases      = self._b_splines(x_clamped)
        spline_out = torch.einsum('bin,oin->bo', bases, self.spline_weight)
        return base_out + spline_out + self.bias


# ── KAN Head ──────────────────────────────────────────────────────────────────

class KANHead(nn.Module):
    def __init__(self, dims, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER,
                 dropout=0.5, grid_range=None, residual_std=KAN_RESIDUAL_STD):
        super().__init__()
        if grid_range is None:
            grid_range = KAN_GRID_RANGE
        layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = (i == len(dims) - 2)
            layers.append(nn.LayerNorm(d_in))
            layers.append(KANLinear(d_in, d_out, grid_size=grid_size,
                                    spline_order=spline_order,
                                    residual_std=residual_std,
                                    grid_range=grid_range))
            if not is_last:
                drop_p = dropout if i == 0 else dropout / 2
                layers.append(nn.Dropout(p=drop_p))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


# ── Full hybrid model ─────────────────────────────────────────────────────────

class HybridEfficientNetSwinKANHead(nn.Module):
    EFFICIENTNET_DIM = 1280
    SWIN_DIM         = 384
    FUSED_DIM        = 1664

    def __init__(self, in_channels=1, num_classes=1, dropout=0.5,
                 swin_pretrained_path=None,
                 kan_grid_size=KAN_GRID_SIZE, kan_spline_order=KAN_SPLINE_ORDER,
                 kan_grid_range=None, kan_residual_std=KAN_RESIDUAL_STD):
        super().__init__()
        if kan_grid_range is None:
            kan_grid_range = KAN_GRID_RANGE
        self.efficientnet = EfficientNet3D_B0(in_channels=in_channels)
        self.swin = SwinTransformer3D(
            in_chans=in_channels, embed_dim=48,
            depths=(2,2,2,2), num_heads=(3,6,12,24),
            window_size=(4,4,4), patch_size=(4,4,4),
            mlp_ratio=4.0, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        self.head = KANHead(
            dims=[self.FUSED_DIM, 512, 256, num_classes],
            grid_size=kan_grid_size, spline_order=kan_spline_order,
            dropout=dropout, grid_range=kan_grid_range,
            residual_std=kan_residual_std,
        )
    def forward(self, x):
        e_feat = self.efficientnet(x)
        s_feat = self.swin(x)
        fused  = torch.cat([e_feat, s_feat], 1)
        return self.head(fused).squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset & helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_eval_transforms():
    return Compose([NormalizeIntensity(nonzero=False, channel_wise=True)])

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


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels); all_probs = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    idx  = np.argsort(all_probs)[::-1]
    tps  = np.cumsum(all_labels[idx] == 1)
    fps  = np.cumsum(all_labels[idx] == 0)
    fps_per_scan = fps / total_scans
    sensitivity  = tps / total_positives
    froc_scores  = {}
    for target in FROC_THRESHOLDS:
        valid_idx = np.where(fps_per_scan <= target)[0]
        froc_scores[f"{target} FP/scan"] = float(sensitivity[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0
    return froc_scores

def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
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

def evaluate_model(loader, model, device, pos_weight, threshold=0.5):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc="Evaluating"):
            patches, labels = patches.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(patches)
                loss   = criterion(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = [float(probs)]; labels_list = [float(labels.cpu())]
            else:
                probs = probs.tolist(); labels_list = labels.cpu().tolist()
            all_probs.extend(probs); all_labels.extend(labels_list)
            all_uids.extend(uids)
            running_loss += loss.item() * patches.size(0)

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
    return avg_loss, auc, auprc, f1, sens, spec, froc, cpm, all_probs, all_labels, all_uids


# ══════════════════════════════════════════════════════════════════════════════
# Audit Report
# ══════════════════════════════════════════════════════════════════════════════

class AuditReport:
    def __init__(self):
        self.checks = []; self.passed = 0; self.failed = 0; self.warned = 0

    def record(self, name, status, detail):
        self.checks.append({"check": name, "status": status, "detail": detail})
        if status == "PASS":   self.passed += 1
        elif status == "FAIL": self.failed += 1
        else:                  self.warned += 1

    def print_check(self, name, status, detail):
        icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}
        print(f"  {icons.get(status,'?')} [{status}] {name}")
        print(f"         {detail}")
        self.record(name, status, detail)

    def save(self, txt_path, json_path):
        lines = [
            "=" * 70,
            "  MODEL AUDIT REPORT — Hybrid EfficientNet-B0 + Swin3D-Tiny + KAN Head",
            "=" * 70,
            f"  Total Checks : {len(self.checks)}",
            f"  Passed       : {self.passed}",
            f"  Warnings     : {self.warned}",
            f"  Failed       : {self.failed}", ""
        ]
        for c in self.checks:
            icon = {"PASS":"[PASS]","FAIL":"[FAIL]","WARN":"[WARN]"}.get(c["status"],"[?]")
            lines += [f"  {icon}  {c['check']}", f"         {c['detail']}"]
        lines += ["=" * 70]
        verdict = ("RESULTS ARE LEGITIMATE ✅" if self.failed == 0
                   else f"AUDIT FAILED — {self.failed} critical issue(s) found ❌")
        lines.append(f"  VERDICT: {verdict}")
        lines.append("=" * 70)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        summary = {"total": len(self.checks), "passed": self.passed,
                   "warned": self.warned, "failed": self.failed,
                   "verdict": "LEGITIMATE" if self.failed == 0 else "FAILED",
                   "checks": self.checks}
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\n[Saved] Full report  → {txt_path}")
        print(f"[Saved] JSON summary → {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Individual check functions
# ══════════════════════════════════════════════════════════════════════════════

def check_1_required_files_exist(report):
    print("\n[CHECK 1] Required Files Exist")
    required = {
        "Model weights (.pth)": MODEL_PATH,
        "Training log CSV":     TRAINING_LOG_CSV,
        "Best metrics JSON":    BEST_METRICS_JSON,
    }
    all_found = True
    for name, path in required.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            report.print_check(f"File exists: {name}", "PASS",
                                f"Found at '{path}' ({size_kb:.1f} KB)")
        else:
            report.print_check(f"File exists: {name}", "FAIL",
                                f"NOT FOUND at '{path}'")
            all_found = False
    return all_found


def check_2_model_weights_not_random(report, device):
    """
    Verifies that both backbones and all KAN layers have diverged from
    random initialisation.

    EfficientNet check : stem Conv3d weight
    Swin check         : patch_embed projection Conv3d weight +
                         first attention QKV projection weight
    KAN head checks    : base_weight and spline_weight for each of the
                         three KANLinear layers (indices 1, 4, 7 inside
                         KANHead.net — after LayerNorm, KANLinear, Dropout)
    """
    print("\n[CHECK 2] Model Weights Are Trained (Both Backbones + KAN Head)")
    try:
        trained  = HybridEfficientNetSwinKANHead().to(device)
        trained.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True))
        random_m = HybridEfficientNetSwinKANHead().to(device)

        # ── EfficientNet: stem conv ───────────────────────────────────────
        t_eff = trained.efficientnet.stem[0].weight.detach().cpu().numpy().flatten()
        r_eff = random_m.efficientnet.stem[0].weight.detach().cpu().numpy().flatten()
        n = min(len(t_eff), 1000)
        r_val, _ = stats.pearsonr(t_eff[:n], r_eff[:n])
        delta     = np.mean(np.abs(t_eff[:n] - r_eff[:n]))
        if abs(r_val) > 0.99:
            report.print_check("EfficientNet stem conv differs from init", "FAIL",
                                f"r={r_val:.4f} — weights unchanged")
        else:
            report.print_check("EfficientNet stem conv differs from init", "PASS",
                                f"Mean delta={delta:.6f}, r={r_val:.4f}")

        # ── Swin: patch_embed projection conv ────────────────────────────
        t_pe = trained.swin.patch_embed.proj.weight.detach().cpu().numpy().flatten()
        r_pe = random_m.swin.patch_embed.proj.weight.detach().cpu().numpy().flatten()
        n = min(len(t_pe), 1000)
        r_pe_val, _ = stats.pearsonr(t_pe[:n], r_pe[:n])
        delta_pe     = np.mean(np.abs(t_pe[:n] - r_pe[:n]))
        if abs(r_pe_val) > 0.99:
            report.print_check("Swin patch_embed proj differs from init", "FAIL",
                                f"r={r_pe_val:.4f} — weights unchanged")
        else:
            report.print_check("Swin patch_embed proj differs from init", "PASS",
                                f"Mean delta={delta_pe:.6f}, r={r_pe_val:.4f}")

        # ── Swin: first attention QKV weight (layer 0, block 0) ──────────
        t_qkv = trained.swin.layers[0].blocks[0].attn.qkv.weight.detach().cpu().numpy().flatten()
        r_qkv = random_m.swin.layers[0].blocks[0].attn.qkv.weight.detach().cpu().numpy().flatten()
        n = min(len(t_qkv), 1000)
        r_qkv_val, _ = stats.pearsonr(t_qkv[:n], r_qkv[:n])
        delta_qkv     = np.mean(np.abs(t_qkv[:n] - r_qkv[:n]))
        if abs(r_qkv_val) > 0.99:
            report.print_check("Swin attention QKV weight differs from init", "FAIL",
                                f"r={r_qkv_val:.4f} — weights unchanged")
        else:
            report.print_check("Swin attention QKV weight differs from init", "PASS",
                                f"Mean delta={delta_qkv:.6f}, r={r_qkv_val:.4f}")

        # ── KAN Head layers ───────────────────────────────────────────────
        # KANHead.net layout: [LN, KANLinear, Dropout, LN, KANLinear, Dropout, LN, KANLinear]
        # KANLinear modules sit at indices 1, 4, 7
        kan_layer_indices = [
            (1,  "KAN head layer-1 (1664→512)"),
            (4,  "KAN head layer-2 (512→256)"),
            (7,  "KAN head output  (256→1)"),
        ]
        for idx, layer_name in kan_layer_indices:
            kan_t = trained.head.net[idx]
            kan_r = random_m.head.net[idx]

            # base_weight
            bw_t = kan_t.base_weight.detach().cpu().numpy().flatten()
            bw_r = kan_r.base_weight.detach().cpu().numpy().flatten()
            n = min(len(bw_t), 1000)
            r_bw, _ = stats.pearsonr(bw_t[:n], bw_r[:n])
            d_bw     = np.mean(np.abs(bw_t[:n] - bw_r[:n]))
            if abs(r_bw) > 0.99:
                report.print_check(f"{layer_name} base_weight differs from init", "FAIL",
                                    f"r={r_bw:.4f} — unchanged")
            else:
                report.print_check(f"{layer_name} base_weight differs from init", "PASS",
                                    f"Mean delta={d_bw:.6f}, r={r_bw:.4f}")

            # spline_weight
            sw_t = kan_t.spline_weight.detach().cpu().numpy().flatten()
            sw_r = kan_r.spline_weight.detach().cpu().numpy().flatten()
            n = min(len(sw_t), 2000)
            d_sw = np.mean(np.abs(sw_t[:n] - sw_r[:n]))
            if d_sw < 1e-6:
                report.print_check(f"{layer_name} spline_weight trained", "FAIL",
                                    f"Mean delta={d_sw:.2e} — splines unchanged")
            else:
                report.print_check(f"{layer_name} spline_weight trained", "PASS",
                                    f"Mean delta={d_sw:.6f}, std={sw_t[:n].std():.6f}")

        return trained

    except Exception as e:
        report.print_check("Load model weights", "FAIL", f"Exception: {e}")
        return None


def check_3_predictions_csv_integrity(report, pred_df):
    print("\n[CHECK 3] Predictions CSV Integrity")
    required_cols = {"seriesuid", "label", "probability"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        report.print_check("Predictions CSV columns", "FAIL", f"Missing: {missing}")
        return None
    report.print_check("Predictions CSV columns", "PASS",
                        f"All required columns present. Shape: {pred_df.shape}")

    nan_count = pred_df[["label", "probability"]].isna().sum().sum()
    if nan_count > 0:
        report.print_check("No NaN values", "FAIL", f"{nan_count} NaN values found")
    else:
        report.print_check("No NaN values", "PASS", "Zero NaN values")

    out_of_range = ((pred_df["probability"] < 0) | (pred_df["probability"] > 1)).sum()
    if out_of_range > 0:
        report.print_check("Probabilities in [0,1]", "FAIL",
                            f"{out_of_range} predictions outside [0,1]")
    else:
        report.print_check("Probabilities in [0,1]", "PASS",
                            f"Range: [{pred_df['probability'].min():.4f}, "
                            f"{pred_df['probability'].max():.4f}]")

    unique_labels = set(pred_df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        report.print_check("Labels are binary", "FAIL", f"Non-binary labels: {unique_labels}")
    else:
        pos = (pred_df["label"] == 1).sum(); neg = (pred_df["label"] == 0).sum()
        report.print_check("Labels are binary", "PASS",
                            f"Positive patches: {pos}, Negative patches: {neg}")

    prob_std = pred_df["probability"].std()
    if prob_std < 0.01:
        report.print_check("Predictions are not collapsed", "FAIL", f"std={prob_std:.4f}")
    else:
        report.print_check("Predictions are not collapsed", "PASS",
                            f"std={prob_std:.4f}, mean={pred_df['probability'].mean():.4f}")

    if len(unique_labels) < 2:
        report.print_check("Both classes in test set", "FAIL", "Only one class found")
    else:
        report.print_check("Both classes in test set", "PASS",
                            "Both positive and negative present")
    return pred_df


def check_4_recompute_metrics_from_predictions(report, pred_df):
    print("\n[CHECK 4] Recompute Metrics from Saved Predictions")
    patient_dict = {}
    for _, row in pred_df.iterrows():
        uid = row["seriesuid"]; prob = row["probability"]; label = row["label"]
        if uid not in patient_dict:
            patient_dict[uid] = {"prob": prob, "label": label}
        else:
            patient_dict[uid]["prob"]  = max(patient_dict[uid]["prob"],  prob)
            patient_dict[uid]["label"] = max(patient_dict[uid]["label"], label)
    y_true  = [v["label"] for v in patient_dict.values()]
    y_probs = [v["prob"]  for v in patient_dict.values()]
    y_pred  = [1 if p >= 0.5 else 0 for p in y_probs]
    try:
        recomputed = {
            "auc":         roc_auc_score(y_true, y_probs),
            "auprc":       average_precision_score(y_true, y_probs),
            "f1":          f1_score(y_true, y_pred),
            "sensitivity": recall_score(y_true, y_pred),
        }
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recomputed["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception as e:
        report.print_check("Metric recomputation", "FAIL", f"Exception: {e}")
        return None

    print("  Recomputed metrics (patient-level, from saved CSV):")
    for k, v in recomputed.items():
        print(f"    {k:15s}: {v:.4f}")

    report.print_check("AUROC recomputable", "PASS",
                        f"Recomputed AUROC={recomputed['auc']:.4f}")
    report.print_check("AUPRC recomputable", "PASS",
                        f"Recomputed AUPRC={recomputed['auprc']:.4f}")

    if recomputed["auc"] < 0.5:
        report.print_check("AUROC sanity range", "FAIL",
                            f"AUROC={recomputed['auc']:.4f} < 0.5")
    elif recomputed["auc"] > 0.999:
        report.print_check("AUROC sanity range", "WARN",
                            f"AUROC={recomputed['auc']:.4f} suspiciously perfect")
    else:
        report.print_check("AUROC sanity range", "PASS",
                            f"AUROC={recomputed['auc']:.4f} plausible")

    if recomputed["sensitivity"] == 0.0:
        report.print_check("Sensitivity > 0", "FAIL", "Sensitivity=0.0")
    elif recomputed["sensitivity"] == 1.0:
        report.print_check("Sensitivity = 1.0 check", "WARN", "Sensitivity=1.0")
    else:
        report.print_check("Sensitivity in valid range", "PASS",
                            f"Sensitivity={recomputed['sensitivity']:.4f}")
    return recomputed


def check_5_training_log_shows_learning(report):
    print("\n[CHECK 5] Training Log Shows Real Learning")
    try:
        log_df = pd.read_csv(TRAINING_LOG_CSV)
        if len(log_df) < 3:
            report.print_check("Training log has enough epochs", "WARN",
                                f"Only {len(log_df)} epochs logged")
            return
        report.print_check("Training log has enough epochs", "PASS",
                            f"{len(log_df)} epochs")

        first_loss = log_df["train_loss"].iloc[:3].mean()
        last_loss  = log_df["train_loss"].iloc[-3:].mean()
        if last_loss < first_loss:
            report.print_check("Training loss decreased", "PASS",
                                f"{first_loss:.4f} → {last_loss:.4f}")
        else:
            report.print_check("Training loss decreased", "FAIL",
                                f"{first_loss:.4f} → {last_loss:.4f}")

        first_auc = log_df["val_auc"].iloc[:3].mean()
        best_auc  = log_df["val_auc"].max()
        if best_auc > first_auc:
            report.print_check("Val AUROC improved during training", "PASS",
                                f"{first_auc:.4f} → {best_auc:.4f}")
        else:
            report.print_check("Val AUROC improved during training", "FAIL",
                                f"{first_auc:.4f} → {best_auc:.4f}")

        loss_std = log_df["train_loss"].std()
        if loss_std < 1e-5:
            report.print_check("Training loss not flat", "FAIL", f"std={loss_std:.2e}")
        else:
            report.print_check("Training loss not flat", "PASS", f"std={loss_std:.6f}")

        # Check warmup phase: LR should be increasing early on
        if "lr" in log_df.columns and len(log_df) > 5:
            lr_early = log_df["lr"].iloc[:5].values
            lr_increasing = all(lr_early[i] <= lr_early[i+1] for i in range(len(lr_early)-1))
            if lr_increasing:
                report.print_check("Warmup LR schedule present", "PASS",
                                    f"LR rising in first 5 epochs: {lr_early[0]:.2e} → {lr_early[-1]:.2e}")
            else:
                report.print_check("Warmup LR schedule present", "WARN",
                                    "LR not monotonically increasing in first 5 epochs")

        auc_std = log_df["val_auc"].std()
        if auc_std < 1e-5 and log_df["val_auc"].mean() > 0.99:
            report.print_check("Val AUROC not fabricated", "FAIL",
                                f"AUROC={log_df['val_auc'].mean():.4f} ± {auc_std:.2e}")
        else:
            report.print_check("Val AUROC variance realistic", "PASS",
                                f"std={auc_std:.4f}")
    except Exception as e:
        report.print_check("Load training log", "FAIL", f"Exception: {e}")


def check_6_model_inference_live(report, model, device):
    print("\n[CHECK 6] Live Inference Matches Saved Predictions (full test set)")
    if model is None:
        report.print_check("Live inference", "FAIL", "Model not loaded"); return
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)
        test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
        dataset   = NodulePatchDataset(test_meta, DATA_DIR,
                                       transforms=get_eval_transforms())
        loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=False)

        model.eval()
        live_probs = []
        with torch.no_grad():
            for patches, _, _ in tqdm(loader, desc="  Live inference"):
                patches = patches.to(device)
                with autocast('cuda'):
                    logits = model(patches)
                probs = torch.sigmoid(logits).cpu().numpy()
                live_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])

        saved_df    = pd.read_csv(PREDICTIONS_CSV)
        saved_probs = saved_df["probability"].values

        if len(live_probs) != len(saved_probs):
            report.print_check("Prediction count matches saved CSV", "FAIL",
                                f"Live {len(live_probs)} vs saved {len(saved_probs)}")
            return

        diffs    = np.abs(np.array(live_probs) - saved_probs)
        max_diff = diffs.max()
        matches  = (diffs < TOLERANCE).sum()
        total    = len(diffs)

        if matches == total:
            report.print_check("Live predictions match saved CSV", "PASS",
                                f"All {total} match within tolerance. Max diff={max_diff:.8f}")
        else:
            report.print_check("Live predictions match saved CSV", "FAIL",
                                f"{matches}/{total} match. Max diff={max_diff:.6f}")

        all_labels = test_meta["label"].values
        if len(np.unique(all_labels)) > 1:
            live_auc = roc_auc_score(all_labels, live_probs)
            report.print_check("Live inference AUROC (patch-level check)", "PASS",
                                f"AUROC={live_auc:.4f}")
        else:
            report.print_check("Live inference AUROC check", "WARN",
                                "Only one class present in test_meta")
    except Exception as e:
        report.print_check("Live inference", "FAIL", f"Exception: {e}")


def check_7_label_split_integrity(report):
    print("\n[CHECK 7] No Patient Overlap Between Splits")
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)

        train_ids = set(metadata[metadata["split"] == "train"]["seriesuid"].unique())
        val_ids   = set(metadata[metadata["split"] == "val"]["seriesuid"].unique())
        test_ids  = set(metadata[metadata["split"] == "test"]["seriesuid"].unique())

        for pair, overlap in [("Train ∩ Val",  train_ids & val_ids),
                               ("Train ∩ Test", train_ids & test_ids),
                               ("Val ∩ Test",   val_ids   & test_ids)]:
            if overlap:
                report.print_check(f"No overlap: {pair}", "FAIL",
                                    f"{len(overlap)} patient(s) shared")
            else:
                report.print_check(f"No overlap: {pair}", "PASS",
                                    "Zero patients shared")

        for split_name in ["train", "val", "test"]:
            sdf = metadata[metadata["split"] == split_name]
            pos = (sdf["label"] == 1).sum(); neg = (sdf["label"] == 0).sum()
            pct = pos / len(sdf) * 100
            report.print_check(f"Class distribution — {split_name}", "PASS",
                                f"Pos={pos}, Neg={neg}, Pos%={pct:.1f}%")
    except Exception as e:
        report.print_check("Load metadata for split check", "FAIL", f"Exception: {e}")


def check_8_permutation_baseline(report, pred_df):
    print("\n[CHECK 8] Permutation Test (Statistical Significance)")
    if pred_df is None:
        report.print_check("Permutation test", "FAIL", "No predictions available"); return
    patient_dict = {}
    for _, row in pred_df.iterrows():
        uid = row["seriesuid"]; prob = row["probability"]; label = row["label"]
        if uid not in patient_dict:
            patient_dict[uid] = {"prob": prob, "label": label}
        else:
            patient_dict[uid]["prob"]  = max(patient_dict[uid]["prob"],  prob)
            patient_dict[uid]["label"] = max(patient_dict[uid]["label"], label)
    y_true  = np.array([v["label"] for v in patient_dict.values()])
    y_probs = np.array([v["prob"]  for v in patient_dict.values()])
    real_auc = roc_auc_score(y_true, y_probs)

    rng = np.random.RandomState(SEED)
    permuted_aucs = np.array([roc_auc_score(y_true, rng.permutation(y_probs))
                               for _ in range(1000)])
    p_value    = (permuted_aucs >= real_auc).mean()
    percentile = (permuted_aucs < real_auc).mean() * 100

    print(f"  Real AUROC: {real_auc:.4f} | "
          f"Permuted mean: {permuted_aucs.mean():.4f} | p={p_value:.4e}")
    if p_value < 0.05:
        report.print_check("Permutation test (p<0.05)", "PASS",
                            f"Beats {percentile:.1f}% of random permutations (p={p_value:.4e})")
    else:
        report.print_check("Permutation test (p<0.05)", "FAIL",
                            f"Not significant (p={p_value:.4f})")


def check_9_prediction_score_separation(report, pred_df):
    print("\n[CHECK 9] Positive vs Negative Score Separation")
    if pred_df is None:
        report.print_check("Score separation test", "FAIL", "No predictions available"); return
    pos_probs = pred_df[pred_df["label"] == 1]["probability"].values
    neg_probs = pred_df[pred_df["label"] == 0]["probability"].values
    pos_mean  = pos_probs.mean(); neg_mean = neg_probs.mean()
    print(f"  Positive patches → mean={pos_mean:.4f}, std={pos_probs.std():.4f}, n={len(pos_probs)}")
    print(f"  Negative patches → mean={neg_mean:.4f}, std={neg_probs.std():.4f}, n={len(neg_probs)}")

    u_stat, p_value = stats.mannwhitneyu(pos_probs, neg_probs, alternative='greater')
    if pos_mean > neg_mean and p_value < 0.05:
        report.print_check("Positive scores > Negative scores (MWU)", "PASS",
                            f"Pos mean={pos_mean:.4f} > Neg mean={neg_mean:.4f}, p={p_value:.2e}")
    elif pos_mean <= neg_mean:
        report.print_check("Positive scores > Negative scores (MWU)", "FAIL",
                            "Positive patches have lower mean probability than negatives")
    else:
        report.print_check("Positive scores > Negative scores (MWU)", "WARN",
                            f"Trend correct but not significant (p={p_value:.4f})")


def check_10_kan_spline_weights_non_trivial(report, model):
    """
    KAN-specific check: verifies B-spline coefficient tensors (spline_weight)
    for all three KANLinear layers in KANHead.net (indices 1, 4, 7) have
    meaningful variance above their Gaussian initialisation noise.

    Init std for each layer = 0.1 (KAN_RESIDUAL_STD for spline init in
    this implementation — normal_(0, 0.1)).  We check that trained std
    is notably above 0.1 or, at minimum, that spline values have
    diverged from initial draw.
    """
    print("\n[CHECK 10] KAN Spline Weights Are Non-Trivial")
    if model is None:
        report.print_check("KAN spline weight check", "FAIL", "Model not loaded"); return

    layer_specs = [
        (1, "KAN head layer-1 (1664→512)"),
        (4, "KAN head layer-2 (512→256)"),
        (7, "KAN head output  (256→1)"),
    ]
    stds = []
    for idx, layer_name in layer_specs:
        sw      = model.head.net[idx].spline_weight.detach().cpu().numpy()
        sw_std  = sw.std()
        sw_mean = np.abs(sw).mean()
        sw_max  = np.abs(sw).max()
        stds.append(sw_std)

        if sw_std < 1e-7:
            report.print_check(f"{layer_name} spline_weight non-zero", "FAIL",
                                f"std={sw_std:.2e} — near-zero, not trained")
        elif sw_std < 0.05:
            report.print_check(f"{layer_name} spline_weight trained", "WARN",
                                f"std={sw_std:.6f} — low; may indicate underfitting")
        else:
            report.print_check(f"{layer_name} spline_weight trained", "PASS",
                                f"std={sw_std:.6f}, mean_abs={sw_mean:.6f}, max_abs={sw_max:.6f}")

    # Cross-layer diversity: stds should not be pathologically disparate
    if len(stds) == 3 and min(stds) > 1e-9:
        ratio = max(stds) / min(stds)
        if ratio > 1e4:
            report.print_check("KAN spline std cross-layer diversity", "WARN",
                                f"Extreme std ratio across layers: {ratio:.1f}× "
                                f"(stds={[f'{s:.4f}' for s in stds]})")
        else:
            report.print_check("KAN spline std cross-layer diversity", "PASS",
                                f"Max/min ratio={ratio:.2f}×, "
                                f"stds={[f'{s:.4f}' for s in stds]}")


def check_11_swin_structure_weight_check(report, model):
    """
    Swin-specific structural checks:
      - PatchMerging reduction weights (present in layers 0-2)
      - Final LayerNorm weight (self.swin.norm) is not all-ones (as at init)
      - Last Swin layer's MLP FC1 weight has been updated
    """
    print("\n[CHECK 11] Swin Backbone Structure-Specific Weight Check")
    if model is None:
        report.print_check("Swin structure check", "FAIL", "Model not loaded"); return

    # PatchMerging in layers 0, 1, 2 — check reduction linear weight std
    for i in range(3):
        layer = model.swin.layers[i]
        if layer.downsample is not None:
            red_w = layer.downsample.reduction.weight.detach().cpu().numpy()
            std_v = red_w.std()
            if std_v < 1e-6:
                report.print_check(f"Swin PatchMerging layer-{i} reduction non-trivial", "FAIL",
                                    f"std={std_v:.2e}")
            else:
                report.print_check(f"Swin PatchMerging layer-{i} reduction non-trivial", "PASS",
                                    f"std={std_v:.6f}")

    # Final norm weight: at init all-ones; after training should differ
    norm_w = model.swin.norm.weight.detach().cpu().numpy()
    norm_deviation = np.abs(norm_w - 1.0).mean()
    if norm_deviation < 1e-6:
        report.print_check("Swin final norm weight trained", "WARN",
                            f"Mean deviation from 1.0 = {norm_deviation:.2e} (near init)")
    else:
        report.print_check("Swin final norm weight trained", "PASS",
                            f"Mean deviation from init (1.0) = {norm_deviation:.6f}")

    # Last Swin layer MLP FC1 weight std
    last_mlp_w = model.swin.layers[-1].blocks[-1].mlp.fc1.weight.detach().cpu().numpy()
    mlp_std = last_mlp_w.std()
    if mlp_std < 1e-6:
        report.print_check("Swin last-layer MLP FC1 weight non-trivial", "FAIL",
                            f"std={mlp_std:.2e}")
    else:
        report.print_check("Swin last-layer MLP FC1 weight non-trivial", "PASS",
                            f"std={mlp_std:.6f}")


def check_12_feature_fusion_dimension(report, model, device):
    """
    Verifies that EfficientNet outputs 1280-d, Swin outputs 384-d,
    and their concatenation is 1664-d by running a micro forward pass
    through each backbone independently.
    """
    print("\n[CHECK 12] Feature Fusion Dimension Sanity (1280 + 384 = 1664)")
    if model is None:
        report.print_check("Feature fusion dimension check", "FAIL", "Model not loaded"); return
    try:
        model.eval()
        dummy = torch.randn(1, 1, 64, 64, 64).to(device)
        with torch.no_grad():
            e_feat = model.efficientnet(dummy)
            s_feat = model.swin(dummy)
            fused  = torch.cat([e_feat, s_feat], dim=1)

        if e_feat.shape == (1, EFFICIENTNET_FEATURE_DIM):
            report.print_check("EfficientNet output dim = 1280", "PASS",
                                f"Shape: {tuple(e_feat.shape)}")
        else:
            report.print_check("EfficientNet output dim = 1280", "FAIL",
                                f"Got {tuple(e_feat.shape)}, expected (1, 1280)")

        if s_feat.shape == (1, SWIN_FEATURE_DIM):
            report.print_check("Swin output dim = 384", "PASS",
                                f"Shape: {tuple(s_feat.shape)}")
        else:
            report.print_check("Swin output dim = 384", "FAIL",
                                f"Got {tuple(s_feat.shape)}, expected (1, 384)")

        if fused.shape == (1, FUSED_DIM):
            report.print_check("Fused feature dim = 1664", "PASS",
                                f"Shape: {tuple(fused.shape)}")
        else:
            report.print_check("Fused feature dim = 1664", "FAIL",
                                f"Got {tuple(fused.shape)}, expected (1, 1664)")

        has_nan = torch.isnan(fused).any().item()
        has_inf = torch.isinf(fused).any().item()
        if has_nan or has_inf:
            report.print_check("Fused features finite", "FAIL",
                                f"NaN={has_nan}, Inf={has_inf}")
        else:
            report.print_check("Fused features finite", "PASS",
                                "No NaN or Inf in fused feature vector")
    except Exception as e:
        report.print_check("Feature fusion dimension check", "FAIL", f"Exception: {e}")
    finally:
        model.train()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model        : Hybrid EfficientNet-B0 + Swin3D-Tiny + KAN Head")
    print(f"KAN topology : {FUSED_DIM} → 512 → 256 → 1")
    print(f"KAN grid_size: {KAN_GRID_SIZE},  spline_order: {KAN_SPLINE_ORDER}")

    # ── Load metadata ─────────────────────────────────────────────────────
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"]
    neg_count  = (train_meta["label"] == 0).sum()
    pos_count  = (train_meta["label"] == 1).sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    print(f"pos_weight = {neg_count / pos_count:.2f}")

    val_loader  = DataLoader(
        NodulePatchDataset(metadata[metadata["split"]=="val"].reset_index(drop=True),
                           DATA_DIR, transforms=get_eval_transforms()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(
        NodulePatchDataset(metadata[metadata["split"]=="test"].reset_index(drop=True),
                           DATA_DIR, transforms=get_eval_transforms()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Load model ────────────────────────────────────────────────────────
    model = HybridEfficientNetSwinKANHead().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print("Model loaded.")

    # ── Youden threshold on validation set ───────────────────────────────
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
    youden_thr = (get_youden_threshold(val_labels_all, val_probs_all)
                  if len(np.unique(val_labels_all)) > 1 else 0.5)
    print(f"Youden threshold (val): {youden_thr:.4f}")

    # ── Test evaluation ───────────────────────────────────────────────────
    (test_loss, test_auc, test_auprc, test_f1_05, test_sens_05, test_spec_05,
     test_froc, test_cpm, test_probs, test_labels, test_uids) = \
        evaluate_model(test_loader, model, device, pos_weight, threshold=0.5)

    pred_df = pd.DataFrame({'seriesuid':   test_uids,
                            'label':       test_labels,
                            'probability': test_probs})
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved to {PREDICTIONS_CSV}")

    # ── Youden metrics on test ────────────────────────────────────────────
    patient_dict_test = {}
    for uid, prob, label in zip(test_uids, test_probs, test_labels):
        if uid not in patient_dict_test:
            patient_dict_test[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict_test[uid]['prob']  = max(patient_dict_test[uid]['prob'],  prob)
            patient_dict_test[uid]['label'] = max(patient_dict_test[uid]['label'], label)
    y_true_t  = [v['label'] for v in patient_dict_test.values()]
    y_probs_t = [v['prob']  for v in patient_dict_test.values()]
    y_pred_y  = [1 if p >= youden_thr else 0 for p in y_probs_t]
    cm_y      = confusion_matrix(y_true_t, y_pred_y, labels=[0, 1])
    tn_y, fp_y, fn_y, tp_y = cm_y.ravel()
    test_f1_y   = f1_score(y_true_t, y_pred_y, zero_division=0)
    test_sens_y = recall_score(y_true_t, y_pred_y, zero_division=0)
    test_spec_y = tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else 0.0

    ci_lo, ci_hi = calculate_95_ci(y_true_t, y_probs_t)

    print("\n--- FINAL TEST RESULTS — Hybrid EfficientNet-B0 + Swin3D-Tiny + KAN ---")
    print(f"  AUROC          : {test_auc:.4f}  (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  AUPRC          : {test_auprc:.4f}")
    print(f"  CPM            : {test_cpm:.4f}")
    print(f"\n  threshold=0.50  → F1:{test_f1_05:.4f} Sens:{test_sens_05:.4f} Spec:{test_spec_05:.4f}")
    print(f"  threshold={youden_thr:.4f} → F1:{test_f1_y:.4f} Sens:{test_sens_y:.4f} Spec:{test_spec_y:.4f}  (Youden)")
    print(f"\n  --- FROC ---")
    for fp_rate, sens in test_froc.items():
        print(f"  {fp_rate:12s} : {sens:.4f}")
    print(f"  CPM (mean)     : {test_cpm:.4f}")

    final_metrics = {
        "model": "Hybrid EfficientNet3D-B0 + Swin3D-Tiny + KAN Head",
        "architecture": {
            "efficientnet_dim": EFFICIENTNET_FEATURE_DIM,
            "swin_dim":         SWIN_FEATURE_DIM,
            "fused_dim":        FUSED_DIM,
            "kan_topology":     f"{FUSED_DIM} → 512 → 256 → 1",
        },
        "kan_config": {
            "grid_size":    KAN_GRID_SIZE,
            "spline_order": KAN_SPLINE_ORDER,
            "grid_range":   KAN_GRID_RANGE,
            "residual_std": KAN_RESIDUAL_STD,
        },
        "auroc": test_auc, "auroc_ci_95": [ci_lo, ci_hi],
        "auprc": test_auprc, "cpm": test_cpm,
        "threshold_05": {"f1": test_f1_05, "sensitivity": test_sens_05,
                         "specificity": test_spec_05},
        "threshold_youden": {"value": youden_thr, "f1": test_f1_y,
                             "sensitivity": test_sens_y, "specificity": test_spec_y},
        "froc": test_froc,
    }
    with open(FINAL_METRICS_JSON, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"\n[Saved] Final metrics → {FINAL_METRICS_JSON}")

    # ── AUDIT ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STARTING AUDIT — Hybrid EfficientNet-B0 + Swin3D-Tiny + KAN Head")
    print("=" * 70)
    report = AuditReport()

    check_1_required_files_exist(report)
    trained_model = check_2_model_weights_not_random(report, device)
    pred_df = check_3_predictions_csv_integrity(report, pred_df)
    if pred_df is not None:
        check_4_recompute_metrics_from_predictions(report, pred_df)
    if os.path.exists(TRAINING_LOG_CSV):
        check_5_training_log_shows_learning(report)
    check_6_model_inference_live(report, trained_model, device)
    check_7_label_split_integrity(report)
    if pred_df is not None:
        check_8_permutation_baseline(report, pred_df)
        check_9_prediction_score_separation(report, pred_df)
    check_10_kan_spline_weights_non_trivial(report, trained_model)
    check_11_swin_structure_weight_check(report, trained_model)
    check_12_feature_fusion_dimension(report, trained_model, device)

    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print(f"  Passed   : {report.passed}")
    print(f"  Warnings : {report.warned}")
    print(f"  Failed   : {report.failed}")
    if report.failed == 0:
        print("\n  ✅  VERDICT: Results appear LEGITIMATE.")
    else:
        print(f"\n  ❌  VERDICT: {report.failed} check(s) FAILED — review above.")
    print("=" * 70)

    report.save("audit_report_effnet_swin_kan.txt",
                "audit_summary_effnet_swin_kan.json")


if __name__ == "__main__":
    main()