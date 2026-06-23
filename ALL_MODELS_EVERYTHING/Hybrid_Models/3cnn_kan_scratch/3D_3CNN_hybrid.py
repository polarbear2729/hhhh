"""
3D Hybrid CNN with KAN Head for Lung Nodule Classification
===========================================================
Combines three parallel 3D CNN feature extractors whose outputs are
concatenated and fused by a shared KAN classification head.

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
    KAN([2816, 512, 256, 1])

Output files produced:
    latest_checkpoint.pth         — full resume state (saved after every epoch)
    best_model_hybrid.pth         — best checkpoint (by val AUROC)
    best_metrics_hybrid.json      — metrics at best epoch
    hybrid_training_log.csv       — epoch-by-epoch history
    hybrid_test_predictions.csv   — candidate-level test predictions
    hybrid_calibration_curve.csv  — calibration data for reliability plot

Resume behaviour
────────────────
If `latest_checkpoint.pth` is present, the script automatically resumes from
the next epoch, restoring model, optimizer, scheduler, scaler, best AUROC,
patience counter, and history.  Safe to interrupt with Ctrl-C or a dropped
RunPod connection — just re-run the same command.
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT      = os.path.dirname(SCRIPT_DIR)
DATA_DIR          = os.path.join(PROJECT_ROOT, "data", "LUNA16_processed_64")
METADATA_PATH     = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH= os.path.join(DATA_DIR, "patient_split.csv")

# ── Tuned for RunPod A100 80GB / H100 (32 vCPUs, 251 GB RAM).
#    Three large 3D CNNs run in parallel, roughly tripling peak GPU memory.
#    If you encounter OOM, lower BATCH_SIZE to 8 (or 4).
BATCH_SIZE         = 8
NUM_WORKERS        = 16
PIN_MEMORY         = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR    = 4
USE_TF32           = True    # ~2× speed-up on Ampere+ (A100/H100) with negligible accuracy loss
USE_COMPILE        = True    # torch.compile() — wraps the model for ~20-30% extra speed

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

# ═══════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True
    if USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

# ═══════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════
class NodulePatchDataset(Dataset):
    def __init__(self, metadata_df, data_dir, transforms=None, preload: bool = True):
        self.metadata   = metadata_df.reset_index(drop=True)
        self.data_dir   = data_dir
        self.transforms = transforms
        self.preload    = preload
        self.patches    = None
        self.labels     = None
        self.seriesuids = None

        if self.preload:
            print(f"    [preload] loading {len(self.metadata):,} patches into RAM…", flush=True)
            patches, labels, seriesuids = [], [], []
            for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata),
                               desc="    [preload]", leave=False):
                filename  = os.path.basename(row["filepath"])
                split     = row["split"]
                label     = int(row["label"])
                subfolder = "pos" if label == 1 else "neg"
                local_path = os.path.join(self.data_dir, split, subfolder, filename)
                arr = np.load(local_path).astype(np.float32)
                patches.append(np.expand_dims(arr, axis=0))   # (1, D, H, W)
                labels.append(label)
                seriesuids.append(row["seriesuid"])
            self.patches    = patches
            self.labels     = labels
            self.seriesuids = seriesuids
            gb = sum(p.nbytes for p in self.patches) / 1024**3
            print(f"    [preload] done — {gb:.2f} GB resident in RAM", flush=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.patches is not None:
            patch = self.patches[idx].copy()
            label = self.labels[idx]
            uid   = self.seriesuids[idx]
        else:
            row       = self.metadata.iloc[idx]
            filename  = os.path.basename(row["filepath"])
            split     = row["split"]
            label     = int(row["label"])
            subfolder = "pos" if label == 1 else "neg"
            local_path = os.path.join(self.data_dir, split, subfolder, filename)
            patch = np.load(local_path).astype(np.float32)
            patch = np.expand_dims(patch, axis=0)       # (1, D, H, W)
            uid   = row["seriesuid"]

        if self.transforms is not None:
            patch = self.transforms(patch)

        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)

        return patch, torch.tensor(label, dtype=torch.float32), uid

# ═══════════════════════════════════════════════════════════════════════
#  3D EfficientNet-B0  (custom implementation — MONAI has no 3D version)
#
#  Faithfully ports EfficientNet-B0 MBConv blocks into 3D with:
#    nn.Conv3d, nn.BatchNorm3d, Squeeze-and-Excitation in 3D.
#  All B0 scaling coefficients (width=1.0, depth=1.0) are preserved.
#  Final feature dimension: 1280  (identical to 2D EfficientNet-B0).
# ═══════════════════════════════════════════════════════════════════════

def _make_divisible(v, divisor, min_value=None):
    """Round `v` up to the nearest multiple of `divisor`."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SwishActivation(nn.Module):
    """Swish: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation3D(nn.Module):
    """3D Squeeze-and-Excitation block."""
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
    """
    3D Mobile Inverted Bottleneck Conv (MBConv) with stochastic depth.

    Structure (expand_ratio > 1):
        Expand → Depthwise3D → SE3D → Project
    Residual skip is used when stride=1 and in_channels==out_channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_skip          = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate

        mid_ch = _make_divisible(in_channels * expand_ratio, 8)
        layers = []

        # Expansion pointwise conv (skipped when expand_ratio == 1)
        if expand_ratio != 1:
            layers += [
                nn.Conv3d(in_channels, mid_ch, 1, bias=False),
                nn.BatchNorm3d(mid_ch, momentum=0.01, eps=1e-3),
                SwishActivation(),
            ]

        # Depthwise conv
        pad = (kernel_size - 1) // 2
        layers += [
            nn.Conv3d(mid_ch, mid_ch, kernel_size,
                      stride=stride, padding=pad, groups=mid_ch, bias=False),
            nn.BatchNorm3d(mid_ch, momentum=0.01, eps=1e-3),
            SwishActivation(),
            SqueezeExcitation3D(mid_ch, se_ratio),
            # Projection pointwise conv
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
    """
    3D EfficientNet-B0 backbone.

    Stage configs mirror the original 2D B0 specification:
        Stage | Op           | Stride | Out-ch | Layers
        1     | MBConv1 3×3  |   1    |  16    |   1
        2     | MBConv6 3×3  |   2    |  24    |   2
        3     | MBConv6 5×5  |   2    |  40    |   2
        4     | MBConv6 3×3  |   2    |  80    |   3
        5     | MBConv6 5×5  |   1    |  112   |   3
        6     | MBConv6 5×5  |   2    |  192   |   4
        7     | MBConv6 3×3  |   1    |  320   |   1

    Followed by: Conv1×1 (320→1280) → BN → Swish → GlobalAvgPool3D
    Output: (B, 1280) feature vector.
    """
    def __init__(self, in_channels=1):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=0.01, eps=1e-3),
            SwishActivation(),
        )

        # (in_ch, out_ch, kernel, stride, expand_ratio, num_layers)
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

        # Head conv: 320 → 1280
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
        return x.flatten(1)            # (B, 1280)

# ═══════════════════════════════════════════════════════════════════════
#  Hybrid Model
# ═══════════════════════════════════════════════════════════════════════
class HybridCNNWithMLPHead(nn.Module):
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
           ─────────────────
           Linear(2816 → 512)
           LayerNorm(512)
           ReLU  |  Dropout(p=0.5)
           Linear(512 → 256)
           LayerNorm(256)
           ReLU  |  Dropout(p=0.5)
           Linear(256 → 1)
                   │
               Logit (B,)   ──►  BCEWithLogitsLoss

    How feature extraction works per branch
    ─────────────────────────────────────────
    ResNet18     : MONAI ResNet(layers=[2,2,2,2]).  The original FC
                   (Linear 512→num_classes) is replaced with nn.Identity()
                   so we receive the 512-d global-avg-pool output directly.

    DenseNet121  : MONAI DenseNet121.  class_layers.out
                   (Linear 1024→1) is replaced with nn.Identity()
                   so we receive the 1024-d flattened feature vector.

    EfficientNet : Custom 3D re-implementation.  EfficientNet3D_B0 already
                   returns (B, 1280) before any classification layer.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1,
                 dropout: float = 0.5):
        super().__init__()

        # ── Branch 1: ResNet18 → 512-d ─────────────────────────────────
        self.resnet18 = ResNet(
            block            = ResNetBlock,        # BasicBlock (no bottleneck)
            layers           = [2, 2, 2, 2],       # ResNet18 depth per stage
            block_inplanes   = [64, 128, 256, 512],
            spatial_dims     = 3,
            n_input_channels = in_channels,
            num_classes      = RESNET18_FEATURE_DIM,  # creates FC(512→512)
        )
        assert hasattr(self.resnet18, 'fc'), \
            "MONAI ResNet structure changed: 'fc' attribute missing."
        # Replace FC with Identity → output is 512-d GAP feature vector
        self.resnet18.fc = nn.Identity()

        # ── Branch 2: DenseNet121 → 1024-d ─────────────────────────────
        self.densenet121 = DenseNet121(
            spatial_dims = 3,
            in_channels  = in_channels,
            out_channels = 1,   # creates Linear(1024→1); will be overridden
        )
        assert hasattr(self.densenet121, 'class_layers'), \
            "MONAI DenseNet structure changed: 'class_layers' missing."
        assert hasattr(self.densenet121.class_layers, 'out'), \
            "MONAI DenseNet structure changed: 'out' missing in class_layers."
        # Replace Linear(1024→1) with Identity → output is 1024-d feature vector
        self.densenet121.class_layers.out = nn.Identity()

        # ── Branch 3: EfficientNet-B0 → 1280-d ─────────────────────────
        # EfficientNet3D_B0 already returns raw (B, 1280) features
        self.efficientnet_b0 = EfficientNet3D_B0(in_channels=in_channels)

        # ── KAN Fusion Head: 2816-d → 1 ────────────────────────────────
        from efficient_kan import KAN
        self.mlp_head = KAN([TOTAL_FEATURE_DIM, 512, 256, num_classes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_r = self.resnet18(x)           # (B, 512)
        f_d = self.densenet121(x)        # (B, 1024)
        f_e = self.efficientnet_b0(x)   # (B, 1280)

        combined = torch.cat([f_r, f_d, f_e], dim=1)   # (B, 2816)
        out      = self.mlp_head(combined)              # (B, 1)
        return out.squeeze(1)                           # (B,)

# ═══════════════════════════════════════════════════════════════════════
#  Evaluation Utilities
# ═══════════════════════════════════════════════════════════════════════

def calculate_candidate_froc(all_labels, all_probs, total_scans):
    """
    Candidate-level FROC.

    Evaluates every single patch (not max-per-patient) so that the
    FP/scan axis faithfully reflects candidate detection performance.
    Patches are ranked by prediction probability; cumulative TP/FP
    curves are interpolated at each FROC_THRESHOLD.
    """
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
    """Bootstrapped 95 % confidence interval for scan-level AUROC."""
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
    """
    Evaluate model on `loader`.

    Returns (avg_loss, auc, auprc, f1, sensitivity, specificity, froc_dict)
    and, when return_preds=True, appends a candidate-level predictions DataFrame.

    Two levels of evaluation are computed:
      • Scan-level  — max-probability aggregation per patient (primary metrics).
      • Candidate-level — all patches kept; used for FROC computation.
    """
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

    # ── Scan-level aggregation (max-prob per patient) ───────────────────
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

    # ── Scan-level metrics ──────────────────────────────────────────────
    auc  = roc_auc_score(y_true_pat, y_probs_pat) \
           if len(np.unique(y_true_pat)) > 1 else 0.5
    auprc = average_precision_score(y_true_pat, y_probs_pat)
    f1    = f1_score(y_true_pat, y_pred_pat)
    sens  = recall_score(y_true_pat, y_pred_pat)

    cm = confusion_matrix(y_true_pat, y_pred_pat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── Candidate-level FROC ────────────────────────────────────────────
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
    print(f"Using device: {device}", flush=True)

    # ── Metadata ────────────────────────────────────────────────────────
    print("Loading metadata...", flush=True)
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df   = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)

    print(f"  Train: {len(train_meta):,}  |  Val: {len(val_meta):,}"
          f"  |  Test: {len(test_meta):,}", flush=True)

    # ── Transforms ──────────────────────────────────────────────────────
    train_transforms = Compose([
        # Random translation breaks centre-of-patch positional shortcuts
        RandAffine(
            prob            = 0.8,
            translate_range = (5, 5, 5),  # up to 5 voxels per axis
            padding_mode    = 'zeros',
            spatial_size    = None,        # preserves original 64×64×64 size
        ),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandGaussianNoise(prob=0.2, std=0.01),
    ])

    # ── DataLoaders ─────────────────────────────────────────────────────
    train_ds = NodulePatchDataset(train_meta, DATA_DIR, transforms=train_transforms)
    val_ds   = NodulePatchDataset(val_meta,   DATA_DIR, transforms=None)
    test_ds  = NodulePatchDataset(test_meta,  DATA_DIR, transforms=None)

    loader_kwargs = dict(
        batch_size        = BATCH_SIZE,
        num_workers       = NUM_WORKERS,
        pin_memory        = PIN_MEMORY,
        persistent_workers= PERSISTENT_WORKERS,
        prefetch_factor   = PREFETCH_FACTOR,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False,                   **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False,                   **loader_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    print("  Building model…", flush=True)
    model = HybridCNNWithMLPHead().to(device, memory_format=torch.channels_last_3d)
    print("  Model on GPU.\n", flush=True)

    total_p      = sum(p.numel() for p in model.parameters())
    trainable_p  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rn18_p       = sum(p.numel() for p in model.resnet18.parameters())
    dn121_p      = sum(p.numel() for p in model.densenet121.parameters())
    enb0_p       = sum(p.numel() for p in model.efficientnet_b0.parameters())
    mlp_p        = sum(p.numel() for p in model.mlp_head.parameters())

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║           Hybrid 3D CNN — Model Summary             ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Branch 1  ResNet18        {rn18_p:>12,} params  → {RESNET18_FEATURE_DIM}-d  ║")
    print(f"║  Branch 2  DenseNet121     {dn121_p:>12,} params  → {DENSENET121_FEATURE_DIM}-d ║")
    print(f"║  Branch 3  EfficientNet-B0 {enb0_p:>12,} params  → {EFFICIENTNET_FEATURE_DIM}-d ║")
    print(f"║  MLP Head  (2816→512→256→1){mlp_p:>12,} params         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Concat feature dim        {TOTAL_FEATURE_DIM:>12,}               ║")
    print(f"║  Total params              {total_p:>12,}               ║")
    print(f"║  Trainable params          {trainable_p:>12,}               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Loss / Optimizer / Scheduler ────────────────────────────────────
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

    # ── Training Loop ────────────────────────────────────────────────────
    best_val_auc     = 0.0
    patience_counter = 0
    history          = []
    start_epoch      = 1

    CHECKPOINT_PATH = "latest_checkpoint.pth"
    HISTORY_PATH    = "hybrid_training_log.csv"
    BEST_MODEL_PATH = "best_model_hybrid.pth"
    BEST_METRICS    = "best_metrics_hybrid.json"

    # ── Resume from latest checkpoint if one exists ──────────────────────
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n  Found checkpoint '{CHECKPOINT_PATH}' — resuming…", flush=True)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_auc     = ckpt["best_val_auc"]
        patience_counter = ckpt["patience_counter"]
        history          = ckpt["history"]
        print(f"  Resumed from epoch {ckpt['epoch']}  "
              f"(best_val_auc={best_val_auc:.4f}, "
              f"patience={patience_counter}/{EARLY_STOPPING_PATIENCE})", flush=True)
        if start_epoch > MAX_EPOCHS:
            print("  Checkpoint already at MAX_EPOCHS — skipping training.",
                  flush=True)
            return
    else:
        print("  No checkpoint found — starting from scratch.", flush=True)

    print("─" * 58)
    print(f"  Training epochs {start_epoch} → {MAX_EPOCHS}")
    print("─" * 58, flush=True)

    def _save_checkpoint(epoch: int):
        """Atomic save: write to .tmp then rename."""
        tmp = CHECKPOINT_PATH + ".tmp"
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_auc":     best_val_auc,
            "patience_counter": patience_counter,
            "history":          history,
        }, tmp)
        os.replace(tmp, CHECKPOINT_PATH)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for patches, labels, _ in pbar:
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

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

        # Validation
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

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch:3d} │ LR {current_lr:.2e} │ "
              f"Train Loss {avg_train_loss:.4f} │ Val Loss {val_loss:.4f}"
              f" │ {epoch_time:.1f}s", flush=True)
        print(f"         │ AUROC {val_auc:.4f} │ AUPRC {val_auprc:.4f} │ "
              f"F1 {val_f1:.4f} │ Sens {val_sens:.4f} │ Spec {val_spec:.4f}",
              flush=True)

        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ [Saved] New best validation AUROC: {best_val_auc:.4f}",
                  flush=True)

            with open(BEST_METRICS, "w") as f:
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
            print(f"  · No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})",
                  flush=True)
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping triggered after epoch {epoch} ***",
                      flush=True)
                break

        # ── Persist progress after every epoch ───────────────────────────
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
        _save_checkpoint(epoch)
        print(f"  ↳ checkpoint saved to {CHECKPOINT_PATH}", flush=True)

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    print(f"\n[Saved] {HISTORY_PATH}", flush=True)

    # ── Final Test Evaluation ────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  Loading best model for final test evaluation")
    print("═" * 58)

    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
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