"""
3D Hybrid CNN with MLP Head — FROZEN BACKBONES (ResNet18 + DenseNet121 + EfficientNet-B0)
==========================================================================================
Three parallel 3D CNN feature extractors are loaded from pre-trained checkpoints,
frozen, and their outputs concatenated and fused by a trainable MLP head.

FIX (ported from 2-branch version): Weight loading strips the 'backbone.' prefix
that the original single-model training wrappers added when saving checkpoints.
Without this fix, 0/N keys match and all backbones run on random initialisation.

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

MLP Fusion Head (only part that trains):
    Linear(2816→512) → LayerNorm → ReLU → Dropout(0.5)
    Linear(512→256)  → LayerNorm → ReLU → Dropout(0.5)
    Linear(256→1)

Output files produced:
    best_model_hybrid3_frozen.pth         — best checkpoint (by val AUROC)
    best_metrics_hybrid3_frozen.json      — metrics at best epoch
    hybrid3_frozen_training_log.csv       — epoch-by-epoch history
    hybrid3_frozen_test_predictions.csv   — candidate-level test predictions
    hybrid3_frozen_calibration_curve.csv  — calibration data for reliability plot
"""

import os
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
DATA_DIR           = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH      = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

# ── Pre-trained backbone checkpoints ────────────────────────────────────
# Set a path to None to skip loading weights for that branch (random init).
RESNET18_WEIGHTS      = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_Resnet18\best_model_resnet18.pth"
DENSENET121_WEIGHTS   = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_densenet121\best_model_densenet121.pth"
EFFICIENTNET_WEIGHTS  = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_EfficientnetB0\best_model_efficientnetb0.pth"

# ── NOTE: batch size 8 is kept from the original; lower to 4 or 2 if OOM.
#    All three backbones are frozen so memory is lighter than joint training.
BATCH_SIZE         = 8
NUM_WORKERS        = 4
PIN_MEMORY         = True
PERSISTENT_WORKERS = True

MAX_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 15
LR                      = 1e-4
WEIGHT_DECAY            = 1e-4
POS_WEIGHT              = 5115.0 / 822.0   # neg:pos ratio for BCEWithLogitsLoss

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ═══════════════════════════════════════════════════════════════════════
#  Weight Loading (PREFIX-AWARE)
# ═══════════════════════════════════════════════════════════════════════
def load_backbone_weights(backbone_module, ckpt_path, label):
    """
    Load weights from a checkpoint saved by a wrapper model that stored
    the backbone under a 'backbone.' prefix.

    Automatically detects whether to strip the prefix or load directly,
    picks whichever gives more matching keys, then verifies a sample weight
    actually landed correctly.
    """
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_keys = set(backbone_module.state_dict().keys())

    # Build candidate key mappings
    direct_keys   = ckpt
    stripped_keys = {k[len("backbone."):]: v
                     for k, v in ckpt.items() if k.startswith("backbone.")}

    direct_match   = len(model_keys & set(direct_keys.keys()))
    stripped_match = len(model_keys & set(stripped_keys.keys()))

    print(f"\n  ── {label} weight loading ──")
    print(f"     Direct match   : {direct_match} / {len(model_keys)}")
    print(f"     Stripped match : {stripped_match} / {len(model_keys)}")

    if stripped_match > direct_match:
        weights_to_load = stripped_keys
        print(f"     Strategy: strip 'backbone.' prefix from checkpoint keys")
    else:
        weights_to_load = direct_keys
        print(f"     Strategy: direct load (no prefix stripping needed)")

    missing, unexpected = backbone_module.load_state_dict(weights_to_load, strict=False)
    matched = len(model_keys) - len(missing)
    print(f"     Loaded  : {matched}/{len(model_keys)} keys matched")
    print(f"     Missing : {len(missing)}   Unexpected: {len(unexpected)}")

    if missing:
        print("     Still missing: "
              + ", ".join(list(missing)[:5])
              + ("..." if len(missing) > 5 else ""))

    # Verify at least one weight value actually landed
    for key in list(model_keys):
        if key in weights_to_load:
            model_val = backbone_module.state_dict()[key].float().mean().item()
            ckpt_val  = weights_to_load[key].float().mean().item()
            ok = abs(model_val - ckpt_val) < 1e-6
            status = "✓" if ok else "⚠ MISMATCH"
            print(f"     Verify [{key}]: {status}  "
                  f"(model={model_val:.6f}, ckpt={ckpt_val:.6f})")
            if not ok:
                print("     ⚠ WARNING: weight value mismatch after load!")
            break

    return matched

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
#  Hybrid Model — FROZEN BACKBONES
# ═══════════════════════════════════════════════════════════════════════
class HybridThreeBranchFrozen(nn.Module):
    """
    3D Hybrid CNN: three frozen feature extractors + trainable MLP fusion head.

                Input  (B, 1, D, H, W)
                   │
          ┌────────┼─────────────┐
          ▼        ▼             ▼
      ResNet18  DenseNet121  EfficientNet-B0
      [512-d]   [1024-d]      [1280-d]
      FROZEN    FROZEN         FROZEN
          │        │             │
          └────────┴─────────────┘
                   │  torch.cat  →  (B, 2816)
                   ▼
           MLP Fusion Head  ← TRAINABLE
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

    Backbone forward passes run inside torch.no_grad() to avoid storing
    intermediate activations, cutting peak GPU memory substantially.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1,
                 dropout: float = 0.5):
        super().__init__()

        # ── Branch 1: ResNet18 → 512-d ─────────────────────────────────
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

        if RESNET18_WEIGHTS:
            load_backbone_weights(self.resnet18, RESNET18_WEIGHTS, "ResNet18")
        else:
            print("  ⚠ RESNET18_WEIGHTS empty — Branch 1 initialised randomly")

        for param in self.resnet18.parameters():
            param.requires_grad = False

        # ── Branch 2: DenseNet121 → 1024-d ─────────────────────────────
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

        if DENSENET121_WEIGHTS:
            load_backbone_weights(self.densenet121, DENSENET121_WEIGHTS, "DenseNet121")
        else:
            print("  ⚠ DENSENET121_WEIGHTS empty — Branch 2 initialised randomly")

        for param in self.densenet121.parameters():
            param.requires_grad = False

        # ── Branch 3: EfficientNet-B0 → 1280-d ─────────────────────────
        self.efficientnet_b0 = EfficientNet3D_B0(in_channels=in_channels)

        if EFFICIENTNET_WEIGHTS:
            load_backbone_weights(self.efficientnet_b0, EFFICIENTNET_WEIGHTS, "EfficientNet-B0")
        else:
            print("  ⚠ EFFICIENTNET_WEIGHTS empty — Branch 3 initialised randomly")

        for param in self.efficientnet_b0.parameters():
            param.requires_grad = False

        # ── MLP Fusion Head: 2816-d → 1  (only part that trains) ────────
        self.mlp_head = nn.Sequential(
            nn.Linear(TOTAL_FEATURE_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f_r = self.resnet18(x)           # (B, 512)
            f_d = self.densenet121(x)        # (B, 1024)
            f_e = self.efficientnet_b0(x)   # (B, 1280)

        combined = torch.cat([f_r, f_d, f_e], dim=1)   # (B, 2816)
        out      = self.mlp_head(combined)              # (B, 1)
        return out.squeeze(1)                           # (B,)

    def freeze_backbones(self):
        """Re-apply freeze — call after loading a full checkpoint if needed."""
        for p in self.resnet18.parameters():
            p.requires_grad = False
        for p in self.densenet121.parameters():
            p.requires_grad = False
        for p in self.efficientnet_b0.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze all backbones for optional second-stage end-to-end fine-tuning."""
        for p in self.resnet18.parameters():
            p.requires_grad = True
        for p in self.densenet121.parameters():
            p.requires_grad = True
        for p in self.efficientnet_b0.parameters():
            p.requires_grad = True

# ═══════════════════════════════════════════════════════════════════════
#  Evaluation Utilities
# ═══════════════════════════════════════════════════════════════════════

def calculate_candidate_froc(all_labels, all_probs, total_scans):
    """
    Candidate-level FROC.

    Evaluates every single patch (not max-per-patient) so that the
    FP/scan axis faithfully reflects candidate detection performance.
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

    auc   = roc_auc_score(y_true_pat, y_probs_pat) \
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
        }).to_csv("hybrid3_frozen_calibration_curve.csv", index=False)
        print("  [Saved] hybrid3_frozen_calibration_curve.csv")

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

    # ── Metadata ─────────────────────────────────────────────────────────
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

    # ── Transforms ───────────────────────────────────────────────────────
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

    # ── DataLoaders ──────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────────
    model = HybridThreeBranchFrozen().to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p    = total_p - trainable_p
    rn18_p      = sum(p.numel() for p in model.resnet18.parameters())
    dn121_p     = sum(p.numel() for p in model.densenet121.parameters())
    enb0_p      = sum(p.numel() for p in model.efficientnet_b0.parameters())
    mlp_p       = sum(p.numel() for p in model.mlp_head.parameters())

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║      3-Branch Hybrid 3D CNN (Frozen Backbones)          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Branch 1  ResNet18        {rn18_p:>12,} params  → {RESNET18_FEATURE_DIM}-d  ║")
    print(f"║  Branch 2  DenseNet121     {dn121_p:>12,} params  → {DENSENET121_FEATURE_DIM}-d ║")
    print(f"║  Branch 3  EfficientNet-B0 {enb0_p:>12,} params  → {EFFICIENTNET_FEATURE_DIM}-d ║")
    print(f"║  MLP Head  (2816→512→256→1){mlp_p:>12,} params         ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Concat feature dim        {TOTAL_FEATURE_DIM:>12,}               ║")
    print(f"║  Total params              {total_p:>12,}               ║")
    print(f"║  Trainable (MLP only)      {trainable_p:>12,}               ║")
    print(f"║  Frozen (backbones)        {frozen_p:>12,}               ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Loss / Optimizer (MLP params only) / Scheduler ───────────────────
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(device)
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    warmup_epochs    = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler        = SequentialLR(
        optimizer,
        schedulers = [warmup_scheduler, cosine_scheduler],
        milestones = [warmup_epochs],
    )
    scaler = GradScaler('cuda')

    # ── Training Loop ─────────────────────────────────────────────────────
    best_val_auc     = 0.0
    patience_counter = 0
    history          = []

    print("─" * 60)
    print("  Starting Training  (only MLP head updates)")
    print("─" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        # Keep all three frozen branches in eval mode so BN running stats
        # don't drift and dropout inside them doesn't activate.
        model.resnet18.eval()
        model.densenet121.eval()
        model.efficientnet_b0.eval()

        train_loss = 0.0
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
            torch.save(model.state_dict(), "best_model_hybrid3_frozen.pth")
            print(f"  ✓ [Saved] New best validation AUROC: {best_val_auc:.4f}")

            with open("best_metrics_hybrid3_frozen.json", "w") as f:
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

    pd.DataFrame(history).to_csv("hybrid3_frozen_training_log.csv", index=False)
    print("\n[Saved] hybrid3_frozen_training_log.csv")

    # ── Final Test Evaluation ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Loading best model for final test evaluation")
    print("═" * 60)

    model.load_state_dict(
        torch.load("best_model_hybrid3_frozen.pth",
                   map_location=device, weights_only=True)
    )
    # Re-freeze in case load_state_dict resets any requires_grad state
    model.freeze_backbones()

    (test_loss, test_auc, test_auprc, test_f1,
     test_sens, test_spec, test_froc, pred_df) = evaluate_model(
        test_loader, model, device, desc="Testing", return_preds=True
    )

    pred_df.to_csv("hybrid3_frozen_test_predictions.csv", index=False)
    print("  [Saved] hybrid3_frozen_test_predictions.csv")

    print("\n─── FINAL TEST RESULTS (Scan-Level) ─────────────────────")
    print(f"  AUROC (Primary)       : {test_auc:.4f}")
    print(f"  AUPRC                 : {test_auprc:.4f}")
    print(f"  F1 Score              : {test_f1:.4f}")
    print(f"  Sensitivity (Recall)  : {test_sens:.4f}")
    print(f"  Specificity           : {test_spec:.4f}")

    print("\n─── FROC (Candidate-Level: FP/Scan → Sensitivity) ───────")
    for fp_rate, sensitivity in test_froc.items():
        print(f"  {fp_rate:14s} : {sensitivity:.4f}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()