# gradcam_hybrid3_frozen.py
"""
Full evaluation + GradCAM++ for the 3-Branch Frozen Hybrid CNN
==============================================================
Model: HybridThreeBranchFrozen (ResNet18 + DenseNet121 + EfficientNet-B0,
       frozen backbones, MLP head).
Checkpoint: best_model_hybrid3_frozen.pth

One GradCAM++ heatmap is produced per backbone branch, giving three spatial
attention maps per candidate patch.  The analysis image lays them out as:

  Col 0  Original CT slice (central axial)
  Col 1  ResNet18 GradCAM++ heatmap
  Col 2  ResNet18 overlay
  Col 3  DenseNet121 GradCAM++ heatmap
  Col 4  DenseNet121 overlay
  Col 5  EfficientNet-B0 GradCAM++ heatmap
  Col 6  EfficientNet-B0 overlay

Selected cases: up to 4 TP, 3 FN, 3 FP, 3 TN (highest-confidence patch per
patient, same selection strategy as the ResNet18 reference script).

Important implementation note — frozen backbone workaround
----------------------------------------------------------
All three backbones run inside a torch.no_grad() context in
HybridThreeBranchFrozen.forward().  This discards the computation graph so
standard backward-based GradCAM yields zero gradients.

Fix: for each GradCAM call the target backbone is run directly (outside
no_grad) with its parameters temporarily re-enabled, while the other two
backbones are run with torch.no_grad() and their features detached before
concatenation.  The MLP head is then called on the combined features, and
backward() is invoked.  The backbone is immediately re-frozen afterwards.

Output files:
    hybrid3_frozen_test_predictions.csv   — candidate-level predictions
    hybrid3_frozen_calibration_curve.csv  — reliability diagram data
    gradcampp_hybrid3_frozen_analysis.png — GradCAM++ visualisation grid
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.networks.nets import DenseNet121, ResNet
from monai.networks.nets.resnet import ResNetBlock
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    recall_score,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
SEED = 42
DATA_DIR           = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH      = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

# Full hybrid checkpoint (saved by hybrid3_frozen training script)
MODEL_PATH = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\Hybrid_Models\3CNN_MLP_FROZEN\best_model_hybrid3_frozen.pth"

# Individual backbone checkpoints — used only to re-load backbone weights
RESNET18_WEIGHTS     = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_Resnet18\best_model_resnet18.pth"
DENSENET121_WEIGHTS  = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_densenet121\best_model_densenet121.pth"
EFFICIENTNET_WEIGHTS = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_EfficientnetB0\best_model_efficientnetb0.pth"

BATCH_SIZE  = 1
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
POS_WEIGHT      = 5115.0 / 822.0

RESNET18_FEATURE_DIM     = 512
DENSENET121_FEATURE_DIM  = 1024
EFFICIENTNET_FEATURE_DIM = 1280
TOTAL_FEATURE_DIM        = (RESNET18_FEATURE_DIM +
                             DENSENET121_FEATURE_DIM +
                             EFFICIENTNET_FEATURE_DIM)   # 2816

# ═══════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ═══════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════
class NodulePatchDataset(torch.utils.data.Dataset):
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
        patch = np.expand_dims(patch, axis=0)            # (1, D, H, W)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]

# ═══════════════════════════════════════════════════════════════════════
#  Weight Loading (PREFIX-AWARE)
# ═══════════════════════════════════════════════════════════════════════
def load_backbone_weights(backbone_module, ckpt_path, label):
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_keys = set(backbone_module.state_dict().keys())

    direct_keys   = ckpt
    stripped_keys = {k[len("backbone."):]: v
                     for k, v in ckpt.items() if k.startswith("backbone.")}

    direct_match   = len(model_keys & set(direct_keys.keys()))
    stripped_match = len(model_keys & set(stripped_keys.keys()))

    weights_to_load = stripped_keys if stripped_match > direct_match else direct_keys
    strategy = "strip 'backbone.' prefix" if stripped_match > direct_match else "direct"
    print(f"  [{label}] direct={direct_match}, stripped={stripped_match} → {strategy}")

    missing, unexpected = backbone_module.load_state_dict(weights_to_load, strict=False)
    matched = len(model_keys) - len(missing)
    print(f"  [{label}] Loaded {matched}/{len(model_keys)} keys "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return matched

# ═══════════════════════════════════════════════════════════════════════
#  3D EfficientNet-B0  (must match the architecture used at training time)
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
                stride    = s     if i == 0 else 1
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
        x = self.global_pool(x)
        return x.flatten(1)    # (B, 1280)

# ═══════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════
class HybridThreeBranchFrozen(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()

        self.resnet18 = ResNet(
            block=ResNetBlock, layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3, n_input_channels=in_channels,
            num_classes=RESNET18_FEATURE_DIM,
        )
        self.resnet18.fc = nn.Identity()
        if RESNET18_WEIGHTS:
            load_backbone_weights(self.resnet18, RESNET18_WEIGHTS, "ResNet18")
        for p in self.resnet18.parameters(): p.requires_grad = False

        self.densenet121 = DenseNet121(
            spatial_dims=3, in_channels=in_channels, out_channels=1,
        )
        self.densenet121.class_layers.out = nn.Identity()
        if DENSENET121_WEIGHTS:
            load_backbone_weights(self.densenet121, DENSENET121_WEIGHTS, "DenseNet121")
        for p in self.densenet121.parameters(): p.requires_grad = False

        self.efficientnet_b0 = EfficientNet3D_B0(in_channels=in_channels)
        if EFFICIENTNET_WEIGHTS:
            load_backbone_weights(self.efficientnet_b0, EFFICIENTNET_WEIGHTS, "EfficientNet-B0")
        for p in self.efficientnet_b0.parameters(): p.requires_grad = False

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

    def forward(self, x):
        with torch.no_grad():
            f_r = self.resnet18(x)
            f_d = self.densenet121(x)
            f_e = self.efficientnet_b0(x)
        combined = torch.cat([f_r, f_d, f_e], dim=1)
        return self.mlp_head(combined).squeeze(1)

    def freeze_backbones(self):
        for p in self.resnet18.parameters():        p.requires_grad = False
        for p in self.densenet121.parameters():     p.requires_grad = False
        for p in self.efficientnet_b0.parameters(): p.requires_grad = False

    def unfreeze_backbones(self):
        for p in self.resnet18.parameters():        p.requires_grad = True
        for p in self.densenet121.parameters():     p.requires_grad = True
        for p in self.efficientnet_b0.parameters(): p.requires_grad = True

# ═══════════════════════════════════════════════════════════════════════
#  Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    desc_idx        = np.argsort(all_probs)[::-1]
    sorted_labels   = all_labels[desc_idx]
    tps             = np.cumsum(sorted_labels == 1)
    fps             = np.cumsum(sorted_labels == 0)
    fps_per_scan    = fps / total_scans
    sensitivity     = tps / total_positives
    froc_scores = {}
    for target in FROC_THRESHOLDS:
        valid_idx = np.where(fps_per_scan <= target)[0]
        froc_scores[f"{target} FP/scan"] = sensitivity[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
    return froc_scores

def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
    rng = np.random.RandomState(SEED)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[idx])) < 2:
            continue
        scores.append(roc_auc_score(np.array(y_true)[idx], np.array(y_probs)[idx]))
    scores = np.sort(np.array(scores))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

# ═══════════════════════════════════════════════════════════════════════
#  GradCAM++ Core
# ═══════════════════════════════════════════════════════════════════════
def grad_cam_plusplus_3d(fmap, grad):
    """
    GradCAM++ weights from a 3-D feature map and its gradient.

    fmap, grad : (1, C, D, H, W) tensors
    Returns    : (D, H, W) heatmap tensor (ReLU applied, un-normalised)
    """
    grad_2 = grad.pow(2)
    grad_3 = grad_2 * grad
    sum_act = fmap.sum(dim=(2, 3, 4), keepdim=True)
    num     = grad_2 * fmap
    denom   = 2 * grad_2 + sum_act * grad_3
    denom   = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha   = (num / denom).sum(dim=(2, 3, 4), keepdim=True)
    weights = F.relu(grad) * alpha
    heatmap = (weights * fmap).sum(dim=1)
    heatmap = F.relu(heatmap)

    if heatmap.max() < 1e-6:
        w_std   = grad.mean(dim=(2, 3, 4), keepdim=True)
        heatmap = F.relu((w_std * fmap).sum(dim=1))
        print("    ⚠ GradCAM++ zero heatmap — fell back to standard GradCAM.")

    return heatmap.squeeze(0)


def _find_layer(model, candidates):
    """Return the first (name, module) pair whose name contains any candidate string."""
    for c in candidates:
        for name, mod in model.named_modules():
            if c in name:
                return name, mod
    raise ValueError(f"None of the candidate layer names found in model: {candidates}")


def _normalise_to_uint8(arr):
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(arr)
    return np.uint8(255 * arr)


def produce_heatmap_branch(
        model,
        input_tensor,
        branch_name,          # "resnet18" | "densenet121" | "efficientnet_b0"
        layer_candidates,     # ordered list of substring matches
        class_idx=1,
    ):
    """
    Produce a GradCAM++ heatmap for one backbone branch of HybridThreeBranchFrozen.

    Frozen-backbone workaround (same approach as 2-branch version):
      - Re-enable gradients on the target backbone only.
      - Run target backbone directly (outside no_grad).
      - Run the other two branches under no_grad, detach their features.
      - Concatenate in the same order as training (rn18 | dn121 | effnet).
      - Call mlp_head → backward → GradCAM++.
      - Re-freeze target backbone immediately.
    """
    model.eval()

    # ── Identify target layer ────────────────────────────────────────────
    target_name, target_layer = _find_layer(model, layer_candidates)
    print(f"    [{branch_name}] GradCAM++ target layer : {target_name}")

    # ── Temporarily enable gradients on the target backbone ─────────────
    backbone = getattr(model, branch_name)
    for p in backbone.parameters():
        p.requires_grad = True

    features  = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        features['value'] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_backward_hook(bwd_hook)

    input_tensor = input_tensor.to(DEVICE)

    # ── Forward: target branch gets grad; the other two stay frozen ──────
    if branch_name == "resnet18":
        f_r = model.resnet18(input_tensor)                   # grad flows
        with torch.no_grad():
            f_d = model.densenet121(input_tensor)
            f_e = model.efficientnet_b0(input_tensor)
        combined = torch.cat([f_r,
                               f_d.detach(),
                               f_e.detach()], dim=1)

    elif branch_name == "densenet121":
        f_d = model.densenet121(input_tensor)                # grad flows
        with torch.no_grad():
            f_r = model.resnet18(input_tensor)
            f_e = model.efficientnet_b0(input_tensor)
        combined = torch.cat([f_r.detach(),
                               f_d,
                               f_e.detach()], dim=1)

    else:  # efficientnet_b0
        f_e = model.efficientnet_b0(input_tensor)            # grad flows
        with torch.no_grad():
            f_r = model.resnet18(input_tensor)
            f_d = model.densenet121(input_tensor)
        combined = torch.cat([f_r.detach(),
                               f_d.detach(),
                               f_e], dim=1)

    logit    = model.mlp_head(combined).squeeze(1)
    grad_val = 1.0 if class_idx == 1 else -1.0
    model.zero_grad()
    logit.backward(gradient=torch.full_like(logit, grad_val), retain_graph=True)

    fmap = features['value'].detach()
    grad = gradients['value'].detach()
    fwd_handle.remove()
    bwd_handle.remove()

    # ── Re-freeze target backbone ────────────────────────────────────────
    for p in backbone.parameters():
        p.requires_grad = False

    prob = torch.sigmoid(logit).item()
    print(f"    [{branch_name}] Logit {logit.item():.4f} → prob {prob:.4f}")
    print(f"    [{branch_name}] fmap {fmap.shape} | "
          f"grad min/mean/max: {grad.min():.5f}/{grad.mean():.5f}/{grad.max():.5f}")

    # ── GradCAM++ → 2-D slice ────────────────────────────────────────────
    heatmap_3d = grad_cam_plusplus_3d(fmap, grad)        # (D, H, W)
    d_center   = heatmap_3d.shape[0] // 2
    hm_slice   = heatmap_3d[d_center].cpu().numpy()
    hm_u8      = _normalise_to_uint8(hm_slice)

    # Original CT central slice
    vol        = input_tensor.squeeze(0).squeeze(0).cpu().numpy()
    orig_slice = vol[vol.shape[0] // 2]
    orig_u8    = _normalise_to_uint8(orig_slice)

    orig_resized = cv2.resize(orig_u8,  (224, 224))
    hm_resized   = cv2.resize(hm_u8,   (224, 224))
    orig_bgr     = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
    hm_color     = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    overlay      = cv2.addWeighted(orig_bgr, 0.5, hm_color, 0.5, 0)

    # Float [0,1] for matplotlib
    hm_display = hm_resized.astype(np.float32) / 255.0

    return orig_resized, hm_display, overlay, prob

# ═══════════════════════════════════════════════════════════════════════
#  Target Layer Definitions per Branch
# ═══════════════════════════════════════════════════════════════════════
RESNET18_TARGET_CANDIDATES = [
    "resnet18.layer3.1",      # second block of stage 3 (preferred)
    "resnet18.layer4.1",      # second block of stage 4
    "resnet18.layer4.0",      # first block of stage 4
    "resnet18.layer3",        # any sub-module of stage 3
]
DENSENET121_TARGET_CANDIDATES = [
    "densenet121.features.denseblock3",    # rich semantic features (preferred)
    "densenet121.features.denseblock4",    # deepest dense block
    "densenet121.features.transition3",    # after denseblock3
    "densenet121.features.denseblock2",    # fallback
]
EFFICIENTNET_TARGET_CANDIDATES = [
    "efficientnet_b0.stages.5",    # stage 6 (192-ch, preferred)
    "efficientnet_b0.stages.6",    # stage 7 (320-ch)
    "efficientnet_b0.stages.4",    # stage 5 (112-ch, fallback)
    "efficientnet_b0.head_conv",   # final 1×1 conv (last resort)
]

# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # ── Metadata ──────────────────────────────────────────────────────────
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df   = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    test_meta    = metadata[metadata["split"] == "test"].reset_index(drop=True)
    test_dataset = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

    # ── Load Model ────────────────────────────────────────────────────────
    model = HybridThreeBranchFrozen().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.freeze_backbones()
    model.eval()
    print("Model loaded.\n")

    # ── Full Test Inference ───────────────────────────────────────────────
    all_probs, all_labels, all_uids, all_paths = [], [], [], []
    with torch.no_grad():
        for patches, labels, uids, fpaths in tqdm(test_loader, desc="Predicting"):
            patches = patches.to(DEVICE)
            logits  = model(patches)
            prob    = torch.sigmoid(logits).cpu().item()
            all_probs.append(prob)
            all_labels.append(labels.item())
            all_uids.append(uids[0])
            all_paths.append(fpaths[0])

    # ── Scan-level aggregation ────────────────────────────────────────────
    patient_dict = {}
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

    # ── Metrics ───────────────────────────────────────────────────────────
    auc   = roc_auc_score(y_true_pat, y_probs_pat) if len(np.unique(y_true_pat)) > 1 else 0.5
    auprc = average_precision_score(y_true_pat, y_probs_pat)
    f1    = f1_score(y_true_pat, y_pred_pat)
    sens  = recall_score(y_true_pat, y_pred_pat)
    cm    = confusion_matrix(y_true_pat, y_pred_pat, labels=[0, 1])
    tn_v, fp_v, fn_v, tp_v = cm.ravel()
    spec  = tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 0.0
    froc  = calculate_candidate_froc(all_labels, all_probs, total_scans)
    ci_lo, ci_hi = calculate_95_ci(y_true_pat, y_probs_pat)

    print("\n" + "=" * 57)
    print("FINAL TEST METRICS — 3-Branch Hybrid (Frozen)")
    print("=" * 57)
    print(f"AUROC            : {auc:.4f}  (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")
    print(f"AUPRC            : {auprc:.4f}")
    print(f"F1               : {f1:.4f}")
    print(f"Sensitivity      : {sens:.4f}")
    print(f"Specificity      : {spec:.4f}")
    print(f"Confusion Matrix : TP={tp_v}  FN={fn_v}  FP={fp_v}  TN={tn_v}")
    print(f"\nFROC (FP/scan → Sensitivity):")
    for k, v in froc.items():
        print(f"  {k:14s}: {v:.4f}")
    print("=" * 57 + "\n")

    # ── Save Predictions & Calibration ───────────────────────────────────
    pred_df = pd.DataFrame({
        'seriesuid':   all_uids,
        'label':       all_labels,
        'probability': all_probs,
        'filepath':    all_paths,
    })
    pred_df.to_csv("hybrid3_frozen_test_predictions.csv", index=False)
    print("[Saved] hybrid3_frozen_test_predictions.csv")

    frac_pos, mean_pred = calibration_curve(y_true_pat, y_probs_pat, n_bins=10)
    pd.DataFrame({
        'mean_predicted_probability': mean_pred,
        'fraction_of_positives':      frac_pos,
    }).to_csv("hybrid3_frozen_calibration_curve.csv", index=False)
    print("[Saved] hybrid3_frozen_calibration_curve.csv")

    # ── Select GradCAM++ Cases ────────────────────────────────────────────
    patient_pred = pd.DataFrame({
        'seriesuid': list(patient_dict.keys()),
        'label':     y_true_pat,
        'prob':      y_probs_pat,
        'pred':      y_pred_pat,
    })
    tp_df = patient_pred[(patient_pred['label'] == 1) & (patient_pred['pred'] == 1)]
    fn_df = patient_pred[(patient_pred['label'] == 1) & (patient_pred['pred'] == 0)]
    fp_df = patient_pred[(patient_pred['label'] == 0) & (patient_pred['pred'] == 1)]
    tn_df = patient_pred[(patient_pred['label'] == 0) & (patient_pred['pred'] == 0)]
    print(f"GradCAM++ case pool — TP:{len(tp_df)}  FN:{len(fn_df)}  "
          f"FP:{len(fp_df)}  TN:{len(tn_df)}")

    selected_cases = []
    for i in range(min(4, len(tp_df))): selected_cases.append(('TP', i+1, tp_df.iloc[i]))
    for i in range(min(3, len(fn_df))): selected_cases.append(('FN', i+1, fn_df.iloc[i]))
    for i in range(min(3, len(fp_df))): selected_cases.append(('FP', i+1, fp_df.iloc[i]))
    for i in range(min(3, len(tn_df))): selected_cases.append(('TN', i+1, tn_df.iloc[i]))

    # ── GradCAM++ Loop ────────────────────────────────────────────────────
    # Each row: (orig, rn18_hm, rn18_ov, dn121_hm, dn121_ov, eff_hm, eff_ov)
    rows_data, row_titles = [], []

    for case_type, case_num, row in selected_cases:
        uid         = row['seriesuid']
        patient_df  = pred_df[pred_df['seriesuid'] == uid]
        best_patch  = patient_df.loc[patient_df['probability'].idxmax()]
        patch_path  = best_patch['filepath']
        idx_in_meta = test_meta[test_meta['filepath'] == patch_path].index[0]
        img_tensor, label_t, _, _ = test_dataset[idx_in_meta]
        input_t   = img_tensor.unsqueeze(0).to(DEVICE)
        class_idx = 1 if row['label'] == 1 else 0

        print(f"\n{'='*65}")
        print(f"Case {case_type}-{case_num} | UID: {uid[:12]}... | "
              f"Prob: {row['prob']:.4f} | Label: {int(row['label'])}")
        print(f"{'='*65}")

        orig_u8, rn18_hm, rn18_ov, prob_rn18 = produce_heatmap_branch(
            model, input_t.clone(), "resnet18",
            RESNET18_TARGET_CANDIDATES, class_idx=class_idx,
        )
        _, dn121_hm, dn121_ov, prob_dn121 = produce_heatmap_branch(
            model, input_t.clone(), "densenet121",
            DENSENET121_TARGET_CANDIDATES, class_idx=class_idx,
        )
        _, eff_hm, eff_ov, prob_eff = produce_heatmap_branch(
            model, input_t.clone(), "efficientnet_b0",
            EFFICIENTNET_TARGET_CANDIDATES, class_idx=class_idx,
        )

        rows_data.append((orig_u8, rn18_hm, rn18_ov, dn121_hm, dn121_ov, eff_hm, eff_ov))
        row_titles.append(
            f"{case_type}-{case_num} | UID:{uid[:8]}... | "
            f"Prob:{row['prob']:.3f} | Label:{int(row['label'])}"
        )

    # ── Plotting ──────────────────────────────────────────────────────────
    n = len(rows_data)
    if n == 0:
        print("No cases to visualise.")
        return

    col_headers = [
        "Original CT",
        "ResNet18\nGradCAM++", "ResNet18\nOverlay",
        "DenseNet121\nGradCAM++", "DenseNet121\nOverlay",
        "EfficientNet-B0\nGradCAM++", "EfficientNet-B0\nOverlay",
    ]
    n_cols = 7
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 2.8, n * 3.0))
    if n == 1:
        axes = axes.reshape(1, -1)

    for col_i, header in enumerate(col_headers):
        axes[0, col_i].set_title(header, fontsize=8, fontweight='bold', pad=4)

    for i, (orig, rn18_hm, rn18_ov,
            dn121_hm, dn121_ov,
            eff_hm,  eff_ov) in enumerate(rows_data):

        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_ylabel(row_titles[i], fontsize=6.5, rotation=0,
                               labelpad=140, va='center')

        axes[i, 1].imshow(rn18_hm,  cmap='jet', vmin=0, vmax=1)
        axes[i, 2].imshow(cv2.cvtColor(rn18_ov,  cv2.COLOR_BGR2RGB))

        axes[i, 3].imshow(dn121_hm, cmap='jet', vmin=0, vmax=1)
        axes[i, 4].imshow(cv2.cvtColor(dn121_ov, cv2.COLOR_BGR2RGB))

        axes[i, 5].imshow(eff_hm,   cmap='jet', vmin=0, vmax=1)
        axes[i, 6].imshow(cv2.cvtColor(eff_ov,   cv2.COLOR_BGR2RGB))

        for ax in axes[i]:
            ax.axis('off')

    plt.suptitle(
        "GradCAM++ Analysis — 3-Branch Hybrid (Frozen Backbones)\n"
        "Col order: Original | ResNet18 heat | ResNet18 overlay | "
        "DenseNet121 heat | DenseNet121 overlay | "
        "EfficientNet-B0 heat | EfficientNet-B0 overlay",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    plt.savefig("gradcampp_hybrid3_frozen_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[Saved] gradcampp_hybrid3_frozen_analysis.png")


if __name__ == "__main__":
    main()