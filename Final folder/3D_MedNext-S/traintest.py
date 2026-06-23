"""
3D MedNeXt (from scratch) with MLP Head for Lung Nodule Classification
FULLY ALIGNED with TinyViT/DenseNet121/EfficientNet-B0 scripts:
    - Patient-level evaluation (max-prob aggregation per seriesuid)
    - Candidate-level FROC
    - Fixed 0.5 classification threshold
    - Same data split, pos_weight, scheduler, mixed precision, etc.

Architecture:
    - MedNeXt encoder (no pretrain, trained from scratch)
    - Input: 64x64x64 single-channel CT patch
    - MedNeXt uses large-kernel depthwise conv + transformer-style MLP blocks
    - Hierarchical feature extraction: 4 encoding stages
    - Global average pool → MLP head identical to TinyViT

Installation:
    pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
    pip install monai torch timm scikit-learn tqdm
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
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    recall_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from monai.transforms import (
    Compose,
    RandRotate90d,
    RandFlipd,
    RandGaussianNoised,
    Resized,
)
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------------------ Configuration ------------------------------
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE = 8          # 3D volumes are memory-heavy; start at 2, increase if VRAM allows
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = None       # computed from training split below

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# MedNeXt config — deliberately conservative for small dataset
MEDNEXT_BASE_CHANNELS = 32   # 32 → ~18M params; increase to 48 if VRAM allows
MEDNEXT_EXP_R = 2            # expansion ratio in MLP blocks
MEDNEXT_KERNEL_SIZE = 3      # depthwise conv kernel; 3 safest from scratch; 5 if VRAM ok
MEDNEXT_BLOCKS = [2, 2, 2, 2]   # blocks per encoder stage (4 stages)
MEDNEXT_DROPOUT = 0.3        # dropout in MLP head


# ============================== Reproducibility ==============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================== Dataset ======================================
class NodulePatchDataset3D(Dataset):
    """
    Loads the 64x64x64 volume as a single-channel 3D patch.
    No intensity scaling — raw CT HU values preserved (same philosophy as TinyViT).
    Augmentation in MONAI transform pipeline.
    """
    def __init__(self, metadata_df, data_dir, transforms=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = os.path.basename(row["filepath"])
        split = row["split"]
        label = int(row["label"])
        subfolder = "pos" if label == 1 else "neg"

        local_path = os.path.join(self.data_dir, split, subfolder, filename)
        vol = np.load(local_path).astype(np.float32)   # (64, 64, 64)

        # Add channel dim → (1, 64, 64, 64)
        vol = vol[np.newaxis, ...]

        data_dict = {"img": vol}

        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        img = data_dict["img"]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)

        return img, torch.tensor(label, dtype=torch.float32), row["seriesuid"]


# ============================== Transforms ===================================
def get_transforms(mode="train"):
    """
    3D transforms aligned to TinyViT philosophy:
    - Only discrete 90° rotations and flips (no intensity scaling)
    - Gaussian noise with low probability
    - NO resizing — 64x64x64 is MedNeXt's direct input
    - Augmentation only during training
    """
    keys = ["img"]

    if mode == "train":
        return Compose([
            # All three spatial axes for full 3D augmentation
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 2)),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=keys, prob=0.2, std=0.01),
        ])
    else:
        return Compose([])   # identity — no augmentation at val/test


# ============================== MedNeXt Architecture =========================
# We implement MedNeXt from scratch here so there is no dependency on the
# external repo for the core training loop. This mirrors what the TinyViT
# script does (timm model defined inline via TinyViTWithMLPHead).
# The architecture below is faithful to the original MedNeXt paper
# (Roy et al., 2023) encoder path only.

class MedNeXtBlock3D(nn.Module):
    """
    Single MedNeXt block:
        1. Depthwise conv (large kernel) — spatial mixing (token mixer)
        2. LayerNorm
        3. Pointwise expand → GELU → pointwise contract (inverted MLP bottleneck)
        4. Residual connection
        5. Optional stochastic depth (drop path)

    This is the direct 3D analogue of a ConvNeXt block, scaled for medical volumes.
    """
    def __init__(self, in_channels, exp_r=2, kernel_size=3, drop_path_rate=0.0):
        super().__init__()
        mid_channels = int(in_channels * exp_r)

        # Depthwise conv — groups=in_channels means each channel is independent
        # This is the "token mixer" equivalent in transformer terminology
        self.dw_conv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

        # Inverted bottleneck MLP (pointwise)
        self.pw_expand = nn.Linear(in_channels, mid_channels)
        self.act = nn.GELU()
        self.pw_contract = nn.Linear(mid_channels, in_channels)

        # Stochastic depth for regularisation — important from scratch
        self.drop_path_rate = drop_path_rate

    def drop_path(self, x):
        """Per-sample stochastic depth during training."""
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        # shape: (B, 1, 1, 1, 1) — drop entire samples
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

    def forward(self, x):
        # x: (B, C, D, H, W)
        residual = x

        # Depthwise spatial mixing
        x = self.dw_conv(x)

        # Move channel to last for LayerNorm + Linear
        x = x.permute(0, 2, 3, 4, 1)          # (B, D, H, W, C)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_contract(x)
        x = x.permute(0, 4, 1, 2, 3)          # (B, C, D, H, W)

        # Stochastic depth residual
        x = self.drop_path(x) + residual
        return x


class MedNeXtDownBlock3D(nn.Module):
    """
    Downsampling block: strided depthwise conv (stride=2) reduces spatial dims
    while doubling channels — exactly like Swin-T's patch merging.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.dw_conv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        # Pointwise projection to new channel count
        self.pw_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: (B, C_in, D, H, W)
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 4, 1)          # (B, D, H, W, C_in)
        x = self.norm(x)
        x = self.pw_proj(x)                    # (B, D, H, W, C_out)
        x = x.permute(0, 4, 1, 2, 3)          # (B, C_out, D/2, H/2, W/2)
        return x


class MedNeXtStem3D(nn.Module):
    """
    Stem: initial patchification.
    4x4x4 non-overlapping conv → projects raw voxels to embedding space.
    Followed by LayerNorm for stable training from scratch.
    Input: (B, 1, 64, 64, 64)
    Output: (B, base_channels, 16, 16, 16)  — 4x spatial reduction
    """
    def __init__(self, in_channels=1, base_channels=32, patch_size=4):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, base_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(base_channels, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)                       # (B, C, D/4, H/4, W/4)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class MedNeXtEncoder3D(nn.Module):
    """
    MedNeXt Encoder — 4-stage hierarchical feature extraction.

    With base_channels=32, patch_size=4, input=64^3:
        Stem   : (B,  1, 64, 64, 64) → (B, 32, 16, 16, 16)   4x reduction
        Stage 0: (B, 32, 16, 16, 16) → (B, 32, 16, 16, 16)   no spatial change
        Down 0 : (B, 32, 16, 16, 16) → (B, 64,  8,  8,  8)   2x reduction
        Stage 1: (B, 64,  8,  8,  8) → (B, 64,  8,  8,  8)
        Down 1 : (B, 64,  8,  8,  8) → (B,128,  4,  4,  4)   2x reduction
        Stage 2: (B,128,  4,  4,  4) → (B,128,  4,  4,  4)
        Down 2 : (B,128,  4,  4,  4) → (B,256,  2,  2,  2)   2x reduction
        Stage 3: (B,256,  2,  2,  2) → (B,256,  2,  2,  2)
        GAP    : (B,256,  2,  2,  2) → (B,256)

    Total 8x spatial reduction after stem = 64/4/2/2/2 = 2 at bottleneck.
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        exp_r=2,
        kernel_size=3,
        block_counts=(2, 2, 2, 2),
        drop_path_rate=0.1,
        patch_size=4,
    ):
        super().__init__()

        # Stem: patchification
        self.stem = MedNeXtStem3D(in_channels, base_channels, patch_size)

        # Build 4 encoder stages with downsampling between them
        channels = [base_channels * (2 ** i) for i in range(4)]
        # channels = [32, 64, 128, 256] with base_channels=32

        # Stochastic depth linearly increases across blocks
        total_blocks = sum(block_counts)
        dp_rates = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, total_blocks)
        ]
        dp_idx = 0

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for stage_idx, n_blocks in enumerate(block_counts):
            c = channels[stage_idx]
            stage_blocks = nn.Sequential(*[
                MedNeXtBlock3D(
                    in_channels=c,
                    exp_r=exp_r,
                    kernel_size=kernel_size,
                    drop_path_rate=dp_rates[dp_idx + b],
                )
                for b in range(n_blocks)
            ])
            self.stages.append(stage_blocks)
            dp_idx += n_blocks

            # Downsampling after each stage except the last
            if stage_idx < len(block_counts) - 1:
                self.downsamples.append(
                    MedNeXtDownBlock3D(channels[stage_idx], channels[stage_idx + 1], kernel_size)
                )

        # Final norm before pooling
        self.final_norm = nn.LayerNorm(channels[-1], eps=1e-6)
        self.num_features = channels[-1]   # 256 with base_channels=32

    def forward(self, x):
        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # Global average pooling → (B, C)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)          # (B, D, H, W, C)
        x = self.final_norm(x)
        x = x.permute(0, 4, 1, 2, 3)          # (B, C, D, H, W)
        x = x.mean(dim=[2, 3, 4])             # Global average pool → (B, C)
        return x


class MedNeXt3DWithMLPHead(nn.Module):
    """
    Full model: MedNeXt encoder + MLP classification head.
    MLP head is IDENTICAL to TinyViT's head for fair comparison.
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=MEDNEXT_BASE_CHANNELS,
        exp_r=MEDNEXT_EXP_R,
        kernel_size=MEDNEXT_KERNEL_SIZE,
        block_counts=MEDNEXT_BLOCKS,
        drop_path_rate=0.1,
        num_classes=1,
        dropout=MEDNEXT_DROPOUT,
    ):
        super().__init__()

        self.encoder = MedNeXtEncoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            block_counts=block_counts,
            drop_path_rate=drop_path_rate,
        )

        feat_dim = self.encoder.num_features   # 256 with base_channels=32

        # MLP head — IDENTICAL to TinyViT for fair comparison
        self.mlp_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)             # (B, feat_dim)
        out = self.mlp_head(features)          # (B, 1)
        return out.squeeze(1)                  # (B,)


# ============================== Evaluation Metrics ===========================
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    """Candidate-level FROC — exactly as in TinyViT/DenseNet scripts."""
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

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
        sens = sensitivity[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
        froc_scores[f"{target} FP/scan"] = sens

    return froc_scores


def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
    bootstrapped_scores = []
    rng = np.random.RandomState(SEED)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        score = roc_auc_score(
            np.array(y_true)[indices],
            np.array(y_probs)[indices]
        )
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(np.array(bootstrapped_scores))
    return np.percentile(sorted_scores, 2.5), np.percentile(sorted_scores, 97.5)


def evaluate_model(loader, model, device, pos_weight_tensor, desc="Validating", return_preds=False):
    """
    Patient-level evaluation identical to TinyViT/DenseNet/EfficientNet:
        - Aggregate per seriesuid with max prob & max label
        - Fixed 0.5 threshold for classification metrics
        - FROC computed candidate-wise
    """
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc=desc):
            patches, labels = patches.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(patches)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy()

            if probs.ndim == 0:
                probs = [float(probs)]
                labels_list = [float(labels.cpu())]
            else:
                probs = probs.tolist()
                labels_list = labels.cpu().tolist()

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
            patient_dict[uid]['prob'] = max(patient_dict[uid]['prob'], prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)

    y_true_patient = [v['label'] for v in patient_dict.values()]
    y_probs_patient = [v['prob'] for v in patient_dict.values()]
    y_pred_patient = [1 if p >= 0.5 else 0 for p in y_probs_patient]
    total_scans = len(patient_dict)

    avg_loss = running_loss / len(loader.dataset)

    auc = roc_auc_score(y_true_patient, y_probs_patient) if len(np.unique(y_true_patient)) > 1 else 0.5
    auprc = average_precision_score(y_true_patient, y_probs_patient)
    f1 = f1_score(y_true_patient, y_pred_patient)
    sens = recall_score(y_true_patient, y_pred_patient)

    cm = confusion_matrix(y_true_patient, y_pred_patient, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    froc = calculate_candidate_froc(all_labels, all_probs, total_scans)

    if return_preds:
        ci_lower, ci_upper = calculate_95_ci(y_true_patient, y_probs_patient)
        print(f"AUROC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_patient, y_probs_patient, n_bins=10
        )
        cal_df = pd.DataFrame({
            'mean_predicted_probability': mean_predicted_value,
            'fraction_of_positives': fraction_of_positives
        })
        cal_df.to_csv("mednext3d_calibration_curve.csv", index=False)
        print("[Saved] Calibration data exported to 'mednext3d_calibration_curve.csv'")

        pred_df = pd.DataFrame({
            'seriesuid': all_uids,
            'label': all_labels,
            'probability': all_probs
        })
        return avg_loss, auc, auprc, f1, sens, spec, froc, pred_df

    return avg_loss, auc, auprc, f1, sens, spec, froc


# ============================== Main =========================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ Data
    print("Loading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)

    # Compute pos_weight from training split (identical to TinyViT)
    train_df = metadata[metadata["split"] == "train"]
    neg_count = (train_df["label"] == 0).sum()
    pos_count = (train_df["label"] == 1).sum()
    pos_weight_value = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"Class balance (train): neg={neg_count}, pos={pos_count}, pos_weight={pos_weight_value:.2f}")

    # ------------------------------------------------------------ Transforms
    train_transforms = get_transforms("train")
    val_transforms   = get_transforms("val")

    train_dataset = NodulePatchDataset3D(train_meta, DATA_DIR, transforms=train_transforms)
    val_dataset   = NodulePatchDataset3D(val_meta,   DATA_DIR, transforms=val_transforms)
    test_dataset  = NodulePatchDataset3D(test_meta,  DATA_DIR, transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )

    # --------------------------------------------------------------- Model
    model = MedNeXt3DWithMLPHead().to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Model Summary ---")
    print(f"Model          : 3D MedNeXt (from scratch) + MLP Head")
    print(f"Base channels  : {MEDNEXT_BASE_CHANNELS}")
    print(f"Kernel size    : {MEDNEXT_KERNEL_SIZE}")
    print(f"Expansion ratio: {MEDNEXT_EXP_R}")
    print(f"Block counts   : {MEDNEXT_BLOCKS}")
    print(f"Feature dim    : {model.encoder.num_features}")
    print(f"Total Params   : {total_params:,}")
    print(f"Trainable      : {trainable_params:,}")
    print("---------------------\n")

    # ------------------------------------------------ Loss / Optimiser / LR
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Identical warmup + cosine schedule to TinyViT
    warmup_epochs   = 5
    warmup_scheduler  = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler  = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    scaler = GradScaler('cuda')

    # --------------------------------------------------------- Training loop
    best_val_auc    = 0.0
    patience_counter = 0
    history         = []

    print("--- Starting Training ---")
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

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc = evaluate_model(
            val_loader, model, device, pos_weight, desc="Validating"
        )

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

        print(f"\nEpoch {epoch:3d} Results:")
        print(f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Patient-AUROC: {val_auc:.4f} | AUPRC: {val_auprc:.4f} | "
              f"F1: {val_f1:.4f} | Sens: {val_sens:.4f}")

        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_mednext3d.pth")
            print(f"  --> [Saved] New Best Validation AUROC: {best_val_auc:.4f}")

            best_metrics = {
                "epoch":       epoch,
                "auc":         val_auc,
                "auprc":       val_auprc,
                "f1":          val_f1,
                "sensitivity": val_sens,
                "specificity": val_spec,
            }
            with open("best_metrics_mednext3d.json", "w") as f:
                json.dump(best_metrics, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping triggered after {epoch} epochs ***")
                break

    pd.DataFrame(history).to_csv("mednext3d_training_log.csv", index=False)
    print("\n[Saved] Training history exported to 'mednext3d_training_log.csv'")

    # ---------------------------------------------------------- Final Test
    print("\n" + "=" * 50)
    print("LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("=" * 50)

    model.load_state_dict(
        torch.load("best_model_mednext3d.pth", map_location=device, weights_only=True)
    )

    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = \
        evaluate_model(test_loader, model, device, pos_weight, desc="Testing", return_preds=True)

    pred_df.to_csv("mednext3d_test_predictions.csv", index=False)
    print("[Saved] Test predictions exported to 'mednext3d_test_predictions.csv'")

    print("\n--- FINAL TEST RESULTS (SCAN-LEVEL) ---")
    print(f"AUROC (Primary):           {test_auc:.4f}")
    print(f"AUPRC:                     {test_auprc:.4f}")
    print(f"F1 Score:                  {test_f1:.4f}")
    print(f"Sensitivity (Recall):      {test_sens:.4f}")
    print(f"Specificity:               {test_spec:.4f}")

    print(f"\n--- FROC (False Positives per Scan vs Sensitivity) ---")
    for fp_rate, sensitivity in test_froc.items():
        print(f"  {fp_rate:10s} : {sensitivity:.4f}")
    print("====================================================\n")


if __name__ == "__main__":
    main()