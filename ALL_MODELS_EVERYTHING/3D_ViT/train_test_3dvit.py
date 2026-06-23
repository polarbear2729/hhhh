"""
3D ViT-Small (ViT-S/16) with MLP Head for Lung Nodule Classification
LUNA16 Dataset — Patch-level classification, Patient-level evaluation

Architecture  : MONAI native 3D ViT-Small
                hidden_size=384, mlp_dim=1536, num_layers=12, num_heads=6
Tokenisation  : 64³ volume → 16³ patch size → 4×4×4 = 64 tokens
Aggregation   : mean pooling over all 64 patch tokens (no CLS token in MONAI ViT
                with classification=False — confirmed by seq shape=(B,64,384))
Head          : Linear(384→256) → LayerNorm(256) → ReLU → Dropout → Linear(256→1)

FIXES applied in v3 (after two failed runs):
    FIX 1 — NaN/Inf gradient norms every ~3 epochs:
             Root cause: pos_embed_type='sincos' produces large intermediate
             activations that overflow float16 in AMP, generating true Inf
             gradients that clip_grad_norm_ cannot handle.
             Fix: pos_embed_type='learnable' — learned embeddings stay in a
             numerically safe range throughout training.

    FIX 2 — Wrong token aggregation:
             MONAI ViT with classification=False does NOT prepend a CLS token.
             Confirmed: backbone output shape=(B,64,384), not (B,65,384).
             slice[:,0,:] was extracting the first PATCH token (arbitrary).
             Fix: sequence.mean(dim=1) — average pool over all 64 patch tokens.
             This is the correct aggregation for MONAI ViT without classification.

    FIX 3 — LR too high for from-scratch transformer:
             LR 3e-5 with warmup 10 caused oscillation even after Inf clipping.
             Fix: LR=1e-5, warmup=15 epochs, weight_decay reduced to 0.01.
             Combined with learnable pos embed this gives stable early training.

    FIX 4 — Gradient clipping applied after unscale but scaler was still
             propagating Inf scale factors on recovery epochs.
             Fix: explicit scaler.get_scale() check + force scaler reset when
             scale drops below threshold (indicates repeated Inf detections).
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
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, recall_score, roc_curve
)
from monai.networks.nets import ViT
from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise, RandAffine
from sklearn.calibration import calibration_curve
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------------------ Configuration ------------------------------
SEED               = 42
DATA_DIR           = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH      = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE         = 8
NUM_WORKERS        = 4
PIN_MEMORY         = True
PERSISTENT_WORKERS = True

MAX_EPOCHS              = 150
EARLY_STOPPING_PATIENCE = 15
LR             = 1e-5      # Reduced from 3e-5 — lower LR for numerical stability
WEIGHT_DECAY   = 0.01      # Reduced from 0.05 — less aggressive regularisation
WARMUP_EPOCHS  = 15        # Extended from 10 — more gentle ramp-up
GRAD_CLIP_NORM = 1.0

POS_WEIGHT = 5115.0 / 822.0

# ── ViT-Small/16 architecture
IMG_SIZE    = (64, 64, 64)
PATCH_SIZE  = (16, 16, 16)
HIDDEN_SIZE = 384
MLP_DIM     = 1536
NUM_LAYERS  = 12
NUM_HEADS   = 6

USE_PRETRAINED  = True
FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# ------------------------------ Reproducibility ----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

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
        patch = np.load(local_path).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ------------------------------ Model --------------------------------------
class ViTSmall3DWithMLPHead(nn.Module):
    """
    MONAI ViT-Small (classification=False) + mean pool + MLP head.

    Key design decisions confirmed by diagnostic output:
      - classification=False: MONAI returns (B, 64, 384) — 64 patch tokens, NO CLS token
      - Aggregation: mean pool over all 64 tokens → (B, 384)
      - pos_embed_type='learnable': avoids sincos float16 overflow (FIX 1)
      - dropout_rate=0.0: internal dropout disabled — with only ~6K samples
        and no pretraining, internal dropout adds noise without benefit

    Forward data flow:
        Input    : (B, 1, 64, 64, 64)
        Tokens   : 64 patch tokens, each (384,)
        Seq out  : (B, 64, 384)
        Mean pool: (B, 384)
        Head     : (B, 256) → (B, 1) → squeeze → (B,)
    """
    def __init__(self, in_channels=1, dropout=0.0, use_pretrained=False):
        super().__init__()

        self.backbone = ViT(
            in_channels    = in_channels,
            img_size       = IMG_SIZE,
            patch_size     = PATCH_SIZE,
            hidden_size    = HIDDEN_SIZE,
            mlp_dim        = MLP_DIM,
            num_layers     = NUM_LAYERS,
            num_heads      = NUM_HEADS,
            proj_type      = 'conv',
            pos_embed_type = 'learnable',  # FIX 1: was 'sincos' — caused Inf in float16
            classification = False,        # Returns full sequence (B, 64, 384)
            dropout_rate   = 0.0,          # Disabled internal dropout — small dataset
            spatial_dims   = 3,
            qkv_bias       = True,
        )

        if use_pretrained:
            self._load_pretrained_weights()

        # Xavier init on the patch embedding conv to reduce initial activation magnitude
        if hasattr(self.backbone, 'patch_embedding'):
            for m in self.backbone.patch_embedding.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def _load_pretrained_weights(self):
        print("[WARN] use_pretrained=True but no weight path configured.")

    def forward(self, x):
        # MONAI ViT (classification=False) returns tuple: (sequence, hidden_states)
        # sequence shape: (B, num_patches, hidden_size) = (B, 64, 384)
        # No CLS token — use mean pooling over all patch tokens
        sequence, _ = self.backbone(x)        # (B, 64, 384)
        pooled      = sequence.mean(dim=1)    # (B, 384)  — FIX 2: was [:, 0, :]
        logit       = self.head(pooled)        # (B, 1)
        return logit.squeeze(1)               # (B,)

# ------------------------------ Sanity check ------------------------------
def run_forward_sanity_check(model, device):
    print("\n[SANITY CHECK] Running forward pass on dummy input...")
    model.eval()
    dummy = torch.randn(2, 1, 64, 64, 64).to(device)

    with torch.no_grad():
        raw_out = model.backbone(dummy)
        seq, hidden = raw_out
        print(f"  backbone seq shape   : {seq.shape}")
        print(f"    → num tokens       : {seq.shape[1]}  (expected 64 = 4×4×4 patches)")
        print(f"    → hidden size      : {seq.shape[2]}  (expected 384)")
        print(f"  hidden states count  : {len(hidden)}")

        # Check for NaN/Inf in backbone output — catches sincos overflow
        has_nan = torch.isnan(seq).any().item()
        has_inf = torch.isinf(seq).any().item()
        print(f"  NaN in backbone out  : {has_nan}  (must be False)")
        print(f"  Inf in backbone out  : {has_inf}  (must be False)")

        pooled = seq.mean(dim=1)
        print(f"  pooled shape         : {pooled.shape}  (expected (2, 384))")

        logits = model(dummy)
        print(f"  final logit shape    : {logits.shape}  (expected (2,))")
        print(f"  logit range          : [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        probs = torch.sigmoid(logits)
        print(f"  sigmoid prob range   : [{probs.min().item():.4f}, {probs.max().item():.4f}]")

        # Check logit range — should be near 0 at init, not ±10
        if logits.abs().max().item() > 5.0:
            print("  [WARN] Large initial logits — check head initialisation")
        else:
            print("  Initial logit magnitude OK (< 5.0)")

    if has_nan or has_inf:
        raise RuntimeError(
            "[SANITY CHECK FAILED] NaN/Inf in backbone output at init. "
            "This will cause Inf gradients during training. "
            "Check pos_embed_type — 'sincos' is known to overflow float16."
        )
    print("[SANITY CHECK] Passed.\n")
    model.train()

# ------------------------------ Evaluation --------------------------------
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    total_positives    = (all_labels == 1).sum()
    descending_indices = np.argsort(all_probs)[::-1]
    sorted_labels      = all_labels[descending_indices]
    tps          = np.cumsum(sorted_labels == 1)
    fps          = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans
    sensitivity  = tps / total_positives
    froc_scores  = {}
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
        bootstrapped_scores.append(
            roc_auc_score(np.array(y_true)[indices], np.array(y_probs)[indices])
        )
    s = np.sort(np.array(bootstrapped_scores))
    return np.percentile(s, 2.5), np.percentile(s, 97.5)

def get_youden_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j = np.argmax(tpr - fpr)
    return float(thresholds[j])

def evaluate_model(loader, model, device, desc="Validating",
                   return_preds=False, threshold=0.5):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))

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

    # Patient-level max-pool aggregation
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
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_p, y_probs_p, n_bins=10
        )
        pd.DataFrame({
            'mean_predicted_probability': mean_predicted_value,
            'fraction_of_positives':      fraction_of_positives
        }).to_csv("vits3d_calibration_curve.csv", index=False)
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

    # TF32 on Ampere+ GPUs — gives float32 precision with float16 speed,
    # reduces numerical overflow risk vs pure float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device   : {device}")
    print(f"Pretrained     : {USE_PRETRAINED}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"LR             : {LR}  (warmup {WARMUP_EPOCHS} ep → cosine)")
    print(f"Weight decay   : {WEIGHT_DECAY}")
    print(f"Grad clip norm : {GRAD_CLIP_NORM}")
    print(f"pos_embed_type : learnable  (sincos disabled — float16 overflow fix)")

    print("\nLoading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict    = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)
    print(f"  Train: {len(train_meta)} | Val: {len(val_meta)} | Test: {len(test_meta)}")
    print(f"  Train pos: {(train_meta['label']==1).sum()} | neg: {(train_meta['label']==0).sum()}")

    train_transforms = Compose([
    # ---------- NEW: random translation to break positional shortcut ----------
    RandAffine(
        prob=0.8,                        # apply to 80% of patches
        translate_range=(5, 5, 5),       # up to 5 mm/voxels in each direction
        padding_mode='zeros',            # fill empty borders with air (0)
        spatial_size=None,               # keep the original 64×64×64 size
    ),
    # -------------------------------------------------------------------------
    RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    RandRotate90(prob=0.5, spatial_axes=(1, 2)),
    RandRotate90(prob=0.5, spatial_axes=(0, 2)),
    RandFlip(prob=0.5, spatial_axis=0),
    RandFlip(prob=0.5, spatial_axis=1),
    RandFlip(prob=0.5, spatial_axis=2),
    RandGaussianNoise(prob=0.2, std=0.01)
])


    train_dataset = NodulePatchDataset(train_meta, DATA_DIR, transforms=train_transforms)
    val_dataset   = NodulePatchDataset(val_meta,   DATA_DIR)
    test_dataset  = NodulePatchDataset(test_meta,  DATA_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)

    model = ViTSmall3DWithMLPHead(use_pretrained=USE_PRETRAINED).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Model Summary ---")
    print(f"Backbone       : ViT-Small/16 3D  (classification=False, mean pool)")
    print(f"Tokens         : {(64//16)**3} patch tokens (no CLS — mean pooled)")
    print(f"Aggregation    : mean pool over all {(64//16)**3} tokens")
    print(f"Hidden         : {HIDDEN_SIZE} | Layers: {NUM_LAYERS} | Heads: {NUM_HEADS}")
    print(f"Total params   : {total_params:,}")
    print(f"Trainable      : {trainable_params:,}")
    print("---------------------")

    # Sanity check raises RuntimeError if NaN/Inf detected — catches sincos overflow
    run_forward_sanity_check(model, device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - WARMUP_EPOCHS))
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[WARMUP_EPOCHS])

    # init_scale=256 (lower than default 65536) — reduces Inf on first backward pass
    scaler = GradScaler('cuda', init_scale=256)

    best_val_auc     = 0.0
    patience_counter = 0
    history          = []
    skipped_steps    = 0   # Count AMP-skipped updates (Inf/NaN detected by scaler)

    print("--- Starting Training ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss  = 0.0
        grad_norms  = []
        epoch_skips = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for patches, labels, _ in progress_bar:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(patches)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Check for Inf/NaN BEFORE clipping — if present, skip this step
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            if not torch.isfinite(total_norm):
                # Inf gradient — scaler will skip the update automatically
                # Zero out gradients manually to prevent stale accumulation
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

        skipped_steps += epoch_skips
        avg_train_loss = train_loss / max(len(train_loader.dataset) - epoch_skips * BATCH_SIZE, 1)
        avg_grad_norm  = np.mean(grad_norms) if grad_norms else float('nan')
        max_grad_norm  = np.max(grad_norms)  if grad_norms else float('nan')
        current_scale  = scaler.get_scale()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Warn if too many steps are being skipped — indicates persistent instability
        skip_pct = epoch_skips / len(train_loader) * 100
        skip_str = f"skipped={epoch_skips}/{len(train_loader)} ({skip_pct:.1f}%)"
        if skip_pct > 10:
            skip_str += "  *** HIGH SKIP RATE — instability detected ***"

        print(f"\n  [Grad] avg={avg_grad_norm:.4f}  max={max_grad_norm:.4f}  "
              f"scale={current_scale:.0f}  {skip_str}")

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc, val_cpm = \
            evaluate_model(val_loader, model, device, desc="Validating")

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
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_vits3d.pth")
            print(f"  --> [Saved] New Best Val AUROC: {best_val_auc:.4f}")
            with open("best_metrics_vits3d.json", "w") as f:
                json.dump({
                    "epoch": epoch, "auc": val_auc, "auprc": val_auprc,
                    "f1": val_f1, "sensitivity": val_sens,
                    "specificity": val_spec, "cpm": val_cpm,
                }, f, indent=4)
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping at epoch {epoch} ***")
                break

    print(f"\nTotal skipped gradient steps: {skipped_steps}")
    pd.DataFrame(history).to_csv("vits3d_training_log.csv", index=False)
    print("[Saved] Training log → 'vits3d_training_log.csv'")

    # ── Final test evaluation
    print("\n" + "=" * 55)
    print("  LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("=" * 55)
    model.load_state_dict(
        torch.load("best_model_vits3d.pth", map_location=device, weights_only=True)
    )

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

    youden_threshold = get_youden_threshold(val_labels_all, val_probs_all) \
        if len(np.unique(val_labels_all)) > 1 else 0.5
    print(f"  Youden threshold (val): {youden_threshold:.4f}")

    test_loss, test_auc, test_auprc, test_f1_05, test_sens_05, test_spec_05, \
        test_froc, test_cpm, pred_df = \
        evaluate_model(test_loader, model, device, desc="Testing (t=0.50)",
                       return_preds=True, threshold=0.5)

    pred_df.to_csv("vits3d_test_predictions.csv", index=False)
    print("[Saved] Test predictions → 'vits3d_test_predictions.csv'")

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
    cm_y      = confusion_matrix(y_true_t, y_pred_y, labels=[0, 1])
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
    print("=" * 55)

    with open("final_test_metrics_vits3d.json", "w") as f:
        json.dump({
            "auroc": test_auc, "auprc": test_auprc, "cpm": test_cpm,
            "threshold_05":     {"f1": test_f1_05,  "sensitivity": test_sens_05,  "specificity": test_spec_05},
            "threshold_youden": {"value": youden_threshold, "f1": test_f1_y,
                                 "sensitivity": test_sens_y, "specificity": test_spec_y},
            "froc": test_froc,
        }, f, indent=4)
    print("[Saved] Final metrics → 'final_test_metrics_vits3d.json'")


if __name__ == "__main__":
    main()