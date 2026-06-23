"""
2-Branch 3D Hybrid CNN (ResNet18 + DenseNet121) with MLP Head — FROZEN BACKBONES
==================================================================================
FIX: Weight loading strips the 'backbone.' prefix that the original training
wrapper added when saving checkpoints. Previously 0/123 ResNet18 keys and
0/725 DenseNet121 keys were loading — backbones were running on random init.
"""

import os, json, random
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

# ═══════════════════════ Configuration ═══════════════════════
SEED = 42
DATA_DIR           = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH      = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

RESNET18_WEIGHTS    = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_Resnet18\best_model_resnet18.pth"
DENSENET121_WEIGHTS = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_densenet121\best_model_densenet121.pth"

BATCH_SIZE         = 8
NUM_WORKERS        = 4
PIN_MEMORY         = True
PERSISTENT_WORKERS = True

MAX_EPOCHS               = 100
EARLY_STOPPING_PATIENCE  = 15
LR                       = 1e-4
WEIGHT_DECAY             = 1e-4
POS_WEIGHT               = 5115.0 / 822.0

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

RESNET18_FEATURE_DIM    = 512
DENSENET121_FEATURE_DIM = 1024
TOTAL_FEATURE_DIM       = RESNET18_FEATURE_DIM + DENSENET121_FEATURE_DIM  # 1536

# ═══════════════════════ Reproducibility ═══════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ═══════════════════════ Weight Loading (PREFIX-AWARE) ════════
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

# ═══════════════════════ Dataset ══════════════════════════════
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
        patch = np.expand_dims(patch, axis=0)   # (1, 64, 64, 64)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ═══════════════════════ Model ════════════════════════════════
class HybridTwoBranch(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()

        # ── Branch 1: ResNet18 ──────────────────────────────────────────
        self.resnet18 = ResNet(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=RESNET18_FEATURE_DIM,
        )
        self.resnet18.fc = nn.Identity()          # output → (B, 512)

        if RESNET18_WEIGHTS:
            load_backbone_weights(self.resnet18, RESNET18_WEIGHTS, "ResNet18")
        else:
            print("  ⚠ RESNET18_WEIGHTS empty — backbone initialised randomly")

        for param in self.resnet18.parameters():
            param.requires_grad = False

        # ── Branch 2: DenseNet121 ───────────────────────────────────────
        self.densenet121 = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
        )
        self.densenet121.class_layers.out = nn.Identity()  # output → (B, 1024)

        if DENSENET121_WEIGHTS:
            load_backbone_weights(self.densenet121, DENSENET121_WEIGHTS, "DenseNet121")
        else:
            print("  ⚠ DENSENET121_WEIGHTS empty — backbone initialised randomly")

        for param in self.densenet121.parameters():
            param.requires_grad = False

        # ── MLP head: 1536 → 512 → 256 → 1  (only part that trains) ───
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
            f_r = self.resnet18(x)       # (B, 512)
            f_d = self.densenet121(x)    # (B, 1024)
        combined = torch.cat([f_r, f_d], dim=1)   # (B, 1536)
        return self.mlp_head(combined).squeeze(1)

    def freeze_backbones(self):
        """Re-apply freeze — call after loading a full checkpoint if needed."""
        for p in self.resnet18.parameters():
            p.requires_grad = False
        for p in self.densenet121.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze both backbones for optional second-stage fine-tuning."""
        for p in self.resnet18.parameters():
            p.requires_grad = True
        for p in self.densenet121.parameters():
            p.requires_grad = True

# ═══════════════════════ Evaluation ═══════════════════════════
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    desc_idx = np.argsort(all_probs)[::-1]
    sorted_labels = all_labels[desc_idx]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans
    sensitivity  = tps / total_positives
    froc_scores  = {}
    for target in FROC_THRESHOLDS:
        valid_idx = np.where(fps_per_scan <= target)[0]
        sens = sensitivity[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
        froc_scores[f"{target} FP/scan"] = sens
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

def evaluate_model(loader, model, device, desc="Validating", return_preds=False):
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

    avg_loss = running_loss / len(loader.dataset)
    auc   = roc_auc_score(y_true_pat, y_probs_pat) if len(np.unique(y_true_pat)) > 1 else 0.5
    auprc = average_precision_score(y_true_pat, y_probs_pat)
    f1    = f1_score(y_true_pat, y_pred_pat)
    sens  = recall_score(y_true_pat, y_pred_pat)
    cm    = confusion_matrix(y_true_pat, y_pred_pat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    froc  = calculate_candidate_froc(all_labels, all_probs, total_scans)

    if return_preds:
        ci_lo, ci_hi = calculate_95_ci(y_true_pat, y_probs_pat)
        print(f"  AUROC 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        frac_pos, mean_pred = calibration_curve(y_true_pat, y_probs_pat, n_bins=10)
        pd.DataFrame({
            'mean_predicted_probability': mean_pred,
            'fraction_of_positives':      frac_pos,
        }).to_csv("hybrid2_frozen_calibration_curve.csv", index=False)
        pred_df = pd.DataFrame({
            'seriesuid':   all_uids,
            'label':       all_labels,
            'probability': all_probs,
        })
        return avg_loss, auc, auprc, f1, sens, spec, froc, pred_df
    return avg_loss, auc, auprc, f1, sens, spec, froc

# ═══════════════════════ Main ═════════════════════════════════
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df   = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)
    print(f"Train: {len(train_meta):,}  Val: {len(val_meta):,}  Test: {len(test_meta):,}")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transforms = Compose([
        RandAffine(prob=0.8, translate_range=(5, 5, 5), padding_mode='zeros', spatial_size=None),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandGaussianNoise(prob=0.2, std=0.01),
    ])

    train_ds = NodulePatchDataset(train_meta, DATA_DIR, transforms=train_transforms)
    val_ds   = NodulePatchDataset(val_meta,   DATA_DIR, transforms=None)
    test_ds  = NodulePatchDataset(test_meta,  DATA_DIR, transforms=None)

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                         pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HybridTwoBranch().to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    print(f"\n2‑Branch Hybrid (Frozen Backbones)")
    print(f"  Total parameters   : {total_params:,}")
    print(f"  Trainable (MLP)    : {trainable_params:,}")
    print(f"  Frozen (backbones) : {frozen_params:,}\n")

    # ── Training setup — optimizer only touches MLP head ──────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    warmup_epochs    = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler        = SequentialLR(optimizer,
                                    schedulers=[warmup_scheduler, cosine_scheduler],
                                    milestones=[warmup_epochs])
    scaler = GradScaler('cuda')

    best_val_auc     = 0.0
    patience_counter = 0
    history          = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        # Keep frozen branches in eval mode so BN running stats don't drift
        model.resnet18.eval()
        model.densenet121.eval()

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
            "epoch": epoch, "lr": current_lr, "train_loss": avg_train_loss,
            "val_loss": val_loss, "val_auc": val_auc, "val_auprc": val_auprc,
            "val_f1": val_f1, "val_sensitivity": val_sens, "val_specificity": val_spec,
        })

        print(f"\nEpoch {epoch:3d} │ LR {current_lr:.2e} │ "
              f"Train Loss {avg_train_loss:.4f} │ Val Loss {val_loss:.4f}")
        print(f"         │ AUROC {val_auc:.4f} │ AUPRC {val_auprc:.4f} │ "
              f"F1 {val_f1:.4f} │ Sens {val_sens:.4f} │ Spec {val_spec:.4f}")

        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_hybrid2_frozen.pth")
            print(f"  ✓ [Saved] New best val AUROC: {best_val_auc:.4f}")
            with open("best_metrics_hybrid2_frozen.json", "w") as f:
                json.dump({"epoch": epoch, "auc": val_auc, "auprc": val_auprc,
                           "f1": val_f1, "sensitivity": val_sens,
                           "specificity": val_spec}, f, indent=4)
        else:
            patience_counter += 1
            print(f"  · No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping after epoch {epoch} ***")
                break

    pd.DataFrame(history).to_csv("hybrid2_frozen_training_log.csv", index=False)

    # ── Final Test ────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load("best_model_hybrid2_frozen.pth",
                                     map_location=device, weights_only=True))
    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = \
        evaluate_model(test_loader, model, device, desc="Testing", return_preds=True)
    pred_df.to_csv("hybrid2_frozen_test_predictions.csv", index=False)

    print("\n─── 2‑BRANCH HYBRID (FROZEN) TEST RESULTS ───")
    print(f"  AUROC: {test_auc:.4f}   AUPRC: {test_auprc:.4f}")
    print(f"  F1: {test_f1:.4f}   Sensitivity: {test_sens:.4f}   Specificity: {test_spec:.4f}")
    print("FROC:")
    for k, v in test_froc.items():
        print(f"  {k}: {v:.4f}")
    print("═" * 40)

if __name__ == "__main__":
    main()