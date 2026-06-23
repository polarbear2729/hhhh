"""
2-Branch 3D Hybrid CNN (ResNet18 + DenseNet121) with KAN Head
===============================================================
Removes the EfficientNet-B0 branch to reduce capacity.
Now supports resuming training from a full checkpoint.
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
DATA_DIR          = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH     = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH= os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE         = 8         # you may need to reduce if OOM
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
TOTAL_FEATURE_DIM       = RESNET18_FEATURE_DIM + DENSENET121_FEATURE_DIM   # 1536

# ═══════════════════════ Reproducibility ═══════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ═══════════════════════ Dataset ═══════════════════════
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
        patch = np.expand_dims(patch, axis=0)   # (1,64,64,64)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ═══════════════════════ 2‑Branch Hybrid Model ═══════════════════════
class HybridTwoBranch(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        # Branch 1: ResNet18
        self.resnet18 = ResNet(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=RESNET18_FEATURE_DIM,  # creates FC(512→512)
        )
        self.resnet18.fc = nn.Identity()      # output (B, 512)

        # Branch 2: DenseNet121
        self.densenet121 = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,                   # creates FC(1024→1)
        )
        self.densenet121.class_layers.out = nn.Identity()  # output (B, 1024)

        # KAN head: 1536 → 512 → 256 → 1
        from efficient_kan import KAN
        self.mlp_head = KAN([TOTAL_FEATURE_DIM, 512, 256, num_classes])

    def forward(self, x):
        f_r = self.resnet18(x)       # (B, 512)
        f_d = self.densenet121(x)    # (B, 1024)
        combined = torch.cat([f_r, f_d], dim=1)   # (B, 1536)
        out = self.mlp_head(combined).squeeze(1)
        return out

# ═══════════════════════ Checkpoint utilities ═══════════════════════
def save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                    best_val_auc, patience_counter, filename="checkpoint_hybrid2.pth"):
    """Save full training state for resumption."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_auc': best_val_auc,
        'patience_counter': patience_counter,
    }, filename)
    print(f"  ✓ Checkpoint saved at epoch {epoch}")

def load_checkpoint_if_exists(model, optimizer, scheduler, scaler, device):
    """Load checkpoint if present; returns (start_epoch, best_val_auc, patience_counter)."""
    checkpoint_path = "checkpoint_hybrid2.pth"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint['best_val_auc']
        patience_counter = checkpoint['patience_counter']
        print(f"Resumed from epoch {checkpoint['epoch']}, best AUC {best_val_auc:.4f}, patience {patience_counter}")
        return start_epoch, best_val_auc, patience_counter
    else:
        print("No checkpoint found, starting from scratch.")
        return 1, 0.0, 0

# ═══════════════════════ Evaluation (identical to original) ═══════════
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    desc_idx = np.argsort(all_probs)[::-1]
    sorted_labels = all_labels[desc_idx]
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

    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict[uid]['prob'] = max(patient_dict[uid]['prob'], prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)

    y_true_pat = [v['label'] for v in patient_dict.values()]
    y_probs_pat = [v['prob'] for v in patient_dict.values()]
    y_pred_pat = [1 if p >= 0.5 else 0 for p in y_probs_pat]
    total_scans = len(patient_dict)

    avg_loss = running_loss / len(loader.dataset)
    auc = roc_auc_score(y_true_pat, y_probs_pat) if len(np.unique(y_true_pat)) > 1 else 0.5
    auprc = average_precision_score(y_true_pat, y_probs_pat)
    f1 = f1_score(y_true_pat, y_pred_pat)
    sens = recall_score(y_true_pat, y_pred_pat)
    cm = confusion_matrix(y_true_pat, y_pred_pat, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    froc = calculate_candidate_froc(all_labels, all_probs, total_scans)

    if return_preds:
        ci_lo, ci_hi = calculate_95_ci(y_true_pat, y_probs_pat)
        print(f"  AUROC 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        frac_pos, mean_pred = calibration_curve(y_true_pat, y_probs_pat, n_bins=10)
        pd.DataFrame({
            'mean_predicted_probability': mean_pred,
            'fraction_of_positives': frac_pos,
        }).to_csv("hybrid2_calibration_curve.csv", index=False)
        pred_df = pd.DataFrame({
            'seriesuid': all_uids,
            'label': all_labels,
            'probability': all_probs,
        })
        return avg_loss, auc, auprc, f1, sens, spec, froc, pred_df
    return avg_loss, auc, auprc, f1, sens, spec, froc

# ═══════════════════════ Main ═══════════════════════
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Metadata
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)
    print(f"Train: {len(train_meta):,}  Val: {len(val_meta):,}  Test: {len(test_meta):,}")

    # Transforms
    train_transforms = Compose([
        RandAffine(prob=0.8, translate_range=(5,5,5), padding_mode='zeros', spatial_size=None),
        RandRotate90(prob=0.5, spatial_axes=(0,1)),
        RandRotate90(prob=0.5, spatial_axes=(1,2)),
        RandRotate90(prob=0.5, spatial_axes=(0,2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandGaussianNoise(prob=0.2, std=0.01),
    ])

    train_ds = NodulePatchDataset(train_meta, DATA_DIR, transforms=train_transforms)
    val_ds   = NodulePatchDataset(val_meta, DATA_DIR, transforms=None)
    test_ds  = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                         pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # Model
    model = HybridTwoBranch().to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"2‑Branch Hybrid parameters: {total_p:,}")

    # Training setup
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])
    scaler = GradScaler('cuda')

    # ═══════════════════════ Resumption logic ═══════════════════════
    start_epoch, best_val_auc, patience_counter = load_checkpoint_if_exists(
        model, optimizer, scheduler, scaler, device
    )

    history = []

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for patches, labels, _ in pbar:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(patches)
                loss = criterion(logits, labels)
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

        # Save a full checkpoint at the end of each epoch (overwrites previous)
        save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                        best_val_auc, patience_counter)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_hybrid2.pth")
            print(f"  ✓ [Saved] New best val AUROC: {best_val_auc:.4f}")
            with open("best_metrics_hybrid2.json", "w") as f:
                json.dump({"epoch": epoch, "auc": val_auc, "auprc": val_auprc,
                           "f1": val_f1, "sensitivity": val_sens, "specificity": val_spec}, f, indent=4)
        else:
            patience_counter += 1
            print(f"  · No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping after epoch {epoch} ***")
                save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                                best_val_auc, patience_counter)
                break

    pd.DataFrame(history).to_csv("hybrid2_training_log.csv", index=False)

    # Final Test
    model.load_state_dict(torch.load("best_model_hybrid2.pth", map_location=device, weights_only=True))
    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = \
        evaluate_model(test_loader, model, device, desc="Testing", return_preds=True)
    pred_df.to_csv("hybrid2_test_predictions.csv", index=False)

    print("\n─── 2‑BRANCH HYBRID TEST RESULTS ───")
    print(f"  AUROC: {test_auc:.4f}   AUPRC: {test_auprc:.4f}")
    print(f"  F1: {test_f1:.4f}   Sensitivity: {test_sens:.4f}   Specificity: {test_spec:.4f}")
    print("FROC:")
    for k, v in test_froc.items():
        print(f"  {k}: {v:.4f}")
    print("═" * 40)

if __name__ == "__main__":
    main()