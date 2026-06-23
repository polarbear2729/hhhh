"""
3D ResNet18 with MLP Head for Lung Nodule Classification
Identical configuration to ResNet34 script — only the model changes.
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, recall_score
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBlock  # BasicBlock — same as ResNet34
from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise, RandAffine
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------------------ Configuration ------------------------------
# UNCHANGED — identical to ResNet34 script
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 5115.0 / 822.0

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# ResNet18 feature dimension — fixed architecture property
# ResNet18 uses BasicBlock with layers=[2,2,2,2]; final stage also outputs 512 channels
RESNET18_FEATURE_DIM = 512

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------ Dataset ------------------------------
# UNCHANGED — identical to ResNet34 script
class NodulePatchDataset(Dataset):
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
        patch = np.load(local_path).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)

        if self.transforms is not None:
            patch = self.transforms(patch)

        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)

        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ------------------------------ Model ------------------------------
class ResNet18WithMLPHead(nn.Module):
    """
    3D ResNet18 backbone from MONAI with a custom MLP classification head.

    ResNet18 uses BasicBlock (not Bottleneck) with layer config [2, 2, 2, 2].
    Final feature dimension before the classifier is 512 — same as ResNet34
    because both architectures share the same channel-widening schedule
    [64 → 128 → 256 → 512]; only the number of blocks per stage differs.

    The original FC head is replaced with:
        Linear(512 → 256) → LayerNorm → ReLU → Dropout → Linear(256 → 1)

    This mirrors the ResNet34 MLP head structure exactly;
    the input dimension is identical (512) so the head is unchanged.
    """
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()

        # MONAI ResNet with 3D spatial dims
        # block=ResNetBlock → BasicBlock (ResNet18/34 style, no bottleneck)
        # layers=[2,2,2,2]  → ResNet18 depth  ← only change from ResNet34 ([3,4,6,3])
        # block_inplanes unchanged: [64, 128, 256, 512]
        self.backbone = ResNet(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],           # ResNet18 block counts per stage
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=RESNET18_FEATURE_DIM,  # Trick: set num_classes=512
                                                # so MONAI's FC outputs 512-d features
                                                # which we then override below
        )

        # Verify MONAI ResNet structure before overriding
        assert hasattr(self.backbone, 'fc'), \
            "MONAI ResNet structure changed: 'fc' layer missing"

        # Replace the final FC layer with MLP head
        # Input: 512-d (ResNet18 penultimate features after global avg pool)
        # Identical to ResNet34 head — feature dim is the same
        self.backbone.fc = nn.Sequential(
            nn.Linear(RESNET18_FEATURE_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1)

# ------------------------------ Evaluation Metrics ------------------------------
from sklearn.calibration import calibration_curve

# UNCHANGED — identical to ResNet34 script
def calculate_candidate_froc(all_labels, all_probs, total_scans):
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
        score = roc_auc_score(np.array(y_true)[indices], np.array(y_probs)[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(np.array(bootstrapped_scores))
    return np.percentile(sorted_scores, 2.5), np.percentile(sorted_scores, 97.5)


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
                probs, labels = [float(probs)], [float(labels.cpu())]
            else:
                probs, labels = probs.tolist(), labels.cpu().tolist()

            all_probs.extend(probs)
            all_labels.extend(labels)
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
        cal_df.to_csv("resnet18_calibration_curve.csv", index=False)
        print("[Saved] Calibration data exported to 'resnet18_calibration_curve.csv'")

        pred_df = pd.DataFrame({
            'seriesuid': all_uids,
            'label': all_labels,
            'probability': all_probs
        })
        return avg_loss, auc, auprc, f1, sens, spec, froc, pred_df

    return avg_loss, auc, auprc, f1, sens, spec, froc

# ------------------------------ Main Execution ------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)

    # UNCHANGED — identical transforms to ResNet34 script
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
    val_dataset = NodulePatchDataset(val_meta, DATA_DIR, transforms=None)
    test_dataset = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=PERSISTENT_WORKERS)

    model = ResNet18WithMLPHead().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Model Summary ---")
    print(f"Model         : 3D ResNet18 + MLP Head")
    print(f"Total Params  : {total_params:,}")
    print(f"Trainable     : {trainable_params:,}")
    print("---------------------\n")

    # UNCHANGED — identical training config to ResNet34 script
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    scaler = GradScaler('cuda')

    best_val_auc = 0.0
    patience_counter = 0
    history = []

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
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * patches.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader.dataset)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc = evaluate_model(
            val_loader, model, device, desc="Validating"
        )

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_auprc": val_auprc,
            "val_f1": val_f1,
            "val_sensitivity": val_sens,
            "val_specificity": val_spec
        })

        print(f"\nEpoch {epoch:3d} Results:")
        print(f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Patient-AUROC: {val_auc:.4f} | AUPRC: {val_auprc:.4f} | F1: {val_f1:.4f} | Sens: {val_sens:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_resnet18.pth")
            print(f"  --> [Saved] New Best Validation AUROC: {best_val_auc:.4f}")

            best_metrics = {
                "epoch": epoch,
                "auc": val_auc,
                "auprc": val_auprc,
                "f1": val_f1,
                "sensitivity": val_sens,
                "specificity": val_spec
            }
            with open("best_metrics_resnet18.json", "w") as f:
                json.dump(best_metrics, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping triggered after {epoch} epochs ***")
                break

    pd.DataFrame(history).to_csv("resnet18_training_log.csv", index=False)
    print("\n[Saved] Training history exported to 'resnet18_training_log.csv'")

    print("\n" + "=" * 50)
    print("LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("=" * 50)

    model.load_state_dict(torch.load("best_model_resnet18.pth", map_location=device, weights_only=True))

    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = evaluate_model(
        test_loader, model, device, desc="Testing", return_preds=True
    )

    pred_df.to_csv("resnet18_test_predictions.csv", index=False)
    print("[Saved] Test predictions exported to 'resnet18_test_predictions.csv'")

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

