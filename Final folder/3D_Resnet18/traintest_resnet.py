"""
Hybrid 3D ResNet18 + 2.5D TinyViT with MLP Head for Lung Nodule Classification
Concatenates features from both backbones before the final classification layer.
Training / evaluation protocol identical to the standalone scripts.
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
)
import timm
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import (
    Compose,
    RandRotate90,
    RandFlip,
    RandGaussianNoise,
    RandAffine,
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
DATA_DIR = r"D:\Projects\Thesis Lung Malignancy Classification ML & DL\LUNA16 Dataset Preprocessed"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE = 8           # adjustable; keep same as original scripts
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = None        # will be calculated from training split

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Fixed architecture constants
RESNET18_FEATURE_DIM = 512   # ResNet18 penultimate feature size (global avg pool)

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------ Hybrid Dataset ------------------------------
class HybridNoduleDataset(Dataset):
    """
    Loads a 64×64×64 volume, applies 3D augmentations, then extracts
    the three central axial slices for the 2.5D branch. 2D augmentations
    are applied to the slice stack.
    Returns:
        volume_3d  : (1, 64, 64, 64)
        img_2d     : (3, 224, 224)
        label      : float
        seriesuid  : str
    """
    def __init__(self, metadata_df, data_dir, mode="train"):
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode = mode

        # 3D transforms (applied to the whole volume before slicing)
        if mode == "train":
            self.transforms_3d = Compose([
                RandAffine(prob=0.8, translate_range=(5, 5, 5),
                           padding_mode='zeros', spatial_size=None),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                RandRotate90(prob=0.5, spatial_axes=(0, 2)),
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),
                RandGaussianNoise(prob=0.2, std=0.01),
            ])
        else:
            self.transforms_3d = None

        # 2D transforms (applied to the 3-slice stack, expects dict with key "img")
        keys = ["img"]
        if mode == "train":
            self.transforms_2d = Compose([
                RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandGaussianNoised(keys=keys, prob=0.2, std=0.01),
                Resized(keys=keys, spatial_size=(224, 224), mode="bicubic"),
            ])
        else:
            self.transforms_2d = Compose([
                Resized(keys=keys, spatial_size=(224, 224), mode="bicubic"),
            ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = os.path.basename(row["filepath"])
        split = row["split"]
        label = int(row["label"])
        subfolder = "pos" if label == 1 else "neg"
        local_path = os.path.join(self.data_dir, split, subfolder, filename)

        # Load raw volume (64,64,64) and add channel dim -> (1,64,64,64)
        vol = np.load(local_path).astype(np.float32)
        vol = np.expand_dims(vol, axis=0)          # (1, 64, 64, 64)

        # 3D augmentations
        if self.transforms_3d is not None:
            vol = self.transforms_3d(vol)           # output is torch.Tensor

        if not isinstance(vol, torch.Tensor):
            vol = torch.from_numpy(vol)

        # Extract three central axial slices for 2.5D branch (after 3D transforms)
        centre = vol.shape[1] // 2   # 32
        slices = vol[0, centre-1:centre+2, :, :]   # (3, 64, 64)

        # 2D augmentations (expects dict)
        data_dict = {"img": slices}
        data_dict = self.transforms_2d(data_dict)
        img_2d = data_dict["img"]                  # (3, 224, 224)

        return vol, img_2d, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ------------------------------ Hybrid Model ------------------------------
class HybridResNetTinyViT(nn.Module):
    """
    Concatenates features from 3D ResNet18 and 2.5D TinyViT, then an MLP head.
    """
    def __init__(self, num_classes=1, dropout=0.5):
        super().__init__()

        # ----- 3D ResNet18 backbone (MONAI) -----
        self.resnet = ResNet(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],                # ResNet18
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=RESNET18_FEATURE_DIM,    # trick to keep fc output = 512
        )
        # The last fc layer is now Linear(512,512) – we will use it as feature extractor
        # and replace it with identity. (We can also just take the avgpool output,
        # but keeping the current structure avoids extra surgery.)
        self.resnet.fc = nn.Identity()
        resnet_feat_dim = RESNET18_FEATURE_DIM

        # ----- 2.5D TinyViT backbone (from timm) -----
        self.tinyvit = timm.create_model(
            "tiny_vit_21m_224",
            pretrained=False,
            num_classes=0,                       # remove classifier
            in_chans=3,
        )
        tinyvit_feat_dim = self.tinyvit.num_features   # 640 for tiny_vit_21m_224

        # ----- MLP head (identical to the original scripts) -----
        combined_dim = resnet_feat_dim + tinyvit_feat_dim
        self.mlp_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, vol_3d, img_2d):
        # 3D features
        f_3d = self.resnet(vol_3d)               # (B, 512)
        # 2.5D features
        f_2d = self.tinyvit(img_2d)              # (B, 640)

        combined = torch.cat([f_3d, f_2d], dim=1) # (B, 1152)
        out = self.mlp_head(combined)            # (B, 1)
        return out.squeeze(1)                    # (B,)

# ------------------------------ Evaluation Metrics ------------------------------
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    """Candidate‑level FROC (identical to standalone scripts)."""
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

def evaluate_model(loader, model, device, pos_weight_tensor, desc="Validating", return_preds=False):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    with torch.no_grad():
        for vol_3d, img_2d, labels, uids in tqdm(loader, desc=desc):
            vol_3d, img_2d, labels = vol_3d.to(device), img_2d.to(device), labels.to(device)

            with autocast('cuda'):
                logits = model(vol_3d, img_2d)
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
            running_loss += loss.item() * vol_3d.size(0)

    # Patient-level aggregation (max prob per seriesuid)
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

        # Optional calibration curve (same as standalone scripts)
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_patient, y_probs_patient, n_bins=10
        )
        cal_df = pd.DataFrame({
            'mean_predicted_probability': mean_predicted_value,
            'fraction_of_positives': fraction_of_positives
        })
        cal_df.to_csv("hybrid_calibration_curve.csv", index=False)
        print("[Saved] Calibration data exported to 'hybrid_calibration_curve.csv'")

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

    # Calculate pos_weight from training split
    train_meta = metadata[metadata["split"] == "train"]
    neg_count = (train_meta["label"] == 0).sum()
    pos_count = (train_meta["label"] == 1).sum()
    pos_weight_value = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"Class balance (train): neg={neg_count}, pos={pos_count}, pos_weight={pos_weight_value:.2f}")

    # Datasets & Dataloaders
    train_dataset = HybridNoduleDataset(train_meta, DATA_DIR, mode="train")
    val_meta = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    val_dataset = HybridNoduleDataset(val_meta, DATA_DIR, mode="val")
    test_dataset = HybridNoduleDataset(test_meta, DATA_DIR, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=PERSISTENT_WORKERS)

    # Model
    model = HybridResNetTinyViT().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Model Summary ---")
    print(f"Model         : Hybrid 3D ResNet18 + 2.5D TinyViT")
    print(f"Total Params  : {total_params:,}")
    print(f"Trainable     : {trainable_params:,}")
    print("---------------------\n")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        for vol_3d, img_2d, labels, _ in progress_bar:
            vol_3d, img_2d, labels = vol_3d.to(device), img_2d.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(vol_3d, img_2d)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * vol_3d.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader.dataset)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc = evaluate_model(
            val_loader, model, device, pos_weight, desc="Validating"
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
            "val_specificity": val_spec,
        })

        print(f"\nEpoch {epoch:3d} Results:")
        print(f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Patient-AUROC: {val_auc:.4f} | AUPRC: {val_auprc:.4f} | F1: {val_f1:.4f} | Sens: {val_sens:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_hybrid.pth")
            print(f"  --> [Saved] New Best Validation AUROC: {best_val_auc:.4f}")

            best_metrics = {
                "epoch": epoch,
                "auc": val_auc,
                "auprc": val_auprc,
                "f1": val_f1,
                "sensitivity": val_sens,
                "specificity": val_spec,
            }
            with open("best_metrics_hybrid.json", "w") as f:
                json.dump(best_metrics, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping triggered after {epoch} epochs ***")
                break

    pd.DataFrame(history).to_csv("hybrid_training_log.csv", index=False)
    print("\n[Saved] Training history exported to 'hybrid_training_log.csv'")

    # Final Testing
    print("\n" + "="*50)
    print("LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("="*50)

    model.load_state_dict(torch.load("best_model_hybrid.pth", map_location=device, weights_only=True))

    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = evaluate_model(
        test_loader, model, device, pos_weight, desc="Testing", return_preds=True
    )

    pred_df.to_csv("hybrid_test_predictions.csv", index=False)
    print("[Saved] Test predictions exported to 'hybrid_test_predictions.csv'")

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