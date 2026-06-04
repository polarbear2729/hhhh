"""
3D Vision Transformer (ViT) with MLP Head for Lung Nodule Classification
Matches DenseNet121 configuration: same data, optim, metrics, logging.
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, roc_curve, recall_score
from sklearn.calibration import calibration_curve
from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------------------ Configuration ------------------------------
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

# ViT specific settings (adjustable)
IMG_SIZE = (64, 64, 64)            # your patch dimensions
PATCH_SIZE = (16, 16, 16)          # 4×4×4 = 64 patches
EMBED_DIM = 384                    # matches DenseNet feature vector size
DEPTH = 6
NUM_HEADS = 6
MLP_RATIO = 4
DROPOUT = 0.1                      # attention + MLP dropout
ATTN_DROPOUT = 0.0

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------ Dataset ------------------------------
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
        patch = np.expand_dims(patch, axis=0)         # (1, D, H, W)

        if self.transforms is not None:
            patch = self.transforms(patch)

        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)

        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ------------------------------ Model ------------------------------
class ViT3DWithMLPHead(nn.Module):
    """
    3D Vision Transformer with a learnable cls token and a custom MLP head.
    Designed to have ~11M parameters, comparable to DenseNet121.
    """
    def __init__(self, in_channels=1, num_classes=1,
                 img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                 embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS,
                 mlp_ratio=MLP_RATIO, dropout=DROPOUT, attn_dropout=ATTN_DROPOUT):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * \
                           (img_size[1] // patch_size[1]) * \
                           (img_size[2] // patch_size[2])   # e.g., 4*4*4 = 64

        # 3D patch embedding
        self.patch_embed = nn.Conv3d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # learnable positional embedding (1 + num_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True           # Pre‑LN (standard in ViT)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        # MLP head (same as DenseNet's head)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),          # additional dropout before final classifier
            nn.Linear(256, num_classes)
        )

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, 1, 64, 64, 64)
        x = self.patch_embed(x)                # (B, embed_dim, D', H', W')
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, num_patches, embed_dim)

        # add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, 1+num_patches, embed_dim)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)                 # (B, 1+num_patches, embed_dim)

        x = self.norm(x[:, 0])                  # take cls token
        x = self.head(x).squeeze(1)             # (B,) logits
        return x

# ------------------------------ Evaluation Metrics ------------------------------
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    total_positives = (all_labels == 1).sum()

    descending_indices = np.argsort(all_probs)[::-1]
    sorted_probs = all_probs[descending_indices]
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

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        score = roc_auc_score(np.array(y_true)[indices], np.array(y_probs)[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = np.percentile(sorted_scores, 2.5)
    ci_upper = np.percentile(sorted_scores, 97.5)
    return ci_lower, ci_upper

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
        cal_df.to_csv("vit3d_calibration_curve.csv", index=False)
        print("[Saved] Calibration data exported to 'vit3d_calibration_curve.csv'")

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

    train_transforms = Compose([
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

    model = ViT3DWithMLPHead().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Model Summary ---")
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print("---------------------\n")

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

    print("--- Starting Training (3D ViT) ---")
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
            torch.save(model.state_dict(), "best_model_vit3d.pth")
            print(f"  --> [Saved] New Best Validation AUROC: {best_val_auc:.4f}")

            best_metrics = {
                "epoch": epoch,
                "auc": val_auc,
                "auprc": val_auprc,
                "f1": val_f1,
                "sensitivity": val_sens,
                "specificity": val_spec
            }
            with open("best_metrics_vit3d.json", "w") as f:
                json.dump(best_metrics, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping triggered after {epoch} epochs ***")
                break

    pd.DataFrame(history).to_csv("vit3d_training_log.csv", index=False)
    print("\n[Saved] Training history exported to 'vit3d_training_log.csv'")

    # Final Testing
    print("\n" + "="*50)
    print("LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("="*50)

    model.load_state_dict(torch.load("best_model_vit3d.pth", map_location=device, weights_only=True))

    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = evaluate_model(
        test_loader, model, device, desc="Testing", return_preds=True
    )

    pred_df.to_csv("vit3d_test_predictions.csv", index=False)
    print("[Saved] Test predictions exported to 'vit3d_test_predictions.csv'")

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