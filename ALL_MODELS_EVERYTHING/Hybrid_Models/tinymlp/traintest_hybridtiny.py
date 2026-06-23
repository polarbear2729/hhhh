# hybrid_resnet3d_tinyvit2d_train.py
"""
Hybrid 3D ResNet18 + 2D TinyViT with MLP Head for Lung Nodule Classification.

Input:
    - 3D volume (1 × 64 × 64 × 64) → ResNet18 (512-d)
    - Central 3 slices (3 × 224 × 224) → TinyViT (384-d)

Concatenated features (896-d) → MLP → logit

Output files:
    best_model_hybrid_resnet_tinyvit.pth
    best_metrics_hybrid_resnet_tinyvit.json
    hybrid_resnet_tinyvit_training_log.csv
    hybrid_resnet_tinyvit_test_predictions.csv
    hybrid_resnet_tinyvit_calibration_curve.csv
"""

import os, json, random, numpy as np, pandas as pd
from xml.parsers.expat import model
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, confusion_matrix, recall_score)
from sklearn.calibration import calibration_curve
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import (Compose, RandRotate90, RandFlip, RandGaussianNoise,
                              RandAffine, Resized)
import warnings
from tqdm import tqdm
import timm

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"          # adjust if needed
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE = 8              # reduce to 4 or 2 if OOM 
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 5115.0 / 822.0   # neg/pos ratio from training set

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Feature dimensions – DO NOT CHANGE
RESNET18_FEATURE_DIM = 512          # MONAI ResNet18 final GAP output
TINYVIT_FEATURE_DIM = 384           # tiny_vit_21m_224 num_features (timm 0.9.x)
TOTAL_FEATURE_DIM = RESNET18_FEATURE_DIM + TINYVIT_FEATURE_DIM   # 896

# TinyViT config
TINYVIT_IN_CHANS = 3
RESIZE_TO = (224, 224)

# ════════════════════════════════════════════════════════════════
#  Reproducibility
# ════════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ════════════════════════════════════════════════════════════════
#  Dataset – returns both 3D volume and 2D slice stack
# ════════════════════════════════════════════════════════════════
class HybridDataset(Dataset):
    """
    For each sample, returns:
        patch_3d : (1, 64, 64, 64)  – raw 3D CT volume
        img_2d   : (3, 224, 224)    – central axial slices resized
        label    : float (0/1)
        seriesuid: str
    """
    def __init__(self, metadata_df, data_dir, transform_3d=None, transform_2d=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform_3d = transform_3d   # aug for 3D branch (None at test)
        self.transform_2d = transform_2d   # resize for 2D branch

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = os.path.basename(row["filepath"])
        split = row["split"]
        label = int(row["label"])
        subfolder = "pos" if label == 1 else "neg"
        local_path = os.path.join(self.data_dir, split, subfolder, filename)

        # Load full 3D volume (64,64,64)
        vol = np.load(local_path).astype(np.float32)           # (64,64,64)

        # ---- 3D branch: add channel dim → (1,64,64,64) ----
        patch_3d = np.expand_dims(vol, axis=0)                 # (1,64,64,64)

        # ---- 2D branch: extract central 3 slices, stack as channels ----
        centre = vol.shape[0] // 2
        slices = [vol[centre-1, :, :],
                vol[centre,   :, :],
                vol[centre+1, :, :]]                         # each (64,64)
        img_2d = np.stack(slices, axis=0)                      # (3,64,64)

        # 3D transforms – directly on the array (NO dict)
        if self.transform_3d is not None:
            patch_3d = self.transform_3d(patch_3d)             # returns tensor

        # 2D transforms – dict‑based (as in the original 2D script)
        if self.transform_2d is not None:
            data_dict_2d = {"img": img_2d}
            data_dict_2d = self.transform_2d(data_dict_2d)
            img_2d = data_dict_2d["img"]

        # Convert to tensors if not already
        if not isinstance(patch_3d, torch.Tensor):
            patch_3d = torch.from_numpy(patch_3d)
        if not isinstance(img_2d, torch.Tensor):
            img_2d = torch.from_numpy(img_2d)

        return (patch_3d, img_2d), torch.tensor(label, dtype=torch.float32), row["seriesuid"]
# ════════════════════════════════════════════════════════════════
#  Transforms
# ════════════════════════════════════════════════════════════════
def get_transforms(mode="train"):
    """
    Returns 3D augmentation (train only) and 2D resize transform (always).
    """
    # 3D transforms (only augmentation for training)
    if mode == "train":
        transform_3d = Compose([
            RandAffine(prob=0.8, translate_range=(5,5,5),
                       padding_mode='zeros', spatial_size=None),
            RandRotate90(prob=0.5, spatial_axes=(0,1)),
            RandRotate90(prob=0.5, spatial_axes=(1,2)),
            RandRotate90(prob=0.5, spatial_axes=(0,2)),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandGaussianNoise(prob=0.2, std=0.01),
        ])
    else:
        transform_3d = Compose([])   # identity for val/test

    # 2D transforms: resize to 224x224 (always applied)
    transform_2d = Compose([
        Resized(keys=["img"], spatial_size=RESIZE_TO, mode="bicubic"),
    ])

    return transform_3d, transform_2d

# ════════════════════════════════════════════════════════════════
#  Model: Hybrid ResNet18 (3D) + TinyViT (2D)
# ════════════════════════════════════════════════════════════════
class HybridResNetTinyViT(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super().__init__()

        # ----- 3D ResNet18 backbone (MONAI) -----
        self.resnet = ResNet(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=512,                # trick to get 512-d features
        )
        self.resnet.fc = nn.Identity()      # remove final layer

        # ----- 2.5D TinyViT backbone (timm) -----
        self.tinyvit = timm.create_model(
            "tiny_vit_21m_224",             # update to the model you actually use
            pretrained=False,
            num_classes=0,
            in_chans=3,
        )

        # ----- Determine actual feature dimensions automatically -----
        with torch.no_grad():
            dummy_vol = torch.zeros(1, 1, 64, 64, 64)
            dummy_img = torch.zeros(1, 3, 224, 224)
            f_3d = self.resnet(dummy_vol)          # (1, resnet_dim)
            f_2d = self.tinyvit(dummy_img)         # (1, tinyvit_dim)
        resnet_dim = f_3d.shape[1]
        tinyvit_dim = f_2d.shape[1]
        combined_dim = resnet_dim + tinyvit_dim

        print(f"  ResNet18 feature dim: {resnet_dim}")
        print(f"  TinyViT feature dim:  {tinyvit_dim}")
        print(f"  Combined dim:         {combined_dim}")

        # ----- MLP head (identical to the standalone scripts) -----
        self.mlp_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, vol_3d, img_2d):
        f_3d = self.resnet(vol_3d)
        f_2d = self.tinyvit(img_2d)
        combined = torch.cat([f_3d, f_2d], dim=1)
        out = self.mlp_head(combined)
        return out.squeeze(1)

# ════════════════════════════════════════════════════════════════
#  Evaluation utilities (same as before)
# ════════════════════════════════════════════════════════════════
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels); all_probs = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    desc_idx = np.argsort(all_probs)[::-1]
    sorted_labels = all_labels[desc_idx]
    tps = np.cumsum(sorted_labels == 1); fps = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans; sensitivity = tps / total_positives
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
        if len(np.unique(np.array(y_true)[idx])) < 2: continue
        scores.append(roc_auc_score(np.array(y_true)[idx], np.array(y_probs)[idx]))
    scores = np.sort(scores)
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def evaluate_model(loader, model, device, pos_weight_tensor, desc="Validating", return_preds=False):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    with torch.no_grad():
        for (patch_3d, img_2d), labels, uids in tqdm(loader, desc=desc):
            patch_3d = patch_3d.to(device, non_blocking=True)
            img_2d = img_2d.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(patch_3d, img_2d)
                loss = criterion(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = [float(probs)]
            else:
                probs = probs.tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().tolist())
            all_uids.extend(uids)
            running_loss += loss.item() * patch_3d.size(0)

    # Patient-level aggregation
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
        }).to_csv("hybrid_resnet_tinyvit_calibration_curve.csv", index=False)
        print("  [Saved] hybrid_resnet_tinyvit_calibration_curve.csv")
        pred_df = pd.DataFrame({
            'seriesuid': all_uids,
            'label': all_labels,
            'probability': all_probs
        })
        return avg_loss, auc, auprc, f1, sens, spec, froc, pred_df

    return avg_loss, auc, auprc, f1, sens, spec, froc

# ════════════════════════════════════════════════════════════════
#  Main training script
# ════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Metadata ────────────────────────────────────────────
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_meta)}  Val: {len(val_meta)}  Test: {len(test_meta)}")

    # Compute pos_weight from training split
    train_labels = metadata[metadata["split"] == "train"]["label"]
    neg_count = (train_labels == 0).sum()
    pos_count = (train_labels == 1).sum()
    pos_weight_value = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    print(f"Pos weight: {pos_weight_value:.2f}")

    # ── Transforms ──────────────────────────────────────────
    train_3d, train_2d = get_transforms("train")
    val_3d, val_2d     = get_transforms("val")

    train_dataset = HybridDataset(train_meta, DATA_DIR, transform_3d=train_3d, transform_2d=train_2d)
    val_dataset   = HybridDataset(val_meta,   DATA_DIR, transform_3d=val_3d,   transform_2d=val_2d)
    test_dataset  = HybridDataset(test_meta,  DATA_DIR, transform_3d=val_3d,   transform_2d=val_2d)

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                         pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    # ── Model ────────────────────────────────────────────────
    model = HybridResNetTinyViT().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    resnet18_params = sum(p.numel() for p in model.resnet.parameters())
    tinyvit_params = sum(p.numel() for p in model.tinyvit.parameters())
    mlp_params      = sum(p.numel() for p in model.mlp_head.parameters())

    print(f"\n{'='*50}")
    print(f"  Hybrid ResNet18(3D) + TinyViT(2D)  Model Summary")
    print(f"{'='*50}")
    print(f"  ResNet18 params:    {resnet18_params:>12,}")
    print(f"  TinyViT params:     {tinyvit_params:>12,}")
    print(f"  MLP head params:    {mlp_params:>12,}")
    print(f"  Total params:       {total_params:>12,}")
    print(f"  Trainable params:   {trainable_params:>12,}")
    print(f"  Concatenated dim:   {TOTAL_FEATURE_DIM}")
    print(f"{'='*50}\n")

    # ── Loss / Optimizer / Scheduler ────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])
    scaler = GradScaler('cuda')

    # ── Training loop ───────────────────────────────────────
    best_val_auc = 0.0
    patience_counter = 0
    history = []

    print("Starting training...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for (patch_3d, img_2d), labels, _ in pbar:
            patch_3d = patch_3d.to(device, non_blocking=True)
            img_2d = img_2d.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(patch_3d, img_2d)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * patch_3d.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader.dataset)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc = evaluate_model(
            val_loader, model, device, pos_weight, desc="Validating")

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": avg_train_loss, "val_loss": val_loss,
            "val_auc": val_auc, "val_auprc": val_auprc, "val_f1": val_f1,
            "val_sensitivity": val_sens, "val_specificity": val_spec,
        })

        print(f"\nEpoch {epoch:3d} | LR {current_lr:.2e} | "
              f"Train loss {avg_train_loss:.4f} | Val loss {val_loss:.4f}")
        print(f"         | AUROC {val_auc:.4f} | AUPRC {val_auprc:.4f} | "
              f"F1 {val_f1:.4f} | Sens {val_sens:.4f} | Spec {val_spec:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_hybrid_resnet_tinyvit.pth")
            print(f"  ✓ [Saved] New best validation AUROC: {best_val_auc:.4f}")
            with open("best_metrics_hybrid_resnet_tinyvit.json", "w") as f:
                json.dump({"epoch": epoch, "auc": val_auc, "auprc": val_auprc,
                           "f1": val_f1, "sensitivity": val_sens, "specificity": val_spec}, f, indent=4)
        else:
            patience_counter += 1
            print(f"  · No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping at epoch {epoch} ***")
                break

    pd.DataFrame(history).to_csv("hybrid_resnet_tinyvit_training_log.csv", index=False)
    print("\n[Saved] training log.")

    # ── Final test evaluation ──────────────────────────────
    print("\n" + "="*50)
    print("LOADING BEST MODEL FOR TEST EVALUATION")
    print("="*50)
    model.load_state_dict(torch.load("best_model_hybrid_resnet_tinyvit.pth", map_location=device, weights_only=True))

    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = evaluate_model(
        test_loader, model, device, pos_weight, desc="Testing", return_preds=True)

    pred_df.to_csv("hybrid_resnet_tinyvit_test_predictions.csv", index=False)
    print("[Saved] test predictions.\n")
    print("─── FINAL TEST RESULTS (Scan-Level) ───")
    print(f"  AUROC:  {test_auc:.4f}")
    print(f"  AUPRC:  {test_auprc:.4f}")
    print(f"  F1:     {test_f1:.4f}")
    print(f"  Sens:   {test_sens:.4f}")
    print(f"  Spec:   {test_spec:.4f}")
    print("\n─── FROC (Candidate-Level) ───")
    for k, v in test_froc.items():
        print(f"  {k:14s} : {v:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()