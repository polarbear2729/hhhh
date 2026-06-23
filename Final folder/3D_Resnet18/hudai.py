# verify_resnet18_fixed.py
"""
Deep verification of ResNet18 3D model.
Corrected to match the actual checkpoint structure.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.networks.nets import ResNet, ResNetBlock
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, confusion_matrix, recall_score)
import warnings
warnings.filterwarnings("ignore")

# ------------------------- Config -------------------------
SEED = 42
DATA_DIR = r"D:\Projects\Thesis Lung Malignancy Classification ML & DL\LUNA16 Dataset Preprocessed"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_resnet18.pth"   # adjust if necessary

BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# ------------------------- Dataset -------------------------
class NodulePatchDataset(torch.utils.data.Dataset):
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
        patch = np.expand_dims(patch, axis=0)   # (1,64,64,64)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]

# ------------------------- Model (matches training) -------
class ResNet18WithMLPHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        self.backbone = ResNet(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes,
        )
        # Replace backbone.fc with the MLP head (exactly as during training)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1)

# ------------------------- Metrics ------------------------
def calculate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    desc_idx = np.argsort(all_probs)[::-1]
    sorted_labels = all_labels[desc_idx]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans
    sensitivity = tps / total_positives
    froc = {}
    for t in FROC_THRESHOLDS:
        idx = np.where(fps_per_scan <= t)[0]
        s = sensitivity[idx[-1]] if len(idx) > 0 else 0.0
        froc[f"{t} FP/scan"] = s
    return froc

def permutation_test(y_true, y_probs, n_perm=500):
    real_auc = roc_auc_score(y_true, y_probs)
    null_aucs = []
    rng = np.random.RandomState(SEED)
    for _ in range(n_perm):
        shuffled = rng.permutation(y_true)
        if len(np.unique(shuffled)) < 2:
            continue
        null_aucs.append(roc_auc_score(shuffled, y_probs))
    null_aucs = np.array(null_aucs)
    p = (null_aucs >= real_auc).sum() / len(null_aucs) if len(null_aucs) > 0 else 1.0
    return real_auc, null_aucs.mean(), null_aucs.std(), p

# ------------------------- Main ---------------------------
def main():
    print(f"Device: {DEVICE}")

    # 1. Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    print(f"\nTest patches: {len(test_meta)}")

    # 2. Data integrity
    train_uids = set(metadata[metadata["split"] == "train"]["seriesuid"])
    val_uids   = set(metadata[metadata["split"] == "val"]["seriesuid"])
    test_uids  = set(test_meta["seriesuid"])
    print(f"Unique patients – Train: {len(train_uids)}, Val: {len(val_uids)}, Test: {len(test_uids)}")
    assert len(train_uids & test_uids) == 0, "Train/Test overlap detected!"
    assert len(val_uids & test_uids) == 0, "Val/Test overlap detected!"
    print("✅ No patient overlap between splits.")

    # 3. Load model (corrected structure)
    model = ResNet18WithMLPHead().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully.")

    # 4. Run predictions
    test_dataset = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    all_probs, all_labels, all_uids = [], [], []
    with torch.no_grad():
        for patches, labels, uids, _ in tqdm(test_loader, desc="Predicting"):
            logits = model(patches.to(DEVICE))
            prob = torch.sigmoid(logits).cpu().item()
            all_probs.append(prob)
            all_labels.append(labels.item())
            all_uids.append(uids[0])

    # 5. Patient-level aggregation
    patient_dict = {}
    for p, l, u in zip(all_probs, all_labels, all_uids):
        if u not in patient_dict:
            patient_dict[u] = {'prob': p, 'label': l}
        else:
            patient_dict[u]['prob']  = max(patient_dict[u]['prob'], p)
            patient_dict[u]['label'] = max(patient_dict[u]['label'], l)

    y_true = [v['label'] for v in patient_dict.values()]
    y_prob = [v['prob'] for v in patient_dict.values()]
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    total_scans = len(patient_dict)

    # 6. Compute metrics
    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    froc = calculate_froc(all_labels, all_probs, total_scans)

    print("\n========== ResNet18 AUDIT ==========")
    print(f"AUROC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1: {f1:.4f}   Sensitivity: {sens:.4f}   Specificity: {spec:.4f}")
    print(f"Confusion Matrix: TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print("FROC:")
    for k, v in froc.items():
        print(f"  {k}: {v:.4f}")

    # 7. Permutation test
    real_auc, null_mean, null_std, p_val = permutation_test(y_true, y_prob)
    print(f"\nPermutation Test: real AUC={real_auc:.4f}, null mean={null_mean:.4f}±{null_std:.4f}, p={p_val:.4f}")

    # 8. Misclassified patients
    print("\n--- Misclassified Patients ---")
    for uid, info in patient_dict.items():
        if info['label'] == 1 and info['prob'] < 0.5:
            print(f"FN: {uid[:16]}...  prob={info['prob']:.4f}")
        elif info['label'] == 0 and info['prob'] >= 0.5:
            print(f"FP: {uid[:16]}...  prob={info['prob']:.4f}")

    # 9. Suspiciously high negative patches
    neg_high = [(p, u) for p, l, u in zip(all_probs, all_labels, all_uids) if l == 0 and p > 0.9]
    if neg_high:
        print(f"\n⚠️ {len(neg_high)} negative patches with prob > 0.9 – likely hard negatives near the nodule.")
        for p, u in neg_high[:5]:
            print(f"  {u[:16]}... prob={p:.4f}")

    # 10. Save report
        report = {
        "auc": float(auc),
        "auprc": float(auprc),
        "f1": float(f1),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "froc": {k: float(v) for k, v in froc.items()},
        "permutation_p": float(p_val),
        "neg_high_count": int(len(neg_high))
    }
    with open("audit_resnet18.json", "w") as f:
        import json
        json.dump(report, f, indent=2)
        print("[Saved] Audit report exported to 'audit_resnet18.json'")


if __name__ == "__main__":
    main()