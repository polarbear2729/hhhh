# verify_hybrid.py
"""
Deep verification of the Hybrid 3D CNN (ResNet18 + DenseNet121 + EfficientNet-B0 + MLP head).
Evaluates on train/val/test splits, checks overfitting, and diagnoses mistakes.
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.networks.nets import DenseNet121, ResNet
from monai.networks.nets.resnet import ResNetBlock
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, recall_score
)
import warnings
warnings.filterwarnings("ignore")

# ------------------------- Config -------------------------
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_hybrid.pth"
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
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ------------------------- EfficientNet-B0 (3D) -------------
# (copied from your training script – necessary for the hybrid model)
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
        self.use_skip = (stride == 1 and in_channels == out_channels)
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
            nn.Conv3d(mid_ch, mid_ch, kernel_size, stride=stride, padding=pad, groups=mid_ch, bias=False),
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
                keep_prob = 1 - self.drop_connect_rate
                rand_tensor = torch.rand(x.size(0), 1, 1, 1, 1, device=x.device)
                rand_tensor = torch.floor(rand_tensor + keep_prob)
                out = out / keep_prob * rand_tensor
            out = out + x
        return out

class EfficientNet3D_B0(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm3d(32, momentum=0.01, eps=1e-3),
            SwishActivation(),
        )
        stage_configs = [
            (32, 16, 3, 1, 1, 1), (16, 24, 3, 2, 6, 2), (24, 40, 5, 2, 6, 2),
            (40, 80, 3, 2, 6, 3), (80, 112, 5, 1, 6, 3), (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1),
        ]
        total_blocks = sum(cfg[5] for cfg in stage_configs)
        block_idx = 0
        stages = []
        for in_ch, out_ch, k, s, expand, n_layers in stage_configs:
            stage = []
            for i in range(n_layers):
                stride = s if i == 0 else 1
                inch = in_ch if i == 0 else out_ch
                drop_rate = 0.2 * block_idx / total_blocks
                stage.append(MBConv3D(inch, out_ch, k, stride, expand, drop_connect_rate=drop_rate))
                block_idx += 1
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)
        self.head_conv = nn.Sequential(
            nn.Conv3d(320, 1280, 1, bias=False),
            nn.BatchNorm3d(1280, momentum=0.01, eps=1e-3),
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
        return x.flatten(1)   # (B, 1280)

# ------------------------- Hybrid Model (exact copy) ----------
class HybridCNNWithMLPHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        # ResNet18 branch
        self.resnet18 = ResNet(
            block=ResNetBlock, layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3, n_input_channels=in_channels,
            num_classes=512
        )
        self.resnet18.fc = nn.Identity()   # output (B,512)

        # DenseNet121 branch
        self.densenet121 = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1)
        self.densenet121.class_layers.out = nn.Identity()  # output (B,1024)

        # EfficientNet-B0 branch
        self.efficientnet_b0 = EfficientNet3D_B0(in_channels=in_channels)  # output (B,1280)

        # MLP fusion head
        self.mlp_head = nn.Sequential(
            nn.Linear(2816, 512),
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
        f_r = self.resnet18(x)          # (B,512)
        f_d = self.densenet121(x)       # (B,1024)
        f_e = self.efficientnet_b0(x)  # (B,1280)
        combined = torch.cat([f_r, f_d, f_e], dim=1)   # (B,2816)
        out = self.mlp_head(combined)                 # (B,1)
        return out.squeeze(1)

# ------------------------- Metrics ---------------------------
def calculate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels); all_probs = np.array(all_probs)
    total_pos = (all_labels == 1).sum()
    desc_idx = np.argsort(all_probs)[::-1]
    sorted_labels = all_labels[desc_idx]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans
    sens = tps / total_pos
    froc = {}
    for t in FROC_THRESHOLDS:
        idx = np.where(fps_per_scan <= t)[0]
        s = sens[idx[-1]] if len(idx) > 0 else 0.0
        froc[f"{t} FP/scan"] = s
    return froc

def evaluate_split(loader, model, device, split_name):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc=f"Eval {split_name}"):
            patches = patches.to(device)
            logits = model(patches)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
            all_uids.extend(uids)
    # Patient-level max aggregation
    patient_dict = {}
    for p, l, u in zip(all_probs, all_labels, all_uids):
        if u not in patient_dict:
            patient_dict[u] = {'prob': p, 'label': l}
        else:
            patient_dict[u]['prob'] = max(patient_dict[u]['prob'], p)
            patient_dict[u]['label'] = max(patient_dict[u]['label'], l)
    y_true = [v['label'] for v in patient_dict.values()]
    y_prob = [v['prob'] for v in patient_dict.values()]
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    total_scans = len(patient_dict)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    froc = calculate_froc(all_labels, all_probs, total_scans)

    return {
        'auc': auc, 'auprc': auprc, 'f1': f1, 'sensitivity': sens,
        'specificity': spec, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'froc': froc, 'total_scans': total_scans,
        'y_true': y_true, 'y_prob': y_prob
    }

# ------------------------- Main -------------------------------
def main():
    print(f"Device: {DEVICE}")

    # --- Load metadata ---
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)
    print(f"Train: {len(train_meta)}  Val: {len(val_meta)}  Test: {len(test_meta)}")

    # --- Datasets & Loaders ---
    train_ds = NodulePatchDataset(train_meta, DATA_DIR)
    val_ds   = NodulePatchDataset(val_meta, DATA_DIR)
    test_ds  = NodulePatchDataset(test_meta, DATA_DIR)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # --- Load model ---
    model = HybridCNNWithMLPHead().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Hybrid model loaded.\n")

    # --- Evaluate all splits ---
    print("Evaluating Train split...")
    train_stats = evaluate_split(train_loader, model, DEVICE, "Train")
    print("Evaluating Val split...")
    val_stats   = evaluate_split(val_loader, model, DEVICE, "Val")
    print("Evaluating Test split...")
    test_stats  = evaluate_split(test_loader, model, DEVICE, "Test")

    # --- Print comparison ---
    print("\n" + "="*70)
    print("                 Train       Validation   Test")
    print("="*70)
    for metric in ['auc','auprc','f1','sensitivity','specificity']:
        t = train_stats[metric]
        v = val_stats[metric]
        ts = test_stats[metric]
        print(f"{metric:15s}: {t:.4f}      {v:.4f}       {ts:.4f}")
    print(f"{'Confusion':15s}: TP={train_stats['tp']} FN={train_stats['fn']}  "
          f"TP={val_stats['tp']} FN={val_stats['fn']}   TP={test_stats['tp']} FN={test_stats['fn']}")
    print(f"{' ':15s}  FP={train_stats['fp']} TN={train_stats['tn']}  "
          f"FP={val_stats['fp']} TN={val_stats['tn']}   FP={test_stats['fp']} TN={test_stats['tn']}")
    print(f"{'Total Scans':15s}: {train_stats['total_scans']}          "
          f"{val_stats['total_scans']}          {test_stats['total_scans']}")
    print("="*70)

    # --- Overfitting analysis ---
    auc_gap = train_stats['auc'] - test_stats['auc']
    spec_gap = train_stats['specificity'] - test_stats['specificity']
    print(f"\nOverfitting indicators:")
    print(f"  AUROC gap (train - test): {auc_gap:.4f}")
    print(f"  Specificity gap (train - test): {spec_gap:.4f}")
    if auc_gap > 0.02 or spec_gap > 0.05:
        print("  ⚠️ Significant overfitting detected!")
    else:
        print("  Overfitting appears mild.")

    # --- Probability distribution analysis ---
    print("\nProbability distribution per split:")
    for name, stats in zip(["Train","Val","Test"], [train_stats, val_stats, test_stats]):
        y = np.array(stats['y_true'])
        p = np.array(stats['y_prob'])
        pos_p = p[y==1]
        neg_p = p[y==0]
        print(f"  {name}: Pos mean={pos_p.mean():.4f} median={np.median(pos_p):.4f}  "
              f"Neg mean={neg_p.mean():.4f} median={np.median(neg_p):.4f}")

    # --- False positive analysis on test ---
    print("\nTest false positive patient details:")
    for i, (true, prob) in enumerate(zip(test_stats['y_true'], test_stats['y_prob'])):
        if true == 0 and prob >= 0.5:
            uid = list({u: (l,p) for u,l,p in zip(
                [u for u,_,_ in test_loader.dataset],
                test_stats['y_true'], test_stats['y_prob']
            )}.keys())[i]  # ugly but we'll just print index
            print(f"  Patient {i} prob={prob:.4f}")

    # Save detailed report
    report = {
        "train": {k: (float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()}) for k,v in train_stats.items() if k not in ['y_true','y_prob']},
        "val": {k: (float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()}) for k,v in val_stats.items() if k not in ['y_true','y_prob']},
        "test": {k: (float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()}) for k,v in test_stats.items() if k not in ['y_true','y_prob']},
        "overfitting_auc_gap": float(auc_gap),
        "overfitting_spec_gap": float(spec_gap),
    }
    with open("audit_hybrid.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n[Saved] audit_hybrid.json")

if __name__ == "__main__":
    main()