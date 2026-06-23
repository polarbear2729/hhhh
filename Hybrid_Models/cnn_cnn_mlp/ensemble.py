"""
Inference‑Time Ensemble of 3D CNNs for Lung Nodule Classification
=================================================================
Loads the three individually trained backbones (ResNet18, DenseNet121,
EfficientNet‑B0), averages their predictions, and reports final metrics.
This replaces the overfitted monolithic hybrid with a simple, robust ensemble.
"""

import os, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, recall_score,
)
from sklearn.calibration import calibration_curve
from monai.networks.nets import DenseNet121, ResNet
from monai.networks.nets.resnet import ResNetBlock
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════
# Configuration (same as your original script)
# ═══════════════════════════════════════════════════
SEED = 42
DATA_DIR          = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH     = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH= os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE         = 8      # ensemble forward passes one by one
NUM_WORKERS        = 4
PIN_MEMORY         = False
PERSISTENT_WORKERS = False

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Paths to the three pre‑trained checkpoints
RESNET18_PATH   = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_Resnet18\best_model_resnet18.pth"
DENSENET121_PATH = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_densenet121\best_model_densenet121.pth"
EFFICIENTNET_PATH = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_EfficientnetB0\best_model_efficientnetb0.pth"   # adjust if name differs

# ═══════════════════════════════════════════════════
# Dataset (identical)
# ═══════════════════════════════════════════════════
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
        patch = np.expand_dims(patch, axis=0)       # (1, D, H, W)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ═══════════════════════════════════════════════════
# 3D EfficientNet-B0 (exact copy from your script)
# ═══════════════════════════════════════════════════
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
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(in_channels, se_ch), SwishActivation(),
            nn.Linear(se_ch, in_channels), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1, 1)

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

# ═══════════════════════════════════════════════════
# Individual model classes (as trained)
# ═══════════════════════════════════════════════════
class ResNet18Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet(
            block=ResNetBlock, layers=[2,2,2,2],
            block_inplanes=[64,128,256,512],
            spatial_dims=3, n_input_channels=1, num_classes=1
        )
        # Replace FC with your MLP head (same as in training)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1)

class DenseNet121Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
        self.backbone.class_layers.out = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1)

class EfficientNetB0Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet3D_B0(in_channels=1)
        self.mlp_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        features = self.backbone(x)
        out = self.mlp_head(features)
        return out.squeeze(1)

# ═══════════════════════════════════════════════════
# Ensemble model
# ═══════════════════════════════════════════════════
class EnsembleModel(nn.Module):
    def __init__(self, resnet_path, densenet_path, effnet_path, device):
        super().__init__()
        self.resnet = ResNet18Model().to(device)
        self.densenet = DenseNet121Model().to(device)
        self.effnet = EfficientNetB0Model().to(device)

        self.resnet.load_state_dict(torch.load(resnet_path, map_location=device))
        self.densenet.load_state_dict(torch.load(densenet_path, map_location=device))
        self.effnet.load_state_dict(torch.load(effnet_path, map_location=device))

        self.resnet.eval()
        self.densenet.eval()
        self.effnet.eval()

    def forward(self, x):
        with torch.no_grad():
            logit_r = self.resnet(x)
            logit_d = self.densenet(x)
            logit_e = self.effnet(x)
        # Average logits (or you can average probabilities)
        avg_logit = (logit_r + logit_d + logit_e) / 3.0
        return avg_logit

# ═══════════════════════════════════════════════════
# Evaluation (identical to original)
# ═══════════════════════════════════════════════════
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

def evaluate_model(loader, model, device, return_preds=False):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc="Evaluating"):
            patches = patches.to(device)
            logits = model(patches)
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
        }).to_csv("ensemble_calibration_curve.csv", index=False)
        print("  [Saved] ensemble_calibration_curve.csv")
        pred_df = pd.DataFrame({
            'seriesuid': all_uids,
            'label': all_labels,
            'probability': all_probs,
        })
        return auc, auprc, f1, sens, spec, froc, pred_df

    return auc, auprc, f1, sens, spec, froc

# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    set_seed = lambda s: (random.seed(s), np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s))
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Metadata
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    test_ds = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # Load ensemble model
    ensemble = EnsembleModel(RESNET18_PATH, DENSENET121_PATH, EFFICIENTNET_PATH, device)
    print("Loaded three backbones for ensemble inference.")

    # Evaluate
    test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = evaluate_model(
        test_loader, ensemble, device, return_preds=True
    )

    pred_df.to_csv("ensemble_test_predictions.csv", index=False)
    print("  [Saved] ensemble_test_predictions.csv")

    print("\n─── ENSEMBLE TEST RESULTS (Scan-Level) ───")
    print(f"  AUROC (Primary)       : {test_auc:.4f}")
    print(f"  AUPRC                 : {test_auprc:.4f}")
    print(f"  F1 Score              : {test_f1:.4f}")
    print(f"  Sensitivity (Recall)  : {test_sens:.4f}")
    print(f"  Specificity           : {test_spec:.4f}")
    print("\n─── FROC (FP/Scan → Sensitivity) ───")
    for fp_rate, sensitivity in test_froc.items():
        print(f"  {fp_rate:14s} : {sensitivity:.4f}")
    print("═" * 58 + "\n")

if __name__ == "__main__":
    main()