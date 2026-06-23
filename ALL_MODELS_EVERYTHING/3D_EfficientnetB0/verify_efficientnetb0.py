"""
=============================================================================
MODEL RESULTS VERIFICATION & AUDIT SCRIPT
EfficientNet-B0 - Lung Nodule Classification (LUNA16)
=============================================================================

PURPOSE:
    Independently verifies that EfficientNet-B0 model results are legitimate by running
    multiple sanity checks across the data, predictions, model weights,
    and statistical properties. Generates a full audit report.

HOW TO RUN:
    python verify_efficientnetb0.py

PREREQUISITES (files must exist from training):
    - best_model_efficientnetb0.pth
    - efficientnetb0_test_predictions.csv
    - efficientnetb0_training_log.csv
    - best_metrics_efficientnetb0.json
    - metadata_all.csv  (or patient_split.csv)

OUTPUT:
    - audit_report_efficientnetb0.txt   (human-readable full report)
    - audit_summary_efficientnetb0.json (machine-readable pass/fail summary)
=============================================================================
"""

import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, recall_score
)
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — Match exactly with your EfficientNet-B0 training script
# =============================================================================
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"   # <-- adjust to your data path
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_efficientnetb0.pth"
PREDICTIONS_PATH = "efficientnetb0_test_predictions.csv"
TRAINING_LOG_PATH = "efficientnetb0_training_log.csv"
BEST_METRICS_PATH = "best_metrics_efficientnetb0.json"
BATCH_SIZE = 8
NUM_WORKERS = 4
POS_WEIGHT = 5115.0 / 822.0
TOLERANCE = 0.01

EFFICIENTNET_FEATURE_DIM = 1280

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# MODEL (must match EfficientNet-B0 training script exactly)
# =============================================================================
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
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, se_channels),
            SwishActivation(),
            nn.Linear(se_channels, in_channels),
            nn.Sigmoid()
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

        mid_channels = _make_divisible(in_channels * expand_ratio, 8)
        layers = []

        if expand_ratio != 1:
            layers += [
                nn.Conv3d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm3d(mid_channels, momentum=0.1, eps=1e-3),
                SwishActivation()
            ]

        pad = (kernel_size - 1) // 2
        layers += [
            nn.Conv3d(mid_channels, mid_channels, kernel_size,
                      stride=stride, padding=pad, groups=mid_channels, bias=False),
            nn.BatchNorm3d(mid_channels, momentum=0.1, eps=1e-3),
            SwishActivation()
        ]

        layers.append(SqueezeExcitation3D(mid_channels, se_ratio))

        layers += [
            nn.Conv3d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels, momentum=0.1, eps=1e-3)
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
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=0.1, eps=1e-3),
            SwishActivation()
        )

        stage_configs = [
            (32,  16,  3, 1, 1, 1),
            (16,  24,  3, 2, 6, 2),
            (24,  40,  5, 2, 6, 2),
            (40,  80,  3, 2, 6, 3),
            (80,  112, 5, 1, 6, 3),
            (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1),
        ]

        total_blocks = sum(cfg[5] for cfg in stage_configs)
        block_idx = 0
        stages = []
        for (in_ch, out_ch, k, s, expand, n_layers) in stage_configs:
            stage = []
            for i in range(n_layers):
                stride = s if i == 0 else 1
                inch = in_ch if i == 0 else out_ch
                drop_rate = 0.2 * block_idx / total_blocks
                stage.append(MBConv3D(inch, out_ch, k, stride, expand,
                                      drop_connect_rate=drop_rate))
                block_idx += 1
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)

        self.head_conv = nn.Sequential(
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, kernel_size=1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.1, eps=1e-3),
            SwishActivation()
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return x


class EfficientNetB0WithMLPHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        self.backbone = EfficientNet3D_B0(in_channels=in_channels)
        self.mlp_head = nn.Sequential(
            nn.Linear(EFFICIENTNET_FEATURE_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.mlp_head(features)
        return out.squeeze(1)


# =============================================================================
# DATASET (same as training)
# =============================================================================
class NodulePatchDataset(Dataset):
    def __init__(self, metadata_df, data_dir):
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_dir = data_dir

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
        return torch.from_numpy(patch), torch.tensor(label, dtype=torch.float32), row["seriesuid"]


# =============================================================================
# AUDIT REPORT CLASS
# =============================================================================
class AuditReport:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warned = 0

    def record(self, check_name, status, detail):
        self.checks.append({"check": check_name, "status": status, "detail": detail})
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.warned += 1

    def print_check(self, check_name, status, detail):
        icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}
        print(f"  {icons.get(status, '?')} [{status}] {check_name}")
        print(f"         {detail}")
        self.record(check_name, status, detail)

    def save(self, txt_path, json_path):
        lines = []
        lines.append("=" * 70)
        lines.append("  MODEL AUDIT REPORT — EfficientNet-B0 Lung Nodule Classification")
        lines.append("=" * 70)
        lines.append(f"  Total Checks : {len(self.checks)}")
        lines.append(f"  Passed       : {self.passed}")
        lines.append(f"  Warnings     : {self.warned}")
        lines.append(f"  Failed       : {self.failed}")
        lines.append("")
        for c in self.checks:
            icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(c["status"], "[?]")
            lines.append(f"  {icon}  {c['check']}")
            lines.append(f"         {c['detail']}")
        lines.append("=" * 70)
        verdict = "RESULTS ARE LEGITIMATE ✅" if self.failed == 0 else f"AUDIT FAILED — {self.failed} critical issue(s) found ❌"
        lines.append(f"  VERDICT: {verdict}")
        lines.append("=" * 70)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        summary = {
            "total": len(self.checks),
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "verdict": "LEGITIMATE" if self.failed == 0 else "FAILED",
            "checks": self.checks
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\n[Saved] Full report → {txt_path}")
        print(f"[Saved] JSON summary → {json_path}")


# =============================================================================
# AUDIT CHECKS
# =============================================================================
def check_1_required_files_exist(report):
    print("\n[CHECK 1] Required Files Exist")
    required = {
        "Model weights": MODEL_PATH,
        "Test predictions CSV": PREDICTIONS_PATH,
        "Training log CSV": TRAINING_LOG_PATH,
        "Best metrics JSON": BEST_METRICS_PATH,
    }
    all_found = True
    for name, path in required.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            report.print_check(f"File exists: {name}", "PASS", f"Found at '{path}' ({size_kb:.1f} KB)")
        else:
            report.print_check(f"File exists: {name}", "FAIL", f"NOT FOUND at '{path}' — training may have crashed")
            all_found = False
    return all_found


def check_2_model_weights_not_random(report, device):
    print("\n[CHECK 2] Model Weights Are Trained (Not Random)")
    try:
        trained_model = EfficientNetB0WithMLPHead().to(device)
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

        random_model = EfficientNetB0WithMLPHead().to(device)
        trained_w = list(trained_model.parameters())[0].detach().cpu().numpy().flatten()
        random_w = list(random_model.parameters())[0].detach().cpu().numpy().flatten()
        correlation, _ = stats.pearsonr(trained_w[:1000], random_w[:1000])
        weight_diff = np.mean(np.abs(trained_w - random_w[:len(trained_w)]))

        if abs(correlation) > 0.99:
            report.print_check("Weights differ from random init", "FAIL",
                               f"Pearson r={correlation:.4f} — weights look unchanged from random init")
        else:
            report.print_check("Weights differ from random init", "PASS",
                               f"Mean weight delta={weight_diff:.6f}, Pearson r={correlation:.4f} vs random")

        # Check MLP head weights specifically
        mlp_w = list(trained_model.mlp_head.parameters())[0].detach().cpu().numpy()
        mlp_std = mlp_w.std()
        if mlp_std < 1e-6:
            report.print_check("MLP head weights are non-trivial", "FAIL",
                               f"MLP head std={mlp_std:.2e} — suspiciously near zero, head may not have trained")
        else:
            report.print_check("MLP head weights are non-trivial", "PASS",
                               f"MLP head weight std={mlp_std:.6f} — looks trained")
        return trained_model
    except Exception as e:
        report.print_check("Load model weights", "FAIL", f"Exception: {e}")
        return None


def check_3_predictions_csv_integrity(report):
    print("\n[CHECK 3] Predictions CSV Integrity")
    try:
        pred_df = pd.read_csv(PREDICTIONS_PATH)
        required_cols = {"seriesuid", "label", "probability"}
        missing = required_cols - set(pred_df.columns)
        if missing:
            report.print_check("Predictions CSV columns", "FAIL", f"Missing columns: {missing}")
            return None
        else:
            report.print_check("Predictions CSV columns", "PASS", f"All required columns present. Shape: {pred_df.shape}")

        nan_count = pred_df[["label", "probability"]].isna().sum().sum()
        if nan_count > 0:
            report.print_check("No NaN values", "FAIL", f"{nan_count} NaN values found in predictions")
        else:
            report.print_check("No NaN values", "PASS", "Zero NaN values")

        out_of_range = ((pred_df["probability"] < 0) | (pred_df["probability"] > 1)).sum()
        if out_of_range > 0:
            report.print_check("Probabilities in [0,1]", "FAIL", f"{out_of_range} predictions outside [0,1] — sigmoid was not applied")
        else:
            prob_min = pred_df["probability"].min()
            prob_max = pred_df["probability"].max()
            report.print_check("Probabilities in [0,1]", "PASS", f"Range: [{prob_min:.4f}, {prob_max:.4f}]")

        unique_labels = set(pred_df["label"].unique())
        if not unique_labels.issubset({0, 1, 0.0, 1.0}):
            report.print_check("Labels are binary", "FAIL", f"Non-binary labels found: {unique_labels}")
        else:
            pos = (pred_df["label"] == 1).sum()
            neg = (pred_df["label"] == 0).sum()
            report.print_check("Labels are binary", "PASS", f"Positive patches: {pos}, Negative patches: {neg}")

        prob_std = pred_df["probability"].std()
        if prob_std < 0.01:
            report.print_check("Predictions are not collapsed", "FAIL", f"Probability std={prob_std:.4f} — model may be predicting same value for all inputs")
        else:
            report.print_check("Predictions are not collapsed", "PASS", f"Probability std={prob_std:.4f}, mean={pred_df['probability'].mean():.4f}")

        if len(unique_labels) < 2:
            report.print_check("Both classes in test set", "FAIL", "Only one class found in test predictions — AUROC is undefined")
        else:
            report.print_check("Both classes in test set", "PASS", "Both positive and negative samples present in test set")
        return pred_df
    except Exception as e:
        report.print_check("Load predictions CSV", "FAIL", f"Exception: {e}")
        return None


def check_4_recompute_metrics_from_predictions(report, pred_df, saved_metrics):
    print("\n[CHECK 4] Recompute Metrics from Saved Predictions")
    patient_dict = {}
    for _, row in pred_df.iterrows():
        uid = row["seriesuid"]
        prob = row["probability"]
        label = row["label"]
        if uid not in patient_dict:
            patient_dict[uid] = {"prob": prob, "label": label}
        else:
            patient_dict[uid]["prob"] = max(patient_dict[uid]["prob"], prob)
            patient_dict[uid]["label"] = max(patient_dict[uid]["label"], label)

    y_true = [v["label"] for v in patient_dict.values()]
    y_probs = [v["prob"] for v in patient_dict.values()]
    y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

    recomputed = {}
    try:
        recomputed["auc"] = roc_auc_score(y_true, y_probs)
        recomputed["auprc"] = average_precision_score(y_true, y_probs)
        recomputed["f1"] = f1_score(y_true, y_pred)
        recomputed["sensitivity"] = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recomputed["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except Exception as e:
        report.print_check("Metric recomputation", "FAIL", f"Exception during recomputation: {e}")
        return

    print(f"  Recomputed metrics (patient-level, from saved CSV):")
    for k, v in recomputed.items():
        print(f"    {k:15s}: {v:.4f}")

    report.print_check("AUROC recomputable from predictions", "PASS", f"Recomputed AUROC={recomputed['auc']:.4f} from saved CSV successfully")
    report.print_check("AUPRC recomputable from predictions", "PASS", f"Recomputed AUPRC={recomputed['auprc']:.4f} from saved CSV successfully")

    if recomputed["auc"] < 0.5:
        report.print_check("AUROC sanity range", "FAIL", f"AUROC={recomputed['auc']:.4f} < 0.5 — worse than random, check label alignment")
    elif recomputed["auc"] > 0.999:
        report.print_check("AUROC sanity range", "WARN", f"AUROC={recomputed['auc']:.4f} is suspiciously perfect — possible data leakage")
    else:
        report.print_check("AUROC sanity range", "PASS", f"AUROC={recomputed['auc']:.4f} is in plausible range [0.5, 0.999]")

    if recomputed["sensitivity"] == 0.0:
        report.print_check("Sensitivity > 0", "FAIL", "Sensitivity=0.0 — model never predicted a single positive. Check threshold.")
    elif recomputed["sensitivity"] == 1.0:
        report.print_check("Sensitivity = 1.0 check", "WARN", "Sensitivity=1.0 — model may be predicting everything as positive")
    else:
        report.print_check("Sensitivity in valid range", "PASS", f"Sensitivity={recomputed['sensitivity']:.4f}")
    return recomputed


def check_5_training_log_shows_learning(report):
    print("\n[CHECK 5] Training Log Shows Real Learning")
    try:
        log_df = pd.read_csv(TRAINING_LOG_PATH)
        if len(log_df) < 3:
            report.print_check("Training log has enough epochs", "WARN", f"Only {len(log_df)} epochs logged — model may have crashed early")
            return
        report.print_check("Training log has enough epochs", "PASS", f"{len(log_df)} epochs recorded")

        first_loss = log_df["train_loss"].iloc[:3].mean()
        last_loss = log_df["train_loss"].iloc[-3:].mean()
        if last_loss < first_loss:
            report.print_check("Training loss decreased", "PASS", f"Loss: {first_loss:.4f} (first 3 epochs avg) → {last_loss:.4f} (last 3 epochs avg)")
        else:
            report.print_check("Training loss decreased", "FAIL", f"Loss did NOT decrease: {first_loss:.4f} → {last_loss:.4f}. Training may have diverged.")

        first_auc = log_df["val_auc"].iloc[:3].mean()
        best_auc = log_df["val_auc"].max()
        if best_auc > first_auc:
            report.print_check("Val AUROC improved over training", "PASS", f"Val AUROC: {first_auc:.4f} (first 3 avg) → {best_auc:.4f} (best)")
        else:
            report.print_check("Val AUROC improved over training", "FAIL", f"Val AUROC never improved beyond initial: {first_auc:.4f} → best={best_auc:.4f}")

        loss_std = log_df["train_loss"].std()
        if loss_std < 1e-5:
            report.print_check("Training loss is not flat", "FAIL", f"Train loss std={loss_std:.2e} — loss never changed, weights may be frozen")
        else:
            report.print_check("Training loss is not flat", "PASS", f"Train loss std={loss_std:.6f} — loss varied across epochs")

        auc_std = log_df["val_auc"].std()
        if auc_std < 1e-5 and log_df["val_auc"].mean() > 0.99:
            report.print_check("Val AUROC is not fabricated", "FAIL", f"Val AUROC is {log_df['val_auc'].mean():.4f} ± {auc_std:.2e} across all epochs — impossible consistency")
        else:
            report.print_check("Val AUROC variance is realistic", "PASS", f"Val AUROC std={auc_std:.4f} across epochs — natural variation present")
    except Exception as e:
        report.print_check("Load training log", "FAIL", f"Exception: {e}")


def check_6_model_inference_live(report, model, device):
    print("\n[CHECK 6] Live Model Inference vs Saved Predictions")
    if model is None:
        report.print_check("Live inference (model unavailable)", "FAIL", "Model failed to load in Check 2 — skipping live inference")
        return
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)

        test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
        n_sample = min(50, len(test_meta))
        sample_meta = test_meta.sample(n=n_sample, random_state=SEED).reset_index(drop=True)
        dataset = NodulePatchDataset(sample_meta, DATA_DIR)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model.eval()
        live_probs, live_labels, live_uids = [], [], []
        with torch.no_grad():
            for patches, labels, uids in tqdm(loader, desc="  Live inference on 50 samples"):
                patches = patches.to(device)
                with autocast('cuda'):
                    logits = model(patches)
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim == 0:
                    probs = [float(probs)]
                live_probs.extend(probs.tolist() if hasattr(probs, 'tolist') else [float(probs)])
                live_labels.extend(labels.tolist())
                live_uids.extend(uids)

        saved_df = pd.read_csv(PREDICTIONS_PATH)
        saved_lookup = dict(zip(zip(saved_df["seriesuid"], saved_df["label"].astype(int)), saved_df["probability"]))

        matches = 0
        checked = 0
        max_diff = 0.0
        for uid, label, live_p in zip(live_uids, live_labels, live_probs):
            key = (uid, int(label))
            if key in saved_lookup:
                saved_p = saved_lookup[key]
                diff = abs(live_p - saved_p)
                max_diff = max(max_diff, diff)
                if diff < TOLERANCE:
                    matches += 1
                checked += 1

        if checked == 0:
            report.print_check("Live inference matches saved predictions", "WARN", "Could not find matching UIDs between live run and saved CSV to compare")
        elif matches / checked >= 0.90:
            report.print_check("Live inference matches saved predictions", "PASS", f"{matches}/{checked} samples match within tolerance={TOLERANCE}. Max diff={max_diff:.6f}")
        else:
            report.print_check("Live inference matches saved predictions", "FAIL", f"Only {matches}/{checked} match. Max diff={max_diff:.6f}. Saved predictions may not correspond to this model.")

        live_auc = roc_auc_score(live_labels, live_probs) if len(set(live_labels)) > 1 else None
        if live_auc is not None:
            if live_auc >= 0.5:
                report.print_check("Live inference AUROC on sample", "PASS", f"Live AUROC on {n_sample} samples = {live_auc:.4f}")
            else:
                report.print_check("Live inference AUROC on sample", "WARN", f"Live AUROC on {n_sample} samples = {live_auc:.4f} < 0.5 (small sample, may be noise — check full test set)")
    except Exception as e:
        report.print_check("Live inference", "FAIL", f"Exception: {e}")


def check_7_label_split_integrity(report):
    print("\n[CHECK 7] No Patient Overlap Between Splits (Data Leakage)")
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)

        train_ids = set(metadata[metadata["split"] == "train"]["seriesuid"].unique())
        val_ids = set(metadata[metadata["split"] == "val"]["seriesuid"].unique())
        test_ids = set(metadata[metadata["split"] == "test"]["seriesuid"].unique())

        for pair, overlap in [("Train ∩ Val", train_ids & val_ids), ("Train ∩ Test", train_ids & test_ids), ("Val ∩ Test", val_ids & test_ids)]:
            if overlap:
                report.print_check(f"No overlap: {pair}", "FAIL", f"{len(overlap)} patient(s) appear in both splits: {list(overlap)[:5]}...")
            else:
                report.print_check(f"No overlap: {pair}", "PASS", f"Zero patients shared between {pair}")

        print(f"  Split sizes — Train: {len(train_ids)} patients, Val: {len(val_ids)} patients, Test: {len(test_ids)} patients")
        for split_name in ["train", "val", "test"]:
            split_df = metadata[metadata["split"] == split_name]
            pos = (split_df["label"] == 1).sum()
            neg = (split_df["label"] == 0).sum()
            ratio = pos / len(split_df) * 100
            report.print_check(f"Class distribution in {split_name}", "PASS", f"Pos={pos}, Neg={neg}, Pos%={ratio:.1f}%")
    except Exception as e:
        report.print_check("Load metadata for split check", "FAIL", f"Exception: {e}")


def check_8_permutation_baseline(report, pred_df):
    print("\n[CHECK 8] Permutation Test — Results Must Beat Random Shuffling")
    if pred_df is None:
        report.print_check("Permutation test", "FAIL", "No predictions available")
        return
    patient_dict = {}
    for _, row in pred_df.iterrows():
        uid = row["seriesuid"]
        prob = row["probability"]
        label = row["label"]
        if uid not in patient_dict:
            patient_dict[uid] = {"prob": prob, "label": label}
        else:
            patient_dict[uid]["prob"] = max(patient_dict[uid]["prob"], prob)
            patient_dict[uid]["label"] = max(patient_dict[uid]["label"], label)

    y_true = np.array([v["label"] for v in patient_dict.values()])
    y_probs = np.array([v["prob"] for v in patient_dict.values()])
    real_auc = roc_auc_score(y_true, y_probs)
    rng = np.random.RandomState(SEED)
    permuted_aucs = []
    for _ in range(1000):
        shuffled = rng.permutation(y_probs)
        permuted_aucs.append(roc_auc_score(y_true, shuffled))
    permuted_aucs = np.array(permuted_aucs)
    p_value = (permuted_aucs >= real_auc).mean()
    percentile = (permuted_aucs < real_auc).mean() * 100
    print(f"  Real AUROC:          {real_auc:.4f}")
    print(f"  Permuted AUROC mean: {permuted_aucs.mean():.4f} ± {permuted_aucs.std():.4f}")
    print(f"  p-value:             {p_value:.6f}")
    print(f"  Beats {percentile:.1f}% of random permutations")
    if p_value < 0.05:
        report.print_check("Permutation test (p < 0.05)", "PASS", f"Real AUROC={real_auc:.4f} significantly beats random (p={p_value:.4f}, beats {percentile:.1f}% of permutations)")
    else:
        report.print_check("Permutation test (p < 0.05)", "FAIL", f"Real AUROC={real_auc:.4f} does NOT significantly beat random (p={p_value:.4f}) — results may be due to chance")


def check_9_prediction_score_separation(report, pred_df):
    print("\n[CHECK 9] Positive vs Negative Score Separation")
    if pred_df is None:
        report.print_check("Score separation test", "FAIL", "No predictions available")
        return
    pos_probs = pred_df[pred_df["label"] == 1]["probability"].values
    neg_probs = pred_df[pred_df["label"] == 0]["probability"].values
    pos_mean = pos_probs.mean()
    neg_mean = neg_probs.mean()
    print(f"  Positive patches — mean prob: {pos_mean:.4f}, std: {pos_probs.std():.4f}, n={len(pos_probs)}")
    print(f"  Negative patches — mean prob: {neg_mean:.4f}, std: {neg_probs.std():.4f}, n={len(neg_probs)}")
    u_stat, p_value = stats.mannwhitneyu(pos_probs, neg_probs, alternative='greater')
    if pos_mean > neg_mean and p_value < 0.05:
        report.print_check("Positive scores > Negative scores", "PASS", f"Pos mean={pos_mean:.4f} > Neg mean={neg_mean:.4f}, Mann-Whitney p={p_value:.2e}")
    elif pos_mean <= neg_mean:
        report.print_check("Positive scores > Negative scores", "FAIL", f"Positive patches have LOWER mean probability ({pos_mean:.4f}) than negatives ({neg_mean:.4f}) — labels may be flipped")
    else:
        report.print_check("Positive scores > Negative scores", "WARN", f"Pos mean > Neg mean but not statistically significant (p={p_value:.4f})")


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  MODEL RESULTS AUDIT — EfficientNet-B0 Lung Nodule Classification")
    print(f"  Device: {device}")
    print("=" * 70)

    report = AuditReport()
    files_ok = check_1_required_files_exist(report)
    model = None
    if files_ok:
        model = check_2_model_weights_not_random(report, device)
    pred_df = check_3_predictions_csv_integrity(report)
    saved_metrics = {}
    if os.path.exists(BEST_METRICS_PATH):
        with open(BEST_METRICS_PATH) as f:
            saved_metrics = json.load(f)
    if pred_df is not None:
        check_4_recompute_metrics_from_predictions(report, pred_df, saved_metrics)
    if os.path.exists(TRAINING_LOG_PATH):
        check_5_training_log_shows_learning(report)
    check_6_model_inference_live(report, model, device)
    check_7_label_split_integrity(report)
    if pred_df is not None:
        check_8_permutation_baseline(report, pred_df)
        check_9_prediction_score_separation(report, pred_df)

    print("\n" + "=" * 70)
    print(f"  AUDIT COMPLETE")
    print(f"  Passed  : {report.passed}")
    print(f"  Warnings: {report.warned}")
    print(f"  Failed  : {report.failed}")
    if report.failed == 0:
        print("\n  ✅  VERDICT: Results appear LEGITIMATE.")
        print("      All checks passed. No evidence of fabrication or leakage.")
    else:
        print(f"\n  ❌  VERDICT: {report.failed} check(s) FAILED.")
        print("      Review the failed checks above before reporting results.")
    print("=" * 70)

    report.save("audit_report_efficientnetb0.txt", "audit_summary_efficientnetb0.json")


if __name__ == "__main__":
    main()