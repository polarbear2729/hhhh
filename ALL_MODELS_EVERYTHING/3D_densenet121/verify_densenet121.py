"""
=============================================================================
MODEL RESULTS VERIFICATION & AUDIT SCRIPT (with Grad‑CAM++)
DenseNet121 - Lung Nodule Classification (LUNA16)
=============================================================================

PURPOSE:
    Independently verifies model legitimacy + visual interpretability.
    - Checks 1-9: data leakage, training logs, live inference, statistics
    - Check 10: Grad‑CAM++ on 2 high-confidence TPs → saves visual overlays

HOW TO RUN:
    python audit_densenet121_with_gradcam.py

PREREQUISITES:
    best_model_densenet121.pth, metadata_all.csv, patient_split.csv,
    densenet121_test_predictions.csv, densenet121_training_log.csv,
    best_metrics_densenet121.json

OUTPUT:
    audit_report_densenet121.txt
    audit_summary_densenet121.json
    gradcam_densenet121_audit/   (Grad‑CAM++ images & heatmaps)
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, recall_score
)
from scipy import stats
from monai.networks.nets import DenseNet121
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — Match exactly with your training script
# =============================================================================
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_densenet121.pth"
PREDICTIONS_PATH = "densenet121_test_predictions.csv"
TRAINING_LOG_PATH = "densenet121_training_log.csv"
BEST_METRICS_PATH = "best_metrics_densenet121.json"
BATCH_SIZE = 8
NUM_WORKERS = 4
POS_WEIGHT = 5115.0 / 822.0
TOLERANCE = 0.01

# Grad‑CAM output folder
GRADCAM_OUTPUT_DIR = "gradcam_densenet121_audit"
os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)


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
# MODEL (must match training script exactly)
# =============================================================================
class DenseNet121WithMLPHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        self.backbone = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1)
        self.backbone.class_layers.out = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1)


# =============================================================================
# DATASET (must match training script exactly)
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
# Grad‑CAM++ Implementation (hook‑based, 3D)
# =============================================================================
class GradCAMPlusPlus3D:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        target_module = self._find_module(model, target_layer_name)
        target_module.register_forward_hook(self._save_activation)
        target_module.register_backward_hook(self._save_gradient)

    def _find_module(self, module, layer_name):
        parts = layer_name.split('.')
        for p in parts:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p)
        return module

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=1):
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).requires_grad_(True)

        output = self.model(input_tensor)
        score = output[0] if target_class == 1 else -output[0]

        self.model.zero_grad()
        score.backward(retain_graph=False)

        acts = self.activations      # (1, C, D', H', W')
        grads = self.gradients       # (1, C, D', H', W')

        # Grad‑CAM++ weights
        grads_pow = grads.pow(2)
        grads_pow_sum = grads_pow.sum(dim=[2,3,4], keepdim=True)
        acts_pow = acts.pow(2)
        acts_pow_sum = acts_pow.sum(dim=[2,3,4], keepdim=True)

        alpha_num = grads_pow
        alpha_den = 2 * grads_pow + (acts_pow_sum * grads_pow).sum(dim=[2,3,4], keepdim=True) + 1e-8
        alpha = alpha_num / alpha_den

        weights = (alpha * F.relu(grads)).sum(dim=[2,3,4], keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='trilinear', align_corners=True)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0,1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


def find_target_layer_densenet(model):
    """Select the last Conv3d in denseblock3 for better spatial resolution."""
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            candidates.append(name)
    db3 = [n for n in candidates if 'denseblock3' in n]
    if db3:
        return db3[-1]   # last conv in denseblock3
    db4 = [n for n in candidates if 'denseblock4' in n]
    if db4:
        return db4[-1]
    return candidates[-1]  # fallback


# =============================================================================
# AUDIT REPORT CLASS (unchanged)
# =============================================================================
class AuditReport:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warned = 0

    def record(self, check_name, status, detail):
        self.checks.append({"check": check_name, "status": status, "detail": detail})
        if status == "PASS": self.passed += 1
        elif status == "FAIL": self.failed += 1
        else: self.warned += 1

    def print_check(self, check_name, status, detail):
        icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}
        print(f"  {icons.get(status, '?')} [{status}] {check_name}")
        print(f"         {detail}")
        self.record(check_name, status, detail)

    def save(self, txt_path, json_path):
        lines = []
        lines.append("="*70)
        lines.append("  MODEL AUDIT REPORT — DenseNet121 Lung Nodule Classification")
        lines.append("="*70)
        lines.append(f"  Total Checks : {len(self.checks)}")
        lines.append(f"  Passed       : {self.passed}")
        lines.append(f"  Warnings     : {self.warned}")
        lines.append(f"  Failed       : {self.failed}")
        lines.append("")
        for c in self.checks:
            icon = {"PASS":"[PASS]","FAIL":"[FAIL]","WARN":"[WARN]"}.get(c["status"],"[?]")
            lines.append(f"  {icon}  {c['check']}")
            lines.append(f"         {c['detail']}")
        lines.append("="*70)
        verdict = "RESULTS ARE LEGITIMATE ✅" if self.failed == 0 else f"AUDIT FAILED — {self.failed} critical issue(s) found ❌"
        lines.append(f"  VERDICT: {verdict}")
        lines.append("="*70)
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
# ORIGINAL CHECKS 1‑9 (unchanged except Check 6 fixed)
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
        trained_model = DenseNet121WithMLPHead().to(device)
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        random_model = DenseNet121WithMLPHead().to(device)
        trained_w = list(trained_model.parameters())[0].detach().cpu().numpy().flatten()
        random_w = list(random_model.parameters())[0].detach().cpu().numpy().flatten()
        correlation, _ = stats.pearsonr(trained_w[:1000], random_w[:1000])
        weight_diff = np.mean(np.abs(trained_w - random_w[:len(trained_w)]))
        if abs(correlation) > 0.99:
            report.print_check("Weights differ from random init", "FAIL", f"Pearson r={correlation:.4f}")
        else:
            report.print_check("Weights differ from random init", "PASS", f"Mean delta={weight_diff:.6f}, r={correlation:.4f}")
        mlp_w = list(trained_model.backbone.class_layers.out.parameters())[0].detach().cpu().numpy()
        mlp_std = mlp_w.std()
        if mlp_std < 1e-6:
            report.print_check("MLP head weights are non-trivial", "FAIL", f"MLP std={mlp_std:.2e}")
        else:
            report.print_check("MLP head weights are non-trivial", "PASS", f"MLP std={mlp_std:.6f}")
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
            report.print_check("Columns", "FAIL", f"Missing: {missing}")
            return None
        report.print_check("Columns", "PASS", f"All present. Shape: {pred_df.shape}")
        nan_count = pred_df[["label","probability"]].isna().sum().sum()
        if nan_count > 0:
            report.print_check("No NaN", "FAIL", f"{nan_count} NaN")
        else:
            report.print_check("No NaN", "PASS", "Zero NaN")
        out_of_range = ((pred_df["probability"] < 0) | (pred_df["probability"] > 1)).sum()
        if out_of_range > 0:
            report.print_check("Probs in [0,1]", "FAIL", f"{out_of_range} out of range")
        else:
            report.print_check("Probs in [0,1]", "PASS", f"Range: [{pred_df['probability'].min():.4f}, {pred_df['probability'].max():.4f}]")
        unique_labels = set(pred_df["label"].unique())
        if not unique_labels.issubset({0,1}):
            report.print_check("Labels binary", "FAIL", f"Non-binary: {unique_labels}")
        else:
            report.print_check("Labels binary", "PASS", f"Pos={ (pred_df['label']==1).sum()}, Neg={ (pred_df['label']==0).sum()}")
        prob_std = pred_df["probability"].std()
        if prob_std < 0.01:
            report.print_check("Not collapsed", "FAIL", f"Prob std={prob_std:.4f}")
        else:
            report.print_check("Not collapsed", "PASS", f"Prob std={prob_std:.4f}, mean={pred_df['probability'].mean():.4f}")
        if len(unique_labels) < 2:
            report.print_check("Both classes", "FAIL", "Only one class")
        else:
            report.print_check("Both classes", "PASS", "Both present")
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
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        recomputed["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except Exception as e:
        report.print_check("Metric recomputation", "FAIL", f"Exception: {e}")
        return
    report.print_check("AUROC recomputable", "PASS", f"AUC={recomputed['auc']:.4f}")
    report.print_check("AUPRC recomputable", "PASS", f"AUPRC={recomputed['auprc']:.4f}")
    if recomputed["auc"] < 0.5:
        report.print_check("AUROC sanity", "FAIL", f"{recomputed['auc']:.4f} < 0.5")
    elif recomputed["auc"] > 0.999:
        report.print_check("AUROC sanity", "WARN", f"{recomputed['auc']:.4f} suspiciously perfect")
    else:
        report.print_check("AUROC sanity", "PASS", f"{recomputed['auc']:.4f} plausible")
    if recomputed["sensitivity"] == 0:
        report.print_check("Sensitivity > 0", "FAIL", "Sensitivity=0.0")
    elif recomputed["sensitivity"] == 1:
        report.print_check("Sensitivity = 1.0", "WARN", "Perfect sensitivity")
    else:
        report.print_check("Sensitivity", "PASS", f"{recomputed['sensitivity']:.4f}")
    return recomputed

def check_5_training_log_shows_learning(report):
    print("\n[CHECK 5] Training Log Shows Real Learning")
    try:
        log_df = pd.read_csv(TRAINING_LOG_PATH)
        if len(log_df) < 3:
            report.print_check("Epochs enough", "WARN", f"Only {len(log_df)} epochs")
            return
        report.print_check("Epochs enough", "PASS", f"{len(log_df)} epochs recorded")
        first_loss = log_df["train_loss"].iloc[:3].mean()
        last_loss = log_df["train_loss"].iloc[-3:].mean()
        if last_loss < first_loss:
            report.print_check("Train loss decreased", "PASS", f"{first_loss:.4f} → {last_loss:.4f}")
        else:
            report.print_check("Train loss decreased", "FAIL", f"{first_loss:.4f} → {last_loss:.4f}")
        first_auc = log_df["val_auc"].iloc[:3].mean()
        best_auc = log_df["val_auc"].max()
        if best_auc > first_auc:
            report.print_check("Val AUC improved", "PASS", f"{first_auc:.4f} → {best_auc:.4f}")
        else:
            report.print_check("Val AUC improved", "FAIL", f"No improvement")
        loss_std = log_df["train_loss"].std()
        if loss_std < 1e-5:
            report.print_check("Loss not flat", "FAIL", f"std={loss_std:.2e}")
        else:
            report.print_check("Loss not flat", "PASS", f"std={loss_std:.6f}")
        auc_std = log_df["val_auc"].std()
        if auc_std < 1e-5 and log_df["val_auc"].mean() > 0.99:
            report.print_check("Val AUC realistic", "FAIL", "Impossible consistency")
        else:
            report.print_check("Val AUC realistic", "PASS", f"std={auc_std:.4f}")
    except Exception as e:
        report.print_check("Load training log", "FAIL", f"Exception: {e}")

def check_6_model_inference_live(report, model, device):
    print("\n[CHECK 6] Live Model Inference vs Saved Predictions")
    if model is None:
        report.print_check("Live inference", "FAIL", "Model unavailable")
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
        # FIXED: use num_workers=0, pin_memory=False to avoid CUDA fork issues
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
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
                live_probs.extend(probs.tolist())
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
            report.print_check("Live vs saved match", "WARN", "No matching UIDs to compare")
        elif matches / checked >= 0.90:
            report.print_check("Live vs saved match", "PASS", f"{matches}/{checked} match, max diff={max_diff:.6f}")
        else:
            report.print_check("Live vs saved match", "FAIL", f"Only {matches}/{checked} match, max diff={max_diff:.6f}")
        live_auc = roc_auc_score(live_labels, live_probs) if len(set(live_labels)) > 1 else None
        if live_auc is not None:
            if live_auc >= 0.5:
                report.print_check("Live AUROC on sample", "PASS", f"Live AUROC on {n_sample} samples = {live_auc:.4f}")
            else:
                report.print_check("Live AUROC on sample", "WARN", f"Live AUROC={live_auc:.4f} < 0.5 (small sample)")
    except Exception as e:
        report.print_check("Live inference", "FAIL", f"Exception: {e}")

def check_7_label_split_integrity(report):
    print("\n[CHECK 7] No Patient Overlap Between Splits")
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)
        train_ids = set(metadata[metadata["split"]=="train"]["seriesuid"].unique())
        val_ids   = set(metadata[metadata["split"]=="val"]["seriesuid"].unique())
        test_ids  = set(metadata[metadata["split"]=="test"]["seriesuid"].unique())
        for pair, overlap in [("Train ∩ Val", train_ids&val_ids),
                              ("Train ∩ Test", train_ids&test_ids),
                              ("Val ∩ Test", val_ids&test_ids)]:
            if overlap:
                report.print_check(f"No overlap: {pair}", "FAIL", f"{len(overlap)} patients")
            else:
                report.print_check(f"No overlap: {pair}", "PASS", "Zero overlap")
        for split_name in ["train","val","test"]:
            split_df = metadata[metadata["split"]==split_name]
            pos = (split_df["label"]==1).sum()
            neg = (split_df["label"]==0).sum()
            ratio = pos/len(split_df)*100
            report.print_check(f"Class distribution {split_name}", "PASS", f"Pos={pos}, Neg={neg}, Pos%={ratio:.1f}%")
    except Exception as e:
        report.print_check("Load metadata", "FAIL", f"Exception: {e}")

def check_8_permutation_baseline(report, pred_df):
    print("\n[CHECK 8] Permutation Test")
    if pred_df is None:
        report.print_check("Permutation", "FAIL", "No predictions")
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
    if p_value < 0.05:
        report.print_check("Permutation test", "PASS", f"p={p_value:.4f}, beats {percentile:.1f}% of random")
    else:
        report.print_check("Permutation test", "FAIL", f"p={p_value:.4f}, does not beat random significantly")

def check_9_prediction_score_separation(report, pred_df):
    print("\n[CHECK 9] Positive vs Negative Score Separation")
    if pred_df is None:
        report.print_check("Score separation", "FAIL", "No predictions")
        return
    pos_probs = pred_df[pred_df["label"]==1]["probability"].values
    neg_probs = pred_df[pred_df["label"]==0]["probability"].values
    pos_mean = pos_probs.mean()
    neg_mean = neg_probs.mean()
    u_stat, p_value = stats.mannwhitneyu(pos_probs, neg_probs, alternative='greater')
    if pos_mean > neg_mean and p_value < 0.05:
        report.print_check("Pos > Neg scores", "PASS", f"Pos mean={pos_mean:.4f} > Neg mean={neg_mean:.4f}, p={p_value:.2e}")
    elif pos_mean <= neg_mean:
        report.print_check("Pos > Neg scores", "FAIL", f"Pos mean={pos_mean:.4f} <= Neg mean={neg_mean:.4f}")
    else:
        report.print_check("Pos > Neg scores", "WARN", f"Not significant, p={p_value:.4f}")


# =============================================================================
# NEW: Grad‑CAM++ Audit (Check 10)
# =============================================================================
def check_10_gradcam_analysis(report, model, device, test_meta):
    """
    Generate Grad‑CAM++ heatmaps for 2 highest-confidence true positives
    from the test set. Saves 3‑panel figures and raw heatmaps.
    """
    print("\n[CHECK 10] Grad‑CAM++ Interpretability (2 high-confidence TPs)")
    if model is None:
        report.print_check("Grad‑CAM++", "FAIL", "Model not loaded")
        return

    # Select top 2 high-confidence TPs
    # Use patient-level predictions from the saved CSV (already loaded in earlier checks)
    pred_df = pd.read_csv(PREDICTIONS_PATH)
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

    tp_uids = [uid for uid, v in patient_dict.items() if v["label"] == 1 and v["prob"] >= 0.5]
    if len(tp_uids) < 2:
        report.print_check("Grad‑CAM++", "FAIL", f"Less than 2 TPs found ({len(tp_uids)})")
        return

    # Sort by confidence (highest first)
    tp_uids.sort(key=lambda uid: patient_dict[uid]["prob"], reverse=True)
    selected_uids = tp_uids[:2]

    # Find target layer
    target_layer = find_target_layer_densenet(model.backbone)
    print(f"  Using target layer: {target_layer}")
    cam_generator = GradCAMPlusPlus3D(model.backbone, target_layer)

    # For each selected patient, take the first positive patch
    dataset = NodulePatchDataset(test_meta, DATA_DIR)
    patient_patches = {}
    for idx in range(len(dataset)):
        patch, label, uid = dataset[idx]
        if uid in selected_uids and label == 1:
            if uid not in patient_patches:
                patient_patches[uid] = (patch, idx)
                if len(patient_patches) == 2:
                    break

    for rank, uid in enumerate(selected_uids):
        patch, _ = patient_patches[uid]
        input_tensor = patch.unsqueeze(0).to(device)  # (1, 1, D, H, W)
        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()

        print(f"\n  Processing TP {rank+1}: {uid[:20]}... (prob={prob:.4f})")
        heatmap = cam_generator.generate(input_tensor, target_class=1)
        vol = patch.squeeze(0).cpu().numpy()  # (D, H, W)

        # Find peak activation slice
        peak_coord = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        z_peak = peak_coord[0]
        middle = vol.shape[0] // 2

        # Save 3-panel images for peak and middle slices
        for slc, label in [(z_peak, "peak"), (middle, "middle")]:
            if 0 <= slc < vol.shape[0]:
                vmin, vmax = vol.min(), vol.max()
                vol_norm = (vol - vmin) / (vmax - vmin) if vmax > vmin else vol
                slice_img = vol_norm[slc]
                slice_heat = heatmap[slc]

                fig, axes = plt.subplots(1, 3, figsize=(15,5))
                axes[0].imshow(slice_img, cmap='gray')
                axes[0].set_title("Original"); axes[0].axis('off')
                heat_plot = axes[1].imshow(slice_heat, cmap='jet', vmin=0, vmax=1)
                axes[1].set_title("Grad‑CAM++ Heatmap"); axes[1].axis('off')
                plt.colorbar(heat_plot, ax=axes[1], fraction=0.046)
                axes[2].imshow(slice_img, cmap='gray')
                axes[2].imshow(slice_heat, cmap='jet', alpha=0.5)
                axes[2].set_title("Overlay"); axes[2].axis('off')
                fig.suptitle(f"TP{rank+1} – slice {slc} – prob={prob:.4f}")
                plt.tight_layout()
                fname = os.path.join(GRADCAM_OUTPUT_DIR, f"TP{rank+1}_{uid[:12]}_slice{slc}.png")
                plt.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved: {fname}")

        # Save raw heatmap
        np.save(os.path.join(GRADCAM_OUTPUT_DIR, f"TP{rank+1}_{uid[:12]}_heatmap.npy"), heatmap)
        # Print peak location and intensities
        print(f"    Peak at (z,y,x): {peak_coord}, value at peak: {vol[peak_coord]:.6f}")
        print(f"    Value at center: {vol[middle, vol.shape[1]//2, vol.shape[2]//2]:.6f}")

    report.print_check("Grad‑CAM++ images saved", "PASS", f"Saved to {GRADCAM_OUTPUT_DIR}/")


# =============================================================================
# MAIN AUDIT RUNNER
# =============================================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*70)
    print("  MODEL RESULTS AUDIT (with Grad‑CAM++) — DenseNet121")
    print(f"  Device: {device}")
    print("="*70)

    report = AuditReport()

    # Original checks
    files_ok = check_1_required_files_exist(report)
    model = None
    if files_ok:
        model = check_2_model_weights_not_random(report, device)
    pred_df = check_3_predictions_csv_integrity(report)
    saved_metrics = {}
    if os.path.exists(BEST_METRICS_PATH):
        with open(BEST_METRICS_PATH) as f:
            saved_metrics = json.load(f)
    recomputed = None
    if pred_df is not None:
        recomputed = check_4_recompute_metrics_from_predictions(report, pred_df, saved_metrics)
    if os.path.exists(TRAINING_LOG_PATH):
        check_5_training_log_shows_learning(report)
    check_6_model_inference_live(report, model, device)
    check_7_label_split_integrity(report)
    if pred_df is not None:
        check_8_permutation_baseline(report, pred_df)
        check_9_prediction_score_separation(report, pred_df)

    # New Grad‑CAM++ check
    if model is not None and pred_df is not None:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)
        test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
        check_10_gradcam_analysis(report, model, device, test_meta)
    else:
        report.print_check("Grad‑CAM++", "FAIL", "Model or predictions not available")

    # Final summary
    print("\n" + "="*70)
    print(f"  AUDIT COMPLETE")
    print(f"  Passed  : {report.passed}")
    print(f"  Warnings: {report.warned}")
    print(f"  Failed  : {report.failed}")
    if report.failed == 0:
        print("\n  ✅  VERDICT: Results appear LEGITIMATE.")
    else:
        print(f"\n  ❌  VERDICT: {report.failed} check(s) FAILED.")
    print("="*70)
    report.save("audit_report_densenet121.txt", "audit_summary_densenet121.json")


if __name__ == "__main__":
    main()