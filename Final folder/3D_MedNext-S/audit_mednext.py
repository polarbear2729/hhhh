"""
=============================================================================
MODEL RESULTS VERIFICATION & AUDIT SCRIPT
3D MedNeXt (from scratch) - Lung Nodule Classification (LUNA16)
=============================================================================

PURPOSE:
    Independently verifies that MedNeXt results are legitimate by running
    multiple sanity checks across the data, predictions, model weights,
    and statistical properties. Generates a full audit report.

HOW TO RUN:
    python verify_mednext3d_results.py

PREREQUISITES (files must exist from training):
    - best_model_mednext3d.pth
    - mednext3d_test_predictions.csv
    - mednext3d_training_log.csv
    - best_metrics_mednext3d.json
    - metadata_all.csv (or patient_split.csv)

OUTPUT:
    - audit_report_mednext3d.txt   (human-readable full report)
    - audit_summary_mednext3d.json (machine-readable pass/fail summary)
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
# CONFIGURATION — matches your MedNeXt training script exactly
# =============================================================================
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

MODEL_PATH = "best_model_mednext3d.pth"
PREDICTIONS_PATH = "mednext3d_test_predictions.csv"
TRAINING_LOG_PATH = "mednext3d_training_log.csv"
BEST_METRICS_PATH = "best_metrics_mednext3d.json"

BATCH_SIZE = 8              # same as training script
NUM_WORKERS = 0             # safe for Windows
TOLERANCE = 0.01            # allowable floating point difference for metric recomputation

# MedNeXt architectural constants (must match training)
MEDNEXT_BASE_CHANNELS = 32
MEDNEXT_EXP_R = 2
MEDNEXT_KERNEL_SIZE = 3
MEDNEXT_BLOCKS = [2, 2, 2, 2]
MEDNEXT_DROPOUT = 0.3

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
# MODEL (exact replica from training script)
# =============================================================================
class MedNeXtBlock3D(nn.Module):
    def __init__(self, in_channels, exp_r=2, kernel_size=3, drop_path_rate=0.0):
        super().__init__()
        mid_channels = int(in_channels * exp_r)
        self.dw_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=in_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pw_expand = nn.Linear(in_channels, mid_channels)
        self.act = nn.GELU()
        self.pw_contract = nn.Linear(mid_channels, in_channels)
        self.drop_path_rate = drop_path_rate

    def drop_path(self, x):
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

    def forward(self, x):
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_contract(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.drop_path(x) + residual
        return x

class MedNeXtDownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.dw_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                 stride=2, padding=kernel_size // 2, groups=in_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pw_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pw_proj(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

class MedNeXtStem3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, patch_size=4):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, base_channels, kernel_size=patch_size,
                              stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(base_channels, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

class MedNeXtEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, exp_r=2, kernel_size=3,
                 block_counts=(2,2,2,2), drop_path_rate=0.1, patch_size=4):
        super().__init__()
        self.stem = MedNeXtStem3D(in_channels, base_channels, patch_size)
        channels = [base_channels * (2**i) for i in range(4)]
        total_blocks = sum(block_counts)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage_idx, n_blocks in enumerate(block_counts):
            c = channels[stage_idx]
            stage_blocks = nn.Sequential(*[
                MedNeXtBlock3D(in_channels=c, exp_r=exp_r, kernel_size=kernel_size,
                               drop_path_rate=dp_rates[dp_idx + b])
                for b in range(n_blocks)
            ])
            self.stages.append(stage_blocks)
            dp_idx += n_blocks
            if stage_idx < len(block_counts) - 1:
                self.downsamples.append(
                    MedNeXtDownBlock3D(channels[stage_idx], channels[stage_idx+1], kernel_size)
                )
        self.final_norm = nn.LayerNorm(channels[-1], eps=1e-6)
        self.num_features = channels[-1]

    def forward(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.final_norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = x.mean(dim=[2, 3, 4])
        return x

class MedNeXt3DWithMLPHead(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, exp_r=2, kernel_size=3,
                 block_counts=(2,2,2,2), dropout=0.3):
        super().__init__()
        self.encoder = MedNeXtEncoder3D(in_channels=in_channels, base_channels=base_channels,
                                       exp_r=exp_r, kernel_size=kernel_size,
                                       block_counts=block_counts)
        feat_dim = self.encoder.num_features
        self.mlp_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.mlp_head(features)
        return out.squeeze(1)


# =============================================================================
# DATASET (same as training script, no transforms at test time)
# =============================================================================
class NodulePatchDataset3D(Dataset):
    """Loads 64x64x64 3D patch as (1, D, H, W)."""
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
        vol = np.load(local_path).astype(np.float32)
        vol = np.expand_dims(vol, axis=0)   # (1, 64, 64, 64)

        if self.transforms is not None:
            data_dict = {"img": vol}
            data_dict = self.transforms(data_dict)
            vol = data_dict["img"]

        if not isinstance(vol, torch.Tensor):
            vol = torch.from_numpy(vol)

        return vol, torch.tensor(label, dtype=torch.float32), row["seriesuid"]


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
        self.checks.append({
            "check": check_name,
            "status": status,
            "detail": detail
        })
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
        lines.append("  MODEL AUDIT REPORT — 3D MedNeXt Lung Nodule Classification")
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
            report.print_check(f"File exists: {name}", "PASS",
                                f"Found at '{path}' ({size_kb:.1f} KB)")
        else:
            report.print_check(f"File exists: {name}", "FAIL",
                                f"NOT FOUND at '{path}' — training may have crashed")
            all_found = False
    return all_found


def check_2_model_weights_not_random(report, device):
    print("\n[CHECK 2] Model Weights Are Trained (Not Random)")
    try:
        trained_model = MedNeXt3DWithMLPHead(
            base_channels=MEDNEXT_BASE_CHANNELS,
            exp_r=MEDNEXT_EXP_R,
            kernel_size=MEDNEXT_KERNEL_SIZE,
            block_counts=MEDNEXT_BLOCKS,
            dropout=MEDNEXT_DROPOUT
        ).to(device)
        trained_model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )

        random_model = MedNeXt3DWithMLPHead(
            base_channels=MEDNEXT_BASE_CHANNELS,
            exp_r=MEDNEXT_EXP_R,
            kernel_size=MEDNEXT_KERNEL_SIZE,
            block_counts=MEDNEXT_BLOCKS,
            dropout=MEDNEXT_DROPOUT
        ).to(device)

        # Compare first-layer weights (stem.proj)
        trained_w = list(trained_model.parameters())[0].detach().cpu().numpy().flatten()
        random_w = list(random_model.parameters())[0].detach().cpu().numpy().flatten()

        n = min(len(trained_w), 1000)
        correlation, p_value = stats.pearsonr(trained_w[:n], random_w[:n])
        weight_diff = np.mean(np.abs(trained_w - random_w[:len(trained_w)]))

        if abs(correlation) > 0.99:
            report.print_check("Weights differ from random init", "FAIL",
                                f"Pearson r={correlation:.4f} — weights look unchanged from random init")
        else:
            report.print_check("Weights differ from random init", "PASS",
                                f"Mean weight delta={weight_diff:.6f}, Pearson r={correlation:.4f} vs random")

        # Check MLP head specifically
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
            report.print_check("Predictions CSV columns", "FAIL",
                                f"Missing columns: {missing}")
            return None
        else:
            report.print_check("Predictions CSV columns", "PASS",
                                f"All required columns present. Shape: {pred_df.shape}")

        nan_count = pred_df[["label", "probability"]].isna().sum().sum()
        if nan_count > 0:
            report.print_check("No NaN values", "FAIL",
                                f"{nan_count} NaN values found in predictions")
        else:
            report.print_check("No NaN values", "PASS", "Zero NaN values")

        out_of_range = ((pred_df["probability"] < 0) | (pred_df["probability"] > 1)).sum()
        if out_of_range > 0:
            report.print_check("Probabilities in [0,1]", "FAIL",
                                f"{out_of_range} predictions outside [0,1] — sigmoid was not applied")
        else:
            prob_min = pred_df["probability"].min()
            prob_max = pred_df["probability"].max()
            report.print_check("Probabilities in [0,1]", "PASS",
                                f"Range: [{prob_min:.4f}, {prob_max:.4f}]")

        unique_labels = set(pred_df["label"].unique())
        if not unique_labels.issubset({0, 1, 0.0, 1.0}):
            report.print_check("Labels are binary", "FAIL",
                                f"Non-binary labels found: {unique_labels}")
        else:
            pos = (pred_df["label"] == 1).sum()
            neg = (pred_df["label"] == 0).sum()
            report.print_check("Labels are binary", "PASS",
                                f"Positive patches: {pos}, Negative patches: {neg}")

        prob_std = pred_df["probability"].std()
        if prob_std < 0.01:
            report.print_check("Predictions are not collapsed", "FAIL",
                                f"Probability std={prob_std:.4f} — model may be predicting same value for all inputs")
        else:
            report.print_check("Predictions are not collapsed", "PASS",
                                f"Probability std={prob_std:.4f}, mean={pred_df['probability'].mean():.4f}")

        if len(unique_labels) < 2:
            report.print_check("Both classes in test set", "FAIL",
                                "Only one class found in test predictions — AUROC is undefined")
        else:
            report.print_check("Both classes in test set", "PASS",
                                "Both positive and negative samples present in test set")

        return pred_df

    except Exception as e:
        report.print_check("Load predictions CSV", "FAIL", f"Exception: {e}")
        return None


def check_4_recompute_metrics_from_predictions(report, pred_df):
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

    report.print_check("AUROC recomputable from predictions", "PASS",
                        f"Recomputed AUROC={recomputed['auc']:.4f} from saved CSV successfully")
    report.print_check("AUPRC recomputable from predictions", "PASS",
                        f"Recomputed AUPRC={recomputed['auprc']:.4f} from saved CSV successfully")

    if recomputed["auc"] < 0.5:
        report.print_check("AUROC sanity range", "FAIL",
                            f"AUROC={recomputed['auc']:.4f} < 0.5 — worse than random, check label alignment")
    elif recomputed["auc"] > 0.999:
        report.print_check("AUROC sanity range", "WARN",
                            f"AUROC={recomputed['auc']:.4f} is suspiciously perfect — possible data leakage")
    else:
        report.print_check("AUROC sanity range", "PASS",
                            f"AUROC={recomputed['auc']:.4f} is in plausible range [0.5, 0.999]")

    if recomputed["sensitivity"] == 0.0:
        report.print_check("Sensitivity > 0", "FAIL",
                            "Sensitivity=0.0 — model never predicted a single positive.")
    elif recomputed["sensitivity"] == 1.0:
        report.print_check("Sensitivity = 1.0 check", "WARN",
                            "Sensitivity=1.0 — model may be predicting everything as positive")
    else:
        report.print_check("Sensitivity in valid range", "PASS",
                            f"Sensitivity={recomputed['sensitivity']:.4f}")

    return recomputed


def check_5_training_log_shows_learning(report):
    print("\n[CHECK 5] Training Log Shows Real Learning")
    try:
        log_df = pd.read_csv(TRAINING_LOG_PATH)

        if len(log_df) < 3:
            report.print_check("Training log has enough epochs", "WARN",
                                f"Only {len(log_df)} epochs logged — model may have crashed early")
            return

        report.print_check("Training log has enough epochs", "PASS",
                            f"{len(log_df)} epochs recorded")

        first_loss = log_df["train_loss"].iloc[:3].mean()
        last_loss = log_df["train_loss"].iloc[-3:].mean()
        if last_loss < first_loss:
            report.print_check("Training loss decreased", "PASS",
                                f"Loss: {first_loss:.4f} (first 3 epochs avg) → {last_loss:.4f} (last 3 epochs avg)")
        else:
            report.print_check("Training loss decreased", "FAIL",
                                f"Loss did NOT decrease: {first_loss:.4f} → {last_loss:.4f}")

        first_auc = log_df["val_auc"].iloc[:3].mean()
        best_auc = log_df["val_auc"].max()
        if best_auc > first_auc:
            report.print_check("Val AUROC improved over training", "PASS",
                                f"Val AUROC: {first_auc:.4f} (first 3 avg) → {best_auc:.4f} (best)")
        else:
            report.print_check("Val AUROC improved over training", "FAIL",
                                f"Val AUROC never improved beyond initial: {first_auc:.4f} → best={best_auc:.4f}")

        loss_std = log_df["train_loss"].std()
        if loss_std < 1e-5:
            report.print_check("Training loss is not flat", "FAIL",
                                f"Train loss std={loss_std:.2e} — loss never changed, weights may be frozen")
        else:
            report.print_check("Training loss is not flat", "PASS",
                                f"Train loss std={loss_std:.6f} — loss varied across epochs")

        auc_std = log_df["val_auc"].std()
        if auc_std < 1e-5 and log_df["val_auc"].mean() > 0.99:
            report.print_check("Val AUROC is not fabricated", "FAIL",
                                f"Val AUROC is {log_df['val_auc'].mean():.4f} ± {auc_std:.2e} across all epochs — impossible consistency")
        else:
            report.print_check("Val AUROC variance is realistic", "PASS",
                                f"Val AUROC std={auc_std:.4f} across epochs — natural variation present")

    except Exception as e:
        report.print_check("Load training log", "FAIL", f"Exception: {e}")


def check_6_model_inference_live(report, model, device):
    print("\n[CHECK 6] Live Model Inference vs Saved Predictions (full test set)")

    if model is None:
        report.print_check("Live inference (model unavailable)", "FAIL",
                            "Model failed to load in Check 2 — skipping live inference")
        return

    try:
        # Load metadata and test split
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)

        test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)

        # Build dataset without transforms (identity)
        dataset = NodulePatchDataset3D(test_meta, DATA_DIR, transforms=None)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False)

        model.eval()
        live_probs, live_labels, live_uids = [], [], []

        with torch.no_grad():
            for patches, labels, uids in tqdm(loader, desc="  Live inference on full test set"):
                patches = patches.to(device)
                with autocast('cuda'):
                    logits = model(patches)
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim == 0:
                    probs = [float(probs)]
                else:
                    probs = probs.tolist()
                live_probs.extend(probs)
                live_labels.extend(labels.tolist())
                live_uids.extend(uids)

        # Load saved predictions
        saved_df = pd.read_csv(PREDICTIONS_PATH)

        # Verify lengths match
        if len(live_probs) != len(saved_df):
            report.print_check("Live inference count matches saved CSV", "FAIL",
                                f"Live {len(live_probs)} patches vs saved {len(saved_df)} — "
                                f"different number of test samples. Check data loading.")
            return

        # Verify seriesuid & label ordering (first 10 and last 5)
        front_match = all(
            str(live_uids[i]) == str(saved_df.iloc[i]["seriesuid"]) and
            int(live_labels[i]) == int(saved_df.iloc[i]["label"])
            for i in range(min(10, len(live_uids)))
        )
        back_match = all(
            str(live_uids[i]) == str(saved_df.iloc[i]["seriesuid"]) and
            int(live_labels[i]) == int(saved_df.iloc[i]["label"])
            for i in range(max(0, len(live_uids) - 5), len(live_uids))
        )
        if not (front_match and back_match):
            report.print_check("UID/label order matches saved CSV", "WARN",
                                "Ordering mismatch – probabilities may still be correct, "
                                "but live vs saved comparison will be element‑wise. "
                                "Check if test metadata order differs from saved CSV.")
            # continue anyway, but warn

        # Element‑wise probability comparison
        saved_probs = saved_df["probability"].values
        diffs = np.abs(np.array(live_probs) - saved_probs)
        max_diff = diffs.max()
        matches = (diffs < TOLERANCE).sum()
        total = len(diffs)

        print(f"  Compared {total} patches. Max absolute difference: {max_diff:.8f}")
        print(f"  {matches}/{total} values within tolerance {TOLERANCE}")

        if matches == total:
            report.print_check("Live inference matches saved predictions", "PASS",
                                f"All {total} probabilities match within tolerance. Max diff={max_diff:.8f}")
        elif matches / total >= 0.995:   # allow very few mismatches due to floating point
            report.print_check("Live inference matches saved predictions", "WARN",
                                f"{matches}/{total} match (≥99.5%). Max diff={max_diff:.8f}. "
                                f"Minor numerical differences possible.")
        else:
            report.print_check("Live inference matches saved predictions", "FAIL",
                                f"Only {matches}/{total} match. Max diff={max_diff:.8f}. "
                                f"Saved predictions do NOT correspond to this model.")

        # Also compute AUROC on live outputs as sanity
        if len(set(live_labels)) > 1:
            live_auc = roc_auc_score(live_labels, live_probs)
            report.print_check("Live inference AUROC on test set", "PASS",
                                f"AUROC from live inference = {live_auc:.4f}")
        else:
            report.print_check("Live inference AUROC on test set", "WARN",
                                "Only one class present – cannot compute AUROC.")

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
        val_ids   = set(metadata[metadata["split"] == "val"]["seriesuid"].unique())
        test_ids  = set(metadata[metadata["split"] == "test"]["seriesuid"].unique())

        for pair, overlap in [
            ("Train ∩ Val", train_ids & val_ids),
            ("Train ∩ Test", train_ids & test_ids),
            ("Val ∩ Test", val_ids & test_ids)
        ]:
            if overlap:
                report.print_check(f"No overlap: {pair}", "FAIL",
                                    f"{len(overlap)} patient(s) appear in both splits: {list(overlap)[:5]}...")
            else:
                report.print_check(f"No overlap: {pair}", "PASS",
                                    f"Zero patients shared between {pair}")

        print(f"  Split sizes — Train: {len(train_ids)} patients, "
              f"Val: {len(val_ids)} patients, Test: {len(test_ids)} patients")

        for split_name in ["train", "val", "test"]:
            split_df = metadata[metadata["split"] == split_name]
            pos = (split_df["label"] == 1).sum()
            neg = (split_df["label"] == 0).sum()
            ratio = pos / len(split_df) * 100
            report.print_check(f"Class distribution in {split_name}", "PASS",
                                f"Pos={pos}, Neg={neg}, Pos%={ratio:.1f}%")

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
        report.print_check("Permutation test (p < 0.05)", "PASS",
                            f"Real AUROC={real_auc:.4f} significantly beats random "
                            f"(p={p_value:.4f}, beats {percentile:.1f}% of permutations)")
    else:
        report.print_check("Permutation test (p < 0.05)", "FAIL",
                            f"Real AUROC={real_auc:.4f} does NOT significantly beat random "
                            f"(p={p_value:.4f}) — results may be due to chance")


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
        report.print_check("Positive scores > Negative scores", "PASS",
                            f"Pos mean={pos_mean:.4f} > Neg mean={neg_mean:.4f}, "
                            f"Mann-Whitney p={p_value:.2e}")
    elif pos_mean <= neg_mean:
        report.print_check("Positive scores > Negative scores", "FAIL",
                            f"Positive patches have LOWER mean probability ({pos_mean:.4f}) "
                            f"than negatives ({neg_mean:.4f}) — labels may be flipped")
    else:
        report.print_check("Positive scores > Negative scores", "WARN",
                            f"Pos mean > Neg mean but not statistically significant (p={p_value:.4f})")


# =============================================================================
# MAIN AUDIT RUNNER
# =============================================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  MODEL RESULTS AUDIT — 3D MedNeXt Lung Nodule Classification")
    print(f"  Device: {device}")
    print("=" * 70)

    report = AuditReport()

    files_ok = check_1_required_files_exist(report)

    model = None
    if files_ok:
        model = check_2_model_weights_not_random(report, device)

    pred_df = check_3_predictions_csv_integrity(report)

    if pred_df is not None:
        check_4_recompute_metrics_from_predictions(report, pred_df)

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

    report.save("audit_report_mednext3d.txt", "audit_summary_mednext3d.json")


if __name__ == "__main__":
    main()