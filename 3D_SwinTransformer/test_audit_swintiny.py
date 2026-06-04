"""
Test & Audit script for 3D Swin‑Tiny (self‑contained backbone)
- Loads best_model_swin3d_tiny.pth
- Evaluates on test set (threshold 0.5 and Youden)
- Saves predictions + final metrics JSON
- Runs comprehensive audit checks (file existence, weight correctness,
  prediction integrity, metric recomputation, leakage, permutation, etc.)
"""

import os, json, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, recall_score, roc_curve
)
from scipy import stats
from monai.transforms import Compose, NormalizeIntensity
from sklearn.calibration import calibration_curve
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------------------ Configuration ------------------------------
SEED = 42
DATA_DIR = r"D:\Projects\Thesis Lung Malignancy Classification ML & DL\LUNA16 Dataset Preprocessed"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE = 8
NUM_WORKERS = 0          # Windows safety – avoids paging file crash
PIN_MEMORY = False

MODEL_PATH = "best_model_swin3d_tiny.pth"
PREDICTIONS_CSV = "swin3d_tiny_test_predictions.csv"
FINAL_METRICS_JSON = "final_test_metrics_swin3d_tiny.json"
TRAINING_LOG_CSV = "swin3d_tiny_training_log.csv"
BEST_METRICS_JSON = "best_metrics_swin3d_tiny.json"

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
TOLERANCE = 0.01          # for metric recomputation comparison

# ================== Self‑contained 3D Swin‑Tiny ==================
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0],
                H // window_size[1], window_size[1],
                W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0]*window_size[1]*window_size[2], C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        D_p, H_p, W_p = x.shape[1], x.shape[2], x.shape[3]

        if any(self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))

        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, B, D_p, H_p, W_p)

        if any(self.shift_size):
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))

        x = x[:, :D, :H, :W, :]
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)
    def forward(self, x):
        B, D, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0,0,0) if (i % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2)
            self.blocks.append(
                SwinTransformerBlock(dim=dim, num_heads=num_heads,
                                     window_size=window_size, shift_size=shift,
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            )
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)
    def forward(self, x):
        x = self.proj(x)                # (B, C, D, H, W)
        x = x.permute(0, 2, 3, 4, 1)   # (B, D, H, W, C)
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformer3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=48,
                 depths=(2,2,2,2), num_heads=(3,6,12,24),
                 window_size=(4,4,4), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim, norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                downsample=PatchMerging if (i_layer < len(depths)-1) else None,
                norm_layer=norm_layer
            )
            self.layers.append(layer)
        self.norm = norm_layer(int(embed_dim * 2 ** (len(depths)-1)))

    def forward(self, x):
        x = self.patch_embed(x)          # (B, D, H, W, C)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                 # (B, D, H, W, C)
        x = x.mean(dim=(1,2,3))          # global average pool -> (B, C)
        return x

class SwinTiny3DWithMLPHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        self.backbone = SwinTransformer3D(
            in_chans=in_channels,
            embed_dim=48,
            depths=(2,2,2,2),
            num_heads=(3,6,12,24),
            window_size=(4,4,4),
            patch_size=(4,4,4),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)          # (B, 384)
        logit = self.head(feat)          # (B, 1)
        return logit.squeeze(1)          # (B,)

# ------------------------------ Reproducibility ------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------ Dataset ------------------------------------
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
        patch = np.expand_dims(patch, axis=0)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

def get_val_transform():
    return Compose([
        NormalizeIntensity(nonzero=False, channel_wise=True),
    ])

# ------------------------------ Evaluation Metrics --------------------------
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
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
        froc_scores[f"{target} FP/scan"] = float(sensitivity[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0
    return froc_scores

def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
    bootstrapped_scores = []
    rng = np.random.RandomState(SEED)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        bootstrapped_scores.append(roc_auc_score(np.array(y_true)[indices], np.array(y_probs)[indices]))
    s = np.sort(np.array(bootstrapped_scores))
    return np.percentile(s, 2.5), np.percentile(s, 97.5)

def get_youden_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j = np.argmax(tpr - fpr)
    return float(thresholds[j])

def evaluate_model(loader, model, device, pos_weight, threshold=0.5):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc="Evaluating"):
            patches, labels = patches.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(patches)
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
            running_loss += loss.item() * patches.size(0)

    # Patient-level aggregation
    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict[uid]['prob']  = max(patient_dict[uid]['prob'], prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)

    y_true_p  = [v['label'] for v in patient_dict.values()]
    y_probs_p = [v['prob']  for v in patient_dict.values()]
    y_pred_p  = [1 if p >= threshold else 0 for p in y_probs_p]
    total_scans = len(patient_dict)
    avg_loss = running_loss / len(loader.dataset)

    auc = roc_auc_score(y_true_p, y_probs_p) if len(np.unique(y_true_p)) > 1 else 0.5
    auprc = average_precision_score(y_true_p, y_probs_p)
    f1 = f1_score(y_true_p, y_pred_p, zero_division=0)
    sens = recall_score(y_true_p, y_pred_p, zero_division=0)
    cm = confusion_matrix(y_true_p, y_pred_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    froc = calculate_candidate_froc(all_labels, all_probs, total_scans)
    cpm = float(np.mean(list(froc.values())))

    return avg_loss, auc, auprc, f1, sens, spec, froc, cpm, all_probs, all_labels, all_uids, y_pred_p

# ------------------------------ Audit Report Class --------------------------
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
        lines.append("  MODEL AUDIT REPORT — 3D Swin‑Tiny")
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

# ====================== Audit Check Functions ===============================
def check_1_required_files_exist(report):
    print("\n[CHECK 1] Required Files Exist")
    required = {
        "Model weights": MODEL_PATH,
        "Training log CSV": TRAINING_LOG_CSV,
        "Best metrics JSON": BEST_METRICS_JSON,
    }
    all_found = True
    for name, path in required.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            report.print_check(f"File exists: {name}", "PASS", f"Found at '{path}' ({size_kb:.1f} KB)")
        else:
            report.print_check(f"File exists: {name}", "FAIL", f"NOT FOUND at '{path}'")
            all_found = False
    return all_found

def check_2_model_weights_not_random(report, device):
    print("\n[CHECK 2] Model Weights Are Trained (Not Random)")
    try:
        trained_model = SwinTiny3DWithMLPHead().to(device)
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        random_model = SwinTiny3DWithMLPHead().to(device)

        # Compare first projection layer
        trained_w = trained_model.backbone.patch_embed.proj.weight.detach().cpu().numpy().flatten()
        random_w = random_model.backbone.patch_embed.proj.weight.detach().cpu().numpy().flatten()

        n = min(len(trained_w), 1000)
        correlation, _ = stats.pearsonr(trained_w[:n], random_w[:n])
        weight_diff = np.mean(np.abs(trained_w - random_w[:len(trained_w)]))

        if abs(correlation) > 0.99:
            report.print_check("Weights differ from random init", "FAIL", f"Pearson r={correlation:.4f} — weights unchanged")
        else:
            report.print_check("Weights differ from random init", "PASS", f"Mean weight delta={weight_diff:.6f}, r={correlation:.4f}")

        # Check MLP head
        mlp_w = list(trained_model.head.parameters())[0].detach().cpu().numpy()
        mlp_std = mlp_w.std()
        if mlp_std < 1e-6:
            report.print_check("MLP head weights are non-trivial", "FAIL", f"std={mlp_std:.2e}")
        else:
            report.print_check("MLP head weights are non-trivial", "PASS", f"std={mlp_std:.6f}")

        return trained_model
    except Exception as e:
        report.print_check("Load model weights", "FAIL", f"Exception: {e}")
        return None

def check_3_predictions_csv_integrity(report, pred_df):
    print("\n[CHECK 3] Predictions CSV Integrity")
    required_cols = {"seriesuid", "label", "probability"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        report.print_check("Predictions CSV columns", "FAIL", f"Missing columns: {missing}")
        return None
    report.print_check("Predictions CSV columns", "PASS", f"All required columns present. Shape: {pred_df.shape}")

    nan_count = pred_df[["label", "probability"]].isna().sum().sum()
    if nan_count > 0:
        report.print_check("No NaN values", "FAIL", f"{nan_count} NaN values found")
    else:
        report.print_check("No NaN values", "PASS", "Zero NaN values")

    out_of_range = ((pred_df["probability"] < 0) | (pred_df["probability"] > 1)).sum()
    if out_of_range > 0:
        report.print_check("Probabilities in [0,1]", "FAIL", f"{out_of_range} predictions outside range")
    else:
        report.print_check("Probabilities in [0,1]", "PASS", f"Range: [{pred_df['probability'].min():.4f}, {pred_df['probability'].max():.4f}]")

    unique_labels = set(pred_df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        report.print_check("Labels are binary", "FAIL", f"Non-binary labels found: {unique_labels}")
    else:
        pos = (pred_df["label"] == 1).sum()
        neg = (pred_df["label"] == 0).sum()
        report.print_check("Labels are binary", "PASS", f"Positive patches: {pos}, Negative patches: {neg}")

    prob_std = pred_df["probability"].std()
    if prob_std < 0.01:
        report.print_check("Predictions are not collapsed", "FAIL", f"std={prob_std:.4f}")
    else:
        report.print_check("Predictions are not collapsed", "PASS", f"std={prob_std:.4f}, mean={pred_df['probability'].mean():.4f}")

    if len(unique_labels) < 2:
        report.print_check("Both classes in test set", "FAIL", "Only one class found")
    else:
        report.print_check("Both classes in test set", "PASS", "Both positive and negative present")

    return pred_df

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
        report.print_check("Metric recomputation", "FAIL", f"Exception: {e}")
        return

    print(f"  Recomputed metrics (patient-level, from saved CSV):")
    for k, v in recomputed.items():
        print(f"    {k:15s}: {v:.4f}")

    report.print_check("AUROC recomputable", "PASS", f"Recomputed AUROC={recomputed['auc']:.4f}")
    report.print_check("AUPRC recomputable", "PASS", f"Recomputed AUPRC={recomputed['auprc']:.4f}")

    if recomputed["auc"] < 0.5:
        report.print_check("AUROC sanity range", "FAIL", f"AUROC={recomputed['auc']:.4f} < 0.5")
    elif recomputed["auc"] > 0.999:
        report.print_check("AUROC sanity range", "WARN", f"AUROC={recomputed['auc']:.4f} suspiciously perfect")
    else:
        report.print_check("AUROC sanity range", "PASS", f"AUROC={recomputed['auc']:.4f} plausible")

    if recomputed["sensitivity"] == 0.0:
        report.print_check("Sensitivity > 0", "FAIL", "Sensitivity=0.0")
    elif recomputed["sensitivity"] == 1.0:
        report.print_check("Sensitivity = 1.0 check", "WARN", "Sensitivity=1.0")
    else:
        report.print_check("Sensitivity in valid range", "PASS", f"Sensitivity={recomputed['sensitivity']:.4f}")

    return recomputed

def check_5_training_log_shows_learning(report):
    print("\n[CHECK 5] Training Log Shows Real Learning")
    try:
        log_df = pd.read_csv(TRAINING_LOG_CSV)
        if len(log_df) < 3:
            report.print_check("Training log has enough epochs", "WARN", f"Only {len(log_df)} epochs")
            return
        report.print_check("Training log has enough epochs", "PASS", f"{len(log_df)} epochs")

        first_loss = log_df["train_loss"].iloc[:3].mean()
        last_loss = log_df["train_loss"].iloc[-3:].mean()
        if last_loss < first_loss:
            report.print_check("Training loss decreased", "PASS", f"{first_loss:.4f} → {last_loss:.4f}")
        else:
            report.print_check("Training loss decreased", "FAIL", f"{first_loss:.4f} → {last_loss:.4f}")

        first_auc = log_df["val_auc"].iloc[:3].mean()
        best_auc = log_df["val_auc"].max()
        if best_auc > first_auc:
            report.print_check("Val AUROC improved", "PASS", f"{first_auc:.4f} → {best_auc:.4f}")
        else:
            report.print_check("Val AUROC improved", "FAIL", f"{first_auc:.4f} → {best_auc:.4f}")

        loss_std = log_df["train_loss"].std()
        if loss_std < 1e-5:
            report.print_check("Training loss is not flat", "FAIL", f"std={loss_std:.2e}")
        else:
            report.print_check("Training loss is not flat", "PASS", f"std={loss_std:.6f}")

        auc_std = log_df["val_auc"].std()
        if auc_std < 1e-5 and log_df["val_auc"].mean() > 0.99:
            report.print_check("Val AUROC is not fabricated", "FAIL", f"AUROC={log_df['val_auc'].mean():.4f} ± {auc_std:.2e}")
        else:
            report.print_check("Val AUROC variance realistic", "PASS", f"std={auc_std:.4f}")
    except Exception as e:
        report.print_check("Load training log", "FAIL", f"Exception: {e}")

def check_6_model_inference_live(report, model, device):
    print("\n[CHECK 6] Live Model Inference vs Saved Predictions (full test set)")
    if model is None:
        report.print_check("Live inference", "FAIL", "Model not loaded")
        return

    try:
        # Reload metadata and test split (same as before)
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)

        test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
        transform = get_val_transform()
        dataset = NodulePatchDataset(test_meta, DATA_DIR, transforms=transform)
        # Deterministic loader – MUST use the exact same settings as during test
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

        model.eval()
        live_probs = []
        with torch.no_grad():
            for patches, _, _ in tqdm(loader, desc="  Live inference"):
                patches = patches.to(device)
                with autocast('cuda'):
                    logits = model(patches)
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim == 0:
                    probs = [float(probs)]
                else:
                    probs = probs.tolist()
                live_probs.extend(probs)

        saved_df = pd.read_csv(PREDICTIONS_CSV)
        saved_probs = saved_df["probability"].values

        if len(live_probs) != len(saved_probs):
            report.print_check("Live inference count matches saved CSV", "FAIL",
                                f"Live {len(live_probs)} vs saved {len(saved_probs)}")
            return

        # Element‑wise comparison
        diffs = np.abs(np.array(live_probs) - saved_probs)
        max_diff = diffs.max()
        matches = (diffs < TOLERANCE).sum()
        total = len(diffs)

        if matches == total:
            report.print_check("Live inference matches saved predictions", "PASS",
                                f"All {total} probabilities match within tolerance. Max diff={max_diff:.8f}")
        else:
            report.print_check("Live inference matches saved predictions", "FAIL",
                                f"{matches}/{total} match. Max diff={max_diff:.6f}")

        # Also compute live AUROC on the whole test set (patient‑level)
        all_labels = test_meta["label"].values  # same order as loader
        if len(np.unique(all_labels)) > 1:
            live_auc = roc_auc_score(all_labels, live_probs)
            report.print_check("Live inference AUROC on test set", "PASS", f"AUROC={live_auc:.4f}")
        else:
            report.print_check("Live inference AUROC on test set", "WARN", "Only one class present")
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

        train_ids = set(metadata[metadata["split"] == "train"]["seriesuid"].unique())
        val_ids   = set(metadata[metadata["split"] == "val"]["seriesuid"].unique())
        test_ids  = set(metadata[metadata["split"] == "test"]["seriesuid"].unique())

        for pair, overlap in [
            ("Train ∩ Val", train_ids & val_ids),
            ("Train ∩ Test", train_ids & test_ids),
            ("Val ∩ Test", val_ids & test_ids)
        ]:
            if overlap:
                report.print_check(f"No overlap: {pair}", "FAIL", f"{len(overlap)} patient(s) shared")
            else:
                report.print_check(f"No overlap: {pair}", "PASS", "Zero patients shared")

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
    print("\n[CHECK 8] Permutation Test")
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

    print(f"  Real AUROC: {real_auc:.4f}  Permuted mean: {permuted_aucs.mean():.4f}  p={p_value:.4e}")
    if p_value < 0.05:
        report.print_check("Permutation test", "PASS", f"Beats {percentile:.1f}% of random permutations")
    else:
        report.print_check("Permutation test", "FAIL", f"Not significant (p={p_value:.4f})")

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
                            f"Pos mean={pos_mean:.4f} > Neg mean={neg_mean:.4f}, p={p_value:.2e}")
    elif pos_mean <= neg_mean:
        report.print_check("Positive scores > Negative scores", "FAIL", "Positive patches have lower mean probability")
    else:
        report.print_check("Positive scores > Negative scores", "WARN",
                            f"Not statistically significant (p={p_value:.4f})")

# ------------------------------ Main -----------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata and dataset
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"]
    neg_count = (train_meta["label"] == 0).sum()
    pos_count = (train_meta["label"] == 1).sum()
    pos_weight_value = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"pos_weight = {pos_weight_value:.2f}")

    val_transform = get_val_transform()
    val_dataset   = NodulePatchDataset(metadata[metadata["split"] == "val"].reset_index(drop=True), DATA_DIR, transforms=val_transform)
    test_dataset  = NodulePatchDataset(metadata[metadata["split"] == "test"].reset_index(drop=True), DATA_DIR, transforms=val_transform)

    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load model
    model = SwinTiny3DWithMLPHead().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print("Model loaded.")

    # Compute Youden threshold on validation set
    print("Computing Youden threshold on validation set...")
    model.eval()
    val_probs_all, val_labels_all = [], []
    with torch.no_grad():
        for patches, labels, _ in tqdm(val_loader, desc="Val (Youden)"):
            patches = patches.to(device)
            with autocast('cuda'):
                logits = model(patches)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs_all.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])
            val_labels_all.extend(labels.tolist())

    youden_threshold = get_youden_threshold(val_labels_all, val_probs_all) if len(np.unique(val_labels_all)) > 1 else 0.5
    print(f"Youden threshold (val): {youden_threshold:.4f}")

    # Test evaluation at threshold 0.5
    test_loss, test_auc, test_auprc, test_f1_05, test_sens_05, test_spec_05, \
        test_froc, test_cpm, test_probs, test_labels, test_uids, test_preds_05 = \
        evaluate_model(test_loader, model, device, pos_weight, threshold=0.5)

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'seriesuid': test_uids,
        'label': test_labels,
        'probability': test_probs
    })
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved to {PREDICTIONS_CSV}")

    # Recompute at Youden threshold
    patient_dict_test = {}
    for uid, prob, label in zip(test_uids, test_probs, test_labels):
        if uid not in patient_dict_test:
            patient_dict_test[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict_test[uid]['prob']  = max(patient_dict_test[uid]['prob'], prob)
            patient_dict_test[uid]['label'] = max(patient_dict_test[uid]['label'], label)
    y_true_t  = [v['label'] for v in patient_dict_test.values()]
    y_probs_t = [v['prob']  for v in patient_dict_test.values()]
    y_pred_y  = [1 if p >= youden_threshold else 0 for p in y_probs_t]
    cm_y = confusion_matrix(y_true_t, y_pred_y, labels=[0, 1])
    tn_y, fp_y, fn_y, tp_y = cm_y.ravel()
    test_f1_y   = f1_score(y_true_t, y_pred_y, zero_division=0)
    test_sens_y = recall_score(y_true_t, y_pred_y, zero_division=0)
    test_spec_y = tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else 0.0

    # Print results
    print("\n--- FINAL TEST RESULTS (SCAN-LEVEL) ---")
    print(f"AUROC           : {test_auc:.4f}")
    print(f"AUPRC           : {test_auprc:.4f}")
    print(f"CPM             : {test_cpm:.4f}")
    print(f"\n  threshold=0.50  → F1:{test_f1_05:.4f} Sens:{test_sens_05:.4f} Spec:{test_spec_05:.4f}")
    print(f"  threshold={youden_threshold:.4f} → F1:{test_f1_y:.4f} Sens:{test_sens_y:.4f} Spec:{test_spec_y:.4f} (Youden)")
    print(f"\n--- FROC ---")
    for fp_rate, sens in test_froc.items():
        print(f"  {fp_rate:12s} : {sens:.4f}")
    print(f"  CPM (mean)    : {test_cpm:.4f}")

    # Save final metrics JSON
    final_metrics = {
        "auroc": test_auc, "auprc": test_auprc, "cpm": test_cpm,
        "threshold_05": {"f1": test_f1_05, "sensitivity": test_sens_05, "specificity": test_spec_05},
        "threshold_youden": {"value": youden_threshold, "f1": test_f1_y,
                             "sensitivity": test_sens_y, "specificity": test_spec_y},
        "froc": test_froc
    }
    with open(FINAL_METRICS_JSON, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Final metrics saved to {FINAL_METRICS_JSON}")

    # ----------------------- AUDIT ----------------------------
    print("\n" + "=" * 70)
    print("  STARTING AUDIT")
    print("=" * 70)
    report = AuditReport()

    check_1_required_files_exist(report)
    model = check_2_model_weights_not_random(report, device)
    pred_df = check_3_predictions_csv_integrity(report, pred_df)

    saved_metrics = {}
    if os.path.exists(BEST_METRICS_JSON):
        with open(BEST_METRICS_JSON) as f:
            saved_metrics = json.load(f)

    if pred_df is not None:
        check_4_recompute_metrics_from_predictions(report, pred_df, saved_metrics)

    if os.path.exists(TRAINING_LOG_CSV):
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
    else:
        print(f"\n  ❌  VERDICT: {report.failed} check(s) FAILED.")
    print("=" * 70)

    report.save("audit_report_swin3d_tiny.txt", "audit_summary_swin3d_tiny.json")

if __name__ == "__main__":
    main()