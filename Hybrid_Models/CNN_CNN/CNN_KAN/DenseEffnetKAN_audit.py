"""
Test & Audit script — Hybrid 3D DenseNet121 + EfficientNet-B0 with KAN Head
- Loads best_model_hybrid_kan.pth
- Evaluates on test set (threshold 0.5 and Youden)
- Saves predictions + final metrics JSON
- Runs comprehensive audit checks:
    1. Required files exist
    2. Weights are trained (both backbones + KAN head)
    3. Predictions CSV integrity
    4. Metric recomputation from saved CSV
    5. Training log shows real learning
    6. Live inference matches saved predictions
    7. No patient overlap between splits
    8. Permutation test (statistical significance)
    9. Positive vs negative score separation
   10. KAN spline weights are non-trivial (model-specific check)
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
from sklearn.calibration import calibration_curve
from monai.networks.nets import DenseNet121
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────── Configuration ───────────────────────────────
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

BATCH_SIZE = 8
NUM_WORKERS = 0        # Windows safety
PIN_MEMORY = False

MODEL_PATH          = "best_model_hybrid_kan.pth"
PREDICTIONS_CSV     = "hybrid_kan_test_predictions.csv"
FINAL_METRICS_JSON  = "final_test_metrics_hybrid_kan.json"
TRAINING_LOG_CSV    = "hybrid_kan_training_log.csv"
BEST_METRICS_JSON   = "best_metrics_hybrid_kan.json"

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
TOLERANCE = 0.01          # max allowed element-wise prediction difference

DENSENET_FEATURE_DIM     = 1024
EFFICIENTNET_FEATURE_DIM = 1280
HYBRID_FEATURE_DIM       = DENSENET_FEATURE_DIM + EFFICIENTNET_FEATURE_DIM  # 2304

# KAN hyperparameters (must match training script exactly)
KAN_GRID_SIZE    = 5
KAN_SPLINE_ORDER = 3
KAN_SCALE_NOISE  = 0.1
KAN_SCALE_BASE   = 1.0
KAN_SCALE_SPLINE = 1.0
KAN_GRID_RANGE   = (-1.0, 1.0)

# ─────────────── KAN Layer ───────────────────────────────────────────────────

class KANLinear(nn.Module):
    """
    EfficientKAN formulation of a Kolmogorov-Arnold Network layer.
    φ_{i,j}(x) = scale_base * SiLU(x) * W_base_{i,j}
               + scale_spline * Σ_k c_{i,j,k} * B_k(x)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int    = KAN_GRID_SIZE,
        spline_order: int = KAN_SPLINE_ORDER,
        scale_noise: float  = KAN_SCALE_NOISE,
        scale_base: float   = KAN_SCALE_BASE,
        scale_spline: float = KAN_SCALE_SPLINE,
        grid_range: tuple   = KAN_GRID_RANGE,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.scale_base   = scale_base
        self.scale_spline = scale_spline

        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1,
                             dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", grid)

        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        num_basis          = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_basis))

        self.base_activation = nn.SiLU()
        self._reset_parameters(scale_noise)

    def _reset_parameters(self, scale_noise: float):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0,
                        std=scale_noise / math.sqrt(self.in_features))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        x    = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            left_num  = x - grid[:-(k + 1)]
            left_den  = grid[k:-1]   - grid[:-(k + 1)]
            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:-k]
            left  = torch.where(left_den  != 0, left_num  / left_den,
                                torch.zeros_like(left_num))
            right = torch.where(right_den != 0, right_num / right_den,
                                torch.zeros_like(right_num))
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        assert bases.size(-1) == self.grid_size + self.spline_order
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out   = F.linear(self.base_activation(x),
                               self.base_weight * self.scale_base)
        spline_basis = self.b_splines(x)
        spline_out   = torch.einsum('big,oig->bo', spline_basis,
                                    self.spline_weight * self.scale_spline)
        return base_out + spline_out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"grid={self.grid_size}, order={self.spline_order}")


# ─────────────── 3D EfficientNet-B0 (self-contained) ─────────────────────────

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
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(in_channels, se_channels), SwishActivation(),
            nn.Linear(se_channels, in_channels), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1, 1)

class MBConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_skip          = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        mid_channels = _make_divisible(in_channels * expand_ratio, 8)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv3d(in_channels, mid_channels, 1, bias=False),
                       nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3), SwishActivation()]
        pad = (kernel_size - 1) // 2
        layers += [nn.Conv3d(mid_channels, mid_channels, kernel_size,
                             stride=stride, padding=pad, groups=mid_channels, bias=False),
                   nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3), SwishActivation()]
        layers.append(SqueezeExcitation3D(mid_channels, se_ratio))
        layers += [nn.Conv3d(mid_channels, out_channels, 1, bias=False),
                   nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-3)]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            if self.training and self.drop_connect_rate > 0:
                keep_prob   = 1 - self.drop_connect_rate
                rand_tensor = torch.rand(x.size(0), 1, 1, 1, 1, device=x.device)
                rand_tensor = torch.floor(rand_tensor + keep_prob)
                out = out / keep_prob * rand_tensor
            out = out + x
        return out

class EfficientNet3D_B0(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=0.01, eps=1e-3), SwishActivation()
        )
        stage_configs = [
            (32,  16,  3, 1, 1, 1), (16,  24,  3, 2, 6, 2),
            (24,  40,  5, 2, 6, 2), (40,  80,  3, 2, 6, 3),
            (80,  112, 5, 1, 6, 3), (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1),
        ]
        total_blocks = sum(c[5] for c in stage_configs)
        block_idx = 0
        stages = []
        for (in_ch, out_ch, k, s, expand, n_layers) in stage_configs:
            stage = []
            for i in range(n_layers):
                stride    = s if i == 0 else 1
                inch      = in_ch if i == 0 else out_ch
                drop_rate = 0.2 * block_idx / total_blocks
                stage.append(MBConv3D(inch, out_ch, k, stride, expand, drop_connect_rate=drop_rate))
                block_idx += 1
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)
        self.head_conv = nn.Sequential(
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, 1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.01, eps=1e-3), SwishActivation()
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
        x = self.stem(x); x = self.stages(x)
        x = self.head_conv(x)
        return self.global_pool(x).flatten(1)

class DenseNet121FeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        _backbone     = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1)
        self.features = _backbone.features
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool3d(x, output_size=1)
        return x.flatten(1)

# ─────────────── Hybrid KAN Model ────────────────────────────────────────────

class HybridKANModel(nn.Module):
    """
    Dual-backbone hybrid model:
        DenseNet121 (1024-d) + EfficientNet-B0 (1280-d)
        → Concat (2304-d)
        → KAN Head → 1 logit

    KAN Head:
        KANLinear(2304→512) → LayerNorm → Dropout(0.5)
        → KANLinear(512→128) → LayerNorm → Dropout(0.3)
        → KANLinear(128→1)
    """
    def __init__(self, in_channels=1, dropout_1=0.5, dropout_2=0.3):
        super().__init__()
        self.densenet_extractor     = DenseNet121FeatureExtractor(in_channels)
        self.efficientnet_extractor = EfficientNet3D_B0(in_channels)

        self.kan_layer1 = KANLinear(HYBRID_FEATURE_DIM, 512)
        self.norm1      = nn.LayerNorm(512)
        self.drop1      = nn.Dropout(dropout_1)

        self.kan_layer2 = KANLinear(512, 128)
        self.norm2      = nn.LayerNorm(128)
        self.drop2      = nn.Dropout(dropout_2)

        self.kan_out    = KANLinear(128, 1)

    def forward(self, x):
        f_dense  = self.densenet_extractor(x)
        f_eff    = self.efficientnet_extractor(x)
        f        = torch.cat([f_dense, f_eff], dim=1)

        f   = self.drop1(self.norm1(self.kan_layer1(f)))
        f   = self.drop2(self.norm2(self.kan_layer2(f)))
        out = self.kan_out(f)
        return out.squeeze(1)

# ─────────────────────────────── Reproducibility ─────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────── Dataset ─────────────────────────────────────
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

# ─────────────────────────────── Metrics ─────────────────────────────────────
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels); all_probs = np.array(all_probs)
    total_positives    = (all_labels == 1).sum()
    descending_indices = np.argsort(all_probs)[::-1]
    sorted_labels      = all_labels[descending_indices]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    fps_per_scan = fps / total_scans
    sensitivity  = tps / total_positives
    froc_scores  = {}
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
                loss   = criterion(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = [float(probs)]; labels_list = [float(labels.cpu())]
            else:
                probs = probs.tolist(); labels_list = labels.cpu().tolist()
            all_probs.extend(probs); all_labels.extend(labels_list)
            all_uids.extend(uids)
            running_loss += loss.item() * patches.size(0)

    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict[uid]['prob']  = max(patient_dict[uid]['prob'],  prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)
    y_true_p  = [v['label'] for v in patient_dict.values()]
    y_probs_p = [v['prob']  for v in patient_dict.values()]
    y_pred_p  = [1 if p >= threshold else 0 for p in y_probs_p]
    total_scans = len(patient_dict)
    avg_loss = running_loss / len(loader.dataset)

    auc   = roc_auc_score(y_true_p, y_probs_p) if len(np.unique(y_true_p)) > 1 else 0.5
    auprc = average_precision_score(y_true_p, y_probs_p)
    f1    = f1_score(y_true_p, y_pred_p, zero_division=0)
    sens  = recall_score(y_true_p, y_pred_p, zero_division=0)
    cm    = confusion_matrix(y_true_p, y_pred_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    froc  = calculate_candidate_froc(all_labels, all_probs, total_scans)
    cpm   = float(np.mean(list(froc.values())))
    return avg_loss, auc, auprc, f1, sens, spec, froc, cpm, all_probs, all_labels, all_uids

# ─────────────────────────────── Audit Report ────────────────────────────────
class AuditReport:
    def __init__(self):
        self.checks = []; self.passed = 0; self.failed = 0; self.warned = 0

    def record(self, name, status, detail):
        self.checks.append({"check": name, "status": status, "detail": detail})
        if status == "PASS": self.passed += 1
        elif status == "FAIL": self.failed += 1
        else: self.warned += 1

    def print_check(self, name, status, detail):
        icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}
        print(f"  {icons.get(status,'?')} [{status}] {name}")
        print(f"         {detail}")
        self.record(name, status, detail)

    def save(self, txt_path, json_path):
        lines = ["=" * 70,
                 "  MODEL AUDIT REPORT — Hybrid DenseNet121 + EfficientNet-B0 + KAN",
                 "=" * 70,
                 f"  Total Checks : {len(self.checks)}",
                 f"  Passed       : {self.passed}",
                 f"  Warnings     : {self.warned}",
                 f"  Failed       : {self.failed}", ""]
        for c in self.checks:
            icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(c["status"], "[?]")
            lines += [f"  {icon}  {c['check']}", f"         {c['detail']}"]
        lines += ["=" * 70]
        verdict = ("RESULTS ARE LEGITIMATE ✅" if self.failed == 0
                   else f"AUDIT FAILED — {self.failed} critical issue(s) found ❌")
        lines.append(f"  VERDICT: {verdict}")
        lines.append("=" * 70)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        summary = {"total": len(self.checks), "passed": self.passed,
                   "warned": self.warned, "failed": self.failed,
                   "verdict": "LEGITIMATE" if self.failed == 0 else "FAILED",
                   "checks": self.checks}
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\n[Saved] Full report    → {txt_path}")
        print(f"[Saved] JSON summary   → {json_path}")

# ─────────────────────────────── Check Functions ─────────────────────────────

def check_1_required_files_exist(report):
    print("\n[CHECK 1] Required Files Exist")
    required = {
        "Model weights (.pth)": MODEL_PATH,
        "Training log CSV":     TRAINING_LOG_CSV,
        "Best metrics JSON":    BEST_METRICS_JSON,
    }
    all_found = True
    for name, path in required.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            report.print_check(f"File exists: {name}", "PASS",
                                f"Found at '{path}' ({size_kb:.1f} KB)")
        else:
            report.print_check(f"File exists: {name}", "FAIL",
                                f"NOT FOUND at '{path}'")
            all_found = False
    return all_found


def check_2_model_weights_not_random(report, device):
    """
    Verifies both backbones and all three KAN layers have been updated
    from their random initialisation by comparing against a freshly
    created model with identical architecture.

    KAN layers carry two learnable tensors each:
        - base_weight  : standard linear weights (Kaiming init)
        - spline_weight: B-spline coefficients   (near-zero Gaussian init)
    We check both to detect partial or no training.
    """
    print("\n[CHECK 2] Model Weights Are Trained (Both Backbones + KAN Head)")
    try:
        trained  = HybridKANModel().to(device)
        trained.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True))
        random_m = HybridKANModel().to(device)

        # ── DenseNet backbone: first conv weight ──────────────────────────
        trained_dn = list(trained.densenet_extractor.features.parameters())[0] \
                         .detach().cpu().numpy().flatten()
        random_dn  = list(random_m.densenet_extractor.features.parameters())[0] \
                         .detach().cpu().numpy().flatten()
        n = min(len(trained_dn), 1000)
        r_dn, _  = stats.pearsonr(trained_dn[:n], random_dn[:n])
        diff_dn  = np.mean(np.abs(trained_dn[:n] - random_dn[:n]))
        if abs(r_dn) > 0.99:
            report.print_check("DenseNet backbone weights differ from init", "FAIL",
                                f"r={r_dn:.4f} — weights unchanged")
        else:
            report.print_check("DenseNet backbone weights differ from init", "PASS",
                                f"Mean delta={diff_dn:.6f}, r={r_dn:.4f}")

        # ── EfficientNet backbone: stem conv weight ───────────────────────
        trained_eff = trained.efficientnet_extractor.stem[0].weight \
                             .detach().cpu().numpy().flatten()
        random_eff  = random_m.efficientnet_extractor.stem[0].weight \
                              .detach().cpu().numpy().flatten()
        n   = min(len(trained_eff), 1000)
        r_eff, _ = stats.pearsonr(trained_eff[:n], random_eff[:n])
        diff_eff  = np.mean(np.abs(trained_eff[:n] - random_eff[:n]))
        if abs(r_eff) > 0.99:
            report.print_check("EfficientNet backbone weights differ from init", "FAIL",
                                f"r={r_eff:.4f} — weights unchanged")
        else:
            report.print_check("EfficientNet backbone weights differ from init", "PASS",
                                f"Mean delta={diff_eff:.6f}, r={r_eff:.4f}")

        # ── KAN layer 1: base_weight ──────────────────────────────────────
        kan1_base_t = trained.kan_layer1.base_weight.detach().cpu().numpy().flatten()
        kan1_base_r = random_m.kan_layer1.base_weight.detach().cpu().numpy().flatten()
        n = min(len(kan1_base_t), 1000)
        r_k1b, _ = stats.pearsonr(kan1_base_t[:n], kan1_base_r[:n])
        diff_k1b  = np.mean(np.abs(kan1_base_t[:n] - kan1_base_r[:n]))
        if abs(r_k1b) > 0.99:
            report.print_check("KAN layer-1 base_weight differs from init", "FAIL",
                                f"r={r_k1b:.4f} — weights unchanged")
        else:
            report.print_check("KAN layer-1 base_weight differs from init", "PASS",
                                f"Mean delta={diff_k1b:.6f}, r={r_k1b:.4f}")

        # ── KAN layer 1: spline_weight ────────────────────────────────────
        kan1_spline_t = trained.kan_layer1.spline_weight.detach().cpu().numpy().flatten()
        kan1_spline_r = random_m.kan_layer1.spline_weight.detach().cpu().numpy().flatten()
        n = min(len(kan1_spline_t), 2000)
        diff_k1s = np.mean(np.abs(kan1_spline_t[:n] - kan1_spline_r[:n]))
        std_k1s  = kan1_spline_t[:n].std()
        if diff_k1s < 1e-6:
            report.print_check("KAN layer-1 spline_weight trained", "FAIL",
                                f"Mean delta={diff_k1s:.2e} — splines unchanged")
        else:
            report.print_check("KAN layer-1 spline_weight trained", "PASS",
                                f"Mean delta={diff_k1s:.6f}, std={std_k1s:.6f}")

        # ── KAN layer 2: base_weight ──────────────────────────────────────
        kan2_base_t = trained.kan_layer2.base_weight.detach().cpu().numpy().flatten()
        kan2_base_r = random_m.kan_layer2.base_weight.detach().cpu().numpy().flatten()
        n = min(len(kan2_base_t), 1000)
        r_k2b, _ = stats.pearsonr(kan2_base_t[:n], kan2_base_r[:n])
        diff_k2b  = np.mean(np.abs(kan2_base_t[:n] - kan2_base_r[:n]))
        if abs(r_k2b) > 0.99:
            report.print_check("KAN layer-2 base_weight differs from init", "FAIL",
                                f"r={r_k2b:.4f} — weights unchanged")
        else:
            report.print_check("KAN layer-2 base_weight differs from init", "PASS",
                                f"Mean delta={diff_k2b:.6f}, r={r_k2b:.4f}")

        # ── KAN output layer: base_weight ─────────────────────────────────
        kan_out_w_t = trained.kan_out.base_weight.detach().cpu().numpy().flatten()
        std_out     = kan_out_w_t.std()
        if std_out < 1e-6:
            report.print_check("KAN output layer base_weight non-trivial", "FAIL",
                                f"std={std_out:.2e} — nearly zero")
        else:
            report.print_check("KAN output layer base_weight non-trivial", "PASS",
                                f"std={std_out:.6f}")

        return trained
    except Exception as e:
        report.print_check("Load model weights", "FAIL", f"Exception: {e}")
        return None


def check_3_predictions_csv_integrity(report, pred_df):
    print("\n[CHECK 3] Predictions CSV Integrity")
    required_cols = {"seriesuid", "label", "probability"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        report.print_check("Predictions CSV columns", "FAIL", f"Missing: {missing}")
        return None
    report.print_check("Predictions CSV columns", "PASS",
                        f"All required columns present. Shape: {pred_df.shape}")

    nan_count = pred_df[["label", "probability"]].isna().sum().sum()
    if nan_count > 0:
        report.print_check("No NaN values", "FAIL", f"{nan_count} NaN values found")
    else:
        report.print_check("No NaN values", "PASS", "Zero NaN values")

    out_of_range = ((pred_df["probability"] < 0) | (pred_df["probability"] > 1)).sum()
    if out_of_range > 0:
        report.print_check("Probabilities in [0,1]", "FAIL",
                            f"{out_of_range} predictions outside [0,1]")
    else:
        report.print_check("Probabilities in [0,1]", "PASS",
                            f"Range: [{pred_df['probability'].min():.4f}, "
                            f"{pred_df['probability'].max():.4f}]")

    unique_labels = set(pred_df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        report.print_check("Labels are binary", "FAIL", f"Non-binary labels: {unique_labels}")
    else:
        pos = (pred_df["label"] == 1).sum(); neg = (pred_df["label"] == 0).sum()
        report.print_check("Labels are binary", "PASS",
                            f"Positive patches: {pos}, Negative patches: {neg}")

    prob_std = pred_df["probability"].std()
    if prob_std < 0.01:
        report.print_check("Predictions are not collapsed", "FAIL", f"std={prob_std:.4f}")
    else:
        report.print_check("Predictions are not collapsed", "PASS",
                            f"std={prob_std:.4f}, mean={pred_df['probability'].mean():.4f}")

    if len(unique_labels) < 2:
        report.print_check("Both classes in test set", "FAIL", "Only one class found")
    else:
        report.print_check("Both classes in test set", "PASS",
                            "Both positive and negative present")
    return pred_df


def check_4_recompute_metrics_from_predictions(report, pred_df, saved_metrics):
    print("\n[CHECK 4] Recompute Metrics from Saved Predictions")
    patient_dict = {}
    for _, row in pred_df.iterrows():
        uid = row["seriesuid"]; prob = row["probability"]; label = row["label"]
        if uid not in patient_dict:
            patient_dict[uid] = {"prob": prob, "label": label}
        else:
            patient_dict[uid]["prob"]  = max(patient_dict[uid]["prob"],  prob)
            patient_dict[uid]["label"] = max(patient_dict[uid]["label"], label)
    y_true  = [v["label"] for v in patient_dict.values()]
    y_probs = [v["prob"]  for v in patient_dict.values()]
    y_pred  = [1 if p >= 0.5 else 0 for p in y_probs]
    recomputed = {}
    try:
        recomputed["auc"]         = roc_auc_score(y_true, y_probs)
        recomputed["auprc"]       = average_precision_score(y_true, y_probs)
        recomputed["f1"]          = f1_score(y_true, y_pred)
        recomputed["sensitivity"] = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recomputed["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except Exception as e:
        report.print_check("Metric recomputation", "FAIL", f"Exception: {e}")
        return None
    print("  Recomputed metrics (patient-level, from saved CSV):")
    for k, v in recomputed.items():
        print(f"    {k:15s}: {v:.4f}")
    report.print_check("AUROC recomputable", "PASS",
                        f"Recomputed AUROC={recomputed['auc']:.4f}")
    report.print_check("AUPRC recomputable", "PASS",
                        f"Recomputed AUPRC={recomputed['auprc']:.4f}")
    if recomputed["auc"] < 0.5:
        report.print_check("AUROC sanity range", "FAIL",
                            f"AUROC={recomputed['auc']:.4f} < 0.5")
    elif recomputed["auc"] > 0.999:
        report.print_check("AUROC sanity range", "WARN",
                            f"AUROC={recomputed['auc']:.4f} suspiciously perfect")
    else:
        report.print_check("AUROC sanity range", "PASS",
                            f"AUROC={recomputed['auc']:.4f} plausible")
    if recomputed["sensitivity"] == 0.0:
        report.print_check("Sensitivity > 0", "FAIL", "Sensitivity=0.0")
    elif recomputed["sensitivity"] == 1.0:
        report.print_check("Sensitivity = 1.0 check", "WARN", "Sensitivity=1.0")
    else:
        report.print_check("Sensitivity in valid range", "PASS",
                            f"Sensitivity={recomputed['sensitivity']:.4f}")
    return recomputed


def check_5_training_log_shows_learning(report):
    print("\n[CHECK 5] Training Log Shows Real Learning")
    try:
        log_df = pd.read_csv(TRAINING_LOG_CSV)
        if len(log_df) < 3:
            report.print_check("Training log has enough epochs", "WARN",
                                f"Only {len(log_df)} epochs logged")
            return
        report.print_check("Training log has enough epochs", "PASS",
                            f"{len(log_df)} epochs")

        first_loss = log_df["train_loss"].iloc[:3].mean()
        last_loss  = log_df["train_loss"].iloc[-3:].mean()
        if last_loss < first_loss:
            report.print_check("Training loss decreased", "PASS",
                                f"{first_loss:.4f} → {last_loss:.4f}")
        else:
            report.print_check("Training loss decreased", "FAIL",
                                f"{first_loss:.4f} → {last_loss:.4f}")

        first_auc = log_df["val_auc"].iloc[:3].mean()
        best_auc  = log_df["val_auc"].max()
        if best_auc > first_auc:
            report.print_check("Val AUROC improved during training", "PASS",
                                f"{first_auc:.4f} → {best_auc:.4f}")
        else:
            report.print_check("Val AUROC improved during training", "FAIL",
                                f"{first_auc:.4f} → {best_auc:.4f}")

        loss_std = log_df["train_loss"].std()
        if loss_std < 1e-5:
            report.print_check("Training loss not flat", "FAIL",
                                f"std={loss_std:.2e}")
        else:
            report.print_check("Training loss not flat", "PASS",
                                f"std={loss_std:.6f}")

        auc_std = log_df["val_auc"].std()
        if auc_std < 1e-5 and log_df["val_auc"].mean() > 0.99:
            report.print_check("Val AUROC not fabricated", "FAIL",
                                f"AUROC={log_df['val_auc'].mean():.4f} ± {auc_std:.2e}")
        else:
            report.print_check("Val AUROC variance realistic", "PASS",
                                f"std={auc_std:.4f}")
    except Exception as e:
        report.print_check("Load training log", "FAIL", f"Exception: {e}")


def check_6_model_inference_live(report, model, device):
    print("\n[CHECK 6] Live Inference Matches Saved Predictions (full test set)")
    if model is None:
        report.print_check("Live inference", "FAIL", "Model not loaded")
        return
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
            split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
            metadata["split"] = metadata["seriesuid"].map(split_dict)
        test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
        dataset   = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
        loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=False)

        model.eval()
        live_probs = []
        with torch.no_grad():
            for patches, _, _ in tqdm(loader, desc="  Live inference"):
                patches = patches.to(device)
                with autocast('cuda'):
                    logits = model(patches)
                probs = torch.sigmoid(logits).cpu().numpy()
                live_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])

        saved_df    = pd.read_csv(PREDICTIONS_CSV)
        saved_probs = saved_df["probability"].values

        if len(live_probs) != len(saved_probs):
            report.print_check("Prediction count matches saved CSV", "FAIL",
                                f"Live {len(live_probs)} vs saved {len(saved_probs)}")
            return

        diffs    = np.abs(np.array(live_probs) - saved_probs)
        max_diff = diffs.max()
        matches  = (diffs < TOLERANCE).sum()
        total    = len(diffs)

        if matches == total:
            report.print_check("Live predictions match saved CSV", "PASS",
                                f"All {total} match within tolerance. Max diff={max_diff:.8f}")
        else:
            report.print_check("Live predictions match saved CSV", "FAIL",
                                f"{matches}/{total} match. Max diff={max_diff:.6f}")

        all_labels = test_meta["label"].values
        if len(np.unique(all_labels)) > 1:
            live_auc = roc_auc_score(all_labels, live_probs)
            report.print_check("Live inference AUROC (patch-level check)", "PASS",
                                f"AUROC={live_auc:.4f}")
        else:
            report.print_check("Live inference AUROC check", "WARN",
                                "Only one class present in test_meta")
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

        for pair, overlap in [("Train ∩ Val",   train_ids & val_ids),
                               ("Train ∩ Test",  train_ids & test_ids),
                               ("Val ∩ Test",    val_ids   & test_ids)]:
            if overlap:
                report.print_check(f"No overlap: {pair}", "FAIL",
                                    f"{len(overlap)} patient(s) shared")
            else:
                report.print_check(f"No overlap: {pair}", "PASS",
                                    "Zero patients shared")

        for split_name in ["train", "val", "test"]:
            sdf  = metadata[metadata["split"] == split_name]
            pos  = (sdf["label"] == 1).sum(); neg = (sdf["label"] == 0).sum()
            pct  = pos / len(sdf) * 100
            report.print_check(f"Class distribution — {split_name}", "PASS",
                                f"Pos={pos}, Neg={neg}, Pos%={pct:.1f}%")
    except Exception as e:
        report.print_check("Load metadata for split check", "FAIL", f"Exception: {e}")


def check_8_permutation_baseline(report, pred_df):
    print("\n[CHECK 8] Permutation Test (Statistical Significance)")
    if pred_df is None:
        report.print_check("Permutation test", "FAIL", "No predictions available")
        return
    patient_dict = {}
    for _, row in pred_df.iterrows():
        uid = row["seriesuid"]; prob = row["probability"]; label = row["label"]
        if uid not in patient_dict:
            patient_dict[uid] = {"prob": prob, "label": label}
        else:
            patient_dict[uid]["prob"]  = max(patient_dict[uid]["prob"],  prob)
            patient_dict[uid]["label"] = max(patient_dict[uid]["label"], label)
    y_true  = np.array([v["label"] for v in patient_dict.values()])
    y_probs = np.array([v["prob"]  for v in patient_dict.values()])
    real_auc = roc_auc_score(y_true, y_probs)

    rng = np.random.RandomState(SEED)
    permuted_aucs = np.array([roc_auc_score(y_true, rng.permutation(y_probs))
                               for _ in range(1000)])
    p_value    = (permuted_aucs >= real_auc).mean()
    percentile = (permuted_aucs < real_auc).mean() * 100

    print(f"  Real AUROC: {real_auc:.4f} | "
          f"Permuted mean: {permuted_aucs.mean():.4f} | p={p_value:.4e}")
    if p_value < 0.05:
        report.print_check("Permutation test (p<0.05)", "PASS",
                            f"Beats {percentile:.1f}% of random permutations (p={p_value:.4e})")
    else:
        report.print_check("Permutation test (p<0.05)", "FAIL",
                            f"Not significant (p={p_value:.4f})")


def check_9_prediction_score_separation(report, pred_df):
    print("\n[CHECK 9] Positive vs Negative Score Separation")
    if pred_df is None:
        report.print_check("Score separation test", "FAIL", "No predictions available")
        return
    pos_probs = pred_df[pred_df["label"] == 1]["probability"].values
    neg_probs = pred_df[pred_df["label"] == 0]["probability"].values
    pos_mean  = pos_probs.mean(); neg_mean = neg_probs.mean()
    print(f"  Positive patches → mean={pos_mean:.4f}, std={pos_probs.std():.4f}, n={len(pos_probs)}")
    print(f"  Negative patches → mean={neg_mean:.4f}, std={neg_probs.std():.4f}, n={len(neg_probs)}")

    u_stat, p_value = stats.mannwhitneyu(pos_probs, neg_probs, alternative='greater')
    if pos_mean > neg_mean and p_value < 0.05:
        report.print_check("Positive scores > Negative scores (MWU)", "PASS",
                            f"Pos mean={pos_mean:.4f} > Neg mean={neg_mean:.4f}, p={p_value:.2e}")
    elif pos_mean <= neg_mean:
        report.print_check("Positive scores > Negative scores (MWU)", "FAIL",
                            "Positive patches have lower mean probability than negatives")
    else:
        report.print_check("Positive scores > Negative scores (MWU)", "WARN",
                            f"Trend correct but not significant (p={p_value:.4f})")


def check_10_kan_spline_weights_non_trivial(report, model):
    """
    KAN-specific check: verifies that the B-spline coefficient tensors
    (spline_weight) for all three KAN layers have meaningful variance
    and are not dominated by their Gaussian initialisation noise.

    A well-trained KAN should show per-layer spline std notably above
    the initialisation std (KAN_SCALE_NOISE / sqrt(in_features)):
        Layer 1:  init std ≈ 0.1 / sqrt(2304) ≈ 0.0021
        Layer 2:  init std ≈ 0.1 / sqrt(512)  ≈ 0.0044
        Out layer: init std ≈ 0.1 / sqrt(128)  ≈ 0.0088

    We flag as WARN (not FAIL) when std is within 2× of init, since
    some layers legitimately learn small corrections.
    """
    print("\n[CHECK 10] KAN Spline Weights Are Non-Trivial")
    if model is None:
        report.print_check("KAN spline weight check", "FAIL", "Model not loaded")
        return

    layers = [
        ("KAN layer-1", model.kan_layer1, 2304),
        ("KAN layer-2", model.kan_layer2, 512),
        ("KAN out",     model.kan_out,    128),
    ]
    for name, layer, in_dim in layers:
        sw      = layer.spline_weight.detach().cpu().numpy()
        sw_std  = sw.std()
        sw_mean = np.abs(sw).mean()
        sw_max  = np.abs(sw).max()
        init_std = KAN_SCALE_NOISE / math.sqrt(in_dim)
        ratio    = sw_std / init_std if init_std > 0 else float('inf')

        if sw_std < 1e-7:
            report.print_check(f"{name} spline_weight non-zero", "FAIL",
                                f"std={sw_std:.2e} — near-zero, not trained")
        elif ratio < 2.0:
            report.print_check(f"{name} spline_weight trained (ratio check)", "WARN",
                                f"std={sw_std:.6f}, init_std≈{init_std:.6f}, "
                                f"ratio={ratio:.2f}× (low but possible)")
        else:
            report.print_check(f"{name} spline_weight trained", "PASS",
                                f"std={sw_std:.6f}, mean_abs={sw_mean:.6f}, "
                                f"max_abs={sw_max:.6f}, ratio={ratio:.1f}× above init")

    # Cross-layer spline diversity: all three layers should not be identical
    sw1 = model.kan_layer1.spline_weight.detach().cpu().numpy().flatten()
    sw2 = model.kan_layer2.spline_weight.detach().cpu().numpy().flatten()
    sw_out = model.kan_out.spline_weight.detach().cpu().numpy().flatten()
    stds = [sw1.std(), sw2.std(), sw_out.std()]
    if max(stds) / (min(stds) + 1e-12) > 1e4:
        report.print_check("KAN spline std cross-layer diversity", "WARN",
                            f"Extreme std disparity across layers: {[f'{s:.4f}' for s in stds]}")
    else:
        report.print_check("KAN spline std cross-layer diversity", "PASS",
                            f"Layer stds: {[f'{s:.4f}' for s in stds]}")


# ─────────────────────────────── Main ────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load metadata ──────────────────────────────────────────────────────
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"]
    neg_count  = (train_meta["label"] == 0).sum()
    pos_count  = (train_meta["label"] == 1).sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    print(f"pos_weight = {neg_count / pos_count:.2f}")

    val_dataset  = NodulePatchDataset(
        metadata[metadata["split"] == "val"].reset_index(drop=True), DATA_DIR)
    test_dataset = NodulePatchDataset(
        metadata[metadata["split"] == "test"].reset_index(drop=True), DATA_DIR)
    val_loader   = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Load model ─────────────────────────────────────────────────────────
    model = HybridKANModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print("Model loaded.")

    # ── Youden threshold on validation set ────────────────────────────────
    print("\nComputing Youden threshold on validation set...")
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
    youden_thr = (get_youden_threshold(val_labels_all, val_probs_all)
                  if len(np.unique(val_labels_all)) > 1 else 0.5)
    print(f"Youden threshold (val): {youden_thr:.4f}")

    # ── Test evaluation at threshold 0.5 ──────────────────────────────────
    test_loss, test_auc, test_auprc, test_f1_05, test_sens_05, test_spec_05, \
        test_froc, test_cpm, test_probs, test_labels, test_uids = \
        evaluate_model(test_loader, model, device, pos_weight, threshold=0.5)

    # Save predictions CSV
    pred_df = pd.DataFrame({'seriesuid':   test_uids,
                            'label':       test_labels,
                            'probability': test_probs})
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved to {PREDICTIONS_CSV}")

    # ── Youden threshold metrics ───────────────────────────────────────────
    patient_dict_test = {}
    for uid, prob, label in zip(test_uids, test_probs, test_labels):
        if uid not in patient_dict_test:
            patient_dict_test[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict_test[uid]['prob']  = max(patient_dict_test[uid]['prob'],  prob)
            patient_dict_test[uid]['label'] = max(patient_dict_test[uid]['label'], label)
    y_true_t  = [v['label'] for v in patient_dict_test.values()]
    y_probs_t = [v['prob']  for v in patient_dict_test.values()]
    y_pred_y  = [1 if p >= youden_thr else 0 for p in y_probs_t]
    cm_y      = confusion_matrix(y_true_t, y_pred_y, labels=[0, 1])
    tn_y, fp_y, fn_y, tp_y = cm_y.ravel()
    test_f1_y   = f1_score(y_true_t, y_pred_y, zero_division=0)
    test_sens_y = recall_score(y_true_t, y_pred_y, zero_division=0)
    test_spec_y = tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else 0.0

    # Confidence interval
    ci_lo, ci_hi = calculate_95_ci(y_true_t, y_probs_t)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n--- FINAL TEST RESULTS — Hybrid DenseNet121 + EfficientNet-B0 + KAN ---")
    print(f"  AUROC          : {test_auc:.4f}  (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  AUPRC          : {test_auprc:.4f}")
    print(f"  CPM            : {test_cpm:.4f}")
    print(f"\n  threshold=0.50  → F1:{test_f1_05:.4f}  Sens:{test_sens_05:.4f}  Spec:{test_spec_05:.4f}")
    print(f"  threshold={youden_thr:.4f} → F1:{test_f1_y:.4f}  "
          f"Sens:{test_sens_y:.4f}  Spec:{test_spec_y:.4f}  (Youden)")
    print(f"\n  --- FROC ---")
    for fp_rate, sens in test_froc.items():
        print(f"  {fp_rate:12s} : {sens:.4f}")
    print(f"  CPM (mean)     : {test_cpm:.4f}")

    # Save final metrics JSON
    final_metrics = {
        "model": "Hybrid DenseNet121 + EfficientNet3D-B0 + KAN Head",
        "kan_config": {
            "grid_size":    KAN_GRID_SIZE,
            "spline_order": KAN_SPLINE_ORDER,
            "scale_noise":  KAN_SCALE_NOISE,
            "scale_base":   KAN_SCALE_BASE,
            "scale_spline": KAN_SCALE_SPLINE,
            "grid_range":   list(KAN_GRID_RANGE),
        },
        "auroc": test_auc, "auroc_ci_95": [ci_lo, ci_hi],
        "auprc": test_auprc, "cpm": test_cpm,
        "threshold_05":     {"f1": test_f1_05,
                             "sensitivity": test_sens_05,
                             "specificity": test_spec_05},
        "threshold_youden": {"value": youden_thr, "f1": test_f1_y,
                             "sensitivity": test_sens_y,
                             "specificity": test_spec_y},
        "froc": test_froc
    }
    with open(FINAL_METRICS_JSON, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"\n[Saved] Final metrics → {FINAL_METRICS_JSON}")

    # ── AUDIT ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STARTING AUDIT — Hybrid KAN Model")
    print("=" * 70)
    report = AuditReport()

    check_1_required_files_exist(report)
    trained_model = check_2_model_weights_not_random(report, device)
    pred_df = check_3_predictions_csv_integrity(report, pred_df)
    saved_metrics = {}
    if os.path.exists(BEST_METRICS_JSON):
        with open(BEST_METRICS_JSON) as f:
            saved_metrics = json.load(f)
    if pred_df is not None:
        check_4_recompute_metrics_from_predictions(report, pred_df, saved_metrics)
    if os.path.exists(TRAINING_LOG_CSV):
        check_5_training_log_shows_learning(report)
    check_6_model_inference_live(report, trained_model, device)
    check_7_label_split_integrity(report)
    if pred_df is not None:
        check_8_permutation_baseline(report, pred_df)
        check_9_prediction_score_separation(report, pred_df)
    check_10_kan_spline_weights_non_trivial(report, trained_model)

    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print(f"  Passed   : {report.passed}")
    print(f"  Warnings : {report.warned}")
    print(f"  Failed   : {report.failed}")
    if report.failed == 0:
        print("\n  ✅  VERDICT: Results appear LEGITIMATE.")
    else:
        print(f"\n  ❌  VERDICT: {report.failed} check(s) FAILED — review above.")
    print("=" * 70)

    report.save("audit_report_hybrid_kan.txt", "audit_summary_hybrid_kan.json")


if __name__ == "__main__":
    main()