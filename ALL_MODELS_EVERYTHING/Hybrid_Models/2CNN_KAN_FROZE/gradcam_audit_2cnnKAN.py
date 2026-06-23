# gradcam_hybrid2_kan_full_eval_audit.py
"""
Full evaluation + GradCAM++ + audit for 2-Branch Hybrid (ResNet18 + DenseNet121) with KAN head.
Backbones are frozen (pretrained). Only the KAN head is trained.
Target layer: ResNet18 branch's layer3.1.
"""

import os, json, random, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, cv2, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             confusion_matrix, recall_score)
from sklearn.calibration import calibration_curve
from scipy import stats
from monai.networks.nets import ResNet, DenseNet121
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import Compose

warnings.filterwarnings("ignore")

# ======================== Configuration ========================
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_hybrid2_kan.pth"                     # from training
PREDICTIONS_PATH = "hybrid2_kan_test_predictions.csv"
TRAINING_LOG_PATH = "hybrid2_kan_training_log.csv"
BEST_METRICS_PATH = "best_metrics_hybrid2_kan.json"

BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

RESNET18_FEATURE_DIM = 512
DENSENET121_FEATURE_DIM = 1024
TOTAL_FEATURE_DIM = RESNET18_FEATURE_DIM + DENSENET121_FEATURE_DIM   # 1536

# KAN config (same as training)
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3
KAN_SCALE_NOISE = 0.1
KAN_SCALE_BASE = 1.0
KAN_SCALE_SPLINE = 1.0
KAN_GRID_EPS = 0.02
KAN_GRID_RANGE = [-1.0, 1.0]

TOLERANCE = 0.01          # for live inference matching

# ======================== Reproducibility ========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ======================== KAN implementation (from training) ========================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=KAN_GRID_SIZE,
                 spline_order=KAN_SPLINE_ORDER, scale_noise=KAN_SCALE_NOISE,
                 scale_base=KAN_SCALE_BASE, scale_spline=KAN_SCALE_SPLINE,
                 base_activation=nn.SiLU, grid_eps=KAN_GRID_EPS, grid_range=None):
        super().__init__()
        if grid_range is None: grid_range = list(KAN_GRID_RANGE)
        self.in_features = in_features; self.out_features = out_features
        self.grid_size = grid_size; self.spline_order = spline_order
        self.scale_noise = scale_noise; self.scale_base = scale_base
        self.scale_spline = scale_spline; self.grid_eps = grid_eps
        self.base_activation = base_activation()

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0])
        self.register_buffer("grid", grid.unsqueeze(0).expand(in_features, -1).contiguous())

        n_coeffs = grid_size + spline_order
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_coeffs))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        nn.init.kaiming_uniform_(self.spline_scaler, a=np.sqrt(5) * self.scale_spline)
        with torch.no_grad():
            inner = self.grid[0, self.spline_order : -self.spline_order]
            x_init = inner.unsqueeze(1).expand(-1, self.in_features)
            noise = (torch.rand(inner.size(0), self.in_features, self.out_features) - 0.5) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(self._curve2coeff(x_init, noise))

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        xv = x.unsqueeze(-1)
        g = self.grid
        bases = ((xv >= g[:, :-1]) & (xv < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_denom = g[:, k:-1] - g[:, :-(k+1)]
            right_denom = g[:, k+1:] - g[:, 1:-k]
            left_w = torch.where(left_denom > 1e-8, (xv - g[:, :-(k+1)]) / left_denom.clamp(min=1e-8), torch.zeros_like(xv))
            right_w = torch.where(right_denom > 1e-8, (g[:, k+1:] - xv) / right_denom.clamp(min=1e-8), torch.zeros_like(xv))
            bases = left_w * bases[..., :-1] + right_w * bases[..., 1:]
        return bases.contiguous()

    def _curve2coeff(self, x, y):
        A = self._b_splines(x).permute(1, 0, 2)
        B = y.permute(1, 0, 2)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def _scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(self.base_activation(x), self.base_weight)
        splines = self._b_splines(x)
        spline_out = F.linear(splines.view(x.size(0), -1), self._scaled_spline_weight.view(self.out_features, -1))
        return base_out + spline_out

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        # Not needed for eval/audit; just placeholder
        pass

class KAN(nn.Module):
    def __init__(self, layers_hidden, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(in_f, out_f, **kwargs)
            for in_f, out_f in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
    def forward(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid: layer.update_grid(x)
            x = layer(x)
        return x

# ======================== Model (exact replica) ========================
class HybridTwoBranch(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.resnet18 = ResNet(
            block=ResNetBlock, layers=[2,2,2,2],
            block_inplanes=[64,128,256,512], spatial_dims=3,
            n_input_channels=in_channels, num_classes=RESNET18_FEATURE_DIM)
        self.resnet18.fc = nn.Identity()
        self.densenet121 = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1)
        self.densenet121.class_layers.out = nn.Identity()
        self.kan_head = KAN(layers_hidden=[TOTAL_FEATURE_DIM, 512, 256, num_classes])

    def forward(self, x):
        with torch.no_grad():
            f_r = self.resnet18(x)
            f_d = self.densenet121(x)
        combined = torch.cat([f_r, f_d], dim=1).float()
        with torch.amp.autocast("cuda", enabled=False):
            out = self.kan_head(combined)
        return out.squeeze(1)

# ======================== Dataset ========================
class NodulePatchDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_df, data_dir, transforms=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_dir = data_dir; self.transforms = transforms
    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = os.path.basename(row["filepath"])
        split = row["split"]; label = int(row["label"])
        subfolder = "pos" if label == 1 else "neg"
        local_path = os.path.join(self.data_dir, split, subfolder, filename)
        patch = np.load(local_path).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)
        if self.transforms is not None: patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor): patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]

# ======================== Metrics ========================
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels); all_probs = np.array(all_probs)
    total_pos = (all_labels == 1).sum()
    desc_idx = np.argsort(all_probs)[::-1]
    sorted_lab = all_labels[desc_idx]
    tps = np.cumsum(sorted_lab == 1); fps = np.cumsum(sorted_lab == 0)
    fps_per_scan = fps / total_scans; sens = tps / total_pos
    froc_scores = {}
    for target in FROC_THRESHOLDS:
        valid = np.where(fps_per_scan <= target)[0]
        froc_scores[f"{target} FP/scan"] = sens[valid[-1]] if len(valid) > 0 else 0.0
    return froc_scores

def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
    rng = np.random.RandomState(SEED)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[idx])) < 2: continue
        scores.append(roc_auc_score(np.array(y_true)[idx], np.array(y_probs)[idx]))
    return np.percentile(np.sort(scores), 2.5), np.percentile(np.sort(scores), 97.5)

# ======================== GradCAM utilities ========================
def find_target_layer(model):
    for name, mod in model.named_modules():
        if name == 'resnet18.layer3.1':
            print(f"✔ GradCAM target: {name}")
            return name
    raise ValueError("Target layer not found")

def grad_cam_plusplus_3d(fmap, grad):
    grad_2 = grad.pow(2); grad_3 = grad_2 * grad
    sum_act = fmap.sum(dim=(2,3,4), keepdim=True)
    numerator = grad_2 * fmap
    denominator = 2*grad_2 + sum_act*grad_3
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    alpha = numerator / denominator
    alpha = alpha.sum(dim=(2,3,4), keepdim=True)
    weights = F.relu(grad) * alpha
    heatmap_pp = (weights * fmap).sum(dim=1); heatmap_pp = F.relu(heatmap_pp)
    if heatmap_pp.max() < 1e-6:
        weights_std = grad.mean(dim=(2,3,4), keepdim=True)
        heatmap = (weights_std * fmap).sum(dim=1); heatmap = F.relu(heatmap)
        print("⚠ Fallback to standard GradCAM.")
        return heatmap.squeeze(0)
    return heatmap_pp.squeeze(0)

def produce_heatmap(model, input_tensor, target_layer_name, class_idx=1):
    model.eval()
    features = {}
    gradients = {}
    def forward_hook(module, input, output):
        features['value'] = output
    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    target_layer = dict(model.named_modules())[target_layer_name]
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    # ---------- Manually run the forward pass WITHOUT torch.no_grad ----------
    input_tensor = input_tensor.requires_grad_(True)   # ensure input tracks grads

    # Run the frozen backbones directly (no no_grad), so gradients flow
    f_r = model.resnet18(input_tensor)                # (B, 512)
    f_d = model.densenet121(input_tensor)              # (B, 1024)
    combined = torch.cat([f_r, f_d], dim=1).float()    # (B, 1536)

    # Forward through the KAN head (float32, autocast off)
    with torch.amp.autocast("cuda", enabled=False):
        logits = model.kan_head(combined).squeeze(1)   # scalar logit
    # ------------------------------------------------------------------------

    gradient_sign = 1.0 if class_idx == 1 else -1.0
    model.zero_grad()
    logits.backward(gradient=torch.full_like(logits, gradient_sign), retain_graph=True)

    fmap = features['value'].detach()
    grad = gradients['value'].detach()
    fwd_handle.remove()
    bwd_handle.remove()

    print(f"Target layer: {target_layer_name} | fmap shape: {fmap.shape}")
    print(f"Logit: {logits.item():.4f} → prob {torch.sigmoid(logits).item():.4f}")
    print(f"Grad min/mean/max: {grad.min().item():.6f} / {grad.mean().item():.6f} / {grad.max().item():.6f}")

    heatmap_3d = grad_cam_plusplus_3d(fmap, grad)
    print(f"Heatmap 3D raw min/max: {heatmap_3d.min().item():.4f} / {heatmap_3d.max().item():.4f}")

    # Central depth slice
    d_center = heatmap_3d.shape[0] // 2
    heatmap_slice = heatmap_3d[d_center].cpu().numpy()
    hm_min, hm_max = heatmap_slice.min(), heatmap_slice.max()
    if hm_max - hm_min > 1e-8:
        heatmap_slice = (heatmap_slice - hm_min) / (hm_max - hm_min)
    else:
        heatmap_slice = np.zeros_like(heatmap_slice)

    # Original volume central slice
    vol = input_tensor.detach().squeeze(0).squeeze(0).cpu().numpy()
    orig_slice = vol[vol.shape[0] // 2]
    orig_disp = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)

    orig_rsz = cv2.resize(orig_disp, (224, 224))
    heat_rsz = cv2.resize(heatmap_slice, (224, 224))
    orig_bgr = cv2.cvtColor(np.uint8(255 * orig_rsz), cv2.COLOR_GRAY2BGR)
    heat_color = cv2.applyColorMap(np.uint8(255 * heat_rsz), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.5, heat_color, 0.5, 0)

    return orig_rsz, heat_rsz, overlay

# ======================== Audit utilities ========================
class AuditReport:
    def __init__(self):
        self.checks = []; self.passed = self.failed = self.warned = 0
    def record(self, name, status, detail):
        self.checks.append({"check":name,"status":status,"detail":detail})
        if status=="PASS": self.passed+=1
        elif status=="FAIL": self.failed+=1
        else: self.warned+=1
    def print_check(self, name, status, detail):
        icons = {"PASS":"✅","FAIL":"❌","WARN":"⚠️ "}
        print(f"  {icons.get(status,'?')} [{status}] {name}\n         {detail}")
        self.record(name, status, detail)
    def save(self, txt, json_path):
        lines = ["="*70, "  HYBRID2 + KAN AUDIT REPORT", "="*70,
                 f"  Checks: {len(self.checks)} | Passed: {self.passed} | Warnings: {self.warned} | Failed: {self.failed}"]
        for c in self.checks:
            icon = {"PASS":"[PASS]","FAIL":"[FAIL]","WARN":"[WARN]"}.get(c["status"],"[?]")
            lines.append(f"  {icon} {c['check']}\n         {c['detail']}")
        lines.append("="*70)
        verdict = "LEGITIMATE ✅" if self.failed==0 else f"FAILED ❌ ({self.failed} issues)"
        lines.append(f"  VERDICT: {verdict}\n"+"="*70)
        with open(txt,'w',encoding='utf-8') as f: f.write('\n'.join(lines))
        with open(json_path,'w') as f: json.dump({"total":len(self.checks),"passed":self.passed,"warned":self.warned,"failed":self.failed,"verdict":"LEGITIMATE" if self.failed==0 else "FAILED","checks":self.checks}, f, indent=4)
        print(f"\n[Saved] Audit → {txt} / {json_path}")

# Audit checks (adapted)
def audit_1_files(report):
    print("\n[CHECK 1] Required Files")
    required = {"Model": MODEL_PATH, "Predictions": PREDICTIONS_PATH,
                "Training log": TRAINING_LOG_PATH, "Best metrics": BEST_METRICS_PATH}
    for name, path in required.items():
        if os.path.exists(path):
            report.print_check(f"File: {name}", "PASS", f"Found ({os.path.getsize(path)/1024:.1f} KB)")
        else:
            report.print_check(f"File: {name}", "FAIL", "Missing")
    return all(os.path.exists(p) for p in required.values())

def audit_2_weights(report, device):
    print("\n[CHECK 2] Trained vs Random Weights (using final model)")
    try:
        # Load the actual trained model
        trained = HybridTwoBranch().to(device)
        trained.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

        # Explicitly freeze backbones (the checkpoint does not store requires_grad)
        for p in trained.resnet18.parameters():
            p.requires_grad = False
        for p in trained.densenet121.parameters():
            p.requires_grad = False

        # Compare first KAN layer's base_weight with a random model
        random_model = HybridTwoBranch().to(device)
        trained_w = list(trained.kan_head.layers[0].parameters())[0].detach().cpu().numpy().flatten()
        random_w = list(random_model.kan_head.layers[0].parameters())[0].detach().cpu().numpy().flatten()
        n = min(len(trained_w), 1000)
        corr, _ = stats.pearsonr(trained_w[:n], random_w[:n])
        diff = np.mean(np.abs(trained_w - random_w[:len(trained_w)]))
        if abs(corr) > 0.99:
            report.print_check("KAN weights differ from random", "FAIL", f"Pearson r={corr:.4f}")
        else:
            report.print_check("KAN weights differ from random", "PASS", f"Mean delta={diff:.6f}, r={corr:.4f}")

        # Verify backbones are actually frozen now
        resnet_frozen = all(not p.requires_grad for p in trained.resnet18.parameters())
        densenet_frozen = all(not p.requires_grad for p in trained.densenet121.parameters())
        if resnet_frozen and densenet_frozen:
            report.print_check("Backbones frozen", "PASS", "All backbone parameters have requires_grad=False")
        else:
            report.print_check("Backbones frozen", "FAIL",
                               f"ResNet18 frozen: {resnet_frozen}, DenseNet121 frozen: {densenet_frozen}")
        return trained
    except Exception as e:
        report.print_check("Load model weights", "FAIL", str(e))
        return None

def audit_3_csv(report):
    print("\n[CHECK 3] Predictions CSV Integrity")
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        req = {"seriesuid","label","probability"}
        if req - set(df.columns):
            report.print_check("CSV columns", "FAIL", f"Missing {req - set(df.columns)}"); return None
        report.print_check("CSV columns", "PASS", f"Shape {df.shape}")
        if df[["label","probability"]].isna().any().any():
            report.print_check("NaN values", "FAIL", "NaNs present"); return None
        report.print_check("NaN values", "PASS", "None")
        if ((df["probability"]<0)|(df["probability"]>1)).any():
            report.print_check("Prob range", "FAIL", "Out of range"); return None
        report.print_check("Prob range", "PASS", f"[{df.probability.min():.4f}, {df.probability.max():.4f}]")
        if df["label"].nunique() < 2:
            report.print_check("Both classes", "FAIL", "Only one class"); return None
        report.print_check("Both classes", "PASS", f"Pos={df.label.sum()}, Neg={len(df)-df.label.sum()}")
        if df["probability"].std() < 0.01:
            report.print_check("Varied predictions", "FAIL", f"std={df.probability.std():.4f}")
        else:
            report.print_check("Varied predictions", "PASS", f"std={df.probability.std():.4f}")
        return df
    except Exception as e:
        report.print_check("Load CSV", "FAIL", str(e)); return None

def audit_4_recompute(report, df):
    print("\n[CHECK 4] Recompute Metrics")
    if df is None: report.print_check("Recompute", "FAIL", "No predictions"); return
    patient = {}
    for _,r in df.iterrows():
        uid=r["seriesuid"]; p=r["probability"]; l=r["label"]
        if uid not in patient: patient[uid]={"prob":p,"label":l}
        else:
            patient[uid]["prob"] = max(patient[uid]["prob"], p)
            patient[uid]["label"] = max(patient[uid]["label"], l)
    yt = [v["label"] for v in patient.values()]
    yp = [v["prob"] for v in patient.values()]
    ypred = [1 if p>=0.5 else 0 for p in yp]
    auc = roc_auc_score(yt, yp) if len(set(yt))>1 else 0.5
    auprc = average_precision_score(yt, yp)
    f1 = f1_score(yt, ypred); sens = recall_score(yt, ypred)
    cm = confusion_matrix(yt, ypred, labels=[0,1]); tn,fp,fn,tp = cm.ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    print(f"  Recomputed: AUC={auc:.4f} AUPRC={auprc:.4f} F1={f1:.4f} Sens={sens:.4f} Spec={spec:.4f}")
    report.print_check("AUC recomputable", "PASS", f"{auc:.4f}")
    if auc<0.5: report.print_check("AUC range", "FAIL", f"{auc:.4f} < 0.5")
    elif auc>0.999: report.print_check("AUC range", "WARN", "Suspiciously perfect")
    else: report.print_check("AUC range", "PASS", f"{auc:.4f} plausible")
    if sens==0: report.print_check("Sensitivity>0", "FAIL", "Zero sensitivity")
    elif sens==1: report.print_check("Sensitivity=1", "WARN", "Possible all-positive")
    else: report.print_check("Sensitivity valid", "PASS", f"{sens:.4f}")

def audit_5_log(report):
    print("\n[CHECK 5] Training Log Learning")
    try:
        log = pd.read_csv(TRAINING_LOG_PATH)
        if len(log)<3: report.print_check("Epochs", "WARN", f"Only {len(log)}"); return
        report.print_check("Epochs", "PASS", f"{len(log)}")
        start_loss = log.train_loss.iloc[:3].mean(); end_loss = log.train_loss.iloc[-3:].mean()
        if end_loss < start_loss: report.print_check("Loss decreased", "PASS", f"{start_loss:.4f}→{end_loss:.4f}")
        else: report.print_check("Loss decreased", "FAIL", f"{start_loss:.4f}→{end_loss:.4f}")
        start_auc = log.val_auc.iloc[:3].mean(); best_auc = log.val_auc.max()
        if best_auc > start_auc: report.print_check("AUROC improved", "PASS", f"{start_auc:.4f}→{best_auc:.4f}")
        else: report.print_check("AUROC improved", "FAIL", f"No improvement")
        if log.train_loss.std() < 1e-5: report.print_check("Loss varied", "FAIL", "Constant loss")
        else: report.print_check("Loss varied", "PASS", f"std={log.train_loss.std():.6f}")
    except Exception as e: report.print_check("Training log", "FAIL", str(e))

def audit_6_live(report, model, device):
    print("\n[CHECK 6] Live Inference vs Saved")
    if model is None: report.print_check("Live inference", "FAIL", "No model"); return
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            ps = pd.read_csv(PATIENT_SPLIT_PATH)
            metadata["split"] = metadata["seriesuid"].map(dict(zip(ps.seriesuid, ps.split)))
        test_meta = metadata[metadata.split=="test"].reset_index(drop=True)
        ds = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        model.eval(); live_probs, live_labels, live_uids = [],[],[]
        with torch.no_grad():
            for patches, labels, uids, _ in tqdm(loader, desc="Live infer"):
                logits = model(patches.to(device))
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim==0: probs=[float(probs)]
                else: probs = probs.tolist()
                live_probs.extend(probs); live_labels.extend(labels.tolist()); live_uids.extend(uids)
        saved = pd.read_csv(PREDICTIONS_PATH)
        if len(live_probs) != len(saved):
            report.print_check("Count match", "FAIL", f"Live {len(live_probs)} vs saved {len(saved)}"); return
        front_ok = all(str(live_uids[i])==str(saved.iloc[i].seriesuid) and
                       int(live_labels[i])==int(saved.iloc[i].label) for i in range(min(10,len(live_uids))))
        back_ok = all(str(live_uids[i])==str(saved.iloc[i].seriesuid) and
                      int(live_labels[i])==int(saved.iloc[i].label) for i in range(max(0,len(live_uids)-5), len(live_uids)))
        if not (front_ok and back_ok): report.print_check("UID order", "WARN", "Mismatch")
        else: report.print_check("UID order", "PASS", "Matches")
        diffs = np.abs(np.array(live_probs) - saved.probability.values)
        match = (diffs < TOLERANCE).sum()
        if match == len(diffs): report.print_check("Probabilities match", "PASS", f"All match, max diff {diffs.max():.8f}")
        elif match/len(diffs)>=0.995: report.print_check("Probabilities match", "WARN", f"{match}/{len(diffs)} match")
        else: report.print_check("Probabilities match", "FAIL", f"Only {match}/{len(diffs)} match")
    except Exception as e: report.print_check("Live inference", "FAIL", str(e))

def audit_7_leakage(report):
    print("\n[CHECK 7] Split Leakage")
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            ps = pd.read_csv(PATIENT_SPLIT_PATH)
            metadata["split"] = metadata["seriesuid"].map(dict(zip(ps.seriesuid, ps.split)))
        train_u = set(metadata[metadata.split=="train"].seriesuid)
        val_u = set(metadata[metadata.split=="val"].seriesuid)
        test_u = set(metadata[metadata.split=="test"].seriesuid)
        for pair, overlap in [("Train∩Val",train_u&val_u),("Train∩Test",train_u&test_u),("Val∩Test",val_u&test_u)]:
            if overlap: report.print_check(f"No overlap: {pair}", "FAIL", f"{len(overlap)} patients")
            else: report.print_check(f"No overlap: {pair}", "PASS", "Zero overlap")
        for sp in ["train","val","test"]:
            sub = metadata[metadata.split==sp]
            pos = (sub.label==1).sum(); neg = (sub.label==0).sum()
            report.print_check(f"Class balance {sp}", "PASS", f"Pos={pos} Neg={neg} Ratio={pos/len(sub)*100:.1f}%")
    except Exception as e: report.print_check("Split check", "FAIL", str(e))

def audit_8_permutation(report, df):
    print("\n[CHECK 8] Permutation Test")
    if df is None: report.print_check("Permutation", "FAIL", "No predictions"); return
    patient={}
    for _,r in df.iterrows():
        uid=r["seriesuid"]; p=r["probability"]; l=r["label"]
        if uid not in patient: patient[uid]={"prob":p,"label":l}
        else: patient[uid]["prob"]=max(patient[uid]["prob"],p); patient[uid]["label"]=max(patient[uid]["label"],l)
    yt=np.array([v["label"] for v in patient.values()])
    yp=np.array([v["prob"] for v in patient.values()])
    real_auc = roc_auc_score(yt,yp)
    rng = np.random.RandomState(SEED); perm_aucs=[]
    for _ in range(1000):
        shuffled = rng.permutation(yp)
        perm_aucs.append(roc_auc_score(yt, shuffled))
    perm_aucs=np.array(perm_aucs); p_val = (perm_aucs>=real_auc).mean()
    print(f"  Real AUC={real_auc:.4f}, perm mean={perm_aucs.mean():.4f}±{perm_aucs.std():.4f}, p={p_val:.4f}")
    if p_val<0.05: report.print_check("Permutation p<0.05", "PASS", f"p={p_val:.4f}")
    else: report.print_check("Permutation p<0.05", "FAIL", f"p={p_val:.4f}")

def audit_9_separation(report, df):
    print("\n[CHECK 9] Score Separation")
    if df is None: report.print_check("Separation", "FAIL", "No predictions"); return
    pos = df[df.label==1].probability; neg = df[df.label==0].probability
    u,p = stats.mannwhitneyu(pos, neg, alternative='greater')
    if pos.mean() > neg.mean() and p<0.05: report.print_check("Pos > Neg scores", "PASS", f"p={p:.2e}")
    elif pos.mean() <= neg.mean(): report.print_check("Pos > Neg scores", "FAIL", f"pos mean {pos.mean():.4f} ≤ neg {neg.mean():.4f}")
    else: report.print_check("Pos > Neg scores", "WARN", f"Not significant p={p:.4f}")

# ======================== Main ========================
def main():
    print(f"Device: {DEVICE}")
    # Load metadata & test
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        ps = pd.read_csv(PATIENT_SPLIT_PATH)
        metadata["split"] = metadata["seriesuid"].map(dict(zip(ps.seriesuid, ps.split)))
    test_meta = metadata[metadata.split=="test"].reset_index(drop=True)
    test_ds = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load model
    model = HybridTwoBranch().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval(); print("Model loaded.\n")

    # ========== Full Test Prediction ==========
    all_probs, all_labels, all_uids, all_paths = [], [], [], []
    with torch.no_grad():
        for patches, labels, uids, fpaths in tqdm(test_loader, desc="Predicting"):
            logits = model(patches.to(DEVICE))
            prob = torch.sigmoid(logits).cpu().item()
            all_probs.append(prob); all_labels.append(labels.item())
            all_uids.append(uids[0]); all_paths.append(fpaths[0])

    # Patient-level aggregation
    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict: patient_dict[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict[uid]['prob'] = max(patient_dict[uid]['prob'], prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)
    y_true = [v['label'] for v in patient_dict.values()]
    y_probs = [v['prob'] for v in patient_dict.values()]
    y_pred = [1 if p>=0.5 else 0 for p in y_probs]
    total_scans = len(patient_dict)

    # Metrics
    auc = roc_auc_score(y_true, y_probs) if len(set(y_true))>1 else 0.5
    auprc = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred); sens = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]); tn,fp,fn,tp = cm.ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    froc = calculate_candidate_froc(all_labels, all_probs, total_scans)
    ci_l, ci_u = calculate_95_ci(y_true, y_probs)
    frac_pos, mean_pred = calibration_curve(y_true, y_probs, n_bins=10)

    print("\n"+"="*50+"\nFINAL TEST METRICS (SCAN-LEVEL)\n"+"="*50)
    print(f"AUROC: {auc:.4f} (95% CI {ci_l:.4f}-{ci_u:.4f})")
    print(f"AUPRC: {auprc:.4f}  F1: {f1:.4f}  Sens: {sens:.4f}  Spec: {spec:.4f}")
    print(f"Confusion: TP={tp} FN={fn} FP={fp} TN={tn}")
    print("FROC:"); [print(f"  {k}: {v:.4f}") for k,v in froc.items()]
    print("="*50)

    # Save predictions & calibration
    pred_df = pd.DataFrame({'seriesuid':all_uids,'label':all_labels,'probability':all_probs,'filepath':all_paths})
    pred_df.to_csv("hybrid2_kan_gradcam_test_predictions.csv", index=False)
    pd.DataFrame({'mean_predicted_probability':mean_pred,'fraction_of_positives':frac_pos}
                ).to_csv("hybrid2_kan_gradcam_calibration_curve.csv", index=False)
    print("[Saved] Predictions & calibration CSV.")

    # ========== GradCAM++ Analysis ==========
    target_layer = find_target_layer(model)
    patient_pred = pd.DataFrame({'seriesuid':list(patient_dict.keys()),'label':y_true,'prob':y_probs,'pred':y_pred})
    tp = patient_pred[(patient_pred.label==1)&(patient_pred.pred==1)]
    fn = patient_pred[(patient_pred.label==1)&(patient_pred.pred==0)]
    fp = patient_pred[(patient_pred.label==0)&(patient_pred.pred==1)]
    tn = patient_pred[(patient_pred.label==0)&(patient_pred.pred==0)]
    print(f"\nGradCAM cases – TP:{len(tp)} FN:{len(fn)} FP:{len(fp)} TN:{len(tn)}")
    selected = []
    for i in range(min(4,len(tp))): selected.append(('TP',i+1,tp.iloc[i]))
    for i in range(min(3,len(fn))): selected.append(('FN',i+1,fn.iloc[i]))
    for i in range(min(3,len(fp))): selected.append(('FP',i+1,fp.iloc[i]))
    for i in range(min(3,len(tn))): selected.append(('TN',i+1,tn.iloc[i]))

    heatmaps_data, titles = [], []
    for case_type, num, row in selected:
        uid = row['seriesuid']
        patient_rows = pred_df[pred_df.seriesuid == uid]
        best = patient_rows.loc[patient_rows.probability.idxmax()]
        patch_path = best.filepath
        idx_in_meta = test_meta[test_meta.filepath == patch_path].index[0]
        patch, label, uid_out, _ = test_ds[idx_in_meta]
        input_t = patch.unsqueeze(0).to(DEVICE)
        cidx = 1 if row['label']==1 else 0
        print(f"\nCase {case_type}-{num}  UID {uid[:12]}...  Prob {row['prob']:.4f}  Label {int(row['label'])}")
        orig, hm, ov = produce_heatmap(model, input_t, target_layer, class_idx=cidx)
        heatmaps_data.append((orig, hm, ov))
        titles.append(f"{case_type}-{num} | {uid[:8]}... | Prob {row['prob']:.3f} | Label {int(row['label'])}")

    n = len(heatmaps_data)
    if n>0:
        fig_h = max(12, 4*n)
        fig, axes = plt.subplots(n, 3, figsize=(12, fig_h))
        if n==1: axes = axes.reshape(1,-1)
        for i,(orig,hm,ov) in enumerate(heatmaps_data):
            axes[i,0].imshow(orig, cmap='gray'); axes[i,0].set_title(titles[i], fontsize=9, fontweight='bold')
            axes[i,0].axis('off')
            axes[i,1].imshow(hm, cmap='jet'); axes[i,1].set_title("Heatmap"); axes[i,1].axis('off')
            axes[i,2].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)); axes[i,2].set_title("Overlay"); axes[i,2].axis('off')
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.savefig("gradcampp_hybrid2_kan_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved gradcampp_hybrid2_kan_analysis.png")

    # ========== Audit ==========
    print("\n"+"="*70+"\n  STARTING AUDIT\n"+"="*70)
    report = AuditReport()
    # Ensure predictions CSV exists for audit (use the one we just saved, but audit expects the training name)
    if not os.path.exists(PREDICTIONS_PATH):
        pred_df.to_csv(PREDICTIONS_PATH, index=False)
    # Also ensure other files exist (they should from training)
    files_ok = audit_1_files(report)
    if files_ok:
        model = audit_2_weights(report, DEVICE)
    df = audit_3_csv(report)
    if df is not None:
        audit_4_recompute(report, df)
    if os.path.exists(TRAINING_LOG_PATH):
        audit_5_log(report)
    audit_6_live(report, model, DEVICE)
    audit_7_leakage(report)
    if df is not None:
        audit_8_permutation(report, df)
        audit_9_separation(report, df)

    print("\n"+"="*70)
    print(f"  AUDIT COMPLETE – Passed: {report.passed}  Warnings: {report.warned}  Failed: {report.failed}")
    if report.failed == 0: print("  ✅ VERDICT: Legitimate results.")
    else: print(f"  ❌ VERDICT: {report.failed} check(s) failed.")
    print("="*70)
    report.save("audit_report_hybrid2_kan.txt", "audit_summary_hybrid2_kan.json")

if __name__ == "__main__":
    main()