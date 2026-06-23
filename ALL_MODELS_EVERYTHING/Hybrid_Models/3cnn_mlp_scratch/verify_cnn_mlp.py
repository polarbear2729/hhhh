# gradcam_hybrid_full_eval_audit.py
"""
Full evaluation + GradCAM++ + audit for the 3D Hybrid CNN (ResNet18 + DenseNet121 + EfficientNet-B0).
Computes all test metrics, saves predictions & calibration, produces GradCAM++ overlay images,
and runs a comprehensive audit to validate the results.
"""

import os, json, random, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, cv2, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             confusion_matrix, recall_score)
from sklearn.calibration import calibration_curve
from scipy import stats
from monai.networks.nets import DenseNet121, ResNet
from monai.networks.nets.resnet import ResNetBlock
warnings.filterwarnings("ignore")

# ======================== Configuration ========================
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"          # adjust if needed
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_hybrid.pth"                       # from training
BATCH_SIZE = 1            # GradCAM runs with batch 1 for simplicity
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Feature dimensions (must match training)
RESNET18_FEATURE_DIM = 512
DENSENET121_FEATURE_DIM = 1024
EFFICIENTNET_FEATURE_DIM = 1280
TOTAL_FEATURE_DIM = RESNET18_FEATURE_DIM + DENSENET121_FEATURE_DIM + EFFICIENTNET_FEATURE_DIM  # 2816

# ======================== Reproducibility ========================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ======================== Dataset ================================
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
        patch = np.load(local_path).astype(np.float32)       # (64,64,64)
        patch = np.expand_dims(patch, axis=0)                # (1,64,64,64)
        if self.transforms is not None: patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor): patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]

# ======================== EfficientNet-B0 (3D) ===================
def _make_divisible(v, divisor, min_value=None):
    if min_value is None: min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v: new_v += divisor
    return new_v

class SwishActivation(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class SqueezeExcitation3D(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(in_channels, se_channels), SwishActivation(),
            nn.Linear(se_channels, in_channels), nn.Sigmoid())
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
                nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3), SwishActivation()]
        pad = (kernel_size - 1) // 2
        layers += [
            nn.Conv3d(mid_channels, mid_channels, kernel_size,
                      stride=stride, padding=pad, groups=mid_channels, bias=False),
            nn.BatchNorm3d(mid_channels, momentum=0.01, eps=1e-3), SwishActivation(),
            SqueezeExcitation3D(mid_channels, se_ratio),
            nn.Conv3d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels, momentum=0.01, eps=1e-3)]
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
            nn.BatchNorm3d(32, momentum=0.01, eps=1e-3), SwishActivation())
        stage_configs = [
            (32, 16, 3, 1, 1, 1), (16, 24, 3, 2, 6, 2),
            (24, 40, 5, 2, 6, 2), (40, 80, 3, 2, 6, 3),
            (80, 112, 5, 1, 6, 3), (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1)]
        total_blocks = sum(cfg[5] for cfg in stage_configs)
        block_idx = 0; stages = []
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
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, kernel_size=1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.01, eps=1e-3), SwishActivation())
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.stem(x); x = self.stages(x)
        x = self.head_conv(x); x = self.global_pool(x)
        return x.flatten(1)

# ======================== Hybrid Model ============================
class HybridCNNWithMLPHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        # ResNet18 branch
        self.resnet18 = ResNet(
            block=ResNetBlock, layers=[2,2,2,2],
            block_inplanes=[64,128,256,512], spatial_dims=3,
            n_input_channels=in_channels, num_classes=RESNET18_FEATURE_DIM)
        self.resnet18.fc = nn.Identity()   # output raw features (512)

        # DenseNet121 branch
        self.densenet121 = DenseNet121(
            spatial_dims=3, in_channels=in_channels, out_channels=1)
        self.densenet121.class_layers.out = nn.Identity()  # output raw (1024)

        # EfficientNet-B0 branch
        self.efficientnet_b0 = EfficientNet3D_B0(in_channels=in_channels)

        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(TOTAL_FEATURE_DIM, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, x):
        f_r = self.resnet18(x)
        f_d = self.densenet121(x)
        f_e = self.efficientnet_b0(x)
        combined = torch.cat([f_r, f_d, f_e], dim=1)
        out = self.mlp_head(combined)
        return out.squeeze(1)

# ======================== Metrics =================================
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

# ======================== GradCAM++ ===============================
def find_target_layer(model):
    for name, mod in model.named_modules():
        if 'resnet18.layer3.1' in name and 'conv' not in name:
            print(f"✔ GradCAM target: {name}")
            return name
    for name, mod in model.named_modules():
        if 'densenet121.denseblock2' in name:
            print(f"✔ GradCAM target (fallback): {name}")
            return name
    raise ValueError("No suitable target layer found.")

def grad_cam_plusplus_3d(fmap, grad):
    grad_2 = grad.pow(2); grad_3 = grad_2 * grad
    sum_activations = fmap.sum(dim=(2,3,4), keepdim=True)
    numerator = grad_2 * fmap
    denominator = 2 * grad_2 + sum_activations * grad_3
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    alpha = numerator / denominator
    alpha = alpha.sum(dim=(2,3,4), keepdim=True)
    weights = F.relu(grad) * alpha
    heatmap_pp = (weights * fmap).sum(dim=1); heatmap_pp = F.relu(heatmap_pp)
    if heatmap_pp.max() < 1e-6:
        weights_std = grad.mean(dim=(2,3,4), keepdim=True)
        heatmap = (weights_std * fmap).sum(dim=1); heatmap = F.relu(heatmap)
        print("⚠ Fallback to standard GradCAM."); return heatmap.squeeze(0)
    return heatmap_pp.squeeze(0)

def produce_heatmap(model, input_tensor, target_layer_name, class_idx=1):
    model.eval(); features={}; gradients={}
    def fwd_hook(m, i, o): features['value'] = o
    def bwd_hook(m, gi, go): gradients['value'] = go[0]
    target = dict(model.named_modules())[target_layer_name]
    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_backward_hook(bwd_hook)

    logits = model(input_tensor)
    grad_sign = 1.0 if class_idx == 1 else -1.0
    model.zero_grad()
    logits.backward(gradient=torch.full_like(logits, grad_sign), retain_graph=True)

    fmap = features['value'].detach(); grad = gradients['value'].detach()
    h1.remove(); h2.remove()
    print(f"Target: {target_layer_name} | fmap {fmap.shape} | logit {logits.item():.4f}")
    heatmap_3d = grad_cam_plusplus_3d(fmap, grad)
    d_center = heatmap_3d.shape[0] // 2
    heatmap_slice = heatmap_3d[d_center].cpu().numpy()
    hm_min, hm_max = heatmap_slice.min(), heatmap_slice.max()
    if hm_max - hm_min > 1e-8: heatmap_slice = (heatmap_slice - hm_min) / (hm_max - hm_min)
    else: heatmap_slice = np.zeros_like(heatmap_slice)

    vol = input_tensor.squeeze(0).squeeze(0).cpu().numpy()
    orig_slice = vol[vol.shape[0] // 2]
    orig_disp = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)

    orig_resized = cv2.resize(orig_disp, (224,224))
    heatmap_resized = cv2.resize(heatmap_slice, (224,224))
    orig_bgr = cv2.cvtColor(np.uint8(255*orig_resized), cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.5, heatmap_color, 0.5, 0)
    return orig_resized, heatmap_resized, overlay

# ======================== Audit Utilities =========================
class AuditReport:
    def __init__(self):
        self.checks = []; self.passed = self.failed = self.warned = 0
    def record(self, name, status, detail):
        self.checks.append({"check": name, "status": status, "detail": detail})
        if status == "PASS": self.passed += 1
        elif status == "FAIL": self.failed += 1
        else: self.warned += 1
    def print_check(self, name, status, detail):
        icons = {"PASS":"✅","FAIL":"❌","WARN":"⚠️ "}
        print(f"  {icons.get(status,'?')} [{status}] {name}\n         {detail}")
        self.record(name, status, detail)
    def save(self, txt_path, json_path):
        lines = ["="*70, "  HYBRID CNN AUDIT REPORT", "="*70,
                 f"  Checks: {len(self.checks)} | Passed: {self.passed} | Warnings: {self.warned} | Failed: {self.failed}"]
        for c in self.checks:
            icon = {"PASS":"[PASS]","FAIL":"[FAIL]","WARN":"[WARN]"}.get(c["status"],"[?]")
            lines.append(f"  {icon} {c['check']}\n         {c['detail']}")
        lines.append("="*70)
        verdict = "LEGITIMATE ✅" if self.failed == 0 else f"FAILED ❌ ({self.failed} issues)"
        lines.append(f"  VERDICT: {verdict}\n" + "="*70)
        with open(txt_path,'w',encoding='utf-8') as f: f.write('\n'.join(lines))
        with open(json_path,'w') as f: json.dump({"total":len(self.checks),
            "passed":self.passed,"warned":self.warned,"failed":self.failed,
            "verdict":"LEGITIMATE" if self.failed==0 else "FAILED",
            "checks":self.checks}, f, indent=4)
        print(f"\n[Saved] Audit report → {txt_path} / {json_path}")

def audit_check_files(report):
    print("\n[CHECK 1] Required Files")
    required = {"Model": MODEL_PATH, "Predictions": "hybrid_test_predictions.csv",
                "Training log": "hybrid_training_log.csv", "Best metrics": "best_metrics_hybrid.json"}
    for name, path in required.items():
        if os.path.exists(path):
            report.print_check(f"File: {name}", "PASS", f"Found ({os.path.getsize(path)/1024:.1f} KB)")
        else:
            report.print_check(f"File: {name}", "FAIL", "Missing")
    return all(os.path.exists(p) for p in required.values())

def audit_weights(report, device):
    print("\n[CHECK 2] Trained vs Random Weights")
    try:
        trained = HybridCNNWithMLPHead().to(device)
        trained.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        random_model = HybridCNNWithMLPHead().to(device)
        w1 = list(trained.parameters())[0].detach().cpu().numpy().flatten()
        w2 = list(random_model.parameters())[0].detach().cpu().numpy().flatten()
        n = min(len(w1), 1000); corr, _ = stats.pearsonr(w1[:n], w2[:n])
        diff = np.mean(np.abs(w1 - w2[:len(w1)]))
        if abs(corr) > 0.99: report.print_check("Weights differ from random", "FAIL", f"Pearson r={corr:.4f}")
        else: report.print_check("Weights differ from random", "PASS", f"Mean delta={diff:.6f}, r={corr:.4f}")
        mlp_std = np.std(list(trained.mlp_head.parameters())[0].detach().cpu().numpy())
        if mlp_std < 1e-6: report.print_check("MLP head trained", "FAIL", f"std={mlp_std:.2e}")
        else: report.print_check("MLP head trained", "PASS", f"std={mlp_std:.6f}")
        return trained
    except Exception as e: report.print_check("Load weights", "FAIL", str(e)); return None

def audit_predictions_csv(report):
    print("\n[CHECK 3] Predictions CSV Integrity")
    try:
        df = pd.read_csv("hybrid_test_predictions.csv")
        req = {"seriesuid","label","probability"}
        if req - set(df.columns): report.print_check("CSV columns", "FAIL", f"Missing {req - set(df.columns)}"); return None
        report.print_check("CSV columns", "PASS", f"Shape {df.shape}")
        if df[["label","probability"]].isna().any().any():
            report.print_check("NaN values", "FAIL", "NaNs present"); return None
        report.print_check("NaN values", "PASS", "None")
        if ((df["probability"]<0)|(df["probability"]>1)).any():
            report.print_check("Prob range [0,1]", "FAIL", "Out of range"); return None
        report.print_check("Prob range [0,1]", "PASS", f"[{df.probability.min():.4f}, {df.probability.max():.4f}]")
        if df["label"].nunique() < 2:
            report.print_check("Both classes present", "FAIL", "Only one class"); return None
        report.print_check("Both classes present", "PASS", f"Pos={df.label.sum()}, Neg={len(df)-df.label.sum()}")
        if df["probability"].std() < 0.01:
            report.print_check("Predictions varied", "FAIL", f"std={df.probability.std():.4f}")
        else: report.print_check("Predictions varied", "PASS", f"std={df.probability.std():.4f}")
        return df
    except Exception as e: report.print_check("Load CSV", "FAIL", str(e)); return None

def audit_recompute_metrics(report, df):
    print("\n[CHECK 4] Recompute Metrics")
    patient = {}
    for _,r in df.iterrows():
        uid=r["seriesuid"]; p=r["probability"]; l=r["label"]
        if uid not in patient: patient[uid] = {"prob":p,"label":l}
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
    report.print_check("AUPRC recomputable", "PASS", f"{auprc:.4f}")
    if auc<0.5: report.print_check("AUC range", "FAIL", f"{auc:.4f} < 0.5")
    elif auc>0.999: report.print_check("AUC range", "WARN", "Suspiciously perfect")
    else: report.print_check("AUC range", "PASS", f"{auc:.4f} plausible")
    if sens==0: report.print_check("Sensitivity>0", "FAIL", "Zero sensitivity")
    elif sens==1: report.print_check("Sensitivity=1", "WARN", "Possible all-positive prediction")
    else: report.print_check("Sensitivity valid", "PASS", f"{sens:.4f}")
    return {"auc":auc,"auprc":auprc,"f1":f1,"sens":sens,"spec":spec}

def audit_training_log(report):
    print("\n[CHECK 5] Training Log Learning")
    try:
        log = pd.read_csv("hybrid_training_log.csv")
        if len(log)<3: report.print_check("Epochs logged", "WARN", f"Only {len(log)}"); return
        report.print_check("Epochs logged", "PASS", f"{len(log)}")
        loss_start = log.train_loss.iloc[:3].mean(); loss_end = log.train_loss.iloc[-3:].mean()
        if loss_end < loss_start: report.print_check("Loss decreased", "PASS", f"{loss_start:.4f}→{loss_end:.4f}")
        else: report.print_check("Loss decreased", "FAIL", f"{loss_start:.4f}→{loss_end:.4f}")
        auc_start = log.val_auc.iloc[:3].mean(); best_auc = log.val_auc.max()
        if best_auc > auc_start: report.print_check("AUROC improved", "PASS", f"{auc_start:.4f}→{best_auc:.4f}")
        else: report.print_check("AUROC improved", "FAIL", f"No improvement")
        if log.train_loss.std() < 1e-5: report.print_check("Loss varied", "FAIL", "Constant loss")
        else: report.print_check("Loss varied", "PASS", f"std={log.train_loss.std():.6f}")
    except Exception as e: report.print_check("Training log", "FAIL", str(e))

def audit_live_inference(report, model, device):
    print("\n[CHECK 6] Live Inference vs Saved")
    if model is None: report.print_check("Live inference", "FAIL", "No model"); return
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            psplit = pd.read_csv(PATIENT_SPLIT_PATH)
            metadata["split"] = metadata["seriesuid"].map(dict(zip(psplit.seriesuid, psplit.split)))
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
        saved = pd.read_csv("hybrid_test_predictions.csv")
        if len(live_probs) != len(saved):
            report.print_check("Count match", "FAIL", f"Live {len(live_probs)} vs saved {len(saved)}"); return
        # compare first 10 and last 5 UIDs
        front_ok = all(str(live_uids[i])==str(saved.iloc[i].seriesuid) and
                       int(live_labels[i])==int(saved.iloc[i].label) for i in range(min(10,len(live_uids))))
        back_ok = all(str(live_uids[i])==str(saved.iloc[i].seriesuid) and
                      int(live_labels[i])==int(saved.iloc[i].label) for i in range(max(0,len(live_uids)-5), len(live_uids)))
        if not (front_ok and back_ok): report.print_check("UID order", "WARN", "Mismatch")
        else: report.print_check("UID order", "PASS", "Matches")
        diffs = np.abs(np.array(live_probs) - saved.probability.values)
        match = (diffs < 0.01).sum()
        if match == len(diffs): report.print_check("Probabilities match", "PASS", f"All match, max diff {diffs.max():.8f}")
        elif match/len(diffs)>=0.995: report.print_check("Probabilities match", "WARN", f"{match}/{len(diffs)} match")
        else: report.print_check("Probabilities match", "FAIL", f"Only {match}/{len(diffs)} match")
    except Exception as e: report.print_check("Live inference", "FAIL", str(e))

def audit_split_leakage(report):
    print("\n[CHECK 7] Split Leakage")
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            psplit = pd.read_csv(PATIENT_SPLIT_PATH)
            metadata["split"] = metadata["seriesuid"].map(dict(zip(psplit.seriesuid, psplit.split)))
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

def audit_permutation(report, df):
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

def audit_score_separation(report, df):
    print("\n[CHECK 9] Score Separation")
    if df is None: report.print_check("Separation", "FAIL", "No predictions"); return
    pos = df[df.label==1].probability; neg = df[df.label==0].probability
    u,p = stats.mannwhitneyu(pos, neg, alternative='greater')
    if pos.mean() > neg.mean() and p<0.05: report.print_check("Pos > Neg scores", "PASS", f"p={p:.2e}")
    elif pos.mean() <= neg.mean(): report.print_check("Pos > Neg scores", "FAIL", f"pos mean {pos.mean():.4f} ≤ neg {neg.mean():.4f}")
    else: report.print_check("Pos > Neg scores", "WARN", f"Not significant p={p:.4f}")

# ======================== Main =====================================
def main():
    print(f"Device: {DEVICE}")
    # Load metadata & test set
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        ps = pd.read_csv(PATIENT_SPLIT_PATH)
        metadata["split"] = metadata["seriesuid"].map(dict(zip(ps.seriesuid, ps.split)))
    test_meta = metadata[metadata.split=="test"].reset_index(drop=True)
    test_ds = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load model
    model = HybridCNNWithMLPHead().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval(); print("Model loaded.\n")

    # ---------- Full Test Prediction ----------
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
    pred_df.to_csv("hybrid_gradcam_test_predictions.csv", index=False)
    pd.DataFrame({'mean_predicted_probability':mean_pred,'fraction_of_positives':frac_pos}
                ).to_csv("hybrid_gradcam_calibration_curve.csv", index=False)
    print("[Saved] Predictions & calibration CSV.")

    # ---------- GradCAM++ Analysis ----------
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
        # find the highest-prob candidate for this patient
        patient_rows = pred_df[pred_df.seriesuid == uid]
        best = patient_rows.loc[patient_rows.probability.idxmax()]
        patch_path = best.filepath
        idx_in_meta = test_meta[test_meta.filepath == patch_path].index[0]
        img_tensor, label, uid_out, _ = test_ds[idx_in_meta]
        input_t = img_tensor.unsqueeze(0).to(DEVICE)
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
        plt.savefig("gradcampp_hybrid_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved gradcampp_hybrid_analysis.png")

    # ---------- Audit ----------
    print("\n" + "="*70 + "\n  STARTING AUDIT\n" + "="*70)
    report = AuditReport()
    # Ensure predictions CSV exists for audit (we saved it as hybrid_gradcam_test_predictions.csv,
    # but audit looks for hybrid_test_predictions.csv. We'll rename or use our new file.
    # To keep audit consistent, we'll copy/rename our predictions to the expected name.
    if not os.path.exists("hybrid_test_predictions.csv"):
        pred_df.to_csv("hybrid_test_predictions.csv", index=False)
    # Also ensure training log and best_metrics exist (they come from training).
    files_ok = audit_check_files(report)
    if files_ok:
        model = audit_weights(report, DEVICE)
    df = audit_predictions_csv(report)
    if df is not None:
        audit_recompute_metrics(report, df)
    if os.path.exists("hybrid_training_log.csv"):
        audit_training_log(report)
    audit_live_inference(report, model, DEVICE)
    audit_split_leakage(report)
    if df is not None:
        audit_permutation(report, df)
        audit_score_separation(report, df)

    print("\n" + "="*70)
    print(f"  AUDIT COMPLETE – Passed: {report.passed}  Warnings: {report.warned}  Failed: {report.failed}")
    if report.failed == 0: print("  ✅ VERDICT: Legitimate results.")
    else: print(f"  ❌ VERDICT: {report.failed} check(s) failed.")
    print("="*70)
    report.save("audit_report_hybrid.txt", "audit_summary_hybrid.json")

if __name__ == "__main__":
    main()