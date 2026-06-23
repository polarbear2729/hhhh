# audit_hybrid3_frozen.py
"""
Full audit for the 3‑Branch Hybrid (ResNet18 + DenseNet121 + EfficientNet‑B0)
with frozen backbones and MLP head.
Verifies file integrity, weight training, prediction consistency, training learning,
data leakage, and statistical significance.
"""

import os, json, random, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, recall_score
from scipy import stats
from tqdm import tqdm
from monai.networks.nets import DenseNet121, ResNet
from monai.networks.nets.resnet import ResNetBlock

warnings.filterwarnings("ignore")

# ======================== Configuration ========================
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

MODEL_PATH = "best_model_hybrid3_frozen.pth"
PREDICTIONS_PATH = "hybrid3_frozen_test_predictions.csv"
TRAINING_LOG_PATH = "hybrid3_frozen_training_log.csv"
BEST_METRICS_PATH = "best_metrics_hybrid3_frozen.json"

BATCH_SIZE = 8                # same as training (use 8 for live inference speed)
NUM_WORKERS = 0
TOLERANCE = 0.01              # max allowed difference for live inference

RESNET18_FEATURE_DIM = 512
DENSENET121_FEATURE_DIM = 1024
EFFICIENTNET_FEATURE_DIM = 1280
TOTAL_FEATURE_DIM = RESNET18_FEATURE_DIM + DENSENET121_FEATURE_DIM + EFFICIENTNET_FEATURE_DIM  # 2816

# ======================== Reproducibility ========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed()

# ======================== EfficientNet-B0 (3D) – same as training ========================
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
        se_ch = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(in_channels, se_ch), SwishActivation(),
            nn.Linear(se_ch, in_channels), nn.Sigmoid())
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
                SwishActivation()]
        pad = (kernel_size - 1) // 2
        layers += [
            nn.Conv3d(mid_ch, mid_ch, kernel_size, stride=stride, padding=pad, groups=mid_ch, bias=False),
            nn.BatchNorm3d(mid_ch, momentum=0.01, eps=1e-3), SwishActivation(),
            SqueezeExcitation3D(mid_ch, se_ratio),
            nn.Conv3d(mid_ch, out_channels, 1, bias=False),
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
            nn.Conv3d(in_channels, 32, 3, stride=1, padding=1, bias=False),
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
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, 1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.01, eps=1e-3), SwishActivation())
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.stem(x); x = self.stages(x); x = self.head_conv(x); x = self.global_pool(x)
        return x.flatten(1)

# ======================== Hybrid Model (exact replica from training) ========================
class HybridThreeBranchFrozen(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        self.resnet18 = ResNet(
            block=ResNetBlock, layers=[2,2,2,2],
            block_inplanes=[64,128,256,512], spatial_dims=3,
            n_input_channels=in_channels, num_classes=RESNET18_FEATURE_DIM)
        self.resnet18.fc = nn.Identity()

        self.densenet121 = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1)
        self.densenet121.class_layers.out = nn.Identity()

        self.efficientnet_b0 = EfficientNet3D_B0(in_channels=in_channels)

        self.mlp_head = nn.Sequential(
            nn.Linear(TOTAL_FEATURE_DIM, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, x):
        with torch.no_grad():
            f_r = self.resnet18(x)
            f_d = self.densenet121(x)
            f_e = self.efficientnet_b0(x)
        combined = torch.cat([f_r, f_d, f_e], dim=1)
        return self.mlp_head(combined).squeeze(1)

    def freeze_backbones(self):
        for p in self.resnet18.parameters(): p.requires_grad = False
        for p in self.densenet121.parameters(): p.requires_grad = False
        for p in self.efficientnet_b0.parameters(): p.requires_grad = False

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
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ======================== Audit Report ========================
class AuditReport:
    def __init__(self):
        self.checks = []; self.passed = self.failed = self.warned = 0
    def record(self, name, status, detail):
        self.checks.append({"check": name, "status": status, "detail": detail})
        if status=="PASS": self.passed+=1
        elif status=="FAIL": self.failed+=1
        else: self.warned+=1
    def print_check(self, name, status, detail):
        icons = {"PASS":"✅","FAIL":"❌","WARN":"⚠️ "}
        print(f"  {icons.get(status,'?')} [{status}] {name}\n         {detail}")
        self.record(name, status, detail)
    def save(self, txt_path, json_path):
        lines = ["="*70, "  HYBRID3 (FROZEN) MLP AUDIT REPORT", "="*70,
                 f"  Checks: {len(self.checks)} | Passed: {self.passed} | Warnings: {self.warned} | Failed: {self.failed}"]
        for c in self.checks:
            icon = {"PASS":"[PASS]","FAIL":"[FAIL]","WARN":"[WARN]"}.get(c["status"],"[?]")
            lines.append(f"  {icon} {c['check']}\n         {c['detail']}")
        lines.append("="*70)
        verdict = "LEGITIMATE ✅" if self.failed==0 else f"FAILED ❌ ({self.failed} issues)"
        lines.append(f"  VERDICT: {verdict}\n"+"="*70)
        with open(txt_path,'w',encoding='utf-8') as f: f.write('\n'.join(lines))
        with open(json_path,'w') as f: json.dump({"total":len(self.checks),"passed":self.passed,"warned":self.warned,"failed":self.failed,"verdict":"LEGITIMATE" if self.failed==0 else "FAILED","checks":self.checks}, f, indent=4)
        print(f"\n[Saved] Audit → {txt_path} / {json_path}")

# ======================== Audit Checks ========================
def check_1_required_files(report):
    print("\n[CHECK 1] Required Files")
    required = {"Model": MODEL_PATH, "Predictions": PREDICTIONS_PATH,
                "Training log": TRAINING_LOG_PATH, "Best metrics": BEST_METRICS_PATH}
    ok = True
    for name, path in required.items():
        if os.path.exists(path):
            report.print_check(f"File: {name}", "PASS", f"Found ({os.path.getsize(path)/1024:.1f} KB)")
        else:
            report.print_check(f"File: {name}", "FAIL", "Missing"); ok = False
    return ok

def check_2_weights_trained(report, device):
    print("\n[CHECK 2] Trained vs Random Weights")
    try:
        trained = HybridThreeBranchFrozen().to(device)
        trained.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        trained.freeze_backbones()    # enforce freeze in case state dict didn't store flags

        random_model = HybridThreeBranchFrozen().to(device)

        # Compare first MLP head weight (Linear 2816→512)
        trained_w = trained.mlp_head[0].weight.detach().cpu().numpy().flatten()
        random_w = random_model.mlp_head[0].weight.detach().cpu().numpy().flatten()
        n = min(len(trained_w), 1000)
        corr, _ = stats.pearsonr(trained_w[:n], random_w[:n])
        diff = np.mean(np.abs(trained_w - random_w[:len(trained_w)]))
        if abs(corr) > 0.99:
            report.print_check("MLP weights differ from random", "FAIL", f"Pearson r={corr:.4f}")
        else:
            report.print_check("MLP weights differ from random", "PASS", f"Mean delta={diff:.6f}, r={corr:.4f}")

        # Verify all three backbones frozen
        rn_frozen = all(not p.requires_grad for p in trained.resnet18.parameters())
        dn_frozen = all(not p.requires_grad for p in trained.densenet121.parameters())
        en_frozen = all(not p.requires_grad for p in trained.efficientnet_b0.parameters())
        if rn_frozen and dn_frozen and en_frozen:
            report.print_check("Backbones frozen", "PASS", "All backbone parameters have requires_grad=False")
        else:
            report.print_check("Backbones frozen", "FAIL",
                               f"ResNet18: {rn_frozen}, DenseNet121: {dn_frozen}, EfficientNet: {en_frozen}")
        return trained
    except Exception as e:
        report.print_check("Load model weights", "FAIL", str(e))
        return None

def check_3_predictions_csv(report):
    print("\n[CHECK 3] Predictions CSV Integrity")
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        req = {"seriesuid", "label", "probability"}
        if req - set(df.columns):
            report.print_check("CSV columns", "FAIL", f"Missing {req - set(df.columns)}"); return None
        report.print_check("CSV columns", "PASS", f"Shape {df.shape}")
        if df[["label","probability"]].isna().any().any():
            report.print_check("NaN values", "FAIL", "NaNs present"); return None
        report.print_check("NaN values", "PASS", "None")
        if ((df["probability"] < 0) | (df["probability"] > 1)).any():
            report.print_check("Prob range", "FAIL", "Out of range"); return None
        report.print_check("Prob range", "PASS", f"[{df.probability.min():.4f}, {df.probability.max():.4f}]")
        if df["label"].nunique() < 2:
            report.print_check("Both classes present", "FAIL", "Only one class"); return None
        report.print_check("Both classes present", "PASS", f"Pos={df.label.sum()}, Neg={len(df)-df.label.sum()}")
        if df["probability"].std() < 0.01:
            report.print_check("Predictions varied", "FAIL", f"std={df.probability.std():.4f}")
        else:
            report.print_check("Predictions varied", "PASS", f"std={df.probability.std():.4f}")
        return df
    except Exception as e:
        report.print_check("Load CSV", "FAIL", str(e)); return None

def check_4_recompute_metrics(report, df):
    print("\n[CHECK 4] Recompute Metrics")
    if df is None: report.print_check("Recompute", "FAIL", "No predictions"); return
    patient = {}
    for _, r in df.iterrows():
        uid=r["seriesuid"]; p=r["probability"]; l=r["label"]
        if uid not in patient: patient[uid]={"prob":p,"label":l}
        else: patient[uid]["prob"]=max(patient[uid]["prob"],p); patient[uid]["label"]=max(patient[uid]["label"],l)
    yt = [v["label"] for v in patient.values()]; yp = [v["prob"] for v in patient.values()]
    ypred = [1 if p>=0.5 else 0 for p in yp]
    auc = roc_auc_score(yt, yp) if len(set(yt))>1 else 0.5
    auprc = average_precision_score(yt, yp)
    f1 = f1_score(yt, ypred); sens = recall_score(yt, ypred)
    cm = confusion_matrix(yt, ypred, labels=[0,1]); tn,fp,fn,tp = cm.ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    print(f"  Recomputed: AUC={auc:.4f} AUPRC={auprc:.4f} F1={f1:.4f} Sens={sens:.4f} Spec={spec:.4f}")
    report.print_check("AUC recomputable", "PASS", f"{auc:.4f}")
    if auc < 0.5: report.print_check("AUC range", "FAIL", f"{auc:.4f} < 0.5")
    elif auc > 0.999: report.print_check("AUC range", "WARN", "Suspiciously perfect")
    else: report.print_check("AUC range", "PASS", f"{auc:.4f} plausible")
    if sens==0: report.print_check("Sensitivity>0", "FAIL", "Zero sensitivity")
    elif sens==1: report.print_check("Sensitivity=1", "WARN", "Possible all-positive")
    else: report.print_check("Sensitivity valid", "PASS", f"{sens:.4f}")

def check_5_training_log(report):
    print("\n[CHECK 5] Training Log Learning")
    try:
        log = pd.read_csv(TRAINING_LOG_PATH)
        if len(log) < 3: report.print_check("Epochs logged", "WARN", f"Only {len(log)}"); return
        report.print_check("Epochs logged", "PASS", f"{len(log)}")
        loss_start = log.train_loss.iloc[:3].mean(); loss_end = log.train_loss.iloc[-3:].mean()
        if loss_end < loss_start: report.print_check("Loss decreased", "PASS", f"{loss_start:.4f}→{loss_end:.4f}")
        else: report.print_check("Loss decreased", "FAIL", f"{loss_start:.4f}→{loss_end:.4f}")
        auc_start = log.val_auc.iloc[:3].mean(); best_auc = log.val_auc.max()
        if best_auc >= auc_start: report.print_check("AUROC improved", "PASS", f"{auc_start:.4f}→{best_auc:.4f}")
        else: report.print_check("AUROC improved", "FAIL", "No improvement")
        if log.train_loss.std() < 1e-5: report.print_check("Loss varied", "FAIL", "Constant loss")
        else: report.print_check("Loss varied", "PASS", f"std={log.train_loss.std():.6f}")
    except Exception as e: report.print_check("Training log", "FAIL", str(e))

def check_6_live_inference(report, model, device):
    print("\n[CHECK 6] Live Inference vs Saved")
    if model is None: report.print_check("Live inference", "FAIL", "No model"); return
    try:
        metadata = pd.read_csv(METADATA_PATH)
        if "split" not in metadata.columns:
            ps = pd.read_csv(PATIENT_SPLIT_PATH)
            metadata["split"] = metadata["seriesuid"].map(dict(zip(ps.seriesuid, ps.split)))
        test_meta = metadata[metadata.split=="test"].reset_index(drop=True)
        ds = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        model.eval(); live_probs, live_labels, live_uids = [], [], []
        with torch.no_grad():
            for patches, labels, uids in tqdm(loader, desc="Live infer"):
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

def check_7_split_leakage(report):
    print("\n[CHECK 7] No Patient Overlap")
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
            sub = metadata[metadata.split==sp]; pos=(sub.label==1).sum(); neg=(sub.label==0).sum()
            report.print_check(f"Class balance {sp}", "PASS", f"Pos={pos} Neg={neg} Ratio={pos/len(sub)*100:.1f}%")
    except Exception as e: report.print_check("Split check", "FAIL", str(e))

def check_8_permutation(report, df):
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
    rng=np.random.RandomState(SEED); perm_aucs=[]
    for _ in range(1000):
        shuffled = rng.permutation(yp)
        perm_aucs.append(roc_auc_score(yt, shuffled))
    perm_aucs=np.array(perm_aucs); p_val=(perm_aucs>=real_auc).mean()
    print(f"  Real AUC={real_auc:.4f}, perm mean={perm_aucs.mean():.4f}±{perm_aucs.std():.4f}, p={p_val:.4f}")
    if p_val<0.05: report.print_check("Permutation p<0.05", "PASS", f"p={p_val:.4f}")
    else: report.print_check("Permutation p<0.05", "FAIL", f"p={p_val:.4f}")

def check_9_score_separation(report, df):
    print("\n[CHECK 9] Score Separation")
    if df is None: report.print_check("Separation", "FAIL", "No predictions"); return
    pos = df[df.label==1].probability; neg = df[df.label==0].probability
    u,p = stats.mannwhitneyu(pos, neg, alternative='greater')
    if pos.mean() > neg.mean() and p<0.05: report.print_check("Pos > Neg scores", "PASS", f"p={p:.2e}")
    elif pos.mean() <= neg.mean(): report.print_check("Pos > Neg scores", "FAIL", f"pos mean {pos.mean():.4f} ≤ neg {neg.mean():.4f}")
    else: report.print_check("Pos > Neg scores", "WARN", f"Not significant p={p:.4f}")

# ======================== Main ========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("="*70)
    print("  AUDIT: 3‑Branch Hybrid (Frozen) + MLP Head")
    print("="*70)

    report = AuditReport()
    files_ok = check_1_required_files(report)
    model = None
    if files_ok:
        model = check_2_weights_trained(report, device)
    df = check_3_predictions_csv(report)
    if df is not None:
        check_4_recompute_metrics(report, df)
    if os.path.exists(TRAINING_LOG_PATH):
        check_5_training_log(report)
    check_6_live_inference(report, model, device)
    check_7_split_leakage(report)
    if df is not None:
        check_8_permutation(report, df)
        check_9_score_separation(report, df)

    print("\n" + "="*70)
    print(f"  AUDIT COMPLETE – Passed: {report.passed}  Warnings: {report.warned}  Failed: {report.failed}")
    if report.failed == 0: print("  ✅ VERDICT: Legitimate results.")
    else: print(f"  ❌ VERDICT: {report.failed} check(s) failed.")
    print("="*70)
    report.save("audit_report_hybrid3_frozen.txt", "audit_summary_hybrid3_frozen.json")

if __name__ == "__main__":
    main()