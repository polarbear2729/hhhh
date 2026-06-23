# gradcam_hybrid3_full_eval.py
"""
Full evaluation + GradCAM++ for 3-Branch Hybrid (ResNet18 + DenseNet121 + EfficientNet-B0) with KAN head (scratch).
Computes all test metrics, saves predictions, calibration curve,
and generates GradCAM++ analysis image.
Target layer: ResNet18 branch's layer3.1 (same as standalone ResNet18).
Requires efficient_kan (pip install efficient-kan@git+https://github.com/Blealtan/efficient-kan.git).
"""

import os, json, random, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, cv2, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             confusion_matrix, recall_score)
from sklearn.calibration import calibration_curve
from monai.networks.nets import ResNet, DenseNet121
from monai.networks.nets.resnet import ResNetBlock
from efficient_kan import KAN

warnings.filterwarnings("ignore")

# ----------------------------- Configuration -----------------------------
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "latest_checkpoint.pth"
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

RESNET18_FEATURE_DIM = 512
DENSENET121_FEATURE_DIM = 1024
EFFICIENTNET_FEATURE_DIM = 1280
TOTAL_FEATURE_DIM = RESNET18_FEATURE_DIM + DENSENET121_FEATURE_DIM + EFFICIENTNET_FEATURE_DIM

# ----------------------------- Reproducibility -----------------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ----------------------------- EfficientNet-B0 3D -----------------------------
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
            nn.Linear(se_ch, in_channels), nn.Sigmoid(),
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
            nn.BatchNorm3d(mid_ch, momentum=0.01, eps=1e-3), SwishActivation(),
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
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=0.01, eps=1e-3), SwishActivation(),
        )
        stage_configs = [
            (32, 16, 3, 1, 1, 1), (16, 24, 3, 2, 6, 2),
            (24, 40, 5, 2, 6, 2), (40, 80, 3, 2, 6, 3),
            (80, 112, 5, 1, 6, 3), (112, 192, 5, 2, 6, 4),
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
            nn.Conv3d(320, EFFICIENTNET_FEATURE_DIM, kernel_size=1, bias=False),
            nn.BatchNorm3d(EFFICIENTNET_FEATURE_DIM, momentum=0.01, eps=1e-3), SwishActivation(),
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
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x); x = self.stages(x); x = self.head_conv(x)
        x = self.global_pool(x); return x.flatten(1)

# ----------------------------- Dataset -----------------------------
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
        split = row["split"]; label = int(row["label"])
        subfolder = "pos" if label == 1 else "neg"
        local_path = os.path.join(self.data_dir, split, subfolder, filename)
        patch = np.load(local_path).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)
        if self.transforms is not None: patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor): patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]

# ----------------------------- Model (3-Branch + KAN Head) -----------------------------
class HybridThreeBranchKAN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.resnet18 = ResNet(block=ResNetBlock, layers=[2,2,2,2],
            block_inplanes=[64,128,256,512], spatial_dims=3,
            n_input_channels=in_channels, num_classes=RESNET18_FEATURE_DIM)
        self.resnet18.fc = nn.Identity()

        self.densenet121 = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1)
        self.densenet121.class_layers.out = nn.Identity()

        self.efficientnet_b0 = EfficientNet3D_B0(in_channels=in_channels)

        self.mlp_head = KAN([TOTAL_FEATURE_DIM, 512, 256, num_classes])

    def forward(self, x):
        f_r = self.resnet18(x)
        f_d = self.densenet121(x)
        f_e = self.efficientnet_b0(x)
        combined = torch.cat([f_r, f_d, f_e], dim=1)
        out = self.mlp_head(combined).squeeze(1)
        return out

# ----------------------------- Metrics -----------------------------
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

# ----------------------------- GradCAM utilities -----------------------------
def find_target_layer(model):
    for name, mod in model.named_modules():
        if name == 'resnet18.layer3.1':
            print(f"Target layer: {name}")
            return name
    raise ValueError("Target layer 'resnet18.layer3.1' not found.")

def grad_cam_plusplus_3d(fmap, grad):
    grad_2 = grad.pow(2); grad_3 = grad_2 * grad
    sum_activations = fmap.sum(dim=(2,3,4), keepdim=True)
    numerator = grad_2 * fmap
    denominator = 2 * grad_2 + sum_activations * grad_3
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    alpha = numerator / denominator
    alpha = alpha.sum(dim=(2,3,4), keepdim=True)
    weights = F.relu(grad) * alpha
    heatmap_pp = (weights * fmap).sum(dim=1)
    heatmap_pp = F.relu(heatmap_pp)
    if heatmap_pp.max() < 1e-6:
        weights_std = grad.mean(dim=(2,3,4), keepdim=True)
        heatmap = (weights_std * fmap).sum(dim=1); heatmap = F.relu(heatmap)
        print("GradCAM++ gave zero; fell back to standard GradCAM.")
    else:
        heatmap = heatmap_pp
    return heatmap.squeeze(0)

def produce_heatmap(model, input_tensor, target_layer_name, class_idx=1):
    model.eval()
    features, gradients = {}, {}
    def forward_hook(module, input, output): features['value'] = output
    def backward_hook(module, grad_input, grad_output): gradients['value'] = grad_output[0]

    target_layer = dict(model.named_modules())[target_layer_name]
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    logits = model(input_tensor)
    gradient_sign = 1.0 if class_idx == 1 else -1.0
    model.zero_grad()
    logits.backward(gradient=torch.full_like(logits, gradient_sign), retain_graph=True)

    fmap = features['value'].detach(); grad = gradients['value'].detach()
    fwd_handle.remove(); bwd_handle.remove()

    print(f"Target layer: {target_layer_name} | fmap shape: {fmap.shape}")
    print(f"Logit: {logits.item():.4f} -> prob {torch.sigmoid(logits).item():.4f}")

    heatmap_3d = grad_cam_plusplus_3d(fmap, grad)
    d_center = heatmap_3d.shape[0] // 2
    heatmap_slice = heatmap_3d[d_center].cpu().numpy()
    hm_min, hm_max = heatmap_slice.min(), heatmap_slice.max()
    if hm_max - hm_min > 1e-8:
        heatmap_slice = (heatmap_slice - hm_min) / (hm_max - hm_min)
    else:
        heatmap_slice = np.zeros_like(heatmap_slice)

    vol = input_tensor.squeeze(0).squeeze(0).cpu().numpy()
    orig_slice = vol[vol.shape[0] // 2]
    orig_disp = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)

    orig_resized = cv2.resize(orig_disp, (224, 224))
    heatmap_resized = cv2.resize(heatmap_slice, (224, 224))
    orig_bgr = cv2.cvtColor(np.uint8(255 * orig_resized), cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig_bgr, 0.5, heatmap_color, 0.5, 0)
    return orig_resized, heatmap_resized, superimposed

# ----------------------------- Main -----------------------------
def main():
    print(f"Using device: {DEVICE}")
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        ps = pd.read_csv(PATIENT_SPLIT_PATH)
        metadata["split"] = metadata["seriesuid"].map(dict(zip(ps.seriesuid, ps.split)))
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)

    test_dataset = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    model = HybridThreeBranchKAN().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded epoch {checkpoint.get('epoch','?')} (best_val_auc={checkpoint.get('best_val_auc','?'):.4f})")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.\n")

    # --------------------------- Full Test Prediction ---------------------------
    all_probs, all_labels, all_uids, all_paths = [], [], [], []
    with torch.no_grad():
        for patches, labels, uids, fpaths in tqdm(test_loader, desc="Predicting"):
            patches = patches.to(DEVICE)
            logits = model(patches)
            prob = torch.sigmoid(logits).cpu().item()
            all_probs.append(prob); all_labels.append(labels.item())
            all_uids.append(uids[0]); all_paths.append(fpaths[0])

    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {'prob': prob, 'label': label}
        else:
            patient_dict[uid]['prob'] = max(patient_dict[uid]['prob'], prob)
            patient_dict[uid]['label'] = max(patient_dict[uid]['label'], label)

    y_true_patient = [v['label'] for v in patient_dict.values()]
    y_probs_patient = [v['prob'] for v in patient_dict.values()]
    y_pred_patient = [1 if p >= 0.5 else 0 for p in y_probs_patient]
    total_scans = len(patient_dict)

    # ------------------------------- Metrics ------------------------------------
    auc = roc_auc_score(y_true_patient, y_probs_patient) if len(set(y_true_patient)) > 1 else 0.5
    auprc = average_precision_score(y_true_patient, y_probs_patient)
    f1 = f1_score(y_true_patient, y_pred_patient)
    sens = recall_score(y_true_patient, y_pred_patient)
    cm = confusion_matrix(y_true_patient, y_pred_patient, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    froc = calculate_candidate_froc(all_labels, all_probs, total_scans)
    ci_lower, ci_upper = calculate_95_ci(y_true_patient, y_probs_patient)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_patient, y_probs_patient, n_bins=10)

    print("\n" + "="*50)
    print("FINAL TEST METRICS (SCAN-LEVEL) -- 3CNN KAN SCRATCH")
    print("="*50)
    print(f"AUROC (Primary):           {auc:.4f}  (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
    print(f"AUPRC:                     {auprc:.4f}")
    print(f"F1 Score:                  {f1:.4f}")
    print(f"Sensitivity (Recall):      {sens:.4f}")
    print(f"Specificity:               {spec:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TP: {tp:4d}   FN: {fn:4d}")
    print(f"  FP: {fp:4d}   TN: {tn:4d}")
    print(f"\nFROC (False Positives per Scan vs Sensitivity):")
    for fp_rate, sensitivity in froc.items():
        print(f"  {fp_rate:10s} : {sensitivity:.4f}")
    print("="*50 + "\n")

    pred_df = pd.DataFrame({
        'seriesuid': all_uids, 'label': all_labels,
        'probability': all_probs, 'filepath': all_paths
    })
    pred_df.to_csv("3cnn_kan_scratch_test_predictions.csv", index=False)
    print("[Saved] 3cnn_kan_scratch_test_predictions.csv")

    cal_df = pd.DataFrame({
        'mean_predicted_probability': mean_predicted_value,
        'fraction_of_positives': fraction_of_positives
    })
    cal_df.to_csv("3cnn_kan_scratch_calibration_curve.csv", index=False)
    print("[Saved] 3cnn_kan_scratch_calibration_curve.csv")

    # --------------------------- GradCAM++ Analysis ---------------------------
    target_layer_name = find_target_layer(model)

    patient_pred = pd.DataFrame({
        'seriesuid': list(patient_dict.keys()),
        'label': y_true_patient, 'prob': y_probs_patient, 'pred': y_pred_patient
    })

    tp_df = patient_pred[(patient_pred['label']==1) & (patient_pred['pred']==1)]
    fn_df = patient_pred[(patient_pred['label']==1) & (patient_pred['pred']==0)]
    fp_df = patient_pred[(patient_pred['label']==0) & (patient_pred['pred']==1)]
    tn_df = patient_pred[(patient_pred['label']==0) & (patient_pred['pred']==0)]
    print(f"\nGradCAM++ case counts -- TP: {len(tp_df)}, FN: {len(fn_df)}, FP: {len(fp_df)}, TN: {len(tn_df)}")

    selected_cases = []
    for i in range(min(4, len(tp_df))): selected_cases.append(('TP', i+1, tp_df.iloc[i]))
    for i in range(min(3, len(fn_df))): selected_cases.append(('FN', i+1, fn_df.iloc[i]))
    for i in range(min(3, len(fp_df))): selected_cases.append(('FP', i+1, fp_df.iloc[i]))
    for i in range(min(3, len(tn_df))): selected_cases.append(('TN', i+1, tn_df.iloc[i]))

    df_pred = pred_df
    heatmaps_data, titles = [], []
    for case_type, case_num, row in selected_cases:
        uid = row['seriesuid']
        patient_df_sub = df_pred[df_pred['seriesuid'] == uid]
        best_candidate = patient_df_sub.loc[patient_df_sub['probability'].idxmax()]
        patch_path = best_candidate['filepath']
        idx_in_meta = test_meta[test_meta['filepath'] == patch_path].index[0]
        patch, label, uid_out, _ = test_dataset[idx_in_meta]

        input_t = patch.unsqueeze(0).to(DEVICE)
        cidx = 1 if row['label'] == 1 else 0
        print(f"\n{'='*60}")
        print(f"Case: {case_type}-{case_num} | UID: {uid[:12]}... | Prob: {row['prob']:.4f} | True label: {int(row['label'])}")
        print(f"{'='*60}")
        orig_slice, hm_slice, overlay = produce_heatmap(
            model, input_t, target_layer_name, class_idx=cidx)
        heatmaps_data.append((orig_slice, hm_slice, overlay))
        titles.append(f"{case_type}-{case_num} | UID:{uid[:8]}... | Prob:{row['prob']:.3f} | Label:{int(row['label'])}")

    n = len(heatmaps_data)
    if n == 0:
        print("No cases to display.")
        return

    fig_height = max(12, 4 * n)
    fig, axes = plt.subplots(n, 3, figsize=(12, fig_height))
    if n == 1: axes = axes.reshape(1, -1)

    for i, (orig, hm, ov) in enumerate(heatmaps_data):
        axes[i, 0].imshow(orig, cmap='gray'); axes[i, 0].axis('off')
        axes[i, 0].set_title(titles[i], fontsize=9, fontweight='bold', pad=6)
        axes[i, 1].imshow(hm, cmap='jet'); axes[i, 1].set_title("GradCAM++ heatmap", fontsize=9); axes[i, 1].axis('off')
        ov_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)
        axes[i, 2].imshow(ov_rgb); axes[i, 2].set_title("Overlay", fontsize=9); axes[i, 2].axis('off')

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig("gradcampp_3cnn_kan_scratch.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: gradcampp_3cnn_kan_scratch.png")

if __name__ == "__main__":
    main()
