# gradcam_hybrid2_full_eval.py
"""
Full evaluation + GradCAM++ for 2-Branch Hybrid (ResNet18 + DenseNet121) with KAN head.
Computes all test metrics, saves predictions, calibration curve,
and generates GradCAM++ analysis image.
Target layer: ResNet18 branch's layer3.1 (same as standalone ResNet18).
Requires efficient_kan (pip install efficient_kan).
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
from monai.transforms import Compose
from efficient_kan import KAN

warnings.filterwarnings("ignore")

# ----------------------------- Configuration -----------------------------
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_hybrid2.pth"          # from training script
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

RESNET18_FEATURE_DIM = 512
DENSENET121_FEATURE_DIM = 1024
TOTAL_FEATURE_DIM = RESNET18_FEATURE_DIM + DENSENET121_FEATURE_DIM   # 1536

# ----------------------------- Reproducibility -----------------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ----------------------------- Dataset (same as training) -----------------
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
        split = row["split"]
        label = int(row["label"])
        subfolder = "pos" if label == 1 else "neg"
        local_path = os.path.join(self.data_dir, split, subfolder, filename)
        patch = np.load(local_path).astype(np.float32)   # (64,64,64)
        patch = np.expand_dims(patch, axis=0)            # (1,64,64,64)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]

# ----------------------------- Model (exact replica from training) ---------
class HybridTwoBranch(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        # Branch 1: ResNet18 (512-d)
        self.resnet18 = ResNet(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=RESNET18_FEATURE_DIM,
        )
        self.resnet18.fc = nn.Identity()

        # Branch 2: DenseNet121 (1024-d)
        self.densenet121 = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
        )
        self.densenet121.class_layers.out = nn.Identity()

        # KAN head: 1536 -> 512 -> 256 -> 1
        self.mlp_head = KAN([TOTAL_FEATURE_DIM, 512, 256, num_classes])

    def forward(self, x):
        f_r = self.resnet18(x)       # (B, 512)
        f_d = self.densenet121(x)    # (B, 1024)
        combined = torch.cat([f_r, f_d], dim=1)   # (B, 1536)
        out = self.mlp_head(combined).squeeze(1)
        return out

# ----------------------------- Metrics -----------------------------------
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

# ----------------------------- GradCAM utilities ---------------------------
def find_target_layer(model):
    """Target layer in ResNet18 branch: layer3.1 (second BasicBlock of layer3)."""
    for name, mod in model.named_modules():
        if name == 'resnet18.layer3.1':
            print(f"✔ Chosen target layer: {name}")
            return name
    raise ValueError("Target layer 'resnet18.layer3.1' not found in model.")

def grad_cam_plusplus_3d(fmap, grad):
    grad_2 = grad.pow(2)
    grad_3 = grad_2 * grad
    sum_activations = fmap.sum(dim=(2, 3, 4), keepdim=True)
    numerator = grad_2 * fmap
    denominator = 2 * grad_2 + sum_activations * grad_3
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    alpha = numerator / denominator
    alpha = alpha.sum(dim=(2, 3, 4), keepdim=True)
    weights = F.relu(grad) * alpha
    heatmap_pp = (weights * fmap).sum(dim=1)
    heatmap_pp = F.relu(heatmap_pp)
    if heatmap_pp.max() < 1e-6:
        weights_std = grad.mean(dim=(2, 3, 4), keepdim=True)
        heatmap = (weights_std * fmap).sum(dim=1)
        heatmap = F.relu(heatmap)
        print("⚠ GradCAM++ gave zero heatmap; fell back to standard GradCAM.")
    else:
        heatmap = heatmap_pp
    return heatmap.squeeze(0)

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

    logits = model(input_tensor)   # single logit
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

    vol = input_tensor.squeeze(0).squeeze(0).cpu().numpy()   # (64,64,64)
    orig_slice = vol[vol.shape[0] // 2]
    orig_disp = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)

    orig_resized = cv2.resize(orig_disp, (224, 224))
    heatmap_resized = cv2.resize(heatmap_slice, (224, 224))
    orig_bgr = cv2.cvtColor(np.uint8(255 * orig_resized), cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig_bgr, 0.5, heatmap_color, 0.5, 0)

    return orig_resized, heatmap_resized, superimposed

# ----------------------------- Main ---------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    # Load metadata & test split
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        ps = pd.read_csv(PATIENT_SPLIT_PATH)
        metadata["split"] = metadata["seriesuid"].map(dict(zip(ps.seriesuid, ps.split)))

    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)

    # No transforms at test time (identity)
    test_dataset = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    # Load model
    model = HybridTwoBranch().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.\n")

    # --------------------------- Full Test Prediction ---------------------------
    all_probs, all_labels, all_uids, all_paths = [], [], [], []
    with torch.no_grad():
        for patches, labels, uids, fpaths in tqdm(test_loader, desc="Predicting"):
            patches = patches.to(DEVICE)
            logits = model(patches)
            prob = torch.sigmoid(logits).cpu().item()
            all_probs.append(prob)
            all_labels.append(labels.item())
            all_uids.append(uids[0])
            all_paths.append(fpaths[0])

    # Patient-level aggregation
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
        y_true_patient, y_probs_patient, n_bins=10
    )

    # Print metrics
    print("\n" + "="*50)
    print("FINAL TEST METRICS (SCAN-LEVEL)")
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
    print("====================================================\n")

    # Save predictions & calibration
    pred_df = pd.DataFrame({
        'seriesuid': all_uids,
        'label': all_labels,
        'probability': all_probs,
        'filepath': all_paths
    })
    pred_df.to_csv("hybrid2_gradcam_test_predictions.csv", index=False)
    print("[Saved] Test predictions exported to 'hybrid2_gradcam_test_predictions.csv'")

    cal_df = pd.DataFrame({
        'mean_predicted_probability': mean_predicted_value,
        'fraction_of_positives': fraction_of_positives
    })
    cal_df.to_csv("hybrid2_gradcam_calibration_curve.csv", index=False)
    print("[Saved] Calibration curve exported to 'hybrid2_gradcam_calibration_curve.csv'")

    # --------------------------- GradCAM++ Analysis ---------------------------
    target_layer_name = find_target_layer(model)

    patient_pred = pd.DataFrame({
        'seriesuid': list(patient_dict.keys()),
        'label': y_true_patient,
        'prob': y_probs_patient,
        'pred': y_pred_patient
    })

    tp = patient_pred[(patient_pred['label'] == 1) & (patient_pred['pred'] == 1)]
    fn = patient_pred[(patient_pred['label'] == 1) & (patient_pred['pred'] == 0)]
    fp = patient_pred[(patient_pred['label'] == 0) & (patient_pred['pred'] == 1)]
    tn = patient_pred[(patient_pred['label'] == 0) & (patient_pred['pred'] == 0)]
    print(f"\nGradCAM++ case counts – TP: {len(tp)}, FN: {len(fn)}, FP: {len(fp)}, TN: {len(tn)}")

    selected_cases = []
    for i in range(min(4, len(tp))): selected_cases.append(('TP', i+1, tp.iloc[i]))
    for i in range(min(3, len(fn))): selected_cases.append(('FN', i+1, fn.iloc[i]))
    for i in range(min(3, len(fp))): selected_cases.append(('FP', i+1, fp.iloc[i]))
    for i in range(min(3, len(tn))): selected_cases.append(('TN', i+1, tn.iloc[i]))

    df_pred = pred_df
    heatmaps_data, titles = [], []
    for case_type, case_num, row in selected_cases:
        uid = row['seriesuid']
        # Best candidate for this patient
        patient_df = df_pred[df_pred['seriesuid'] == uid]
        best_candidate = patient_df.loc[patient_df['probability'].idxmax()]
        patch_path = best_candidate['filepath']
        idx_in_meta = test_meta[test_meta['filepath'] == patch_path].index[0]
        patch, label, uid_out, _ = test_dataset[idx_in_meta]

        input_t = patch.unsqueeze(0).to(DEVICE)   # (1,1,64,64,64)
        cidx = 1 if row['label'] == 1 else 0
        print(f"\n{'='*60}")
        print(f"Case: {case_type}-{case_num} | UID: {uid[:12]}... | Prob: {row['prob']:.4f} | True label: {int(row['label'])}")
        print(f"{'='*60}")
        orig_slice, hm_slice, overlay = produce_heatmap(
            model, input_t, target_layer_name, class_idx=cidx
        )
        heatmaps_data.append((orig_slice, hm_slice, overlay))
        titles.append(f"{case_type}-{case_num} | UID:{uid[:8]}... | Prob:{row['prob']:.3f} | Label:{int(row['label'])}")

    n = len(heatmaps_data)
    if n == 0:
        print("No cases to display.")
        return

    fig_height = max(12, 4 * n)
    fig, axes = plt.subplots(n, 3, figsize=(12, fig_height))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (orig, hm, ov) in enumerate(heatmaps_data):
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(titles[i], fontsize=9, fontweight='bold', pad=6)

        axes[i, 1].imshow(hm, cmap='jet')
        axes[i, 1].set_title("GradCAM++ heatmap", fontsize=9)
        axes[i, 1].axis('off')

        ov_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)
        axes[i, 2].imshow(ov_rgb)
        axes[i, 2].set_title("Overlay", fontsize=9)
        axes[i, 2].axis('off')

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig("gradcampp_hybrid2_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: gradcampp_hybrid2_analysis.png")

if __name__ == "__main__":
    main()