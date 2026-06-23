# gradcam_mednext3d_full_eval.py
"""
Full evaluation + GradCAM++ for 3D MedNeXt (from scratch).
Computes all test metrics, saves predictions, calibration curve,
and generates GradCAM++ analysis image.

Model architecture: MedNeXt3DWithMLPHead (encoder + MLP head).
Target layer for GradCAM++: encoder.stages.2 (128-channel feature map at 4x4x4).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    recall_score,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- Configuration -----------------------------
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_mednext3d.pth"        # output from training script
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# ----------------------------- Dataset ------------------------
class NodulePatchDataset3D(torch.utils.data.Dataset):
    """Loads a 3D nodule patch and returns (img, label, seriesuid, filepath)."""
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
        vol = np.load(local_path).astype(np.float32)   # (64,64,64)
        vol = np.expand_dims(vol, axis=0)              # (1,64,64,64)
        if self.transforms is not None:
            # transforms expect dict with key "img"
            data_dict = {"img": vol}
            data_dict = self.transforms(data_dict)
            vol = data_dict["img"]
        if not isinstance(vol, torch.Tensor):
            vol = torch.from_numpy(vol)
        return vol, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]

# ----------------------------- MedNeXt Model (excerpt from training script) -----------------
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
        x = x.permute(0, 2, 3, 4, 1)          # (B,D,H,W,C)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_contract(x)
        x = x.permute(0, 4, 1, 2, 3)          # (B,C,D,H,W)
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

# ----------------------------- Metrics -------------------------
def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
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
        sens = sensitivity[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
        froc_scores[f"{target} FP/scan"] = sens
    return froc_scores

def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
    rng = np.random.RandomState(SEED)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        score = roc_auc_score(np.array(y_true)[indices], np.array(y_probs)[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.sort(bootstrapped_scores)
    return np.percentile(sorted_scores, 2.5), np.percentile(sorted_scores, 97.5)

# ----------------------------- GradCAM utilities ---------------------------
def find_target_layer(model):
    """
    MedNeXt target: the third encoder stage (128 channels, spatial 4x4x4).
    Named 'encoder.stages.2' in the model.
    """
    for name, mod in model.named_modules():
        if name == "encoder.stages.2":
            print(f"✔ Chosen target layer: {name}")
            return name
    raise ValueError("Target layer 'encoder.stages.2' not found in model.")

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

    logits = model(input_tensor)
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

    # Extract central slice from heatmap (depth = 4)
    d_center = heatmap_3d.shape[0] // 2
    heatmap_slice = heatmap_3d[d_center].cpu().numpy()
    hm_min, hm_max = heatmap_slice.min(), heatmap_slice.max()
    if hm_max - hm_min > 1e-8:
        heatmap_slice = (heatmap_slice - hm_min) / (hm_max - hm_min)
    else:
        heatmap_slice = np.zeros_like(heatmap_slice)

    # Original volume central slice (depth = 64, center = 32)
    vol = input_tensor.squeeze(0).squeeze(0).cpu().numpy()
    orig_slice = vol[vol.shape[0] // 2]
    orig_disp = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)

    orig_resized = cv2.resize(orig_disp, (224, 224))
    heatmap_resized = cv2.resize(heatmap_slice, (224, 224))
    orig_bgr = cv2.cvtColor(np.uint8(255 * orig_resized), cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig_bgr, 0.5, heatmap_color, 0.5, 0)

    return orig_resized, heatmap_resized, superimposed

# ----------------------------- Main -----------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    np.random.seed(SEED)

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    test_dataset = NodulePatchDataset3D(test_meta, DATA_DIR, transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    # Load model
    model = MedNeXt3DWithMLPHead().to(DEVICE)
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

    # Patient-level aggregation (max probability per scan)
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
    auc = roc_auc_score(y_true_patient, y_probs_patient) if len(np.unique(y_true_patient)) > 1 else 0.5
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

    # ----------------------------- Print Metrics --------------------------------
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

    # -------------------------- Save Prediction & Calibration -----------------
    pred_df = pd.DataFrame({
        'seriesuid': all_uids,
        'label': all_labels,
        'probability': all_probs,
        'filepath': all_paths
    })
    pred_df.to_csv("mednext3d_gradcam_test_predictions.csv", index=False)
    print("[Saved] Test predictions exported to 'mednext3d_gradcam_test_predictions.csv'")

    cal_df = pd.DataFrame({
        'mean_predicted_probability': mean_predicted_value,
        'fraction_of_positives': fraction_of_positives
    })
    cal_df.to_csv("mednext3d_gradcam_calibration_curve.csv", index=False)
    print("[Saved] Calibration curve exported to 'mednext3d_gradcam_calibration_curve.csv'")

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
        patient_df = df_pred[df_pred['seriesuid'] == uid]
        best_candidate = patient_df.loc[patient_df['probability'].idxmax()]
        patch_path = best_candidate['filepath']
        idx_in_meta = test_meta[test_meta['filepath'] == patch_path].index[0]
        img_tensor, label, uid_out, _ = test_dataset[idx_in_meta]
        input_t = img_tensor.unsqueeze(0).to(DEVICE)

        cidx = 1 if row['label'] == 1 else 0
        print(f"\n{'='*60}")
        print(f"Case: {case_type}-{case_num} | UID: {uid[:12]}... | Prob: {row['prob']:.4f} | True label: {int(row['label'])}")
        print(f"{'='*60}")
        orig_slice, hm_slice, overlay = produce_heatmap(model, input_t, target_layer_name, class_idx=cidx)
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
    plt.savefig("gradcampp_mednext3d_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: gradcampp_mednext3d_analysis.png")

if __name__ == "__main__":
    main()