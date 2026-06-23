"""
Grad‑CAM++ for DenseNet121 (before RandAffine retraining)
Uses hook‑based Grad‑CAM++ with automatic layer selection.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- Configuration -----------------------------
SEED = 42
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_DenseNet121\best_model_densenet121.pth"   # adjust path if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "gradcampp_densenet121_before_fix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------- Reproducibility ---------------------------
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ----------------------------- Model Definition -------------------------
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
        return self.backbone(x).squeeze(1)

# ----------------------------- Dataset ----------------------------------
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
        patch = np.load(local_path).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)   # [1, D, H, W]
        if self.transforms:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]

# ----------------------------- Grad‑CAM++ (Hook‑based) -------------------
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

        acts = self.activations
        grads = self.gradients

        # Grad‑CAM++ weight computation
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

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

# ----------------------------- Helper: find good layer -------------------
def find_target_layer_densenet(model):
    """
    Finds a suitable convolutional layer in DenseNet121.
    Prefers the last Conv3d in denseblock3 for better spatial resolution.
    """
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            candidates.append(name)
    # Prefer denseblock3 (larger feature maps)
    db3_layers = [n for n in candidates if 'denseblock3' in n]
    if db3_layers:
        # return the last conv in denseblock3
        return db3_layers[-1]
    # fallback to denseblock4
    db4_layers = [n for n in candidates if 'denseblock4' in n]
    if db4_layers:
        return db4_layers[-1]
    # ultimate fallback: last conv in model
    return candidates[-1]

# ----------------------------- Visualization ----------------------------
def save_standard_gradcam(vol, heatmap, prob, uid, case_type, slice_idx, output_dir):
    vmin, vmax = vol.min(), vol.max()
    vol_norm = (vol - vmin) / (vmax - vmin) if vmax > vmin else vol

    slice_img = vol_norm[slice_idx]
    slice_heat = heatmap[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(slice_img, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    heat_plot = axes[1].imshow(slice_heat, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Grad‑CAM++ Heatmap")
    axes[1].axis('off')
    plt.colorbar(heat_plot, ax=axes[1], fraction=0.046)

    axes[2].imshow(slice_img, cmap='gray')
    axes[2].imshow(slice_heat, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    fig.suptitle(f"{case_type} – Slice {slice_idx} – Prob={prob:.4f}", fontsize=14)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"{case_type}_{uid[:12]}_slice{slice_idx}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad‑CAM++: {fname}")
    np.save(os.path.join(output_dir, f"{case_type}_{uid[:12]}_heatmap.npy"), heatmap)

# ----------------------------- Main Execution ---------------------------
def main():
    print(f"Using device: {DEVICE}")

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    test_dataset = NodulePatchDataset(test_meta, DATA_DIR, transforms=None)

    # Load model
    model = DenseNet121WithMLPHead().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # Patient-level predictions
    def compute_patient_predictions(dataset, model, device):
        model.eval()
        all_probs, all_labels, all_uids = [], [], []
        with torch.no_grad():
            for i in range(len(dataset)):
                patch, label, uid = dataset[i]
                patch_4d = patch.unsqueeze(0).to(device)
                logit = model(patch_4d)
                prob = torch.sigmoid(logit).item()
                all_probs.append(prob)
                all_labels.append(label.item())
                all_uids.append(uid)

        patient_dict = {}
        for prob, label, uid in zip(all_probs, all_labels, all_uids):
            if uid not in patient_dict:
                patient_dict[uid] = {'probs': [prob], 'label': label}
            else:
                patient_dict[uid]['probs'].append(prob)

        y_true = np.array([v['label'] for v in patient_dict.values()])
        y_scores = np.array([max(v['probs']) for v in patient_dict.values()])
        patient_uids = list(patient_dict.keys())
        return y_true, y_scores, patient_uids

    y_true, y_scores, patient_uids = compute_patient_predictions(test_dataset, model, DEVICE)
    y_pred = (y_scores >= 0.5).astype(int)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_uids = [patient_uids[i] for i in np.where(fp_mask)[0]]
    fn_uids = [patient_uids[i] for i in np.where(fn_mask)[0]]

    tp_mask = (y_true == 1) & (y_pred == 1)
    tp_probs = y_scores[tp_mask]
    tp_uids_all = [patient_uids[i] for i in np.where(tp_mask)[0]]

    selected_tp_uids = []
    if len(tp_probs) > 0:
        high_conf_idx = np.argmax(tp_probs)
        selected_tp_uids.append(("high_conf_TP", tp_uids_all[high_conf_idx]))
    if len(tp_probs) > 1:
        target_prob = 0.8
        mod_conf_idx = np.argmin(np.abs(tp_probs - target_prob))
        selected_tp_uids.append(("moderate_TP", tp_uids_all[mod_conf_idx]))

    # Find target layer
    target_layer = find_target_layer_densenet(model.backbone)
    print(f"Using target layer: {target_layer}")
    cam_generator = GradCAMPlusPlus3D(model.backbone, target_layer)

    def process_patient(uid, case_type):
        rows = test_meta[test_meta["seriesuid"] == uid]
        if len(rows) == 0:
            return
        row = rows.iloc[0]
        filename = os.path.basename(row["filepath"])
        subfolder = "pos" if row["label"] == 1 else "neg"
        patch_path = os.path.join(DATA_DIR, row["split"], subfolder, filename)
        patch = np.load(patch_path).astype(np.float32)
        patch = np.expand_dims(patch, 0)
        input_tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()

        print(f"\n--- {case_type} (prob={prob:.4f}) ---")
        heatmap = cam_generator.generate(input_tensor, target_class=1)
        vol = patch.squeeze(0)

        peak_coord = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        print(f"Peak at (z,y,x): {peak_coord}, value at peak: {vol[peak_coord]:.6f}")
        print(f"Value at center: {vol[vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2]:.6f}")

        z_peak = peak_coord[0]
        middle = vol.shape[0] // 2
        for slc in [z_peak, middle]:
            if 0 <= slc < vol.shape[0]:
                save_standard_gradcam(vol, heatmap, prob, uid, case_type, slc, OUTPUT_DIR)

    for case_type, uid in selected_tp_uids:
        process_patient(uid, case_type)
    for uid in fp_uids:
        process_patient(uid, "FP")
    for uid in fn_uids:
        process_patient(uid, "FN")

    print(f"\nAll Grad‑CAM++ outputs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()