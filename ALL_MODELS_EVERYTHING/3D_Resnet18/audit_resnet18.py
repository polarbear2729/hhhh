"""
Extended audit after RandAffine fix.
Verifies data integrity, training sanity, and that the model now focuses on the nodule.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import Compose, RandAffine
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---------- Config ----------
DATA_DIR = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
# Update this path to your new best model
MODEL_PATH = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_Resnet18\best_model_resnet18.pth"
RESNET18_FEATURE_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Model Definition ----------
class ResNet18WithMLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet(
            block=ResNetBlock, layers=[2,2,2,2],
            block_inplanes=[64,128,256,512],
            spatial_dims=3, n_input_channels=1, num_classes=RESNET18_FEATURE_DIM
        )
        self.backbone.fc = nn.Sequential(
            nn.Linear(RESNET18_FEATURE_DIM, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.backbone(x).squeeze(1)

# ---------- Utility: load test predictions ----------
def compute_patient_predictions(metadata, data_dir, model, device):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    test_meta = metadata[metadata["split"]=="test"].reset_index(drop=True)
    with torch.no_grad():
        for idx in range(len(test_meta)):
            row = test_meta.iloc[idx]
            fname = os.path.basename(row["filepath"])
            subfolder = "pos" if row["label"]==1 else "neg"
            path = os.path.join(data_dir, row["split"], subfolder, fname)
            patch = np.load(path).astype(np.float32)
            patch = np.expand_dims(patch, 0)  # [1, D, H, W]
            input_t = torch.from_numpy(patch).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(input_t)).item()
            all_probs.append(prob)
            all_labels.append(row["label"])
            all_uids.append(row["seriesuid"])

    # Patient-level max-pool
    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {'probs': [], 'label': label}
        patient_dict[uid]['probs'].append(prob)

    y_true = np.array([v['label'] for v in patient_dict.values()])
    y_scores = np.array([max(v['probs']) for v in patient_dict.values()])
    return y_true, y_scores

# ---------- 1. Data Leakage (unchanged) ----------
print("\n=== 1. Data Leakage Check ===")
metadata = pd.read_csv(METADATA_PATH)
split_df = pd.read_csv(PATIENT_SPLIT_PATH)
if "split" not in metadata.columns:
    metadata["split"] = metadata["seriesuid"].map(dict(zip(split_df["seriesuid"], split_df["split"])))
train_uids = set(metadata[metadata["split"]=="train"]["seriesuid"])
val_uids   = set(metadata[metadata["split"]=="val"]["seriesuid"])
test_uids  = set(metadata[metadata["split"]=="test"]["seriesuid"])
if train_uids.intersection(test_uids):
    print("❌ TRAIN/TEST LEAKAGE DETECTED")
elif val_uids.intersection(test_uids):
    print("❌ VAL/TEST LEAKAGE DETECTED")
else:
    print("✅ No patient overlap between splits.")

# ---------- 2. Model Parameter Count ----------
print("\n=== 2. Model Parameters ===")
model = ResNet18WithMLPHead()
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total:,} (expected ~33.2M)")

# ---------- 3. Training Curves ----------
print("\n=== 3. Training Log ===")
log = pd.read_csv("resnet18_training_log.csv")  # adjust filename if you saved with a different name
best_epoch = log.loc[log["val_auc"].idxmax()]["epoch"]
print(f"Best epoch: {best_epoch}, best val AUC: {log['val_auc'].max():.4f}")
final_train = log[log["epoch"]==log["epoch"].max()]["train_loss"].values[0]
final_val   = log[log["epoch"]==log["epoch"].max()]["val_loss"].values[0]
print(f"Final Train Loss: {final_train:.4f}, Val Loss: {final_val:.4f}")

# ---------- 4. Test Predictions ----------
print("\n=== 4. Test Predictions ===")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE).eval()
y_true, y_scores = compute_patient_predictions(metadata, DATA_DIR, model, DEVICE)
auc = roc_auc_score(y_true, y_scores)
cm = confusion_matrix(y_true, [1 if p>=0.5 else 0 for p in y_scores])
print(f"Recomputed patient-level AUROC: {auc:.4f}")
print("Confusion matrix:")
print(cm)

# ---------- 5. Augmentation Verification ----------
print("\n=== 5. Augmentation Verification (RandAffine translation) ===")
# Load one positive training patch
train_meta = metadata[metadata["split"]=="train"]
pos_train = train_meta[train_meta["label"]==1]
row = pos_train.iloc[0]
fname = os.path.basename(row["filepath"])
path = os.path.join(DATA_DIR, row["split"], "pos", fname)
patch = np.load(path).astype(np.float32)
patch = np.expand_dims(patch, 0)  # [1, D, H, W]

# Define the same augmentation as in training
transform = Compose([
    RandAffine(prob=1.0, translate_range=(5,5,5), padding_mode='zeros')
])

# Apply 5 times and record argmax (brightest voxel) position
positions = []
for _ in range(5):
    transformed = transform(patch.copy())
    # transformed is tensor or numpy? transform returns numpy/tensor depending on input.
    if isinstance(transformed, torch.Tensor):
        arr = transformed.squeeze().cpu().numpy()
    else:
        arr = np.squeeze(transformed)
    max_idx = np.unravel_index(np.argmax(arr), arr.shape)
    positions.append(max_idx)

print("Argmax positions of brightest voxel after 5 random translations:")
for i, pos in enumerate(positions):
    print(f"  {i}: {pos}")
if len(set(positions)) > 1:
    print("✅ Nodule position varies across translations – augmentation is active.")
else:
    print("⚠️ Position unchanged – check transform probability or range.")

# ---------- 6. Border Occlusion Test ----------
print("\n=== 6. Border Occlusion Test ===")
# Use a TP test sample (first one)
test_meta = metadata[metadata["split"]=="test"]
pos_test = test_meta[test_meta["label"]==1]
if len(pos_test)>0:
    row = pos_test.iloc[0]
    fname = os.path.basename(row["filepath"])
    path = os.path.join(DATA_DIR, row["split"], "pos", fname)
    patch = np.load(path).astype(np.float32)
    patch = np.expand_dims(patch, 0)  # [1, D, H, W]
    input_t = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        orig_prob = torch.sigmoid(model(input_t)).item()
    print(f"Original prob: {orig_prob:.4f}")

    # Remove outer 2-voxel border
    border = 2
    patch_occ = patch.copy()
    patch_occ[0, :border, :, :] = 0
    patch_occ[0, -border:, :, :] = 0
    patch_occ[0, :, :border, :] = 0
    patch_occ[0, :, -border:, :] = 0
    patch_occ[0, :, :, :border] = 0
    patch_occ[0, :, :, -border:] = 0
    input_occ = torch.from_numpy(patch_occ).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob_border = torch.sigmoid(model(input_occ)).item()
    print(f"Prob after border removal: {prob_border:.4f} (drop {orig_prob-prob_border:.4f})")
    if orig_prob - prob_border < 0.2:
        print("✅ Small drop – model no longer relies heavily on borders.")
    else:
        print("⚠️ Significant drop – border may still be used; check further.")

# ---------- 7. Corner Occlusion Test (previous peak region) ----------
print("\n=== 7. Corner Occlusion Test ===")
# We'll occlude a 5x5x5 corner at (0,0,0) – the former peak for FN
patch_corner = patch.copy()
patch_corner[0, :5, :5, :5] = 0   # corner block
input_corner = torch.from_numpy(patch_corner).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    prob_corner = torch.sigmoid(model(input_corner)).item()
print(f"Prob after corner occlusion: {prob_corner:.4f} (drop {orig_prob-prob_corner:.4f})")
if orig_prob - prob_corner < 0.2:
    print("✅ Small drop – model ignores the corner.")
else:
    print("⚠️ Corner still influences prediction – shortcut may persist.")

# ---------- 8. Grad‑CAM Peak‑to‑Nodule Distance ----------
print("\n=== 8. Grad‑CAM Nodule Alignment Check ===")
# Use simple custom Grad‑CAM (light version, same as before but without saving images)
class GradCAM3D:
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
        weights = grads.view(1, acts.shape[1], -1).mean(dim=2)
        weights = weights.view(1, acts.shape[1], 1, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        import torch.nn.functional as F
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='trilinear', align_corners=True)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

# Use the model's backbone with layer3.1 as before
cam_generator = GradCAM3D(model.backbone, "layer3.1")
# Use the same patch as for occlusion (TP)
input_tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
heatmap = cam_generator.generate(input_tensor)

# Nodule center estimated by brightest voxel in original patch (non-border)
arr = patch.squeeze()
# Exclude a border to avoid edge artifacts in nodule center estimate
arr_masked = arr[2:-2, 2:-2, 2:-2]
nodule_center = np.unravel_index(np.argmax(arr_masked), arr_masked.shape)
nodule_center = np.array(nodule_center) + 2  # compensate for cropping

# Heatmap peak
heat_peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
distance = np.linalg.norm(np.array(heat_peak) - nodule_center)
print(f"Nodule center (approx): {nodule_center}")
print(f"Grad‑CAM peak: {heat_peak}")
print(f"Distance between peak and nodule center: {distance:.1f} voxels")
if distance < 10:
    print("✅ Grad‑CAM is tightly focused on the nodule.")
else:
    print("⚠️ Heatmap peak is far from nodule – model may still be distracted.")

print("\n=== Audit complete ===")