# gradcam_tinyvit.py (verbose v4)
"""
GradCAM++ for TinyViT lung nodule classification.
Prints detailed diagnostic values for each heatmap.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.transforms import Compose, Resized
import timm
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- Configuration -----------------------------
DATA_DIR = r"D:\Projects\Thesis Lung Malignancy Classification ML & DL\LUNA16 Dataset Preprocessed"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")
MODEL_PATH = "best_model_tinyvit.pth"
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Dataset (unchanged) ------------------------
class NodulePatchDataset2D(torch.utils.data.Dataset):
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
        vol = np.load(local_path).astype(np.float32)
        centre = vol.shape[0] // 2
        slices = [vol[centre - 1, :, :], vol[centre, :, :], vol[centre + 1, :, :]]
        img_3ch = np.stack(slices, axis=0)
        data_dict = {"img": img_3ch}
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        img = data_dict["img"]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        return img, torch.tensor(label, dtype=torch.float32), row["seriesuid"], row["filepath"]

# ----------------------------- Model (unchanged) --------------------------
class TinyViTWithMLPHead(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            "tiny_vit_21m_224", pretrained=False, num_classes=0, in_chans=3
        )
        feat_dim = self.backbone.num_features
        self.mlp_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.mlp_head(features)
        return out.squeeze(1)

# ----------------------------- GradCAM utilities ---------------------------
def find_target_layer(model):
    """
    Priority: last Conv2d with spatial output > 1x1.
    Fallback: backbone.norm (pre‑pooling LN, requires token reshape).
    """
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)

    target_name = None
    best_size = 0

    def hook_fn(name):
        def hook(module, input, output):
            nonlocal target_name, best_size
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                h, w = output.shape[2], output.shape[3]
                if h > 1 and h * w > best_size:
                    best_size = h * w
                    target_name = name
        return hook

    handles = []
    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(hook_fn(name)))
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()

    if target_name is not None:
        print(f"✔ Chosen target layer (Conv2d): {target_name} (spatial size {best_size})")
        return target_name
    else:
        print("⚠ No Conv2d layer found, using backbone.norm (token layer)")
        return "backbone.norm"

def reshape_token_to_grid(tensor):
    """If tensor is (B, N, C), reshape to (B, C, H, W) assuming N = H*W."""
    if tensor.dim() == 3:
        B, N, C = tensor.shape
        hw = int(np.sqrt(N))
        if hw * hw == N:
            return tensor.transpose(1, 2).view(B, C, hw, hw)
        else:
            raise ValueError(f"Cannot reshape {N} tokens to a square grid.")
    return tensor

def grad_cam_vanilla(fmap, grad):
    """Standard GradCAM: weight = global average of gradients."""
    weights = grad.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
    heatmap = (weights * fmap).sum(dim=1)            # (B, H, W)
    heatmap = F.relu(heatmap)
    return heatmap

def grad_cam_plusplus(fmap, grad):
    """GradCAM++ weighting (returns heatmap, can be zero if all grads negative)."""
    grad_2 = grad.pow(2)
    grad_3 = grad_2 * grad
    sum_activations = fmap.sum(dim=(2, 3), keepdim=True)

    numerator = grad_2 * fmap
    denominator = 2 * grad_2 + sum_activations * grad_3
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))

    alpha = numerator / denominator
    alpha = alpha.sum(dim=(2, 3), keepdim=True)
    weights = F.relu(grad) * alpha               # ReLU kills negative gradients
    heatmap = (weights * fmap).sum(dim=1)        # (B, H, W)
    heatmap = F.relu(heatmap)
    return heatmap

def produce_heatmap(model, input_tensor, target_layer_name, class_idx=1):
    """
    Full pipeline with verbose diagnostics.
    Returns: heatmap_resized (224,224), superimposed (BGR)
    """
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

    # Reshape if needed
    fmap_4d = reshape_token_to_grid(fmap)
    grad_4d = reshape_token_to_grid(grad)

    print(f"\n--- Heatmap Diagnostic for class {class_idx} ---")
    print(f"Target layer     : {target_layer_name}")
    print(f"Logit value      : {logits.item():.4f} → prob {torch.sigmoid(logits).item():.4f}")
    print(f"fmap shape       : {fmap.shape} → 4D: {fmap_4d.shape}")
    print(f"Grad  min/mean/max: {grad.min().item():.4f} / {grad.mean().item():.4f} / {grad.max().item():.4f}")
    print(f"Fmap  min/mean/max: {fmap.min().item():.4f} / {fmap.mean().item():.4f} / {fmap.max().item():.4f}")

    # Try GradCAM++
    heatmap_pp = grad_cam_plusplus(fmap_4d, grad_4d)
    max_pp = heatmap_pp.max().item()
    print(f"GradCAM++ raw heatmap min/max: {heatmap_pp.min().item():.6f} / {max_pp:.6f}")

    if max_pp < 1e-6:
        print("→ GradCAM++ heatmap is ZERO. Falling back to standard GradCAM.")
        heatmap = grad_cam_vanilla(fmap_4d, grad_4d)
        print(f"GradCAM raw heatmap min/max: {heatmap.min().item():.6f} / {heatmap.max().item():.6f}")
    else:
        heatmap = heatmap_pp

    heatmap = heatmap.squeeze(0).cpu().numpy()
    hm_min, hm_max = heatmap.min(), heatmap.max()
    print(f"Final heatmap (before norm) range: [{hm_min:.6f}, {hm_max:.6f}]")

    # Normalise to [0,1]
    if hm_max - hm_min > 1e-8:
        heatmap_norm = (heatmap - hm_min) / (hm_max - hm_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)
        print("⚠ Heatmap is constant – normalised to all zeros.")

    # Resize to input size
    heatmap_resized = cv2.resize(heatmap_norm, (224, 224))

    # Overlay
    input_np = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_disp = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.uint8(255 * img_disp), 0.5, heatmap_color, 0.5, 0)

    return heatmap_resized, superimposed

# ----------------------------- Main -----------------------------------------
# gradcam_tinyvit.py (expanded selection)
# ... all imports, dataset, model, and GradCAM functions remain identical ...

def main():
    print(f"Using device: {DEVICE}")

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        patient_split = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(patient_split["seriesuid"], patient_split["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)

    test_transforms = Compose([Resized(keys=["img"], spatial_size=(224, 224), mode="bicubic")])
    test_dataset = NodulePatchDataset2D(test_meta, DATA_DIR, transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    model = TinyViTWithMLPHead().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.\n")

    target_layer_name = find_target_layer(model)

    # Predict
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

    df = pd.DataFrame({
        'seriesuid': all_uids,
        'label': all_labels,
        'prob': all_probs,
        'filepath': all_paths
    })

    patient_pred = df.groupby('seriesuid').agg(
        prob=('prob', 'max'),
        label=('label', 'max')
    ).reset_index()
    patient_pred['pred'] = (patient_pred['prob'] >= 0.5).astype(int)

    tp = patient_pred[(patient_pred['label'] == 1) & (patient_pred['pred'] == 1)]
    fn = patient_pred[(patient_pred['label'] == 1) & (patient_pred['pred'] == 0)]
    fp = patient_pred[(patient_pred['label'] == 0) & (patient_pred['pred'] == 1)]
    tn = patient_pred[(patient_pred['label'] == 0) & (patient_pred['pred'] == 0)]
    print(f"Available – TP: {len(tp)}, FN: {len(fn)}, FP: {len(fp)}, TN: {len(tn)}")

    # Select multiple cases per category (up to the numbers shown)
    selected_cases = []
    # TP: up to 4
    for i in range(min(4, len(tp))):
        selected_cases.append(('TP', i+1, tp.iloc[i]))
    # FN: up to 3
    for i in range(min(3, len(fn))):
        selected_cases.append(('FN', i+1, fn.iloc[i]))
    # FP: up to 3
    for i in range(min(3, len(fp))):
        selected_cases.append(('FP', i+1, fp.iloc[i]))
    # TN: up to 3
    for i in range(min(3, len(tn))):
        selected_cases.append(('TN', i+1, tn.iloc[i]))

    heatmaps, titles = [], []
    for case_type, case_num, row in selected_cases:
        if row is None:
            continue
        uid = row['seriesuid']
        case_name = f"{case_type}-{case_num}"
        patient_df = df[df['seriesuid'] == uid]
        best_candidate = patient_df.loc[patient_df['prob'].idxmax()]
        patch_path = best_candidate['filepath']
        idx_in_meta = test_meta[test_meta['filepath'] == patch_path].index[0]
        img_tensor, label, uid_out, _ = test_dataset[idx_in_meta]
        input_t = img_tensor.unsqueeze(0).to(DEVICE)

        cidx = 1 if row['label'] == 1 else 0
        print(f"\n{'='*60}")
        print(f"Case: {case_name} | UID: {uid[:12]}... | Prob: {row['prob']:.4f} | True label: {int(row['label'])}")
        print(f"{'='*60}")
        heatmap, overlay = produce_heatmap(model, input_t, target_layer_name, class_idx=cidx)
        heatmaps.append((img_tensor, heatmap, overlay))
        titles.append(f"{case_name} | UID:{uid[:8]}... | Prob:{row['prob']:.3f} | Label:{int(row['label'])}")

    n = len(heatmaps)
    if n == 0:
        print("No cases to display.")
        return

    # Adjust figure size – more rows → taller figure
    fig_height = max(12, 4 * n)
    fig, axes = plt.subplots(n, 3, figsize=(12, fig_height))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (img_t, hm, ov) in enumerate(heatmaps):
        img_np = img_t.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title("Original patch")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(hm, cmap='jet')
        axes[i, 1].set_title("GradCAM heatmap")
        axes[i, 1].axis('off')

        ov_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)
        axes[i, 2].imshow(ov_rgb)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')

        fig.text(0.5, 0.02 + (n - i - 1) / n * 0.9, titles[i],
                 ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("gradcampp_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: gradcampp_analysis.png")

if __name__ == "__main__":
    main()