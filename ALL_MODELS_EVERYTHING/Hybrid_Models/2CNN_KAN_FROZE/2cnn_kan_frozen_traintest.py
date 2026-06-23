"""
2-Branch 3D Hybrid CNN (ResNet18 + DenseNet121) with KAN Head — FROZEN BACKBONES
==================================================================================
Swaps the MLP head for a KAN (Kolmogorov-Arnold Network) head built entirely
from scratch using B-spline basis functions.

KAN references:
  Liu et al., "KAN: Kolmogorov-Arnold Networks" — arXiv 2404.19756 (2024)
  Blealtan/efficient-kan on GitHub (B-spline formulation used here)

Backbone weight-loading retains the prefix-stripping fix from the frozen-MLP
version (strips 'backbone.' from checkpoint keys when necessary).
"""

import os, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, recall_score,
)
from sklearn.calibration import calibration_curve
from monai.networks.nets import DenseNet121, ResNet
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise, RandAffine,
)
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
DATA_DIR           = r"C:\Users\T2520789\LUNGCANCER3D\data"
METADATA_PATH      = os.path.join(DATA_DIR, "metadata_all.csv")
PATIENT_SPLIT_PATH = os.path.join(DATA_DIR, "patient_split.csv")

RESNET18_WEIGHTS    = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_Resnet18\best_model_resnet18.pth"
DENSENET121_WEIGHTS = r"C:\Users\T2520789\LUNGCANCER3D\scripts\Sadit\3D_densenet121\best_model_densenet121.pth"

BATCH_SIZE         = 8
NUM_WORKERS        = 4
PIN_MEMORY         = True
PERSISTENT_WORKERS = True

MAX_EPOCHS               = 100
EARLY_STOPPING_PATIENCE  = 15
LR                       = 1e-4
WEIGHT_DECAY             = 1e-4
POS_WEIGHT               = 5115.0 / 822.0

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

RESNET18_FEATURE_DIM    = 512
DENSENET121_FEATURE_DIM = 1024
TOTAL_FEATURE_DIM       = RESNET18_FEATURE_DIM + DENSENET121_FEATURE_DIM   # 1536

# ── KAN head hyper-parameters ─────────────────────────────────────────────────
# Each edge (i→j) learns:  base_w·SiLU(x_i)  +  scale·Σ coeff_t · B_t(x_i)
# where {B_t} are B-spline basis functions on an adaptive grid.

KAN_GRID_SIZE    = 5      # number of B-spline intervals (more = finer resolution)
KAN_SPLINE_ORDER = 3      # polynomial order (3 = cubic, smoothest common choice)
KAN_SCALE_NOISE  = 0.1    # noise std for initial spline coefficient fitting
KAN_SCALE_BASE   = 1.0    # kaiming init scale for base (linear residual) weights
KAN_SCALE_SPLINE = 1.0    # kaiming init scale for per-edge spline scalers
KAN_GRID_EPS     = 0.02   # 0 = fully adaptive grid update, 1 = fully uniform
KAN_GRID_RANGE   = [-1.0, 1.0]  # initial input domain covered by the grid

# Optional spline regularization from Liu et al. §2.5 — set both to 0 to disable
KAN_REG_ACTIVATION = 0.0   # L1 on spline weights (encourages edge sparsity)
KAN_REG_ENTROPY    = 0.0   # entropy penalty (encourages single dominant spline per edge)

# Adaptive grid update schedule
#   KAN grids are refitted to the real feature distribution periodically.
#   After KAN_GRID_UPDATE_UNTIL epochs the grids are stable enough to leave alone.
KAN_GRID_UPDATE_EVERY   = 5   # update every N epochs
KAN_GRID_UPDATE_UNTIL   = 20  # stop updating after epoch M
KAN_GRID_UPDATE_BATCHES = 50  # number of training batches to sample per update


# ═══════════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Weight loading  (backbone prefix-aware — unchanged from frozen-MLP version)
# ═══════════════════════════════════════════════════════════════════════════════

def load_backbone_weights(backbone_module, ckpt_path, label):
    """
    Load weights from a checkpoint saved by a wrapper model that stored the
    backbone under a 'backbone.' prefix.  Tries both direct and prefix-stripped
    key mappings and picks whichever matches more keys.
    """
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_keys = set(backbone_module.state_dict().keys())

    direct_keys   = ckpt
    stripped_keys = {k[len("backbone."):]: v
                     for k, v in ckpt.items() if k.startswith("backbone.")}

    direct_match   = len(model_keys & set(direct_keys.keys()))
    stripped_match = len(model_keys & set(stripped_keys.keys()))

    print(f"\n  ── {label} weight loading ──")
    print(f"     Direct match   : {direct_match} / {len(model_keys)}")
    print(f"     Stripped match : {stripped_match} / {len(model_keys)}")

    if stripped_match > direct_match:
        weights_to_load = stripped_keys
        print("     Strategy: strip 'backbone.' prefix from checkpoint keys")
    else:
        weights_to_load = direct_keys
        print("     Strategy: direct load (no prefix stripping needed)")

    missing, unexpected = backbone_module.load_state_dict(weights_to_load, strict=False)
    matched = len(model_keys) - len(missing)
    print(f"     Loaded  : {matched}/{len(model_keys)} keys matched")
    print(f"     Missing : {len(missing)}   Unexpected: {len(unexpected)}")
    if missing:
        print("     Still missing: "
              + ", ".join(list(missing)[:5])
              + ("..." if len(missing) > 5 else ""))

    # Verify at least one weight actually landed
    for key in list(model_keys):
        if key in weights_to_load:
            mv = backbone_module.state_dict()[key].float().mean().item()
            cv = weights_to_load[key].float().mean().item()
            ok = abs(mv - cv) < 1e-6
            print(f"     Verify [{key}]: {'✓' if ok else '⚠ MISMATCH'}  "
                  f"(model={mv:.6f}, ckpt={cv:.6f})")
            if not ok:
                print("     ⚠ WARNING: weight value mismatch after load!")
            break
    return matched


# ═══════════════════════════════════════════════════════════════════════════════
#  KAN — built from scratch (B-spline formulation)
# ═══════════════════════════════════════════════════════════════════════════════

class KANLinear(nn.Module):
    """
    Single KAN layer mapping in_features → out_features.

    Every directed edge (i → j) computes:
        edge(x_i) = base_w_ij · SiLU(x_i)  +  scale_ij · Σ_t  c_ij_t · B_t(x_i)

    where {B_t} are cubic B-spline basis functions defined on a learnable
    grid that is periodically adapted to the actual input distribution.

    Learnable parameters
    --------------------
    base_weight   : (out, in)           — linear residual coefficients
    spline_weight : (out, in, n_coeffs) — spline basis coefficients
    spline_scaler : (out, in)           — per-edge global spline scale

    n_coeffs = grid_size + spline_order  (8 for the default 5+3 setting)
    """

    def __init__(
        self,
        in_features:     int,
        out_features:    int,
        grid_size:       int   = KAN_GRID_SIZE,
        spline_order:    int   = KAN_SPLINE_ORDER,
        scale_noise:     float = KAN_SCALE_NOISE,
        scale_base:      float = KAN_SCALE_BASE,
        scale_spline:    float = KAN_SCALE_SPLINE,
        base_activation        = nn.SiLU,
        grid_eps:        float = KAN_GRID_EPS,
        grid_range:      list  = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = list(KAN_GRID_RANGE)

        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.scale_noise  = scale_noise
        self.scale_base   = scale_base
        self.scale_spline = scale_spline
        self.grid_eps     = grid_eps
        self.base_activation = base_activation()

        # ── Uniform extended B-spline grid ────────────────────────────────
        # The grid has (grid_size + 1) inner knots plus spline_order extra
        # knots on each side to support the B-spline boundary conditions.
        # Final shape: (in_features, grid_size + 2·order + 1)
        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
            * h + grid_range[0]
        )
        self.register_buffer(
            "grid",
            grid.unsqueeze(0).expand(in_features, -1).contiguous(),
        )

        # ── Trainable parameters ──────────────────────────────────────────
        n_coeffs = grid_size + spline_order   # number of B-spline basis functions

        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_coeffs))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    # ── Initialisation ────────────────────────────────────────────────────────
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight,   a=np.sqrt(5) * self.scale_base)
        nn.init.kaiming_uniform_(self.spline_scaler, a=np.sqrt(5) * self.scale_spline)
        with torch.no_grad():
            # Evaluate the basis at the inner grid knots, add small random noise,
            # then fit spline coefficients via least-squares.
            inner  = self.grid[0, self.spline_order : -self.spline_order]  # (G+1,)
            x_init = inner.unsqueeze(1).expand(-1, self.in_features)       # (G+1, in)
            noise  = (
                torch.rand(inner.size(0), self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(self._curve2coeff(x_init, noise))

    # ── B-spline basis evaluation (Cox–de Boor recurrence) ────────────────────
    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all B-spline basis functions at every input value.

        Args
            x : (B, in_features)  — must be float32
        Returns
            bases : (B, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        xv = x.unsqueeze(-1)    # (B, in, 1)  — broadcast against grid dim
        g  = self.grid           # (in, n_knots)

        # Order-0: step indicator — 1 where x falls in the half-open interval [g_t, g_{t+1})
        bases = ((xv >= g[:, :-1]) & (xv < g[:, 1:])).to(x.dtype)   # (B, in, n_knots-1)

        # Cox–de Boor recurrence: build degree-k from degree-(k-1)
        # After k steps: bases has shape (B, in, n_knots-1-k)
        # Final (k=spline_order): (B, in, grid_size + spline_order) = (B, in, n_coeffs) ✓
        for k in range(1, self.spline_order + 1):
            left_denom  = g[:, k  : -1    ] - g[:, : -(k + 1)]   # (in, n_knots-1-k)
            right_denom = g[:, k+1:        ] - g[:, 1 : -k    ]   # (in, n_knots-1-k)

            # Guard zero denominators (degenerate knots) with where+clamp
            left_w = torch.where(
                left_denom  > 1e-8,
                (xv - g[:, : -(k + 1)]) / left_denom.clamp(min=1e-8),
                torch.zeros_like(xv),
            )
            right_w = torch.where(
                right_denom > 1e-8,
                (g[:, k + 1:] - xv)     / right_denom.clamp(min=1e-8),
                torch.zeros_like(xv),
            )
            bases = left_w * bases[..., :-1] + right_w * bases[..., 1:]

        return bases.contiguous()   # (B, in, n_coeffs)

    # ── Least-squares coefficient fitting ────────────────────────────────────
    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Find spline coefficients C such that  B(x) @ C ≈ y.

        Solves independently for each input dimension (batched lstsq).

        Args
            x : (N, in_features)
            y : (N, in_features, out_features)   — target spline output per edge
        Returns
            (out_features, in_features, n_coeffs)
        """
        A = self._b_splines(x).permute(1, 0, 2)    # (in, N, n_coeffs)
        B = y.permute(1, 0, 2)                       # (in, N, out)
        # A_i @ C_i ≈ B_i  →  C_i has shape (n_coeffs, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in, n_coeffs, out)
        return solution.permute(2, 0, 1).contiguous()  # (out, in, n_coeffs)

    # ── Scaled weights ────────────────────────────────────────────────────────
    @property
    def _scaled_spline_weight(self) -> torch.Tensor:
        """spline_weight element-wise scaled by per-edge spline_scaler."""
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_features) — float32 expected (caller should cast before here)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # ① Residual path: learnable linear applied to SiLU(x)
        base_out = F.linear(self.base_activation(x), self.base_weight)

        # ② Spline path: evaluate basis, flatten (in · n_coeffs), then project
        splines    = self._b_splines(x)                              # (B, in, n_coeffs)
        spline_out = F.linear(
            splines.view(x.size(0), -1),                             # (B, in·n_coeffs)
            self._scaled_spline_weight.view(self.out_features, -1),  # (out, in·n_coeffs)
        )
        return base_out + spline_out   # (B, out)

    # ── Adaptive grid update ──────────────────────────────────────────────────
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Refit the B-spline knot positions to cover the actual distribution of x,
        then refit spline coefficients so the layer's output is preserved.

        The grid is blended between fully adaptive (quantile-based, grid_eps=0)
        and fully uniform (grid_eps=1).

        Must be called with float32 input — lstsq does not support float16.
        """
        x = x.float()
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # ── Snapshot current spline output before the grid moves ──────────
        splines       = self._b_splines(x)                            # (B, in, n_coeffs)
        orig_coeff    = self._scaled_spline_weight.permute(1, 2, 0)   # (in, n_coeffs, out)
        unreduced_out = torch.bmm(
            splines.permute(1, 0, 2),   # (in, B, n_coeffs)
            orig_coeff,                  # (in, n_coeffs, out)
        ).permute(1, 0, 2)              # (B, in, out)  — what the layer was producing

        # ── Build new grid ────────────────────────────────────────────────
        x_sorted      = torch.sort(x, dim=0)[0]
        idx           = torch.linspace(0, batch - 1, self.grid_size + 1,
                                       dtype=torch.int64, device=x.device)
        grid_adaptive = x_sorted[idx]                                  # (G+1, in)

        step         = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        arange       = torch.arange(self.grid_size + 1,
                                    dtype=x.dtype, device=x.device).unsqueeze(1)
        grid_uniform = arange * step + x_sorted[0] - margin           # (G+1, in)

        # Blend adaptive and uniform
        grid = self.grid_eps * grid_uniform + (1.0 - self.grid_eps) * grid_adaptive

        # Extend by spline_order knots on each side
        lo_ext = grid[:1]  - step * torch.arange(
            self.spline_order, 0, -1, dtype=x.dtype, device=x.device
        ).unsqueeze(1)
        hi_ext = grid[-1:] + step * torch.arange(
            1, self.spline_order + 1, dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        grid_full = torch.cat([lo_ext, grid, hi_ext], dim=0)           # (G+2k+1, in)
        self.grid.copy_(grid_full.T.contiguous())                       # (in, G+2k+1)

        # ── Refit coefficients to preserve outputs on the new grid ────────
        self.spline_weight.data.copy_(self._curve2coeff(x, unreduced_out))

    # ── Regularization (Liu et al. §2.5) ─────────────────────────────────────
    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy:    float = 1.0,
    ) -> torch.Tensor:
        """
        L1 encourages sparse edge activations.
        Entropy encourages each edge to rely on one dominant spline basis.
        Both are computed on spline_weight as a proxy for activation magnitude.
        """
        w       = self.spline_weight.abs()
        l1      = w.mean(dim=-1)                                       # (out, in)
        p       = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1)                  # (out, in)
        return (regularize_activation * l1 + regularize_entropy * entropy).mean()


class KAN(nn.Module):
    """
    Stack of KANLinear layers.

    Example
    -------
        KAN([1536, 512, 256, 1])
        → KANLinear(1536→512), KANLinear(512→256), KANLinear(256→1)

    Calling forward(x, update_grid=True) propagates update_grid() through
    every layer in sequence using the intermediate activations as input.
    """

    def __init__(
        self,
        layers_hidden:  list,
        grid_size:      int   = KAN_GRID_SIZE,
        spline_order:   int   = KAN_SPLINE_ORDER,
        scale_noise:    float = KAN_SCALE_NOISE,
        scale_base:     float = KAN_SCALE_BASE,
        scale_spline:   float = KAN_SCALE_SPLINE,
        base_activation       = nn.SiLU,
        grid_eps:       float = KAN_GRID_EPS,
        grid_range:     list  = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = list(KAN_GRID_RANGE)

        self.layers = nn.ModuleList([
            KANLinear(
                in_f, out_f,
                grid_size=grid_size, spline_order=spline_order,
                scale_noise=scale_noise, scale_base=scale_base,
                scale_spline=scale_spline, base_activation=base_activation,
                grid_eps=grid_eps, grid_range=list(grid_range),
            )
            for in_f, out_f in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        for layer in self.layers:
            if update_grid:
                # Each layer updates its grid using its own input,
                # then produces output that becomes the next layer's input.
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(
        self,
        regularize_activation: float = KAN_REG_ACTIVATION,
        regularize_entropy:    float = KAN_REG_ENTROPY,
    ) -> torch.Tensor:
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class NodulePatchDataset(Dataset):
    def __init__(self, metadata_df, data_dir, transforms=None):
        self.metadata   = metadata_df.reset_index(drop=True)
        self.data_dir   = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row        = self.metadata.iloc[idx]
        filename   = os.path.basename(row["filepath"])
        split      = row["split"]
        label      = int(row["label"])
        subfolder  = "pos" if label == 1 else "neg"
        local_path = os.path.join(self.data_dir, split, subfolder, filename)
        patch      = np.load(local_path).astype(np.float32)
        patch      = np.expand_dims(patch, axis=0)    # (1, 64, 64, 64)
        if self.transforms is not None:
            patch = self.transforms(patch)
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch)
        return patch, torch.tensor(label, dtype=torch.float32), row["seriesuid"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

class HybridTwoBranch(nn.Module):
    """
    Two frozen 3-D CNN backbones whose feature vectors are concatenated and fed
    into a KAN head (the only part that is trained).

    Architecture
    ────────────
    ResNet18    → (B, 512)  ─┐
                              ├─ cat → (B, 1536) → KAN[1536→512→256→1]
    DenseNet121 → (B, 1024) ─┘

    NOTE ON MIXED PRECISION
    ───────────────────────
    The backbones run under autocast (float16/bfloat16 is fine for them).
    The KAN head is forced to float32 because:
      • The B-spline basis evaluation uses comparisons sensitive to rounding
      • torch.linalg.lstsq (used in grid updates) is not supported in float16
    We handle this by casting the concatenated features to float32 before the
    KAN head and wrapping the KAN call with autocast(enabled=False).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()

        # ── Branch 1: ResNet18 (frozen) ──────────────────────────────────────
        self.resnet18 = ResNet(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=RESNET18_FEATURE_DIM,
        )
        self.resnet18.fc = nn.Identity()    # output → (B, 512)

        if RESNET18_WEIGHTS:
            load_backbone_weights(self.resnet18, RESNET18_WEIGHTS, "ResNet18")
        else:
            print("  ⚠ RESNET18_WEIGHTS empty — backbone initialised randomly")
        for p in self.resnet18.parameters():
            p.requires_grad = False

        # ── Branch 2: DenseNet121 (frozen) ───────────────────────────────────
        self.densenet121 = DenseNet121(
            spatial_dims=3, in_channels=in_channels, out_channels=1,
        )
        self.densenet121.class_layers.out = nn.Identity()   # output → (B, 1024)

        if DENSENET121_WEIGHTS:
            load_backbone_weights(self.densenet121, DENSENET121_WEIGHTS, "DenseNet121")
        else:
            print("  ⚠ DENSENET121_WEIGHTS empty — backbone initialised randomly")
        for p in self.densenet121.parameters():
            p.requires_grad = False

        # ── KAN head (the only trained component) ────────────────────────────
        # 1536 → 512 → 256 → 1
        # Parameter count (default grid_size=5, order=3):
        #   Layer 0 (1536→512): 512·1536·(8+1+1) ≈ 7.9 M
        #   Layer 1  (512→256): 256· 512·(8+1+1) ≈ 1.3 M
        #   Layer 2   (256→1 ): 1  · 256·(8+1+1) ≈ 2.6 K
        #   Total ≈ 9.2 M  (vs ≈ 0.9 M for the MLP head)
        self.kan_head = KAN(layers_hidden=[TOTAL_FEATURE_DIM, 512, 256, num_classes])

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f_r = self.resnet18(x)       # (B, 512)
            f_d = self.densenet121(x)    # (B, 1024)

        # Cast to float32 and disable autocast so the KAN B-spline math is stable
        combined = torch.cat([f_r, f_d], dim=1).float()    # (B, 1536)
        with torch.amp.autocast("cuda", enabled=False):
            out = self.kan_head(combined)
        return out.squeeze(1)

    # ── Backbone freeze / unfreeze helpers ────────────────────────────────────
    def freeze_backbones(self):
        for p in self.resnet18.parameters():    p.requires_grad = False
        for p in self.densenet121.parameters(): p.requires_grad = False

    def unfreeze_backbones(self):
        for p in self.resnet18.parameters():    p.requires_grad = True
        for p in self.densenet121.parameters(): p.requires_grad = True

    # ── KAN grid update ────────────────────────────────────────────────────────
    @torch.no_grad()
    def update_kan_grids(
        self,
        loader,
        device,
        n_batches: int = KAN_GRID_UPDATE_BATCHES,
    ):
        """
        Collect backbone feature vectors from n_batches of training data, then
        call update_grid() on every KANLinear layer so the spline knots match
        the real feature distribution.

        Call this:
          • ONCE before training starts (grids are initially uniform on [-1,1])
          • Every KAN_GRID_UPDATE_EVERY epochs up to KAN_GRID_UPDATE_UNTIL
        """
        self.resnet18.eval()
        self.densenet121.eval()
        feats = []
        for i, (patches, _, _) in enumerate(loader):
            if i >= n_batches:
                break
            patches = patches.to(device)
            f_r = self.resnet18(patches).float()
            f_d = self.densenet121(patches).float()
            feats.append(torch.cat([f_r, f_d], dim=1).cpu())
        features = torch.cat(feats, dim=0).to(device)
        # update_grid=True: each KANLinear updates on its input, then forwards
        self.kan_head(features, update_grid=True)
        print(f"  ✓ KAN grids updated using {features.size(0)} feature vectors")


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation  (identical logic to frozen-MLP version, output filenames changed)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_candidate_froc(all_labels, all_probs, total_scans):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    total_positives = (all_labels == 1).sum()
    desc_idx      = np.argsort(all_probs)[::-1]
    sorted_labels = all_labels[desc_idx]
    tps           = np.cumsum(sorted_labels == 1)
    fps           = np.cumsum(sorted_labels == 0)
    fps_per_scan  = fps / total_scans
    sensitivity   = tps / total_positives
    froc_scores   = {}
    for target in FROC_THRESHOLDS:
        valid = np.where(fps_per_scan <= target)[0]
        froc_scores[f"{target} FP/scan"] = sensitivity[valid[-1]] if len(valid) > 0 else 0.0
    return froc_scores


def calculate_95_ci(y_true, y_probs, n_bootstraps=1000):
    rng = np.random.RandomState(SEED)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(np.array(y_true)[idx])) < 2:
            continue
        scores.append(roc_auc_score(np.array(y_true)[idx], np.array(y_probs)[idx]))
    scores = np.sort(np.array(scores))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def evaluate_model(loader, model, device, desc="Validating", return_preds=False):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))

    with torch.no_grad():
        for patches, labels, uids in tqdm(loader, desc=desc):
            patches, labels = patches.to(device), labels.to(device)
            with autocast("cuda"):
                logits = model(patches)
                loss   = criterion(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs  = [float(probs)]
                labels = [float(labels.cpu())]
            else:
                probs  = probs.tolist()
                labels = labels.cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_uids.extend(uids)
            running_loss += loss.item() * patches.size(0)

    # Patient-level aggregation: max probability per scan
    patient_dict = {}
    for prob, label, uid in zip(all_probs, all_labels, all_uids):
        if uid not in patient_dict:
            patient_dict[uid] = {"prob": prob, "label": label}
        else:
            patient_dict[uid]["prob"]  = max(patient_dict[uid]["prob"],  prob)
            patient_dict[uid]["label"] = max(patient_dict[uid]["label"], label)

    y_true_pat  = [v["label"] for v in patient_dict.values()]
    y_probs_pat = [v["prob"]  for v in patient_dict.values()]
    y_pred_pat  = [1 if p >= 0.5 else 0 for p in y_probs_pat]
    total_scans = len(patient_dict)

    avg_loss = running_loss / len(loader.dataset)
    auc   = roc_auc_score(y_true_pat, y_probs_pat) if len(np.unique(y_true_pat)) > 1 else 0.5
    auprc = average_precision_score(y_true_pat, y_probs_pat)
    f1    = f1_score(y_true_pat, y_pred_pat)
    sens  = recall_score(y_true_pat, y_pred_pat)
    cm    = confusion_matrix(y_true_pat, y_pred_pat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    froc  = calculate_candidate_froc(all_labels, all_probs, total_scans)

    if return_preds:
        ci_lo, ci_hi = calculate_95_ci(y_true_pat, y_probs_pat)
        print(f"  AUROC 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        frac_pos, mean_pred = calibration_curve(y_true_pat, y_probs_pat, n_bins=10)
        pd.DataFrame({
            "mean_predicted_probability": mean_pred,
            "fraction_of_positives":      frac_pos,
        }).to_csv("hybrid2_kan_calibration_curve.csv", index=False)
        pred_df = pd.DataFrame({
            "seriesuid":   all_uids,
            "label":       all_labels,
            "probability": all_probs,
        })
        return avg_loss, auc, auprc, f1, sens, spec, froc, pred_df
    return avg_loss, auc, auprc, f1, sens, spec, froc


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    metadata = pd.read_csv(METADATA_PATH)
    if "split" not in metadata.columns:
        split_df   = pd.read_csv(PATIENT_SPLIT_PATH)
        split_dict = dict(zip(split_df["seriesuid"], split_df["split"]))
        metadata["split"] = metadata["seriesuid"].map(split_dict)

    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta   = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta  = metadata[metadata["split"] == "test"].reset_index(drop=True)
    print(f"Train: {len(train_meta):,}  Val: {len(val_meta):,}  Test: {len(test_meta):,}")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transforms = Compose([
        RandAffine(prob=0.8, translate_range=(5, 5, 5), padding_mode="zeros", spatial_size=None),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandGaussianNoise(prob=0.2, std=0.01),
    ])

    train_ds = NodulePatchDataset(train_meta, DATA_DIR, transforms=train_transforms)
    val_ds   = NodulePatchDataset(val_meta,   DATA_DIR, transforms=None)
    test_ds  = NodulePatchDataset(test_meta,  DATA_DIR, transforms=None)

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                         pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HybridTwoBranch().to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p    = total_p - trainable_p
    kan_p       = sum(p.numel() for p in model.kan_head.parameters())
    print(f"\n2-Branch Hybrid (Frozen Backbones + KAN Head)")
    print(f"  Total parameters      : {total_p:,}")
    print(f"  Trainable (KAN head)  : {trainable_p:,}  ({kan_p:,} in KAN layers)")
    print(f"  Frozen (backbones)    : {frozen_p:,}")
    for i, layer in enumerate(model.kan_head.layers):
        lp = sum(p.numel() for p in layer.parameters())
        print(f"    KAN layer {i}: {layer.in_features}→{layer.out_features}  ({lp:,} params)")
    print()

    # ── Training setup ────────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    warmup_epochs    = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - warmup_epochs))
    scheduler        = SequentialLR(optimizer,
                                    schedulers=[warmup_scheduler, cosine_scheduler],
                                    milestones=[warmup_epochs])
    scaler = GradScaler("cuda")

    # ── Initial KAN grid update ───────────────────────────────────────────────
    # The default grid is uniform on [-1, 1], but backbone features are unlikely
    # to be normalised to that range.  Adapt before the first gradient step.
    print("Performing initial KAN grid update…")
    model.update_kan_grids(train_loader, device)

    best_val_auc     = 0.0
    patience_counter = 0
    history          = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        # Frozen branches stay in eval mode so BN running stats don't drift
        model.resnet18.eval()
        model.densenet121.eval()

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]")
        for patches, labels, _ in pbar:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(patches)
                loss   = criterion(logits, labels)
                # Optional KAN spline regularization (both are 0 by default)
                if KAN_REG_ACTIVATION > 0 or KAN_REG_ENTROPY > 0:
                    loss = loss + model.kan_head.regularization_loss()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * patches.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader.dataset)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        val_loss, val_auc, val_auprc, val_f1, val_sens, val_spec, val_froc = \
            evaluate_model(val_loader, model, device, desc="Validating")

        history.append({
            "epoch": epoch, "lr": current_lr, "train_loss": avg_train_loss,
            "val_loss": val_loss, "val_auc": val_auc, "val_auprc": val_auprc,
            "val_f1": val_f1, "val_sensitivity": val_sens, "val_specificity": val_spec,
        })

        print(f"\nEpoch {epoch:3d} │ LR {current_lr:.2e} │ "
              f"Train Loss {avg_train_loss:.4f} │ Val Loss {val_loss:.4f}")
        print(f"         │ AUROC {val_auc:.4f} │ AUPRC {val_auprc:.4f} │ "
              f"F1 {val_f1:.4f} │ Sens {val_sens:.4f} │ Spec {val_spec:.4f}")

        # Periodic KAN grid update (early epochs only)
        if epoch <= KAN_GRID_UPDATE_UNTIL and epoch % KAN_GRID_UPDATE_EVERY == 0:
            print(f"  Updating KAN grids (epoch {epoch})…")
            model.update_kan_grids(train_loader, device)

        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_hybrid2_kan.pth")
            print(f"  ✓ [Saved] New best val AUROC: {best_val_auc:.4f}")
            with open("best_metrics_hybrid2_kan.json", "w") as f:
                json.dump({
                    "epoch": epoch, "auc": val_auc, "auprc": val_auprc,
                    "f1": val_f1, "sensitivity": val_sens, "specificity": val_spec,
                }, f, indent=4)
        else:
            patience_counter += 1
            print(f"  · No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early stopping after epoch {epoch} ***")
                break

    pd.DataFrame(history).to_csv("hybrid2_kan_training_log.csv", index=False)

    # ── Final test ────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load("best_model_hybrid2_kan.pth",
                                     map_location=device, weights_only=True))
    test_loss, test_auc, test_auprc, test_f1, test_sens, test_spec, test_froc, pred_df = \
        evaluate_model(test_loader, model, device, desc="Testing", return_preds=True)
    pred_df.to_csv("hybrid2_kan_test_predictions.csv", index=False)

    print("\n─── 2-BRANCH HYBRID (KAN HEAD) TEST RESULTS ───")
    print(f"  AUROC: {test_auc:.4f}   AUPRC: {test_auprc:.4f}")
    print(f"  F1: {test_f1:.4f}   Sensitivity: {test_sens:.4f}   Specificity: {test_spec:.4f}")
    print("FROC:")
    for k, v in test_froc.items():
        print(f"  {k}: {v:.4f}")
    print("═" * 40)


if __name__ == "__main__":
    main()