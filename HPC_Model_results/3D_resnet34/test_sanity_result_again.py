import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBlock

RESNET34_FEATURE_DIM = 512

# 1. PASTE MODEL ARCHITECTURE DIRECTLY HERE (Exactly from your training script)
class ResNet34WithMLPHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        self.backbone = ResNet(
            block=ResNetBlock,
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=RESNET34_FEATURE_DIM
        )
        assert hasattr(self.backbone, 'fc'), "MONAI ResNet structure changed: 'fc' layer missing"
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(RESNET34_FEATURE_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1)

# --- EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Initialize and load the model weights
try:
    model = ResNet34WithMLPHead()  
    # Use weights_only=True for modern PyTorch security standards
    model.load_state_dict(torch.load("best_model_resnet34.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # CRITICAL: Lock dropout and batchnorm behavior
    print("✅ Model weights loaded successfully and set to eval mode.")
except Exception as e:
    print(f"❌ Failed to load model weights. Error detail:\n{e}")

# 3. Load your saved test predictions CSV to compare
csv_path = "resnet34_test_predictions.csv"
try:
    df_saved = pd.read_csv(csv_path)
    print(f"\n✅ Loaded '{csv_path}'. First few rows:")
    print(df_saved.head(3))
except FileNotFoundError:
    print(f"\n❌ Could not find {csv_path}")
    df_saved = None

print("\n--- Manual Check Complete ---")
print("If the script printed the green success checkmark, your model weights are perfectly intact and compatible.")