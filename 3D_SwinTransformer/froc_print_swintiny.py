import pandas as pd
import numpy as np

FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

pred_df = pd.read_csv("swin3d_tiny_test_predictions.csv")

labels = pred_df["label"].values
probs = pred_df["probability"].values

num_scans = pred_df["seriesuid"].nunique()
total_pos = np.sum(labels == 1)

# sort by confidence
idx = np.argsort(-probs)
labels = labels[idx]

tp = np.cumsum(labels == 1)
fp = np.cumsum(labels == 0)

fps_per_scan = fp / num_scans
sens = tp / total_pos

froc_scores = {}

for thr in FROC_THRESHOLDS:
    valid = np.where(fps_per_scan <= thr)[0]
    
    if len(valid) == 0:
        froc_scores[thr] = 0.0
    else:
        froc_scores[thr] = float(np.max(sens[valid]))

print("\n===== FROC RESULTS =====")
for k, v in froc_scores.items():
    print(f"{k:>5} FP/scan : {v:.4f}")
print("========================\n")

pd.DataFrame({
    "FP_per_scan": list(froc_scores.keys()),
    "Sensitivity": list(froc_scores.values())
}).to_csv("swin3d_tiny_froc.csv", index=False)

print("Saved: swin3d_tiny_froc.csv")

# ---- CPM (Competition Performance Metric) ----
cpm = np.mean(list(froc_scores.values()))

print("===== CPM SCORE =====")
print(f"CPM: {cpm:.4f}")
print("=====================\n")