import pandas as pd
import numpy as np

# Same thresholds used in training code
FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Load saved predictions
pred_df = pd.read_csv("densenet121_test_predictions.csv")

all_labels = pred_df["label"].values
all_probs = pred_df["probability"].values

# Number of unique CT scans
total_scans = pred_df["seriesuid"].nunique()

# Total positive nodules
total_positives = np.sum(all_labels == 1)

# Sort by confidence
descending_indices = np.argsort(all_probs)[::-1]

sorted_probs = all_probs[descending_indices]
sorted_labels = all_labels[descending_indices]

# Cumulative TP and FP
tps = np.cumsum(sorted_labels == 1)
fps = np.cumsum(sorted_labels == 0)

fps_per_scan = fps / total_scans
sensitivity = tps / total_positives

froc_scores = {}

for target in FROC_THRESHOLDS:
    valid_idx = np.where(fps_per_scan <= target)[0]

    if len(valid_idx) > 0:
        sens = sensitivity[valid_idx[-1]]
    else:
        sens = 0.0

    froc_scores[target] = sens

# Print Results
print("\n===== FROC RESULTS =====")

for fp_rate, sens in froc_scores.items():
    print(f"{fp_rate:>5} FP/scan : {sens:.4f}")

print("========================\n")

# Save CSV for plotting
froc_df = pd.DataFrame({
    "FP_per_scan": list(froc_scores.keys()),
    "Sensitivity": list(froc_scores.values())
})

froc_df.to_csv(
    "densenet121_froc.csv",
    index=False
)

print("Saved: densenet121_froc.csv")

# ---- CPM (Competition Performance Metric) ----
cpm = np.mean(list(froc_scores.values()))

print("===== CPM SCORE =====")
print(f"CPM: {cpm:.4f}")
print("=====================\n")