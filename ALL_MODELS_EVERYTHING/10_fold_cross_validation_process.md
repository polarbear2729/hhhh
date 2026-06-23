# 10-Fold Cross-Validation Implementation

## Why We Did This

Our original evaluation used a single **70/15/15** train/val/test split. The problem:

- The test set (15%) is just **one random draw**
- Other papers using the same dataset might have **easier or harder cases** in their test set even with the same split ratio
- Comparing results across papers becomes **unfair** — your high scores might just be luck

**10-fold cross-validation solves this** by testing every sample exactly once across 10 different train/test splits, then reporting the **mean ± standard deviation**. This is the standard evaluation protocol for LUNA16.

---

## How 10-Fold CV Works

### The LUNA16 Dataset

LUNA16 comes pre-divided into **10 subsets** (subset0–subset9). These were created stratified by the original challenge organizers, so class balance is consistent across all subsets (~13–14% positive). **No shuffling needed.**

| Subset | Negative | Positive | % Positive |
|--------|:--------:|:--------:|:----------:|
| subset0 | 670 | 112 | 14.3% |
| subset1 | 780 | 128 | 14.1% |
| subset2 | 805 | 128 | 13.7% |
| subset3 | 715 | 119 | 14.3% |
| subset4 | 775 | 128 | 14.2% |
| subset5 | 715 | 108 | 13.1% |
| subset6 | 775 | 129 | 14.3% |
| subset7 | 730 | 111 | 13.2% |
| subset8 | 730 | 118 | 13.9% |
| subset9 | 670 | 105 | 13.5% |

These subsets are stored in the `subset` column of `metadata_all.csv`.

### Fold Assignment

For each fold k (0 through 9):

| Fold | Test (1 subset) | Validate (1 subset) | Train (8 subsets) |
|:----:|:---------------:|:-------------------:|:-----------------:|
| 0 | subset0 | subset9 | subsets 1–8 |
| 1 | subset1 | subset9 | subsets 0, 2–8 |
| 2 | subset2 | subset9 | subsets 0–1, 3–8 |
| 3 | subset3 | subset9 | subsets 0–2, 4–8 |
| 4 | subset4 | subset9 | subsets 0–3, 5–8 |
| 5 | subset5 | subset9 | subsets 0–4, 6–8 |
| 6 | subset6 | subset9 | subsets 0–5, 7–8 |
| 7 | subset7 | subset9 | subsets 0–6, 8 |
| 8 | subset8 | subset9 | subsets 0–7 |
| 9 | subset9 | subset8 | subsets 0–7 |

**Every sample gets tested exactly once.** The validation set is always subset9 (or subset8 for fold 9) to keep the 8/1/1 split consistent.

### What We Average

After all 10 folds, we compute **mean ± std** for:
- AUROC
- AUPRC
- F1 Score
- Sensitivity
- Specificity
- FROC at each FP/scan level (0.125, 0.25, 0.5, 1, 2, 4, 8)

---

## Code Structure

```
scripts/Sadit/10_fold_CV/
├── __init__.py
├── data.py                          # CV dataset, fold loaders, transforms
├── evaluate.py                      # Shared evaluation metrics
├── run_2cnn_mlp_frozen.py           # 2-branch hybrid (ResNet18 + DenseNet121 + MLP head)
├── run_2cnn_kan_frozen.py           # 2-branch hybrid (ResNet18 + DenseNet121 + KAN head)
├── run_3cnn_mlp_frozen.py           # 3-branch hybrid (ResNet18 + DenseNet121 + EfficientNet + MLP head)
├── run_3cnn_kan_frozen.py           # 2-branch hybrid + KAN head (mirrors original 3CNN_KAN code)
├── run_resnet18_10fold.py           # Standalone 3D ResNet18
├── run_densenet121_10fold.py        # Standalone 3D DenseNet121
├── run_efficientnetb0_10fold.py     # Standalone 3D EfficientNet-B0
├── run_all_frozen.bat               # Batch runner: all 4 frozen hybrids
├── run_all_models.bat               # Batch runner: all 7 models
└── results/
    ├── summary_all_frozen_10fold.txt       # Results for all frozen hybrids
    ├── summary_all_models_10fold.txt       # Results for all 7 models
    ├── {model_name}_10fold_results.txt     # Per-model detailed results
    └── {model_name}_per_fold_metrics.csv   # Per-fold raw metrics
```

---

## Key Implementation Details

### 1. Data Loading (`data.py`)

The original dataset loads files from fixed folders (`data/train/`, `data/val/`, `data/test/`) based on a `split` column. For 10-fold CV, we reassign which samples are train/val/test per fold, so the files may be in any of the three folders.

**Solution:** `CVPatchDataset` searches all three split folders for each `.npy` file:

```python
class CVPatchDataset(Dataset):
    def __getitem__(self, idx):
        ...
        for split_folder in ["train", "val", "test"]:
            candidate = os.path.join(self.data_dir, split_folder, subfolder, filename)
            if os.path.exists(candidate):
                local_path = candidate
                break
```

This allows any sample to be loaded regardless of its original split assignment.

### 2. Fold Split Generation (`data.py`)

The `get_fold_loaders()` function assigns samples to train/val/test based only on the `subset` column, completely ignoring the original `split` column:

```python
def get_fold_loaders(metadata, fold, transforms=None):
    test_mask = metadata["subset"] == f"subset{fold}"
    val_subset = "subset9" if fold != 9 else "subset8"
    val_mask = metadata["subset"] == val_subset
    train_mask = ~test_mask & ~val_mask
    ...
```

### 3. Evaluation (`evaluate.py`)

A shared `evaluate_model()` function handles all models identically:

- **Scan-level metrics**: Max-probability aggregation per `seriesuid`
- **Candidate-level metrics**: Used for FROC computation (all patches kept)
- **Metrics computed**: AUROC, AUPRC, F1, Sensitivity, Specificity, FROC
- **Loss**: BCEWithLogitsLoss with positive weight = 5115/822

### 4. Training Loop

Each model script follows the same pattern:

```python
for fold in range(10):
    train_loader, val_loader, test_loader = get_fold_loaders(metadata, fold)
    model = ModelClass()
    optimizer = AdamW(...)
    scheduler = LinearWarmup → CosineAnnealing
    scaler = GradScaler()

    for epoch in range(MAX_EPOCHS):
        train one epoch
        validate
        if val_auc improved: reset patience
        else: patience += 1; if patience >= 15: early stop

    test on held-out fold
    save fold metrics

aggregate and save results
```

### 5. Frozen Hybrid Models vs Standalone Models

| Aspect | Frozen Hybrids | Standalone 3D Models |
|--------|:--------------:|:--------------------:|
| **What trains** | Only MLP/KAN head (backbones frozen) | Entire network |
| **Trainable params** | ~1M (MLP) / ~9M (KAN) | ~4M–11M |
| **Time per fold** | ~10 minutes | ~45 minutes |
| **Backbone modes** | Backbones forced to `.eval()` in training | Normal `.train()` |
| **Pretrained weights** | Loaded from saved .pth files | Random init |

### 6. KAN-Specific Logic

The KAN models (`run_2cnn_kan_frozen.py` and `run_3cnn_kan_frozen.py`) have extra steps:

- **Initial grid update**: Before training starts, the KAN spline grids are adapted to the real feature distribution
- **Periodic grid updates**: Every 5 epochs (up to epoch 20), grids are refitted
- **Float32 precision**: KAN forward pass runs with `autocast(enabled=False)` because B-spline math and `torch.linalg.lstsq` require float32

```python
# Before training
model.update_kan_grids(train_loader, device)

# During training (periodic)
if epoch <= KAN_GRID_UPDATE_UNTIL and epoch % KAN_GRID_UPDATE_EVERY == 0:
    model.update_kan_grids(train_loader, device)
```

### 7. Results Aggregation

After all 10 folds, each script:

1. Computes **mean ± std** for each metric across 10 folds
2. Saves per-fold values in CSV format
3. Appends to a shared `summary_all_models_10fold.txt` file
4. Prints a formatted table to console

Example output:

```
  AUROC      : 0.9923 ± 0.0047
  AUPRC      : 0.9961 ± 0.0020
  F1         : 0.9875 ± 0.0058
  SENSITIVITY: 0.9840 ± 0.0075
  SPECIFICITY: 0.9902 ± 0.0068

  FROC:
    0.125 FP/scan: 0.9450 ± 0.0120
    0.25 FP/scan : 0.9620 ± 0.0095
    0.5 FP/scan  : 0.9850 ± 0.0060
    1 FP/scan    : 0.9890 ± 0.0050
    2 FP/scan    : 0.9940 ± 0.0035
    4 FP/scan    : 0.9970 ± 0.0020
    8 FP/scan    : 1.0000 ± 0.0000
```

---

## Why This Is Better Than a Single Split

| Single 70/15/15 Split | 10-Fold CV |
|-----------------------|:----------:|
| One test set (15% of data) | Every sample tested exactly once |
| Results depend on which samples land in test | Results averaged over 10 different test sets |
| High variance across different random splits | Low variance (mean ± std reported) |
| Unfair comparison with other papers | Fair comparison — everyone uses same 10 LUNA16 subsets |
| Cannot detect overfitting to the specific split | Overfitting apparent from high std across folds |

---

## How to Run

```bash
cd scripts/Sadit/10_fold_CV

# Run all 7 models sequentially (overnight):
.\run_all_models.bat

# Run just the 4 frozen hybrids (~3 hours):
.\run_all_frozen.bat

# Run individual models:
python run_2cnn_mlp_frozen.py
python run_resnet18_10fold.py
python run_densenet121_10fold.py
python run_efficientnetb0_10fold.py
```

All results are saved to `results/` subfolder. Open `summary_all_models_10fold.txt` for the complete comparison.
