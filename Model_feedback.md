---

## **COMPREHENSIVE EVALUATION OF YOUR LUNA16 MODELS**

### **TEST RESULTS SUMMARY TABLE**

| Model | Test AUROC | Test AUPRC | Test F1 | Sensitivity | Specificity | CPM | Epochs Trained | Status |
|-------|-----------|-----------|---------|-------------|-------------|-----|---------------|--------|
| **3D DenseNet121** | **0.9911** | **0.9964** | **0.9727** | **0.9780** | **0.9302** | ~0.96 | 43 | ✅ Excellent |
| **3D ResNet34** | **0.9885** | **0.9953** | **0.9424** | **0.9890** | **0.7674** | ~0.96 | 33 | ✅ Excellent |
| **3D EfficientNet-B0** | **0.9959** | **0.9983** | **0.9574** | **0.9890** | **0.8372** | ~0.98 | 58 | ✅ Best Overall |
| **3D Swin-Tiny** | **0.9677** | **0.9875** | **0.9036** | **0.8242** | **1.0000** | **0.9012** | 94 | ✅ Good |
| **3D ViT-Small** | **0.7112** (no pretrain) / **0.7147** (pretrain) | **0.8040** / **0.8058** | **0.8089** | **1.0000** | **0.0000** | **0.3090** / **0.2965** | 18 | ❌ **FAILED** |
| **2D TinyViT** | **0.9451** | **0.9766** | **0.9081** | **0.9231** | **0.7674** | N/A | 28 | ⚠️ Mediocre for 2D |

---

## **1. WHICH VALUES WERE USED AND WHY**

### **A. Common Values Across All Models**

| Parameter | Value | Why It Was Chosen |
|-----------|-------|-------------------|
| **SEED = 42** | Fixed random seed | Ensures reproducibility. Same train/val/test splits, same weight initialization, same augmentation every run. Critical for scientific validity. |
| **BATCH_SIZE = 8** | 8 patches per batch | Limited by GPU memory (3D volumes are large: 64×64×64). With 8GB-16GB VRAM, 8 is the safe maximum for 3D models. Your DenseNet uses ~45MB model file, ResNet ~248MB. |
| **NUM_WORKERS = 4** | 4 CPU workers for data loading | Keeps GPU fed without CPU bottleneck. Windows sometimes has issues with higher values. |
| **PIN_MEMORY = True** | Tensor pinned in CPU RAM | Faster CPU→GPU transfer. Standard practice. |
| **PERSISTENT_WORKERS = True** | Workers stay alive between epochs | Avoids worker spawn overhead. Saves ~2-3 seconds per epoch. |
| **POS_WEIGHT = 5115/822 = 6.22** | Class imbalance correction | Your dataset has ~6.2× more negative patches than positive. Without this, the model would predict "negative" for everything and get 86% accuracy while being useless. |
| **FROC_THRESHOLDS** | [0.125, 0.25, 0.5, 1, 2, 4, 8] | Standard LUNA16 evaluation protocol. These are false positives per scan thresholds used in the official challenge. |
| **EARLY_STOPPING_PATIENCE = 15** | Wait 15 epochs for improvement | Prevents overfitting. If validation AUROC doesn't improve for 15 epochs, training stops. |

### **B. Why These Specific Values Make Sense for Medical Data**

- **POS_WEIGHT = 6.22**: In cancer detection, false negatives (missing cancer) are far more dangerous than false positives. The weight forces the model to pay 6× more attention to positive (cancer) cases. This is standard in medical imaging.

- **Patient-level aggregation (max-prob per seriesuid)**: A patient may have multiple nodules. You take the maximum probability across all their patches. This is clinically correct — if ANY nodule is malignant, the patient needs follow-up.

- **Fixed 0.5 threshold**: For reporting consistency. In practice, you'd use Youden's J statistic (which your Swin and ViT scripts compute), but 0.5 is the neutral baseline.

---

## **2. WHY MODELS FAILED — DETAILED FAILURE ANALYSIS**

### **❌ MODEL THAT FAILED: 3D ViT-Small**

**What happened:**
- Test AUROC: **0.7112** (barely better than random guessing at 0.5)
- Sensitivity: **1.0000** but Specificity: **0.0000** at threshold 0.5
- The model predicts ALL samples as positive — it's essentially broken

**Root Causes from the Code and Logs:**

| Problem | Evidence from Your Logs | Why It Happened |
|---------|------------------------|-----------------|
| **1. No actual learning** | Val AUROC oscillates between 0.55-0.70 for all 18 epochs | The model never learned to distinguish positive from negative |
| **2. All predictions collapse to ~0.5** | `pos: mean=0.5077 std=0.0678` vs `neg: mean=0.4981 std=0.0760` | The model outputs nearly identical probabilities for both classes |
| **3. "Pretrained" weights don't exist** | Code says `USE_PRETRAINED = True` but prints: `"[WARN] use_pretrained=True but no weight path configured."` | You set pretrained=True but provided NO actual pretrained weights. The model trains from scratch with random initialization. |
| **4. No intensity normalization** | Your ViT code has NO `NormalizeIntensity` transform (unlike Swin which has it) | Raw CT values (Hounsfield Units, roughly -1000 to +1000) have extreme ranges. Transformers are sensitive to input scale. CNNs handle this better via BatchNorm layers. |
| **5. Wrong architecture choices** | `dropout_rate=0.0`, `pos_embed_type='learnable'` (changed from 'sincos' due to NaN) | No regularization + the "fix" for NaN may have hurt representation learning |

**Why the "Fixes" Didn't Actually Fix It:**

- **FIX 1 (learnable pos embed instead of sincos)**: Sincos was causing NaN in float16. Learnable embeddings avoid NaN but may not capture spatial relationships as well, especially with tiny 64³ volumes.
- **FIX 2 (mean pooling instead of CLS token)**: MONAI ViT doesn't use CLS token. Mean pooling 64 patch tokens dilutes any strong signal — like averaging 64 weak classifiers.
- **FIX 3 (LR reduced to 1e-5)**: This is EXTREMELY low for training from scratch. With only ~6K training samples and 30M parameters, the model needed stronger signals.
- **FIX 4 (gradient clipping)**: The gradients were clipping every epoch (`max=439`, `scale=256` then `8192`). This indicates the loss landscape is unstable.

**The Fundamental Problem:**

Transformers need **MASSIVE data** or **pretrained weights**. Your ViT-Small has **30 million parameters** trained on only **~6,000 patches** (822 positive, 5,115 negative). That's **~5,000 parameters per training sample** — severe overparameterization.

Compare to CNNs:
- DenseNet121: ~11.5M params, works because of dense skip connections
- ResNet34: ~63M params, works because of residual learning
- EfficientNet-B0: ~8M params, works because of compound scaling

Your ViT has **30M params with NO inductive bias** (no locality, no translation invariance built-in). It needs 100× more data.

---

### **⚠️ MODEL THAT UNDERPERFORMED: 2D TinyViT**

**Test AUROC: 0.9451** — sounds good, but let's be honest about the problems:

| Issue | Explanation |
|-------|-------------|
| **Lost 3D spatial information** | You took only 3 central axial slices (centre-1, centre, centre+1). A 64×64×64 nodule has rich 3D structure — z-axis texture, shape in coronal/sagittal views. You threw away 61/64 of the volume. |
| **Pretrained=False** | `pretrained=False` in `timm.create_model()`. No ImageNet transfer learning. Training from scratch on 3 slices. |
| **Resized 64→224** | Bicubic upsampling of 64×64 slices to 224×224 creates artificial smoothness. Tiny details (spiculation, margin irregularity) get blurred. |
| **No intensity normalization** | Raw HU values passed through. TinyViT expects ImageNet-normalized inputs (~0-255 or normalized mean/std). Your CT values are completely different. |

**Why 0.9451 is actually disappointing:**
- Your worst 3D CNN (DenseNet) gets 0.9911
- Your 2D model loses 0.046 AUROC — that's **huge** in medical terms
- The 2D model can't see 3D shape features that radiologists use (lobulation, spiculation in 3D)

---

## **3. CNN SETTINGS vs TRANSFORMER SETTINGS**

| Aspect | **CNNs** (DenseNet, ResNet, EfficientNet) | **Transformers** (Swin, ViT, TinyViT) |
|--------|------------------------------------------|--------------------------------------|
| **Learning Rate** | **1e-4** (DenseNet, ResNet, EfficientNet) | **5e-5** (Swin), **1e-5** (ViT) — **10× lower** |
| **Why different?** | CNNs have strong inductive bias (locality, translation invariance). They learn faster and can handle higher LR. | Transformers start from random attention patterns. High LR causes divergence. Need gentle warm-up. |
| **Warmup Epochs** | **5 epochs** (CNNs) | **10 epochs** (Swin), **15 epochs** (ViT) |
| **Why?** | CNNs stabilize quickly. | Transformers need longer to "discover" spatial relationships through attention. |
| **Weight Decay** | **1e-4** (CNNs) | **1e-4** (Swin), **0.01** (ViT) |
| **Normalization** | **None** (raw HU values) for DenseNet/ResNet/EfficientNet | **NormalizeIntensity** for Swin, **None** for ViT |
| **Why?** | CNN BatchNorm layers adapt to any input distribution automatically. | Transformers lack BatchNorm in backbone. Need manual normalization or they explode. |
| **Gradient Clipping** | **Not needed** (no mention in CNN code) | **GRAD_CLIP_NORM = 1.0** (Swin, ViT) |
| **Why?** | CNN gradients are naturally bounded by local receptive fields. | Self-attention can produce exploding gradients (Q@K^T matrix can have extreme values). |
| **Dropout** | **0.5 in MLP head only** | **0.0 internal** (ViT), **0.1 drop_path** (Swin) |
| **Label Smoothing** | **None** | **0.1** (Swin only) |
| **Why?** | CNNs are less prone to overconfidence. | Transformers with enough capacity memorize. Label smoothing prevents "hard" 0/1 predictions. |
| **Data Augmentation** | RandRotate90, RandFlip, RandGaussianNoise (all 3D axes) | Same for 3D; for 2D TinyViT: only 2D rotations + flips |
| **Mixed Precision** | **Yes** (autocast + GradScaler) | **Yes, but with init_scale=256** (lower than default 65536) |
| **Why lower scale?** | CNNs handle default scaling fine. | Transformers are numerically unstable. Lower init_scale prevents Inf on first backward pass. |

### **Key Insight: Transformers Need "Kid Gloves"**

Your CNNs are like experienced workers — give them tools and they get the job done. Your Transformers are like new interns — need constant supervision, lower expectations, and gentler feedback.

The fact that **Swin-Tiny worked reasonably well (AUROC 0.9677)** while **ViT-Small completely failed (AUROC 0.7112)** shows:

| | **Swin Transformer** | **Plain ViT** |
|---|---|---|
| Has local windows (4×4×4 patches attend locally) | ✅ Yes | ❌ No — global attention |
| Hierarchical feature maps | ✅ Yes (4 stages) | ❌ No — single scale |
| Shifted windows for cross-connection | ✅ Yes | ❌ No |
| Parameters | ~5.6M | ~30M |
| **Result** | **0.9677 AUROC** | **0.7112 AUROC** |

**Swin works better because:**
1. **Local windows** = built-in locality bias similar to CNNs
2. **Fewer parameters** (5.6M vs 30M) = less overfitting
3. **Hierarchical design** = captures multi-scale features like CNNs
4. **Your Swin had NormalizeIntensity** = stable inputs

**ViT failed because:**
1. **Global attention on 64 tokens** = every token attends to every other. With random init, it's noise attending to noise.
2. **30M parameters on 6K samples** = severe overfitting (though it didn't even memorize — it collapsed)
3. **No pretrained weights** = started from complete randomness
4. **No input normalization** = transformer attention scores were dominated by raw HU value magnitude

---

## **4. WHY 2D TINYVIT IS NOT FEASIBLE FOR A HYBRID MODEL**

You asked specifically about this. Here are the **concrete, technical reasons:**

### **A. Dimensionality Mismatch — The Fundamental Problem**

| | **2D TinyViT** | **3D CNN/Transformer** |
|---|---|---|
| Input | 3 channels × 224 × 224 (3 axial slices) | 1 channel × 64 × 64 × 64 (full volume) |
| Feature space | 2D spatial + "fake" 3rd channel | True 3D spatial |
| What it sees | Flat 2D texture patterns | Volumetric shape, depth relationships |

**The "3 channels" in TinyViT are NOT RGB — they're 3 adjacent CT slices.** This is a hack. The model has NO way to know these are spatially related slices. It processes them like color channels in a photo.

### **B. Information Loss is Catastrophic**

```
Full 3D volume:     64 × 64 × 64  = 262,144 voxels
TinyViT input:      3 × 224 × 224 = 150,528 pixels (but upsampled from 3×64×64)
Actual information: 3 × 64 × 64   = 12,288 voxels (before upsampling)
                    ↑
            You kept only 4.7% of the original volume!
```

**What gets lost:**
- **Coronal and sagittal views**: A nodule might look benign in axial but clearly spiculated in coronal. TinyViT never sees these.
- **3D shape descriptors**: Volume, surface area, sphericity — key malignancy indicators.
- **Growth patterns**: How the nodule extends through adjacent slices.

### **C. The "Fake 3-Channel" Problem**

Your code:
```python
slices = [vol[centre - 1, :, :], vol[centre, :, :], vol[centre + 1, :, :]]
img_3ch = np.stack(slices, axis=0)  # (3, 64, 64)
```

This stacks slices as "channels." But:
- **Channel 1** (centre-1) and **Channel 3** (centre+1) are 2 voxels apart in reality
- The model treats them like R, G, B — independent color information
- **No 3D convolutions** ever happen. No 3×3×3 kernels. Only 2D 3×3 kernels that happen to see 3 "color" planes.

### **D. Pretraining Mismatch**

| | **ImageNet Pretraining** | **Your Medical Data** |
|---|---|---|
| Image distribution | Natural images (cats, dogs, cars) | CT scans (Hounsfield Units, -1000 to +1000) |
| Texture patterns | Fur, edges, colors | Tissue density, noise, partial volume effects |
| Object semantics | Semantic categories | Pathological features (spiculation, lobulation) |

**Even if you used pretrained=True:**
- ImageNet features detect edges, corners, textures — useful for photos
- Medical features need to detect subtle density variations, smooth vs. irregular margins
- **Transfer learning from ImageNet to CT is weak** — studies show minimal benefit

### **E. If You Still Want a Hybrid: What Would Actually Work**

A true hybrid needs **feature-level fusion**, not input-level hack:

```
Option 1: 3D CNN encoder + 2D Transformer encoder → fusion at feature level
Option 2: Process full 3D with 3D Swin, process 2D MIP/CPR views with 2D CNN → late fusion
Option 3: 3D volume → 3D CNN backbone → project to sequence → Transformer decoder
```

Your current "hybrid" idea (2D TinyViT + 3D something) doesn't fuse features — it replaces 3D processing with 2D processing. That's **not hybrid, it's downgrade**.

---

## **5. HYPERPARAMETERS: SAME vs DIFFERENT — MEDICAL SUITABILITY**

### **WHAT STAYED THE SAME (And Whether That's Good)**

| Parameter | Same Value | Good for Medical? | Verdict |
|-----------|-----------|-------------------|---------|
| **AdamW optimizer** | All models | Yes. Decoupled weight decay works better than SGD for deep nets. Medical data has noisy labels, AdamW's adaptive LR helps. | ✅ Good |
| **Cosine annealing after warmup** | All models | Yes. Smooth LR decay prevents sudden drops that could destabilize training. | ✅ Good |
| **BCEWithLogitsLoss** | All models | **Excellent choice.** Binary classification with logits is numerically stable. Pos_weight handles imbalance. | ✅ Perfect |
| **Max epochs 100** | CNNs and Swin | Yes. With early stopping (patience=15), this is just an upper bound. | ✅ Good |
| **No learning rate restart** | All models | Fine. Medical training is short; restarts unnecessary. | ✅ Acceptable |

### **WHAT DIFFERED (And Whether the Differences Are Justified)**

| Parameter | CNN Value | Transformer Value | Medical Justification | Verdict |
|-----------|-----------|-------------------|----------------------|---------|
| **Base LR** | 1e-4 | 5e-5 (Swin), 1e-5 (ViT) | Transformers need lower LR due to no inductive bias. **But 1e-5 for ViT was too low** — model never moved from initialization. | ⚠️ Swin OK, ViT too low |
| **Warmup epochs** | 5 | 10 (Swin), 15 (ViT) | Longer warmup for transformers is standard. **15 was excessive for 18 total epochs** — 83% of training was just warming up! | ❌ ViT warmup too long |
| **Weight decay** | 1e-4 | 1e-4 (Swin), 0.01 (ViT) | ViT's 0.01 is 100× higher. With 30M params and tiny data, this crushed the model's capacity to learn. | ❌ ViT WD too high |
| **Gradient clipping** | None | 1.0 | Needed for transformers. But ViT's max gradients were 439 — clipping at 1.0 may have been too aggressive. | ⚠️ Swin OK, ViT maybe too harsh |
| **Dropout** | 0.5 (head only) | 0.0 internal (ViT), 0.1 path (Swin) | ViT with 0.0 dropout and 30M params = overfitting machine. But it didn't overfit — it collapsed. | ❌ ViT needed more regularization |
| **Label smoothing** | None | 0.1 (Swin only) | Good for preventing overconfidence. Should have been used in ViT too. | ⚠️ Inconsistent |
| **Input normalization** | None (raw HU) | NormalizeIntensity (Swin), None (ViT) | **This was critical.** Swin worked partly because of normalization. ViT failed partly because of its absence. | ❌ ViT should have had it |

### **WHAT SHOULD HAVE BEEN DIFFERENT**

| Setting | What You Did | What You Should Have Done | Why |
|---------|-------------|---------------------------|-----|
| **ViT pretrained weights** | Set `USE_PRETRAINED=True` but provided NO weights | Either: (a) Download actual pretrained 3D ViT weights, or (b) Set `USE_PRETRAINED=False` and increase LR to 1e-4 | Empty flag did nothing. Model trained from scratch with cripplingly low LR. |
| **ViT dropout** | 0.0 | 0.1-0.3 | 30M parameters, 6K samples. Without dropout, model should overfit. The fact that it didn't means training failed entirely. |
| **ViT max epochs** | 150 | 300+ with stronger regularization | With LR=1e-5, the model needed 10× more epochs just to start learning. But that would overfit without dropout. |
| **All models: intensity normalization** | Inconsistent | **Standardize for all** | CT Hounsfield Units have defined meaning. Normalizing per-image (`NormalizeIntensity`) preserves relative contrast while stabilizing neural network inputs. |

---

## **MEDICAL ACCEPTABILITY: HOW GOOD ARE THESE RESULTS?**

### **The Gold Standards in Lung Nodule Detection**

| Metric | Clinical/Competition Benchmark | Your Best Model (EfficientNet-B0) | Assessment |
|--------|-------------------------------|-------------------------------------|------------|
| **AUROC** | >0.95 considered excellent | **0.9959** | ✅ **Outstanding** — among best published |
| **AUPRC** | >0.90 good for imbalanced data | **0.9983** | ✅ **Exceptional** |
| **Sensitivity at 0.125 FP/scan** | >0.80 in LUNA16 challenge | **0.9651** | ✅ **Excellent** |
| **CPM (Competition Performance Metric)** | ~0.80-0.85 for top algorithms | **~0.98** | ✅ **Top-tier** |
| **Specificity** | >0.90 desired for screening | **0.8372** | ⚠️ **Acceptable but could be better** |

### **Clinical Interpretation**

Your **EfficientNet-B0 (AUROC 0.9959)** means:
- If you randomly pick one malignant and one benign nodule, the model correctly ranks the malignant one higher **99.6% of the time**.
- This is **better than many radiologists** in studies (radiologist AUROC typically 0.85-0.95 for nodule classification).

**However, specificity of 0.8372 means:**
- **16.3% of benign nodules are flagged as suspicious**
- In a screening program with 2% prevalence, this means **~91% of positive screens would be false positives**
- This is the **classic screening paradox**: high sensitivity saves lives but causes anxiety and unnecessary procedures

### **Are These Results "Acceptable" in Medicine?**

| Context | Acceptable? | Notes |
|---------|-------------|-------|
| **Research publication** | ✅ Yes | Results are excellent, properly validated with patient-level splits, no data leakage |
| **FDA/CE approval as CADe (Computer-Aided Detection)** | ⚠️ Maybe | Need multi-site validation, prospective study, clinical reader study |
| **Clinical deployment without radiologist** | ❌ No | No AI is approved for standalone diagnosis. Must be "second reader" or "triaging" tool |
| **Screening program triage** | ✅ Potentially | Could prioritize urgent cases, but false positive rate needs management |

### **What Would Make This Clinically Robust**

1. **External validation** on another dataset (LIDC-IDRI, NLST)
2. **Reader study**: Radiologists with vs. without AI assistance
3. **Calibration improvement**: Your probabilities should reflect true likelihood (calibration curves help)
4. **Uncertainty quantification**: Report confidence intervals (you do this — good!)
5. **Explainability**: Grad-CAM or attention maps showing which voxels drove the decision

---

## **FINAL VERDICT**

| Model | Grade | Key Takeaway |
|-------|-------|--------------|
| **3D EfficientNet-B0** | **A+** | Best overall. Compound scaling works brilliantly for 3D medical volumes. |
| **3D DenseNet121** | **A** | Excellent, most balanced (best specificity). Dense connections prevent gradient vanishing. |
| **3D ResNet34** | **A-** | Excellent sensitivity but lower specificity. 63M params may overfit to positives. |
| **3D Swin-Tiny** | **B+** | Good but not great. Hierarchical design helps, but still behind CNNs without pretraining. |
| **2D TinyViT** | **C+** | Mediocre. Lost too much 3D information. Not suitable for hybrid without major redesign. |
| **3D ViT-Small** | **F** | Complete failure. Wrong architecture for data size, no pretraining, bad hyperparameters. |

**Your CNNs significantly outperform transformers** — this is consistent with medical imaging literature. CNNs' built-in inductive biases (locality, translation invariance, hierarchical features) match radiological reasoning better than transformer's global attention, especially with limited data.

**For our thesis:** The CNN results are publication-ready. The transformer results (except ViT) are acceptable as "exploratory comparison." The ViT failure is actually a **valuable finding** — it demonstrates that transformers aren't automatically superior and require careful tuning or pretraining for medical 3D data.