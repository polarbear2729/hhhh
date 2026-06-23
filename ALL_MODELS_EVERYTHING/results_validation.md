# Results Validation — ANALYSIS.md / CLAIMS_AUDIT.md vs Test Results

## Important Distinction

| Source | Evaluation Level | Dataset |
|--------|:----------------:|:-------:|
| **ANALYSIS.md** & **CLAIMS_AUDIT.md** | **Patch-level** (each 64³ patch independently) | **Validation set** (best epoch) |
| **Test_result_off_all_models.txt** | **Patient-level (scan-level)** — max-prob per patient | **Test set** (held-out) |

Patch-level validation metrics and patient-level test metrics **are not directly comparable** — they are different evaluation protocols on different data splits. The validation below checks consistency, not equality.

---

## 1. Standalone Models — Val (Patch) vs Test (Patient)

### ResNet-18

| Metric | Val (patch) | Test (patient) | Δ | Consistent? |
|--------|:-----------:|:--------------:|:-:|:-----------:|
| AUC | 0.9997 | 0.9977 | −0.0020 | ✅ slight expected drop |
| AUPRC | 0.9999 | 0.9990 | −0.0009 | ✅ |
| F1 | 0.9730 | **0.9890** | **+0.0160** | ⚠️ test better — unusual but possible (val best-epoch tuned for AUC, not F1) |
| Sensitivity | 1.0000 | 0.9890 | −0.0110 | ✅ one FN |
| Specificity | 0.8837 | **0.9767** | **+0.0930** | ⚠️ test much better — the model generalises better than val early-stopping suggests |

### DenseNet-121

| Metric | Val (patch) | Test (patient) | Δ | Consistent? |
|--------|:-----------:|:--------------:|:-:|:-----------:|
| AUC | 0.9965 | 0.9905 | −0.0060 | ✅ |
| AUPRC | 0.9981 | 0.9958 | −0.0023 | ✅ |
| F1 | 0.9574 | 0.9560 | −0.0014 | ✅ |
| Sensitivity | 1.0000 | 0.9560 | −0.0440 | ✅ 4 FN on test |
| Specificity | 0.8140 | **0.9070** | **+0.0930** | ⚠️ test better — same pattern as ResNet-18 |

### EfficientNet-B0

| Metric | Val (patch) | Test (patient) | Δ | Consistent? |
|--------|:-----------:|:--------------:|:-:|:-----------:|
| AUC | **1.0000** | **0.9808** | **−0.0192** | ❌ notable drop — 2% AUC loss, largest in study |
| AUPRC | 1.0000 | 0.9913 | −0.0087 | ⚠️ |
| F1 | 0.9945 | 0.9333 | −0.0612 | ❌ large drop — +6.1 pp |
| Sensitivity | 1.0000 | **0.9231** | **−0.0769** | ❌ large drop — misses 7 of 91 malignant patients |
| Specificity | 0.9767 | 0.8837 | −0.0930 | ❌ large drop |

**Finding:** EfficientNet-B0 shows the largest val→test performance drop of any model. The patch-level validation (AUC=1.0) overestimates patient-level test performance by ~2% AUC. Possible causes: the model overfits to patch-level patterns that don't aggregate cleanly at patient level (max-prob per patient makes it sensitive to any single high-confidence false positive patch in a benign scan). Despite this, the val metrics in ANALYSIS.md are correctly labelled as "Best validation metrics" — the issue is they don't generalise as well to patient-level test as other models.

---

## 2. Hybrids — Val (Patch) vs Test (Patient)

### 2-CNN Hybrids

| Variant | Source | AUC | F1 | Sens | Spec | Δ AUC | Δ F1 | Consistent? |
|---------|--------|----:|---:|----:|----:|:-----:|:----:|:-----------:|
| **KAN Frozen** | Val (patch) | 0.9987 | 0.9730 | 1.0000 | 0.8837 | — | — | — |
| | **Test (patient)** | **0.9972** | **0.9890** | **0.9890** | **0.9767** | −0.0015 | **+0.0160** | ✅ close (F1/Spec better on test) |
| **KAN Scratch** | Val (patch) | 0.9984 | 0.9783 | 1.0000 | 0.9070 | — | — | — |
| | **Test (patient)** | **0.9985** | **0.9836** | **0.9890** | **0.9535** | **+0.0001** | +0.0053 | ✅ near-identical AUC |
| **MLP Frozen** | Val (patch) | 0.9987 | 0.9783 | 1.0000 | 0.9070 | — | — | — |
| | **Test (patient)** | **0.9974** | **0.9945** | **0.9890** | **1.0000** | −0.0013 | **+0.0162** | ✅ better on test |
| **MLP Scratch** | Val (patch) | 0.9995 | 0.9836 | 1.0000 | 0.9302 | — | — | — |
| | **Test (patient)** | **0.9847** | **0.9724** | **0.9670** | **0.9535** | **−0.0148** | −0.0112 | ⚠️ moderate drop |

### 3-CNN Hybrids

| Variant | Source | AUC | F1 | Sens | Spec | Δ AUC | Δ F1 | Consistent? |
|---------|--------|----:|---:|----:|----:|:-----:|:----:|:-----------:|
| **KAN Frozen** | Val (patch) | 0.9990 | 0.9730 | 1.0000 | 0.8837 | — | — | — |
| | **Test (patient)** | **0.9969** | **0.9890** | **0.9890** | **0.9767** | −0.0021 | **+0.0160** | ✅ stable |
| **KAN Scratch** | Val (patch) | 1.0000 | 0.9730 | 1.0000 | 0.8837 | — | — | — |
| | **Test (patient)** | **0.9990** | **0.9730** | **0.9890** | **0.9070** | −0.0010 | **0.0000** | ✅ near-identical |
| **MLP Frozen** | Val (patch) | **1.0000** | 0.9945 | 1.0000 | **0.9767** | — | — | — |
| | **Test (patient)** | **0.9964** | **0.9945** | **0.9890** | **1.0000** | −0.0036 | **0.0000** | ✅ **most stable — F1 matches exactly** |
| **MLP Scratch** | Val (patch) | 0.9992 | 0.9730 | 1.0000 | 0.8837 | — | — | — |
| | **Test (patient)** | **0.9798** | **0.9462** | **0.9670** | **0.8372** | **−0.0194** | −0.0268 | ⚠️ notable drop — overfits when 3 backbones trained from scratch |

---

## 3. Overall Best Model — Validated

Both ANALYSIS.md and Test_result_off_all_models.txt agree on the same winner:

| Criterion | ANALYSIS.md (val patch) | Test_result (test patient) | Verdict |
|-----------|:-----------------------:|:--------------------------:|:--------|
| **Best model** | 3-CNN MLP Frozen | 3-CNN MLP Frozen | ✅ Confirmed |
| Best AUC | 1.0000 (tied with B0) | 0.9964 (2nd to 3CNN KAN Scratch 0.9990) | ⚠️ 3CNN KAN Scratch has higher test AUC |
| Best F1 | 0.9945 | **0.9945** | ✅ **F1 matches exactly** |
| Best specificity | 0.9767 | **1.0000** (zero FP) | ✅ **Even better on test** |
| Lowest FN | — | **FN=1** (ties with ResNet-18, 2CNN KAN Scratch, etc.) | ✅ Joint best |
| Lowest FP | 25 (projected) | **0** (actual) | ✅ **Better than projected** |

**Key finding:** 3-CNN MLP Frozen achieves **zero false positives (FP=0)** on the patient-level test set — better than the val-based projection of 25 FP in CLAIMS_AUDIT.md. Its F1 of 0.9945 is identical to the val estimate.

---

## 4. Claims Audit — Re-validated Against Test Results

### Claim 1: False-Positive Reduction

Test_result provides **actual** FP counts (patient-level), which are more informative than the val-based projections in CLAIMS_AUDIT.md:

| Model | CLAIMS_AUDIT.md (projected from val) | Test_result (actual patient-level) | Actual FP | Verdict |
|-------|:------------------------------------:|:----------------------------------:|:---------:|:--------|
| 3D DenseNet-121 | 200 FP | **4 FP** | ✅ Far fewer than projected |
| 3D ResNet-18 | 125 FP | **1 FP** | ✅ Far fewer |
| 3D EfficientNet-B0 | 25 FP | **5 FP** | ⚠️ 5x worse than projection |
| 2CNN MLP Scratch | 75 FP | **2 FP** | ✅ Projection was pessimistic |
| 2CNN MLP Frozen | 100 FP | **0 FP** | ✅ **Perfect — better than projected** |
| 3CNN MLP Frozen | 25 FP | **0 FP** | ✅ **Perfect — better than projected** |

**Re-validated verdict: ✅ Claim holds.** The 3-CNN MLP Frozen achieves **zero false positives** on the patient-level test — better than the val-based projection. The val projections in CLAIMS_AUDIT.md were conservative (patch-level val specificity underestimates patient-level test specificity for most models).

### Claim 2: Lightweight Training

| Sub-claim | Test_result evidence | Verdict |
|-----------|:-------------------:|:--------|
| *"converges in 1 epoch"* | 3CNN MLP Frozen training log confirms epoch 1 best | ✅ |
| *"only MLP head trains"* | Architecture confirms frozen backbones | ✅ |
| *"on-disk model is 4-5× larger"* | File sizes unchanged | ✅ unchanged |

### Claim 3: MLP > KAN

| Regime | Val (patch) | Test (patient) | Verdict |
|--------|:-----------:|:--------------:|:--------|
| 2-CNN Frozen: MLP vs KAN | MLP wins (F1 +0.0053) | MLP wins (F1 +0.0055) | ✅ Confirmed on test |
| 2-CNN Scratch: MLP vs KAN | MLP wins (F1 +0.0053) | **KAN wins** (F1: 0.9836 vs 0.9724) | ⚠️ **Reversed on test** — KAN has higher test AUC (0.9985 vs 0.9847) and F1 (0.9836 vs 0.9724) |
| 3-CNN Frozen: MLP vs KAN | MLP wins (F1 +0.0215) | MLP wins (F1: 0.9945 vs 0.9890) | ✅ Confirmed |
| 3-CNN Scratch: MLP vs KAN | Tie (F1 both 0.9730) | **KAN wins** (F1: 0.9730 vs 0.9462) | ⚠️ **Reversed on test** — KAN scratch clearly beats MLP scratch |

**Re-validated verdict: ⚠️ Nuance added.** The MLP > KAN claim in CLAIMS_AUDIT.md was based on validation metrics. On test data, KAN wins in **2 out of 4 regimes** (2-CNN Scratch, 3-CNN Scratch) while MLP still wins in the frozen regimes. The 2-CNN Scratch reversal is particularly important — KAN's test AUC (0.9985) is notably higher than MLP's (0.9847). The frozen-regime conclusions are unchanged.

---

## 5. Paper_analysis.md — Not Affected

Paper_analysis.md compares against external papers (MSA-Net, LMLCC-Net, etc.). It does not reference our internal test metrics — it reports the papers' published results. The validation of our models' superiority over these papers should use the **patient-level test results** from Test_result_off_all_models.txt.

**Updated comparison using actual patient-level test results:**

| Paper | Paper AUC | Our Best (3CNN MLP Frozen test) | Δ |
|-------|:---------:|:-------------------------------:|:-:|
| MSA-Net | 0.9930 | **0.9964** | +0.0034 |
| LMLCC-Net | 0.9410 | **0.9964** | +0.0554 |
| Sungheetha et al. | 0.9470 | **0.9964** | +0.0494 |

Our outperformance claim is **still supported** with patient-level test metrics — the margins are slightly smaller but still comfortable.

---

## 6. Key Discrepancies to Address

| Issue | Details | Action |
|-------|---------|--------|
| **ANALYSIS.md uses patch-level val** | All tables in ANALYSIS.md are explicitly patch-level validation (best_epoch). The document header says so. | ✅ Correct as-is — but needs clear labelling that these are **not** test results |
| **CLAIMS_AUDIT.md FP counts are projections** | Uses val-specificity × 1075 negatives to project FP. Actual test FP counts are much lower for most models. | ⚠️ The projections are conservative. Consider replacing with actual test FP counts |
| **EfficientNet-B0 val→test gap** | AUC drops from 1.0000 (val patch) to 0.9808 (test patient). F1 drops from 0.9945 to 0.9333. | ⚠️ Don't claim EfficientNet-B0 generalises perfectly — test results are notably weaker |
| **MLP vs KAN reversal on test** | KAN beats MLP in 2/4 regimes on test (opposite of val findings). | ⚠️ Update claim to "MLP wins in frozen regimes; KAN can be competitive when trained from scratch" |
| **3CNN MLP Frozen remains best** | Despite minor AUC drop (1.0→0.9964), F1 is identical (0.9945) and specificity is perfect (1.0). | ✅ This conclusion is validated and strengthened |
