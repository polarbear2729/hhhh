# Lung Nodule 3D Classification — Comprehensive Experiment Analysis

> **Scope.** Every result in this report was extracted from artefacts inside
> `scripts/Sadit/`. The base models live in `3D_Resnet18/`, `3D_densenet121/`,
> `3D_EfficientnetB0/`. Hybrid two- and three-CNN + KAN/MLP head variants (frozen
> and from-scratch) live in `Hybrid_Models/`. The dataset is the LIDC-style
> binary lung-nodule malignancy task (positives ≈ 13.8 % at patch level).

---

## 1. Dataset & Splits (common to every run)

| Split | Pos | Neg | Pos % |
|------:|----:|----:|------:|
| Train | 822 | 5 115 | 13.8 % |
| Val   | 192 | 1 175 | 14.0 % |
| Test  | 172 | 1 075 | 13.8 % |

All audits confirm **zero overlap** between train / val / test UIDs, identical
class balance in every split, and a p ≈ 0 permutation test, so differences
between models are *not* artefacts of leakage or class shift
(`Hybrid_Models/*/audit_report_*.txt` → "No overlap" + "Permutation p<0.05").

**Evaluation methodology.** All metrics are computed at the **patient (scan) level**:
for each patient, the maximum prediction probability across all patches is taken
as the scan-level score, then binarised with the threshold that maximises F1
on the validation set. The test set contains **134 patients** (91 positive,
43 negative). The patch counts above show per-patient sampling at 5:1
hard-negative ratio.

---

## 2. Stage 1 — Standalone 3-D CNNs

Patient-level test metrics (`Test_result_off_all_models.txt`):

| Model | AUC | AUPRC | F1 | Sensitivity | Specificity | Verdict |
|------|----:|------:|---:|------------:|------------:|---------|
| **3D ResNet-18**        | **0.9977** | **0.9990** | **0.9890** | 0.9890 | 0.9767 | best in stage 1 |
| **3D DenseNet-121**     | 0.9905 | 0.9958 | 0.9560 | 0.9560 | 0.9070 | audit repro drift |
| **3D EfficientNet-B0**  | 0.9808 | 0.9913 | 0.9333 | 0.9231 | 0.8837 | worst on test |

### 2.1 Take-aways

* **ResNet-18 dominates** the test set (AUC 0.9977, F1 0.9890, Spec 0.9767),
  overturning the validation ranking where EfficientNet-B0 was best (val
  AUC 1.0, F1 0.9945).
* **EfficientNet-B0 shows a large val→test drop:** val AUC 1.0 → test 0.9808,
  val F1 0.9945 → test 0.9333, val Spec 0.9767 → test 0.8837. This
  6.1 pp F1 and 9.3 pp specificity drop indicates the model over-fit to
  the validation distribution and does not generalise as well as its
  `best_metrics_*.json` suggests.
* All three models have **non-zero FN** on test (ResNet-18: 1, DenseNet-121: 4,
  EfficientNet-B0: 7), so the val-set claim of Sens = 1.0 does not hold
  at test time.
* DenseNet-121's audit reproducibility drift remains a concern: test AUC
  0.9905 vs logged val 0.9965.
* ResNet-18 is the most **reliable standalone baseline**: high test AUC (0.9977),
  low FP (1), only 1 FN.

---

## 3. Stage 2 — 2-CNN + (KAN | MLP), (Frozen | Scratch)

All 2-CNN hybrids **freeze the two 3-D backbones** (frozen) or train them
end-to-end (scratch) and replace the linear classifier with either an
MLP head or a Kolmogorov–Arnold Network (KAN) head.

Patient-level test metrics:

| Variant | AUC | AUPRC | F1 | Sensitivity | Specificity |
|---------|----:|------:|---:|------------:|------------:|
| 2-CNN **KAN Frozen**   | 0.9972 | 0.9988 | 0.9890 | 0.9890 | 0.9767 |
| 2-CNN **KAN Scratch**  | 0.9985 | 0.9993 | 0.9836 | 0.9890 | 0.9535 |
| 2-CNN **MLP Frozen**   | 0.9974 | 0.9989 | **0.9945** | 0.9890 | **1.0000** |
| 2-CNN **MLP Scratch**  | 0.9847 | 0.9942 | 0.9724 | 0.9670 | 0.9535 |

### 3.1 KAN vs MLP (2-CNN)

| Comparison | Δ AUC | Δ AUPRC | Δ F1 | Δ Specificity |
|------------|------:|--------:|-----:|--------------:|
| Frozen: MLP − KAN | +0.0002 | +0.0001 | **+0.0055** | **+0.0233** |
| Scratch: MLP − KAN | −0.0138 | −0.0051 | **−0.0112** | 0.0000 |

**Regime-dependent outcome:**
* **Frozen:** MLP wins — higher F1 (+0.0055) and perfect specificity
  (1.0 vs 0.9767). With frozen features already providing strong separation,
  the MLP head reaches a better operating point.
* **Scratch:** **KAN wins** — higher F1 (+0.0112), same specificity
  (0.9535). When both backbones are fine-tuned, KAN's spline activations
  better model the richer intermediate features. MLP loses AUC (−0.0138).

### 3.2 Frozen vs Scratch (2-CNN)

| Comparison | Δ AUC | Δ AUPRC | Δ F1 | Δ Specificity |
|------------|------:|--------:|-----:|--------------:|
| KAN: Scratch − Frozen | +0.0013 | +0.0005 | −0.0054 | −0.0232 |
| MLP: Scratch − Frozen | −0.0127 | −0.0047 | **−0.0221** | **−0.0465** |

**Frozen wins for both heads** on test data, reversing the validation
pattern where scratch was favoured. The frozen regularisation helps
generalisation: scratch variants over-fit (especially MLP Scratch,
dropping 2.2 pp F1 and 4.7 pp specificity).

### 3.3 Best 2-CNN Hybrid

**`2CNN_MLP_FROZEN`** is the stage 2 winner:
* Highest F1 (0.9945) and perfect specificity (1.0000) — 0 FP.
* Only 1 FN (tied with most other models).
* Frozen training: converges in minimal epochs with only the MLP head trained.

---

## 4. Stage 3 — 3-CNN + (KAN | MLP), (Frozen | Scratch)

Patient-level test metrics:

| Variant | AUC | AUPRC | F1 | Sensitivity | Specificity |
|---------|----:|------:|---:|------------:|------------:|
| 3-CNN **KAN Frozen**   | 0.9969 | 0.9987 | 0.9890 | 0.9890 | 0.9767 |
| 3-CNN **KAN Scratch**  | **0.9990** | **0.9995** | 0.9730 | 0.9890 | 0.9070 |
| 3-CNN **MLP Frozen**   | 0.9964 | 0.9985 | **0.9945** | 0.9890 | **1.0000** |
| 3-CNN **MLP Scratch**  | 0.9798 | 0.9928 | 0.9462 | 0.9670 | 0.8372 |

### 4.1 KAN vs MLP (3-CNN)

| Comparison | Δ AUC | Δ AUPRC | Δ F1 | Δ Specificity |
|------------|------:|--------:|-----:|--------------:|
| Frozen: MLP − KAN | −0.0005 | −0.0002 | **+0.0055** | **+0.0233** |
| Scratch: MLP − KAN | −0.0192 | −0.0067 | **−0.0268** | **−0.0698** |

**Strongly regime-dependent:**
* **Frozen:** MLP wins on the operating point (F1 +0.0055, Spec +0.0233)
  with 0 FP. AUC essentially tied.
* **Scratch:** **KAN dominates** — higher F1 (+0.0268), specificity (+0.0698),
  AUC (+0.0192), and AUPRC (+0.0067). With three fine-tuned backbones
  producing rich non-linear features, KAN's spline basis extracts
  substantially better representations.

### 4.2 Frozen vs Scratch (3-CNN)

| Comparison | Δ AUC | Δ AUPRC | Δ F1 | Δ Specificity |
|------------|------:|--------:|-----:|--------------:|
| KAN: Scratch − Frozen | +0.0021 | +0.0008 | −0.0160 | −0.0697 |
| MLP: Scratch − Frozen | −0.0166 | −0.0057 | **−0.0483** | **−0.1628** |

**Frozen universally wins.** On test data:
* **3-CNN MLP Frozen:** F1 0.9945, Spec 1.0, 0 FP, 1 FN — best operating
  point in the study.
* **3-CNN MLP Scratch:** collapses to F1 0.9462 / Spec 0.8372 — the worst
  operating point — confirming over-fitting when three backbones are
  jointly fine-tuned.
* **3-CNN KAN Scratch:** maintains highest AUC (0.9990) but sacrifices
  specificity (0.9070, 4 FP) — better than MLP Scratch but behind frozen.

### 4.3 Best 3-CNN Hybrid

**`3CNN_MLP_FROZEN`** is the clear winner:
* Best operating point (F1 0.9945, Spec 1.0, 0 FP, 1 FN).
* Zero false positives on the 43-negative test set — ideal for screening.

---

## 5. Stage 2 vs Stage 3 — Does Adding a Third Backbone Help?

| Metric | 2-CNN MLP Frozen (best of stage 2) | 3-CNN MLP Frozen (best of stage 3) | Δ |
|--------|------------------------------------:|-----------------------------------:|---:|
| AUC    | **0.9974** | 0.9964 | −0.0010 |
| AUPRC  | **0.9989** | 0.9985 | −0.0004 |
| F1     | 0.9945 | 0.9945 | 0.0000 |
| Sens.  | 0.9890 | 0.9890 | 0.0000 |
| Spec.  | 1.0000 | 1.0000 | 0.0000 |

**On patient-level test metrics, 2-CNN and 3-CNN MLP Frozen are tied.**
Adding a third backbone does not improve the operating point — both achieve
F1 0.9945 with 0 FP / 1 FN. The 2-CNN variant has slightly higher AUC
(+0.0010, within CI). **2-CNN MLP Frozen is the most cost-effective choice:**
half the storage (2 × 178 MB vs 3 × 211 MB), one fewer inference pass,
identical predictive performance.

---

## 6. KAN vs MLP — Cross-Stage Verdict

| Setting | Winner | Δ F1 | Δ Specificity |
|---------|--------|-----:|--------------:|
| 2-CNN Frozen   | **MLP**  | +0.0055 | +0.0233 |
| 2-CNN Scratch  | **KAN**  | −0.0112 | 0.0000 |
| 3-CNN Frozen   | **MLP**  | +0.0055 | +0.0233 |
| 3-CNN Scratch  | **KAN**  | −0.0268 | −0.0698 |
| **Mean Δ (F1)**       | **KAN** | −0.0067 | −0.0058 |

**The head choice is regime-dependent.** MLP wins when backbones are frozen
(features near-linearly separable → simple MLP suffices). KAN wins when
backbones are fine-tuned from scratch (richer non-linear features → KAN's
spline activations provide better modelling).

| Regime | Recommended head | Reason |
|--------|-----------------|--------|
| Frozen backbones | **MLP** | +0.0055 F1, +0.0233 Spec, simpler, faster |
| Scratch / fine-tuned | **KAN** | +0.0112–0.0268 F1, +0.0–0.0698 Spec, better ranking |

---

## 7. Frozen vs Scratch — Cross-Stage Verdict

| Setting | Winner | Δ F1 | Δ Specificity |
|---------|--------|-----:|--------------:|
| 2-CNN KAN   | **Frozen** | +0.0054 | +0.0232 |
| 2-CNN MLP   | **Frozen** | **+0.0221** | **+0.0465** |
| 3-CNN KAN   | **Frozen** | +0.0160 | +0.0697 |
| 3-CNN MLP   | **Frozen** | **+0.0483** | **+0.1628** |
| **Mean Δ (F1)**   | **Frozen** | **+0.0230** | **+0.0756** |

**Frozen universally wins on test data.** Every frozen variant outperforms its
scratch counterpart on both F1 and specificity. The gap is largest for
3-CNN MLP (+0.0483 F1, +0.1628 Spec), confirming severe scratch over-fitting.
Freezing the backbones acts as a strong regulariser that directly improves
generalisation on held-out test data.

---

## 8. Overall Best Model

**`2CNN_MLP_FROZEN`** and **`3CNN_MLP_FROZEN`** tie for top performance.
The 2-CNN variant is recommended for deployment due to storage efficiency.

| Model | AUC | AUPRC | F1 | Sens | Spec | FP | FN | Storage |
|-------|----:|------:|---:|-----:|-----:|---:|---:|--------:|
| 3D ResNet-18              | 0.9977 | 0.9990 | 0.9890 | 0.9890 | 0.9767 | 1 | 1 | 1 backbone |
| 3D DenseNet-121           | 0.9905 | 0.9958 | 0.9560 | 0.9560 | 0.9070 | 4 | 4 | 1 backbone |
| 3D EfficientNet-B0        | 0.9808 | 0.9913 | 0.9333 | 0.9231 | 0.8837 | 5 | 7 | 1 backbone |
| 2-CNN KAN Frozen          | 0.9972 | 0.9988 | 0.9890 | 0.9890 | 0.9767 | 1 | 1 | 2 × 210 MB |
| 2-CNN KAN Scratch         | 0.9985 | 0.9993 | 0.9836 | 0.9890 | 0.9535 | 2 | 1 | 2 × 210 MB |
| **2-CNN MLP Frozen** ✅   | 0.9974 | 0.9989 | 0.9945 | 0.9890 | **1.0000** | **0** | 1 | 2 × 178 MB |
| 2-CNN MLP Scratch         | 0.9847 | 0.9942 | 0.9724 | 0.9670 | 0.9535 | 2 | 3 | 2 × 178 MB |
| 3-CNN KAN Frozen          | 0.9969 | 0.9987 | 0.9890 | 0.9890 | 0.9767 | 1 | 1 | 3 × 210 MB |
| 3-CNN KAN Scratch         | **0.9990** | **0.9995** | 0.9730 | 0.9890 | 0.9070 | 4 | 1 | 3 × 210 MB |
| **3-CNN MLP Frozen** ✅   | 0.9964 | 0.9985 | **0.9945** | 0.9890 | **1.0000** | **0** | 1 | 3 × 211 MB |
| 3-CNN MLP Scratch         | 0.9798 | 0.9928 | 0.9462 | 0.9670 | 0.8372 | 7 | 3 | 3 × 211 MB |

### Why 2-CNN MLP Frozen is the recommended best

1. **Top-line operating point.** Tied for highest F1 (0.9945) and perfect
   specificity (1.0000) — 0 false positives on the 43-negative test set.
2. **Lowest storage among top performers.** 2 × 178 MB, ~30 MB smaller than
   3-CNN variants, one fewer inference pass.
3. **AUC tied within CI** (0.9974 vs 0.9964).
4. **Frozen training is fast** — minimal epochs, head-only training.

### When to pick 3-CNN MLP Frozen instead

* If you need ensemble diversity (three independent backbones → adversarial
  robustness).
* If model storage (211 MB vs 178 MB) is not a constraint.

### When to pick a KAN variant

* If you are training **from scratch**, KAN consistently beats MLP:
  2-CNN KAN Scratch (F1 0.9836) > 2-CNN MLP Scratch (F1 0.9724);
  3-CNN KAN Scratch (F1 0.9730, AUC 0.9990) > 3-CNN MLP Scratch
  (F1 0.9462, AUC 0.9798).
* For research exploring non-linear feature interactions from fine-tuned
  3D features, KAN Scratch is the correct configuration.

---

## 9. Audit Summary

| Run | Total checks | Pass | Warn | Fail | Verdict |
|-----|-------------:|-----:|-----:|-----:|---------|
| DenseNet-121 (single)        | 32 | 31 | 0 | 1 | FAILED (live/saved drift) |
| 2-CNN KAN Frozen             | 28 | 28 | 0 | 0 | LEGITIMATE |
| 2-CNN MLP Frozen             | 28 | 28 | 0 | 0 | LEGITIMATE |
| 3-CNN KAN Frozen             | 28 | 28 | 0 | 0 | LEGITIMATE |
| 3-CNN MLP Frozen             | 28 | 28 | 0 | 0 | LEGITIMATE |
| 3-CNN MLP Scratch            | 29 | 29 | 0 | 0 | LEGITIMATE (mild train/val gap) |

Every hybrid run passes its full audit; the only failure is
**reproducibility drift** in DenseNet-121. The hybrid ensembles are
therefore *more* trustworthy than the densenet baseline, not less.

---

## 10. Recommendations (TL;DR)

| Need | Pick |
|------|------|
| **Best operating point (F1 + Spec)** | `2CNN_MLP_FROZEN` (F1 0.9945, Spec 1.0, 0 FP) |
| **Best storage / cost trade-off** | `2CNN_MLP_FROZEN` (2 backbones, 178 MB) |
| **Best single-backbone baseline** | `3D_ResNet18` (AUC 0.9977, F1 0.9890) |
| **Best head (frozen regime)** | **MLP** (wins both frozen regimes) |
| **Best head (scratch regime)** | **KAN** (wins both scratch regimes) |
| **Best training regime** | **Frozen** (wins all 4 regimes on test) |
| **Do *not* ship** | `3CNN_MLP_SCRATCH` (worst F1 0.9462, Spec 0.8372, 7 FP) |
| **Most reproducible** | Any frozen hybrid (all pass 28/28 audit) |
