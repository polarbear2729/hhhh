The following table provides a comparison of recent (2025–2026) research papers directly comparable to your **LungCancer3D** project, focusing on model architectures, evaluation protocols, and performance metrics on LUNA16 and LIDC-IDRI datasets.

### Comparison Table: 3D Lung Nodule Classification & Detection (2025–2026)

| Paper | Year | Model Architecture | Dataset | AUC | Sens | Spec | F1 | Evaluation | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MSA-Net** | 2025 | 3D RTConvBlock (Residual Conv + Transformer) | LUNA16 | 0.993 | 0.963 | 0.947* | 0.955 | Patch-level (48x48x48) | Captures fine-grained internal features via multi-head self-attention. |
| **LMLCC-Net** | 2025 | Multi-branched 3D CNN (Learnable HU filters) | LUNA16 | 0.941 | 0.929 | 0.922 | NR | Patch-level (32x32x32) | Uses semi-supervised learning to label ambiguous cases. |
| **CPLOYO** | 2025 | YOLOv8 + KAN-Bottleneck + RepViT | LUNA16 | NR | 0.898 | NR | NR | Nodule-level (2D slices) | Directly compares **KAN vs MLP**; KAN improved small nodule detection. |
| **Alzahrani et al.** | 2025 | Faster R-CNN + ResNet50 (2D) | LUNA16 | NR | 0.999** | NR | NR | 10-Fold CV | Focuses on False Positive Reduction (FPR) using pre-trained backbones. |
| **Sungheetha et al.** | 2026 | Ensemble (CNN + LSTM + Transformer) | LIDC-IDRI | 0.947 | 0.891 | 0.876 | NR | 10-Fold CV | Part of a "Digital Twin" framework; LUNA16 external AUC reached 0.967. |
| **Ilse et al.** | 2025 | Foundation Models (MedImageInsight & RAD-DINO) | Holdout VinDR | NR | NR | NR | NR | **Frozen Backbone** | Studied scaling laws; MI2 scaled better for findings than RAD-DINO. |
| **MedKAN** | 2025 | Local & Global Information KAN (MedKAN) | 9 Public Datasets | NR | NR | NR | NR | NR | Explores KAN for medical image texture and context. |
| :--- | :-- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LungCancer3D ★** | **2026** | **2 frozen 3D backbones + MLP head** | **LUNA16** | **0.9974** | **0.9890** | **1.0000** | **0.9945** | **Patient-level test** | **Zero FP on 43 negatives; best model from 11-run sweep** |
| **LungCancer3D** | 2026 | 3 frozen 3D backbones + MLP head | LUNA16 | 0.9964 | 0.9890 | 1.0000 | 0.9945 | Patient-level test | Tied best F1/Spec; 0 FP |
| **LungCancer3D** | 2026 | 3D ResNet-18 | LUNA16 | 0.9977 | 0.9890 | 0.9767 | 0.9890 | Patient-level test | Best single-model baseline |

*\*Reported as Precision in source; NR = Not Reported; \*\*Sensitivity at 8 FP/scan.*

---

### Key Methodological Overlaps with LungCancer3D

*   **KAN vs MLP Comparisons:** Similar to your project, **CPLOYO** replaced traditional MLP structures with a **KAN-Bottleneck** to capture complex nonlinear relationships, specifically to improve accuracy for small nodules. **MedKAN** and **CEST-LKAN** also investigate the robustness of KAN over MLP in medical imaging settings.
*   **Frozen Backbones:** The **MedImageInsight (MI2)** study systematically evaluated **frozen backbones** for findings classification, aligning with your frozen multi-backbone ensemble approach. They found that for some tasks, as few as 30k in-domain samples were sufficient for a frozen encoder to surpass open-weights baselines.
*   **3D Multi-Scale / Hybrid Architectures:** **MSA-Net** utilizes 3D RTConvBlocks to combine the local feature extraction of residual convolutions with the global dependency modeling of Transformers, mirroring your hybrid ensemble's objective of capturing both fine-grained and contextual details.
*   **Hounsfield Unit (HU) Intensity Filtering:** **LMLCC-Net** introduces a "Learnable Dynamic Range Layer" that optimizes HU intensity boundaries during training to focus on tissue densities (e.g., soft tissue, fibrotic cores), which relates to your hard negative sampling from candidate data.
*   **Data Leakage and Patient-Level Splits:** Multiple papers emphasized the necessity of patient-level stratification. For instance, **MSA-Net** and **LMLCC-Net** utilized nodule-ID-based splits and rigorous screening to ensure consistent and reliable evaluation.

Based on the documents in your notebook, here are the links and identifiers for the research papers and articles directly comparable to your **LungCancer3D** project:

### Primary Research Papers (2025–2026)

*   **MSA-Net: multiple self-attention mechanism for 3D lung nodule classification in CT images** (Pan et al., 2025)
    *   Link: [https://doi.org/10.1186/s12880-025-01725-x](https://doi.org/10.1186/s12880-025-01725-x)
*   **LMLCC-Net: A Semi-Supervised Deep Learning Model for Lung Nodule Malignancy Prediction using Hounsfield Unit-Based Intensity Filtering** (Mamun et al., 2025)
    *   Link: [https://doi.org/10.1109/ACCESS.2024.0429000](https://doi.org/10.1109/ACCESS.2024.0429000)
*   **Accurate Detection of Pulmonary Nodule and False Positive Reduction with Faster R-CNN and ResNet Models** (Alzahrani & Solaiman, 2025)
    *   Link: [https://doi.org/10.22937/IJCSNS.2025.25.2.1](https://doi.org/10.22937/IJCSNS.2025.25.2.1)
*   **Explainable AI digital twin framework for early lung disease detection** (Sungheetha et al., 2026)
    *   Link: [https://doi.org/10.3389/fcomp.2026.1652980](https://doi.org/10.3389/fcomp.2026.1652980)
*   **MedKAN: An Advanced Kolmogorov-Arnold Network for Medical Image Classification** (Yang et al., 2025)
    *   Link: [https://doi.org/10.48550/arXiv.2502.18416](https://doi.org/10.48550/arXiv.2502.18416)

### Specialized Methodological Studies

*   **Data Scaling Laws for Radiology Foundation Models** (Ilse et al., 2025)
    *   *Note: This source includes the MedImageInsight (MI2) and RAD-DINO frozen backbone analysis.*
    *   MI2 Link: [http://arxiv.org/abs/2410.06542](http://arxiv.org/abs/2410.06542)
*   **CEST MRI data analysis using Kolmogorov-Arnold network (KAN) and Lorentzian-KAN (LKAN) models** (Wang et al., 2025)
    *   Link: [https://pubmed.ncbi.nlm.nih.gov/40468586/](https://pubmed.ncbi.nlm.nih.gov/40468586/)
*   **CPLOYO: A Pulmonary Nodule Detection Model with Multi-Scale Feature Fusion and Nonlinear Feature Learning** (Wang et al., 2025)
    *   Publication: [arXiv preprint](https://arxiv.org/abs/2502.00709)

