KAN ,MLP in final model for classification  
Multiple models for feature extraction \- resnet 50, densenet 121, efficienet b0 and tiny vit  
These are the ideas given by our supervisor.

Input (2.5D 3-channel patch)  
        │  
   ┌────┴────┬──────────────┐  
CNN-1   CNN-2    \[Transformer Extractors\]  
                  ├── TinyViT-21M   (global attn, hierarchical windows)  
                  ├── Swin-Tiny     (shifted window, multi-scale spatial)  
                  └── MobileViT-S  (CNN+local transformer hybrid)  
        │  
   Concatenate features  
        │  
   MLP / KAN Classifier

from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise, RandAffine  
train\_transforms \= Compose(\[  
    \# \---------- NEW: random translation to break positional shortcut \----------  
    RandAffine(  
        prob=0.8,                        \# apply to 80% of patches  
        translate\_range=(5, 5, 5),       \# up to 5 mm/voxels in each direction  
        padding\_mode='zeros',            \# fill empty borders with air (0)  
        spatial\_size=None,               \# keep the original 64×64×64 size  
    ),  
    \# \-------------------------------------------------------------------------  
    RandRotate90(prob=0.5, spatial\_axes=(0, 1)),  
    RandRotate90(prob=0.5, spatial\_axes=(1, 2)),  
    RandRotate90(prob=0.5, spatial\_axes=(0, 2)),  
    RandFlip(prob=0.5, spatial\_axis=0),  
    RandFlip(prob=0.5, spatial\_axis=1),  
    RandFlip(prob=0.5, spatial\_axis=2),  
    RandGaussianNoise(prob=0.2, std=0.01)  
\])

### **1\. Dataset Summary & Preprocessing Check**

Based on the `LUNA2_0_updated_final.ipynb` file you uploaded, your data pipeline logic is robust and fundamentally solid for medical imaging standards.

**What is Done (The Summary):**

·         **Patient-Level Splitting:** You correctly grouped by `seriesuid`. Your split yielded 888 patients (601 with nodules, 287 without). They are split into Train (70%), Val (15%), and Test (15%). (no leakage between train/val/test)

·         **CT Standardization:** You resampled the CT scans to a 1x1x1 mm isotropic resolution and normalized the Hounsfield Units (HU) between \-1000 and \+400, scaled to \[0, 1\].

·         **Patch Extraction:** Extracted exactly $64 \\times 64 \\times 64$ patches around the physical world coordinates mapped to Voxel indices.

·         **Hard Negative Mining:** You enforced a strict 5-negatives-per-positive ratio and implemented a crucial spatial deduplication step to ensure none of the false-positive candidates overlap with the physical radius of true nodules.

**LUNA16 Dataset Preprocessed\\**

**├── train\\**

**│   ├── pos\\**      	

**│   └── neg\\**     	

**├── val\\**

**│   ├── pos\\**

**│   └── neg\\**

**├── test\\**

**│   ├── pos\\**

**│   └── neg\\**

**├── metadata\_all.csv   (8,551 rows, includes filepath, split, label, world coordinates, nodule diameter)**

**└── metadata\_subset0.csv**

**…**

**└── metadata\_subset9.csv**

**└── patient\_split.csv  (patient-level split mapping)**

| Positive per split | Train: 822, Val: 192, Test: 172 |
| :---- | :---- |
| **Negative per split** | Train: 5,115, Val: 1,175, Test: 1,075 |

**Metadata\_all.csv first 3 lines look like this \-**

**filepath,seriesuid,subset,split,label,class\_name,coord\_world\_X,coord\_world\_Y,coord\_world\_Z,diameter\_mm**

**/content/drive/MyDrive/LUNA16\_processed\_64/test/pos/1.3.6.1.4.1.14519.5.2.1.6279.6001.395623571499047043765181005112\_pos\_966.npy,1.3.6.1.4.1.14519.5.2.1.6279.6001.395623571499047043765181005112,subset0,test,1,pos,-64.11832825,-4.887440096,-85.9024469,5.105120807**

**/content/drive/MyDrive/LUNA16\_processed\_64/test/neg/1.3.6.1.4.1.14519.5.2.1.6279.6001.395623571499047043765181005112\_neg\_0.npy,1.3.6.1.4.1.14519.5.2.1.6279.6001.395623571499047043765181005112,subset0,test,0,neg,-100.1600002,46.1493498,-196.2087502,**

**Patient\_split forst 2 contents –**

**seriesuid,has\_nodule,split**

**1.3.6.1.4.1.14519.5.2.1.6279.6001.385151742584074711135621089321,0,train**

**The Plan:** Individual Model Shootout → Hybrid → KAN vs MLP  
**Phase A – Individual Model Benchmark**  
Train each candidate solo with a simple MLP head. Keep everything identical: same data splits, same epochs, same optimizer. Then rank them by validation AUROC.

**Phase B – Hybrid Fusion**  
Pick the best performing 3D model and the best performing 2D or 3D Transformer (they capture different aspects). Combine them using the fusion strategy I described earlier. (If the top two are both CNNs, pick the best CNN and the best Transformer anyway – complementarity matters.)

**Phase C – The KAN Showdown with GRADCAM verification**  
With that fixed hybrid backbone, train MLP vs KAN. Clean comparison, strong thesis.  
Also use gradcam for xAI purposes.

### **Phase 1: Data Handoff & Pipeline Integration (Days 1-4)**

*Your groupmate does the heavy lifting with the raw .mhd files. Your job is to catch their output and build the PyTorch pipeline.*

* **Day 1: Environment & Architecture Setup**  
  * **Task:** Set up your PyTorch environment with CUDA optimized for the RTX 4090\. Install monai (for 3D models) and efficient-kan (for the KAN classifier).  
  * **Groupmate Task:** Execute the World-to-Voxel coordinate conversions and HU normalization.  
* **Day 2: The "Clean Slate" Handoff**  
  * **Task:** Receive the preprocessed data from your groupmate. Verify that the data is split strictly into Train (70%), Validation (15%), and Test (15%) at the **patient level** (no shared seriesuid). Lock the Test folder completely.  
  * **Groupmate Task:** Deliver the extracted $64 \\times 64 \\times 64$ balanced patches (Nodules vs. Hard Negatives).  
* **Day 3: Dataloaders & Augmentation**  
  * **Task:** Build your PyTorch Dataset and DataLoader. Set num\_workers=4 to protect your 32GB of RAM. Implement MONAI's 3D augmentations (random rotations, flips) *only* on the training set dataloader.  
* **Day 4: The Sanity Check**  
  * **Task:** Write a script to plot a batch of data from the dataloader. Visually confirm the patches look correct and the augmentations are applying properly before feeding them to the models.

---

### **Phase 2: Offline Feature Extraction (Days 5-9)**

*We leverage the RTX 4090 to chew through the 3D convolutions once, saving the features as flat vectors to make the classifier training lightning-fast.*

* **Day 5: DenseNet-121 Extraction**  
  * **Task:** Load DenseNet121 (3D version from MONAI). Strip the classification head. Run your entire Train, Val, and Test datasets through it without calculating gradients (torch.no\_grad()).  
  * **Outcome:** Save the output as densenet\_train\_features.npy, densenet\_val.npy, etc.  
* **Day 6: EfficientNet Extraction**  
  * **Task:** Load EfficientNet3D (via MONAI or a specialized PyTorch 3D repository). Strip the head, run the data through, and extract the features.  
  * **Outcome:** Save efficientnet\_train\_features.npy, etc.  
* **Day 7: Vision Transformer (ViT) Extraction**  
  * **Task:** Implement the 3D ViT (or extract the central 2.5D slices if using standard TinyViT). Run the data through the self-attention blocks and extract the final global feature vectors.  
  * **Outcome:** Save vit\_train\_features.npy, etc.  
* **Days 8 & 9: Feature Validation & Clean-up**  
  * **Task:** Check your .npy files. Ensure the tensor shapes match your dataset size (e.g., if you have 2,000 training patches, your DenseNet feature file should be shape (2000, 1024\)). Clear out your GPU VRAM and prepare for Phase 3\.

how to statistically prove the top 2 models are significantly better (DeLong test / bootstrap CI comparison) 

---

### **Phase 3: The Classifier Showdown (Days 10-14)**

*Because you are using offline features, training these classifiers will take minutes, not hours. This is where you run your $3 \\times 2$ matrix.*

* **Day 10: Building the MLP Baseline**  
  * **Task:** Construct a standard Multi-Layer Perceptron (nn.Linear $\\rightarrow$ nn.ReLU $\\rightarrow$ nn.Dropout $\\rightarrow$ nn.Linear). Set up Binary Cross Entropy Loss and the AdamW optimizer.  
* **Day 11: Building the KAN Novelty**  
  * **Task:** Integrate the Kolmogorov-Arnold Network using the efficient-kan library. Ensure its input layer matches the dimension of your saved feature vectors.  
* **Day 12: Matrix Execution (DenseNet & EfficientNet)**  
  * **Task:** Load the DenseNet features. Train the MLP and save the metrics. Train the KAN and save the metrics. Clear memory. Load the EfficientNet features. Train both classifiers and save metrics.  
* **Day 13: Matrix Execution (ViT & Hyperparameter Tuning)**  
  * **Task:** Load the ViT features. Train both the MLP and the KAN.  
  * **Outcome:** You now have the validation accuracy/loss for all 6 model combinations.  
* **Day 14: Selection & Finalization**  
  * **Task:** Look at the validation results. Identify the absolute best MLP combination and the absolute best KAN combination. Freeze their weights.

---

### **Phase 4: Evaluation, Interpretability, & Writing (Days 15-20)**

*Generating the proof that your models work and writing the final chapters.*

* **Day 15: The Locked Test Set Evaluation**  
  * **Task:** For the first time, run the Test Set features through your top models. Calculate the clinical metrics: Sensitivity (Recall), Specificity, and total Accuracy.  
* **Day 16: Graph Generation**  
  * **Task:** Write a Python script using matplotlib and sklearn. Plot the ROC Curves for the MLP vs. KAN on the exact same graph so the difference is visually obvious. Generate Confusion Matrices.  
* **Day 17: Interpretability (The "Why")**  
  * **Task:** For the winning backbone, generate a 3D Grad-CAM heatmap on a few Test patches to show *where* it looked. For the winning KAN, export a visualization of its learned activation splines to show *how* it made its decision.  
* **Day 18: Updating Chapter 4 (Methodology)**  
  * **Task:** Write the technical explanation of your pipeline. Detail the offline feature extraction strategy, the 3 backbones used, and explicitly contrast the mathematics of the MLP vs. the KAN.  
* **Day 19: Drafting Chapter 5 (Results)**  
  * **Task:** Insert your 6-model comparison table. Add the ROC graphs and Confusion Matrices. Write the discussion explicitly answering whether KAN outperformed MLP in reducing False Positives.  
* **Day 20: Slide Deck & Buffer**  
  * **Task:** Build your defense slides. Keep text minimal; rely on your Grad-CAM heatmaps and ROC curves to tell the story. Rehearse your timing.

