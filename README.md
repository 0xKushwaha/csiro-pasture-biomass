# CSIRO Pasture Biomass Prediction Challenge

🥈 **Silver Medal Solution - Rank 138/3802 teams**

Deep learning solution for predicting pasture biomass components from aerial imagery using vision transformers and biological constraint enforcement.

---

## 📊 Competition Overview

**Competition**: [CSIRO Pasture Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass)

**Objective**: Predict five biomass components from 2000×1000 pixel pasture images:
- Dry Green Biomass (g/m²)
- Dry Dead Biomass (g/m²)
- Dry Clover Biomass (g/m²)
- Green Dry Matter (GDM, g/m²)
- Total Dry Biomass (g/m²)

**Evaluation Metric**: Globally weighted R² with component-specific weights
- Dry_Total_g: 0.5
- GDM_g: 0.2
- Dry_Green_g, Dry_Dead_g, Dry_Clover_g: 0.1 each

**Dataset**: 357 unique training images from Australian pastures across NSW, Tasmania, Victoria, and WA

---

## 🏆 Results

- **Public Leaderboard**: Rank 138 (Silver Medal)
- **Final Score**: ~0.62+ R²
- **Key Innovation**: Biological constraint enforcement through architectural design

---

## 🔬 Solution Approach

### Key Insights

1. **Biological Constraints Matter**
   - Only 3 of 5 targets are mathematically independent
   - GDM = Green + Clover
   - Total = Green + Dead + Clover
   - Enforcing these constraints improved performance significantly

2. **Multi-Crop Architecture**
   - Wide 2000×1000 images require careful handling
   - Dual-crop processing (left/right halves) preserves spatial detail
   - Significantly outperformed single-crop approaches

3. **Transform-Space Optimization Pitfalls**
   - Training in log-space or Box-Cox transformed space misaligned with R² evaluation
   - Original-scale training with robust scaling worked best

### Model Architecture

**Backbone**: DINOv3 Huge Plus (`vit_huge_plus_patch16_dinov3.lvd1689m`)
- State-of-the-art vision transformer pre-trained on massive datasets
- Gradient checkpointing enabled for memory efficiency
- Initially frozen for 5 epochs, then unfrozen for fine-tuning

**Architecture Pipeline**:
```
Input Image (2000×1000)
    ↓
Dual-Crop Processing (2× 512×512 from left/right halves)
    ↓
DINOv3 Feature Extraction (per crop)
    ↓
LocalMambaBlock Fusion (2 layers)
    ↓
Adaptive Pooling + MLP Head
    ↓
Single Target Output
```

**Separate Models Strategy**:
- Train **3 independent models** (Dead, Clover, Green)
- Each model has its own backbone, fusion layers, and regression head
- Sequential training: Dead → Clover → Green
- StratifiedKFold binning per target for balanced folds

**Target Prediction Strategy**:
- **Direct Predictions**: 3 separate models predict Dry_Green_g, Dry_Dead_g, Dry_Clover_g
- **Derived Predictions** (at inference): 
  - GDM_g = Dry_Green_g + Dry_Clover_g
  - Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g

### Training Strategy

**Regularization**:
- Backbone frozen for first 5 epochs
- Gradient checkpointing for memory efficiency
- Dropout: 0.35 in final MLP, 0.2 in LocalMambaBlock
- Data augmentation: HorizontalFlip, VerticalFlip, RandomRotate90, Rotation(±10°)
- Gradient clipping: 0.5
- Early stopping with patience=10

**Cross-Validation**:
- 4-Fold StratifiedKFold
- Separate binning for each target (Dead, Clover, Green)
- Ensures balanced target distribution per fold

**Optimization**:
- Loss function: RMSE in original scale
- AdamW optimizer with differential learning rates:
  - Backbone: 5e-5
  - Head: 2e-4
- Weight decay: 5e-4
- Cosine warmup scheduler (3 epochs)
- Gradient accumulation: 4 steps
- Mixed precision (AMP) training

**Sequential Training**:
1. Train Dead model (all 4 folds)
2. Train Clover model (all 4 folds)
3. Train Green model (all 4 folds)
4. Final validation using all 3 models together

### Ensemble Methods

**Multi-Model Architecture**:
- 3 separate models (Dead, Clover, Green)
- Each trained independently with 4-fold CV
- Total of 12 model checkpoints (3 targets × 4 folds)

**Inference Strategy**:
- Load all 3 models for each fold
- Generate predictions for Green, Dead, Clover independently
- Mathematically derive GDM and Total at inference time
- Average predictions across 4 folds

---

## 🛠️ Technical Stack

- **Framework**: PyTorch
- **Vision Models**: timm, transformers (HuggingFace)
- **Augmentation**: Albumentations
- **Ensemble**: XGBoost, LightGBM, CatBoost
- **Compute**: Kaggle (2× Tesla T4 / A100 80GB)

---

## 📁 Repository Structure

```
csiro-pasture-biomass/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── data/
│   └── README.md             # Data download instructions
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
├── src/
│   ├── dataset.py            # Custom dataset classes
│   ├── models/
│   │   ├── dual_crop_vit.py  # Dual-crop architecture
│   │   └── constraint_head.py # Biological constraint layer
│   ├── train.py              # Training pipeline
│   └── inference.py          # Inference script
├── configs/
│   └── dinov2_base.yaml      # Model configuration
└── submissions/
    └── best_submission.csv    # Final predictions
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/csiro-pasture-biomass.git
cd csiro-pasture-biomass
pip install -r requirements.txt
```

### Data Setup

Download competition data from [Kaggle](https://www.kaggle.com/competitions/csiro-pasture-biomass/data) and place in `data/` directory.

### Training

```bash
python src/train.py --config configs/dinov2_base.yaml
```

### Inference

```bash
python src/inference.py --checkpoint path/to/checkpoint.pth --output submissions/
```

---

## 📈 Key Learnings

1. **Small datasets require aggressive regularization**: With only 357 images, overfitting was the primary challenge

2. **Biological constraints are powerful inductive biases**: Enforcing known relationships between targets eliminated gradient conflicts

3. **Spatial resolution matters for wide images**: Multi-crop processing was essential for preserving fine-grained spatial information

4. **Evaluation metric alignment is critical**: Training transforms must align with evaluation metrics

5. **Simple often beats complex**: Frozen backbones with simple regression heads outperformed end-to-end fine-tuning

---

## 🙏 Acknowledgments

- CSIRO for organizing the competition and providing the dataset
- Kaggle community for discussions and insights
- Pre-trained vision transformer authors (DINOv2, DINOv3)

---
