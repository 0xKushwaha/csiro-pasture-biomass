# CSIRO Pasture Biomass Prediction Challenge

**Silver Medal Solution - Rank 140/3802 teams**

Deep learning solution for predicting pasture biomass components from aerial imagery using a dual-crop Vision Transformer with a Selective State Space Model (SSM) fusion layer.

---

## Competition Overview

**Competition**: [CSIRO Pasture Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass)

**Objective**: Predict five biomass components from 2000×1000 pixel aerial pasture images:
- Dry Green Biomass (`Dry_Green_g`, g/m²)
- Dry Dead Biomass (`Dry_Dead_g`, g/m²)
- Dry Clover Biomass (`Dry_Clover_g`, g/m²)
- Green Dry Matter (`GDM_g`, g/m²)
- Total Dry Biomass (`Dry_Total_g`, g/m²)

**Evaluation Metric**: Globally weighted R²
- `Dry_Total_g`: 0.5
- `GDM_g`: 0.2
- `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`: 0.1 each

**Dataset**: 357 unique training images from Australian pastures across NSW, Tasmania, Victoria, and WA

---

## Results

- **Public Leaderboard**: Rank 138 / 3802 (Silver Medal)
- **Final Score**: ~0.62+ weighted R²

---

## Solution Approach

### Key Insights

1. **Biological constraints as inductive bias**
   - Only 3 of 5 targets are mathematically independent: Green, Dead, Clover
   - `GDM = Green + Clover`, `Total = Green + Dead + Clover`
   - Deriving GDM and Total from base predictions at inference guarantees consistency and outperformed training a direct model for each

2. **Separate models per target**
   - Green, Dead, and Clover have very different distributions (Dead has ~40% zeros)
   - A shared multi-task backbone caused gradient conflicts
   - Three independent models each specialize for their target's distribution

3. **Dual-crop processing**
   - Directly resizing 2000×1000 → 512×512 distorts the 2:1 aspect ratio and halves effective resolution
   - Splitting into left/right 1000×1000 crops and resizing each to 512×512 preserves both
   - Improved R² by ~0.03 over single-crop

4. **Original-scale training**
   - Training in log-space or Box-Cox space misaligns with the R² evaluation metric
   - RMSE loss in original scale worked best

---

## Model Architecture

**Backbone**: DINOv3 Huge Plus (`vit_huge_plus_patch16_dinov3.lvd1689m`)
- ~1.3B parameter vision transformer
- Frozen for first 5 epochs, then fine-tuned end-to-end
- Gradient checkpointing enabled for memory efficiency

**LocalMambaBlock** (SSM fusion):
- Simplified selective state space model: `h_t = A·h_{t-1} + B_t·x_t`, `y_t = C_t·h_t + D·x_t`
- Input-dependent B and C make it selective (Mamba-style)
- Applied over concatenated left+right crop token sequences
- 2 layers, each with residual connection and sigmoid gate

**Architecture Pipeline**:
```
Input Image (2000×1000)
        ↓
Dual-Crop Split → left (1000×1000), right (1000×1000)
        ↓
Resize each to 512×512
        ↓
DINOv3 Feature Extraction (shared backbone, per crop)
        ↓
Concatenate token sequences [2×seq_len, dim]
        ↓
LocalMambaBlock Fusion (2 layers)
        ↓
Adaptive Average Pooling + MLP Head
        ↓
Single target prediction
```

**Prediction strategy**:
- 3 models trained independently: Dead → Clover → Green
- At inference: `GDM = Green + Clover`, `Total = Green + Dead + Clover`
- 12 checkpoints total (3 targets × 4 folds), averaged at inference

---

## Training Strategy

**Regularization**:
- Backbone frozen for first 5 epochs to stabilize head before fine-tuning
- Dropout: 0.35 in MLP head, 0.2 in LocalMambaBlock
- Early stopping with patience=10
- Gradient clipping: 0.5

**Augmentation** (training only, synchronized across both crops):
- HorizontalFlip (p=0.5), VerticalFlip (p=0.5), RandomRotate90 (p=0.5)
- Rotation ±10° (p=0.3)
- ColorJitter disabled — found to hurt performance

**Cross-Validation**:
- 4-Fold StratifiedKFold with separate quantile binning per target
- Ensures balanced target distribution across folds despite small dataset

**Optimization**:
- Loss: RMSE in original scale
- AdamW with differential LR — backbone: 5e-5, head: 2e-4
- Weight decay: 5e-4
- Cosine warmup (3 epochs) + cosine annealing
- Gradient accumulation: 4 steps (effective batch size 16)
- Mixed precision (AMP)

---

## Technical Stack

- **Framework**: PyTorch
- **Vision Models**: timm (DINOv3 Huge Plus)
- **Augmentation**: Albumentations
- **CV**: 4-fold StratifiedKFold with prediction averaging
- **Compute**: Kaggle (2× Tesla T4 / A100 80GB)

---

## Repository Structure

```
csiro-pasture-biomass/
├── README.md
├── requirements.txt
├── configs/
│   └── dinov2_base.yaml        # Hyperparameter config (overrides CFG defaults)
├── data/
│   ├── README.md               # Data download instructions
│   ├── train.csv
│   ├── test.csv
│   ├── train_images/
│   └── test_images/
├── src/
│   ├── dataset.py              # BiomassDataset, transforms, collate_fn
│   ├── models/
│   │   └── dual_crop_vit.py    # LocalMambaBlock + BiomassModelSingle
│   ├── train.py                # Training pipeline (3 targets × 4 folds)
│   └── inference.py            # Generates submission CSV
└── submissions/
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/csiro-pasture-biomass.git
cd csiro-pasture-biomass
pip install -r requirements.txt
```

### Data Setup

Download competition data from [Kaggle](https://www.kaggle.com/competitions/csiro-pasture-biomass/data) and place in `data/`:

```
data/
├── train.csv
├── test.csv
├── train_images/
└── test_images/
```

### Training

```bash
# Using default config (CFG class in train.py)
python src/train.py

# Override config with YAML
python src/train.py --config configs/dinov2_base.yaml

# Debug mode (100 samples, 2 epochs)
python src/train.py --debug

# Specific folds only
python src/train.py --folds 0,1

# Skip training, run final validation only
python src/train.py --skip_training
```

Checkpoints saved to `./v2_trained/fold{fold}_{target}_best.pth`

### Inference

```bash
python src/inference.py
```

Output: `submissions/submission.csv`

---

## Key Learnings

1. **Separate models beat multi-task on small data**: With 357 images and targets with very different distributions, gradient conflicts in a shared backbone were damaging. Independent models per target removed this entirely.

2. **Dual-crop preserves resolution**: Wide images should not be naively resized. Splitting and processing each half independently maintains spatial detail critical for biomass estimation.

3. **Derive don't predict**: GDM and Total are mathematical combinations of base targets. Deriving them post-hoc avoids inconsistency (e.g. Total < GDM) and improves both heavily-weighted derived metrics.

4. **Metric alignment matters**: RMSE in original scale aligns with the R² evaluation metric. Log-space training does not.

5. **Freeze then fine-tune**: Freezing the large pretrained backbone for early epochs prevents the randomly initialized head from destroying pretrained representations before it stabilizes.

---

## Acknowledgments

- CSIRO for organizing the competition and providing the dataset
- Kaggle community for discussions and insights
- DINOv2/DINOv3 authors for the pretrained backbone
