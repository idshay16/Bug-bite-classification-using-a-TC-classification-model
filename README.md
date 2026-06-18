# Bug-Bite Classification via Tropical Cyclone Transfer Learning

**Braude College of Engineering — Software Engineering Capstone**

## Overview

This project investigates whether pre-training a CNN backbone on tropical cyclone (TC) satellite imagery improves classification of bug-bite skin images. The core hypothesis: cyclone images share visual properties with skin macro-photography (radial textures, soft gradients, subtle pattern differences between severity categories), so a TC-pretrained backbone may encode more useful inductive biases than ImageNet alone for the bug-bite task.

## Problem Statement

Dermatological bug-bite classification is hard — inter-class similarity is high, intra-class variation is large, and medical image datasets are small. Standard ImageNet pre-training leaves the backbone biased toward object-centric features. We test whether a structurally different domain (cyclone satellite imagery) provides a better feature prior.

## Research Design

Two training pipelines are compared across 3 architectures, 3 label-smoothing values, 3 learning rates, and 7 random seeds (63 configs total):

| Pipeline | Pre-training path |
|---|---|
| **Control** | ImageNet → Bug-bite fine-tune |
| **TC** | ImageNet → Cyclone pre-train → Bug-bite fine-tune |

### Models

| Architecture | timm ID | Input size |
|---|---|---|
| ConvNeXt-Tiny | `convnext_tiny.fb_in22k_ft_in1k` | 256 × 256 |
| DenseNet-121 | `densenet121.ra_in1k` | 256 × 256 |
| InceptionV3 | `inception_v3.tv_in1k` | 299 × 299 |

### Hyperparameter Sweep (cyclone pre-training phase only)

- `label_smoothing`: 0.0, 0.1, 0.2
- `phase2_lr`: 1e-6, 5e-6, 1e-5
- Seeds: 7 (0–6)

Bug-bite fine-tuning always uses `label_smoothing=0.0`, `phase2_lr=5e-6`.

## Datasets

### Bug-Bite (target task)
- **5 classes**: `ants`, `bed_bugs`, `mosquitos`, `spiders`, `ticks_fleas`
- Located at: `/home/test/bug_data/{train,val}` (WSL native path)
- Also includes a binary split (`bites` / `no_bites`) used in early experiments

### Cyclone (auxiliary pre-training task)
- **Source**: HURSAT-B1 infrared satellite imagery matched to IBTrACS intensity labels
- **Years**: 1995–2016
- **5 intensity classes** (mapped to Saffir-Simpson categories)
- Up to 4000 images per class (train), 500 per class (val)
- Fetch script: `miscellaneous_code/fetch_cyclone_dataset.py`
- Located at: `/home/test/cyclone_data_split/{train,val}`

## Repository Structure

```
.
├── notebooks/
│   ├── Multiclass_Classification.ipynb   # Main unified notebook (4-section pipeline)
│   └── Binary_Classification.ipynb       # Early binary (bite / no-bite) notebook
├── Stacked_Model.ipynb                   # YOLO + classifier stacked approach (legacy)
├── scripts/
│   ├── shared.py                         # Shared imports, paths, metrics, GradCAM helpers
│   ├── 01_train_control.py               # Section 1: control training
│   ├── 02_train_tc.py                    # Section 2: TC pre-train + bug-bite fine-tune
│   ├── 03_evaluate.py                    # Section 3: feature maps + GradCAM comparison
│   └── 04_xai.py                         # Section 4: LIME, SHAP, DiCE, t-SNE, ensemble
├── miscellaneous_code/
│   ├── run_experiments.py                # Hyperparameter sweep runner (main entry point)
│   ├── pytorch_utils.py                  # DataLoaders, train loop, backbone extraction
│   ├── fetch_cyclone_dataset.py          # Download + label HURSAT-B1 cyclone images
│   ├── cyclone_preprocessing.py          # Grid-line artifact removal from satellite PNGs
│   └── ...                              # Dataset inspection, augmentation, utility scripts
├── Bug-Data/
│   ├── Multiclass_Bug_data/             # 5-class bug-bite dataset
│   └── Binary_Bug_Data/                 # Binary dataset (bites / no_bites)
└── deploy/
    └── app.py                           # Legacy Streamlit demo (binary classifier)
```

## How to Run

### 1. Fetch and preprocess the cyclone dataset

```bash
python miscellaneous_code/fetch_cyclone_dataset.py            # full (~4000/class)
python miscellaneous_code/fetch_cyclone_dataset.py --trial    # quick test (5/class)
```

### 2. Run the full hyperparameter sweep

```bash
# from repo root, with WSL datasets mounted at /home/test/
python miscellaneous_code/run_experiments.py
```

**Pause/resume**: `touch results/PAUSE` or `Ctrl-C` — checkpoints are preserved, re-running picks up where it left off.

**Status**: `python miscellaneous_code/run_experiments.py --status`

### 3. Run individual pipeline stages

```bash
# Control training
python scripts/01_train_control.py \
    --run-id s0_ls0.10_lr5.0e-06 --seed 0 \
    --ctrl-convnext results/checkpoints/control_seed_0/control_convnext.pt \
    --ctrl-densenet results/checkpoints/control_seed_0/control_densenet.pt \
    --ctrl-inception results/checkpoints/control_seed_0/control_inception.pt

# TC pre-training + fine-tuning
python scripts/02_train_tc.py \
    --run-id s0_ls0.10_lr5.0e-06 --seed 0 \
    --label-smoothing 0.1 --phase2-lr 5e-6 \
    --cyc-convnext results/checkpoints/config_000/cyclone_convnext.pt \
    --cyc-densenet results/checkpoints/config_000/cyclone_densenet.pt \
    --cyc-inception results/checkpoints/config_000/cyclone_inception.pt \
    --tc-convnext  results/checkpoints/config_000/tc_bug_convnext.pt \
    --tc-densenet  results/checkpoints/config_000/tc_bug_densenet.pt \
    --tc-inception results/checkpoints/config_000/tc_bug_inception.pt

# Feature maps + GradCAM visualization
python scripts/03_evaluate.py --run-id <run-id> \
    --ctrl-convnext <path> --ctrl-densenet <path> --ctrl-inception <path> \
    --cyc-convnext  <path> --cyc-densenet  <path> --cyc-inception  <path> \
    --tc-convnext   <path> --tc-densenet   <path> --tc-inception   <path>

# XAI: LIME, SHAP, DiCE, t-SNE, ensemble
python scripts/04_xai.py --run-id <run-id> \
    --ctrl-convnext <path> --ctrl-densenet <path> --ctrl-inception <path> \
    --tc-convnext   <path> --tc-densenet   <path> --tc-inception   <path>
```

### 4. Results

- Training curves, per-class F1, ROC curves → `results/plots/<run-id>/`
- Feature map comparisons → `results/feature_maps/<run-id>/`
- GradCAM overlays → same as feature maps
- XAI outputs (SHAP, LIME, DiCE, t-SNE) → `results/xai/<run-id>/`
- Per-run metrics JSON → `results/metrics/<run-id>_{control,tc}.json`
- Sweep results → `results/experiment_results.json`

## XAI Methods

| Method | Purpose |
|---|---|
| **GradCAM** | Visual comparison of control vs TC attention maps |
| **SHAP** | Per-pixel attribution (GradientExplainer) |
| **LIME** | Superpixel-based local explanation |
| **DiCE** | Counterfactual explanations in embedding space (PCA-reduced) |
| **t-SNE** | Embedding cluster visualization (silhouette score reported) |
| **Ensemble** | Soft-vote across all 3 architectures, control vs TC compared |

## Dependencies

Core: `torch`, `torchvision`, `timm`, `numpy`, `Pillow`, `opencv-python`, `scikit-learn`, `matplotlib`

Optional XAI: `shap`, `lime`, `scikit-image`, `dice-ml`

Cyclone fetch: `netCDF4`, `requests`, `pandas`, `tqdm`

## Model Weights

Trained checkpoints are not stored in this repository. To obtain:

- **Selected weights** (best-performing configs used in the paper)
- **Full sweep weights** (all 63 configs × 3 models × 2 pipelines)

Contact: **ishay.yulzary@gmail.com**

## Environment Note

Dataset paths are hardcoded to WSL native paths (`/home/test/bug_data/`, `/home/test/cyclone_data_split/`). Edit `scripts/shared.py` and `miscellaneous_code/run_experiments.py` to change them.
