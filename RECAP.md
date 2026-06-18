# Project Recap — Bug-Bite Classification via TC Transfer Learning

Concise handoff document for collaborators. Covers the full commit history grouped by phase, what changed, and why.

---

## Phase 0 — Source Material (Aug 2022 – Feb 2026)

**Goal:** The pre-existing commits (before the team's work) are not ours. They are a base model and pipeline from a prior academic project that we gutted and used as a starting foundation. Binary classification (bite / no-bite), YOLOv3 detection, early ensemble notebooks — all inherited.

**Key artifacts carried forward:**
- `Bug-Data/` — bug-bite image dataset
- `deploy/app.py` — Streamlit binary demo (legacy, not maintained)
- Initial notebook structure and `pytorch_utils.py` skeleton

---

## Phase 1 — Initial Runs + Control Baseline (Mar 2026 – Apr 2026)

**Goal:** Run the inherited pipeline to establish a baseline and understand its performance before making changes.

**Key changes:**
- 2026-03-15: Colab notebook created, code structure refactored
- 2026-03-27: Fixed binary data pathing, PROJECT_ROOT constant, .gitattributes (notebook output stripping)
- 2026-04-14: First control run (bug-data only) to document baseline before modifying the classifier
- 2026-04-27: Multiclass notebook refactored — added ConvNeXtSmall, ResNet101, EfficientNetV2M with two-phase training; added YOLO annotation tools

---

## Phase 2 — PoC + TC Pipeline Build (May 2026)

**Goal (two-part):** (1) Prove the hypothesis could work on transformers as a PoC, then roll back to CNNs due to time and hardware limits. (2) Build and stabilize the full TC transfer-learning pipeline end-to-end on the `transfer-learn-implementation` branch (merged to `main` 2026-05-29).

### 2026-05-08 to 2026-05-12 — Data + YOLO cleanup
- Added `augment_data.py` for bug-bite dataset augmentation
- Refactored multiclass notebook: preprocessing, model selection, ensemble fixes
- Cleaned Stacked_Model notebook (YOLO pipeline) — later removed entirely as unneeded
- Removed YOLOv8 config, weights, results
- Updated YOLO to v3, then cleaned that too

### 2026-05-15 to 2026-05-18 — Transformer PoC
- Replaced CNNs with transformer-based models (ViT-class)
- Goal: quick PoC to validate TC hypothesis on attention-based architectures
- Outcome: rolled back — insufficient time and GPU hardware for transformer training at scale

### 2026-05-23 — Cyclone Data Quality Fix
- Fixed TC satellite image grid-line artifact removal (brightness-level detection + high-pass filtering)
- `cyclone_preprocessing.py` — `_highpass()`, `_flag_ranges()`, `residual_check()` with overlap exclusion
- Added compiled `pytorch_utils.py` (DataLoaders, train loop, backbone extraction)
- This was a critical fix: without it, models learned grid-line patterns instead of cyclone features

### 2026-05-27 — Architecture Lock-in + Bug Fixes
- Replaced transformers with VGG16/ResNet50/InceptionV3 first, then immediately swapped to ConvNeXt-Tiny (final choice: IN22k pretrained, 28M params)
- Fixed CPU data-loading bottleneck (num_workers)
- Fixed phase-boundary early stopping bug (val loss was checked at wrong epoch boundary)
- Fixed `grad_checkpointing` assertion crash for models that declare but don't support it
- Fixed cyclone class collapse: Cat-1 never predicted — added `auto_class_weights`
- Softened class weights to sqrt, set bug-bite batch=16

### 2026-05-28 — Dataset + Training Stabilization
- Moved dataset paths to WSL native FS (`/home/test/`) — eliminated I/O bottleneck from Windows→WSL mount
- Balanced cyclone dataset: all years, max 3000/class (later raised to 4000 train, 500 val)
- Swapped ResNet50 → DenseNet-121 (final model trio: ConvNeXt-Tiny, DenseNet-121, InceptionV3)
- Fixed cross-filesystem split bug (`os.link` → `shutil.copy2`)
- Synced all fixes between `main` and `transfer-learn-implementation` branches

### 2026-05-29 — Merge + Notebook Restructure
- `transfer-learn-implementation` merged into `main` (TC notebook version taken as canonical)
- `Multiclass_Classification.ipynb` restructured into 4 unified sections:
  1. Control training (ImageNet → Bug-bite)
  2. TC training (ImageNet → Cyclone → Bug-bite)
  3. Feature map + GradCAM comparison
  4. XAI: LIME, SHAP, DiCE, t-SNE, ensemble

---

## Phase 3 — Automated Sweep + Finalization (Late May – Jun 2026)

**Goal:** Clean repo to only what's actually used; extract scripts to run the full data collection (hyperparameter sweep across all seeds) automatically without manual intervention.

### 2026-05-30 — XAI Fix
- Fixed DiCE counterfactual cell — was crashing on embedding extraction; now uses `reset_classifier(0)` to get embeddings then PCA-reduces to 50D for DiCE

### 2026-05-31 — Script Extraction
- Extracted notebook sections into standalone scripts: `scripts/01_train_control.py`, `02_train_tc.py`, `03_evaluate.py`, `04_xai.py`, `shared.py`
- `miscellaneous_code/run_experiments.py` — hyperparameter sweep runner:
  - Grid: label_smoothing ∈ {0.0, 0.1, 0.2} × phase2_lr ∈ {1e-6, 5e-6, 1e-5} × 7 seeds = **63 configs**
  - Pause/resume via `results/PAUSE` file or Ctrl-C; crash-safe via checkpoint reuse
- Removed redundant files

### 2026-06-12 — Repo Cleanup
- Removed all YOLOv3-related scripts, custom data tools, annotation utilities
- Deleted old `CHANGES_RECAP_*` files
- Removed `results/` directory from tracking (added to `.gitignore`)
- Cleaned `__pycache__` directories

### 2026-06-17–18 — Final Touches
- Added results dirs to `.gitignore`
- Added `scripts/run_gradcam_book.py` — standalone GradCAM script with fixed gradient computation

---

## Current State (2026-06-18)

- **Main branch** is the working branch. `transfer-learn-implementation` is merged and inactive.
- **Pipeline entry point**: `miscellaneous_code/run_experiments.py`
- **Datasets**: WSL native paths `/home/test/bug_data/` and `/home/test/cyclone_data_split/`
- **Cyclone data fetch**: `miscellaneous_code/fetch_cyclone_dataset.py` (HURSAT-B1 + IBTrACS, 1995–2016)
- **63-config sweep** may or may not be complete — check `results/experiment_state.json` with `--status`
- **Models**: ConvNeXt-Tiny, DenseNet-121, InceptionV3 (all via `timm`)
- **XAI**: GradCAM, SHAP, LIME, DiCE, t-SNE, ensemble — all in `scripts/04_xai.py` and `03_evaluate.py`
