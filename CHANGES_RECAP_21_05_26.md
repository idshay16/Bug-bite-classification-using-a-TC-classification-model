# Changes Recap — 21 May 2026

> TC Domain Pre-Training pipeline implementation across `notebooks/Multiclass_Classification.ipynb` and `miscellaneous_code/pytorch_utils.py`.

---

## Overview — What Changed and Why

The multiclass ensemble was previously trained directly: **ImageNet → Bug bites**.  
Today's work introduces an intermediate domain warm-up step:

```
ImageNet weights
    ↓  Phase 0  — Cyclone classification (5 TC classes, 1 844 images)
                  [backbone acquires radial/gradient feature detectors]
Cyclone backbone extracted  (cyclone head discarded, backbone weights only)
    ↓  Phase 1  — Freeze backbone → train fresh bug-bite head (5 bug classes)
    ↓  Phase 2  — Unfreeze backbone → full fine-tune on bug bites at low lr
TC-pretrained bug-bite model saved
```

The hypothesis: cyclone satellite images share morphological properties with bug bites (radial, ring-shaped, gradient circular patterns). A backbone that has learned to detect these features on cyclone data should require less fine-tuning data to generalise to bug bites.

The binary gate (DINOv2 ViT-S) is **not** retrained — TC pre-training applies only to the Stage 2 multiclass ensemble.

---

## File 1 — `miscellaneous_code/pytorch_utils.py`

### 1a. `get_pytorch_loaders` — augmentation support

**Before:**
```python
def get_pytorch_loaders(train_dir, val_dir, img_size=310, batch_size=4):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
```

**After:**
```python
def get_pytorch_loaders(train_dir, val_dir, img_size=310, batch_size=4, augment=False):
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([...])  # unchanged
```

Existing callers that omit `augment` are unaffected. Cyclone and bug-bite loaders pass `augment=True`.

---

### 1b. `train_pytorch_model` — AMP dtype fix (bfloat16)

**Before:**
```python
use_amp = device.type == 'cuda'
scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
...
with torch.autocast(device_type=device.type, enabled=use_amp):
```

**After:**
```python
use_amp = device.type == 'cuda'
# bfloat16 has float32-range exponents — avoids NaN in transformer SDPA under AMP
scaler  = torch.amp.GradScaler('cuda', enabled=False)
...
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
```

**Why:** Both SwinV2-Large and DINOv2-Large use scaled dot-product attention internally. Under the default `float16` AMP, SDPA can produce NaN gradients that silently corrupt training or hang the CUDA kernel. `bfloat16` has the same dynamic range as `float32` (8-bit exponent), eliminating underflow/overflow — no gradient scaler is needed.

The global SDP flags in the notebook (cell 2) complement this by forcing the math backend:
```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

---

### 1c. `train_pytorch_model` — gradient checkpointing + `phase2_batch_size`

**Before:**
```python
def train_pytorch_model(model, train_loader, val_loader, device,
                        phase1_epochs=5, phase2_epochs=20, patience=5, save_path=None,
                        phase1_batch_size=None):
    ...
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)
    patience_ctr = 0
    run_phase(phase2_epochs, optimizer, 'Phase 2: fine-tuning full backbone',
              train_loader, val_loader)
```

**After:**
```python
def train_pytorch_model(model, train_loader, val_loader, device,
                        phase1_epochs=5, phase2_epochs=20, patience=5, save_path=None,
                        phase1_batch_size=None, phase2_batch_size=None):
    ...
    for param in model.parameters():
        param.requires_grad = True
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(True)
    if phase2_batch_size is not None:
        p2_train_loader = DataLoader(train_loader.dataset, batch_size=phase2_batch_size,
                                     shuffle=True,  num_workers=0)
        p2_val_loader   = DataLoader(val_loader.dataset,   batch_size=phase2_batch_size,
                                     shuffle=False, num_workers=0)
    else:
        p2_train_loader, p2_val_loader = train_loader, val_loader
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)
    patience_ctr = 0
    run_phase(phase2_epochs, optimizer, 'Phase 2: fine-tuning full backbone',
              p2_train_loader, p2_val_loader)
```

**Why:** Phase 1 freezes the backbone so only the head's gradients are computed (~5 GB VRAM on A5000). Phase 2 unfreezes the full backbone — PyTorch must store every intermediate activation for the backward pass, jumping to ~23 GB and overflowing to system RAM, which saturates the PCIe bus and hangs the kernel rather than throwing a clean OOM.

- **Gradient checkpointing** (`set_grad_checkpointing`) recomputes activations during the backward pass instead of storing them. Cost: ~30% extra compute. Benefit: activation memory reduced 5–10×. DINOv2-Large (307 M params, 518×518) stabilised at 8 GB VRAM on the A5000 with this fix.
- **`phase2_batch_size`** provides a second control knob. SwinV2-Large uses `SWINV2_P2_BATCH = 4` (half the default of 8) because windowed attention at 384×384 retains more intermediate spatial feature maps than the uniform ViT architecture.

---

### 1d. New helper functions

**`get_pt_probs_preds`** — returns `(true_labels, argmax_preds, softmax_probs)` for full metric computation:
```python
def get_pt_probs_preds(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out   = model(imgs.to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_labels, all_preds, all_probs
```

**`get_backbone_state_dict`** — strips the classifier head from a cyclone checkpoint so only the backbone transfers into a fresh bug-bite model:
```python
def get_backbone_state_dict(state_dict, head_prefix='head.'):
    return {k: v for k, v in state_dict.items() if not k.startswith(head_prefix)}
```

---

## File 2 — `notebooks/Multiclass_Classification.ipynb`

### Structure — complete rewrite (54 cells → 28 cells)

The previous notebook trained three Keras/PyTorch models sequentially with no shared infrastructure. The new notebook is restructured into five sections:

| Section | Cells | Purpose |
|---|---|---|
| Imports & Setup | 0–2 | All library imports, device detection, SDP flags |
| Helper Functions | 3–5 | Custom Keras layers, `full_metrics()`, `_BASELINE` |
| Configuration | 6–8 | All tunable variables, pytorch_utils import, data setup |
| Model Training & Eval | 9–20 | 4 cells × 3 models (header / loading / training / eval) |
| Feature Map Inspection | 21–25 | Cyclone backbone visualisation for all 3 models |
| TC Ensemble Evaluation | 26–27 | Load all 3 TC models, soft-vote, compare vs baseline |

---

### 2a. Separate epoch budgets for cyclone vs bug-bite phases

**Before** (single set of epoch vars, same count for both domains):
```python
swinv2_p1_epochs  = 5
swinv2_p2_epochs  = 20
```

**After** (split into two sets):
```python
# Cyclone (Phase 0) — lighter touch to avoid over-specialising to TC features
swinv2_tc_p1_epochs = 3;   swinv2_tc_p2_epochs = 10
dinov2_tc_p1_epochs = 3;   dinov2_tc_p2_epochs = 10
efficientnet_tc_p1_epochs = 3; efficientnet_tc_p2_epochs = 10

# Bug-bite (Phase 1+2) — the actual task, full budget
swinv2_p1_epochs  = 5;   swinv2_p2_epochs  = 20
dinov2_p1_epochs  = 5;   dinov2_p2_epochs  = 20
efficientnet_p1_epochs = 5; efficientnet_p2_epochs = 20
```

---

### 2b. `full_metrics()` evaluation helper

Replaces `evaluate_pytorch_model` (which only printed a classification report and a confusion matrix) with a unified function that covers all decision-support metrics in one call:

```python
_BASELINE = {          # Dataset B (Bites Only) — existing ensemble without TC pre-training
    'accuracy':      0.930,
    'balanced_acc':  0.9370,
    'macro_roc_auc': 0.9839,
    'kappa':         0.9122,
    'mcc':           0.9128,
}

def full_metrics(true_labels, preds, probs, class_names, title='', compare_baseline=False):
    # 1. classification_report (per-class precision/recall/F1)
    # 2. Scalar metrics: accuracy, balanced accuracy, macro ROC-AUC, Cohen's kappa, MCC
    #    — with optional Δ vs _BASELINE when compare_baseline=True
    # 3. Three-panel plot:
    #    Panel 1 — Confusion matrix (viridis, normalised colour scale)
    #    Panel 2 — Per-class F1 bar chart (horizontal, sorted)
    #    Panel 3 — ROC curves OvR for each class + macro average
```

Individual model eval uses `compare_baseline=False` (no per-model baseline exists).  
TC ensemble eval uses `compare_baseline=True` to show Δ against the Dataset B baseline directly.

---

### 2c. Backbone extraction pattern (per PyTorch model)

Each PyTorch model's training cell follows the same pattern:

```python
# Phase 0: train on cyclone data
tc_swinv2_c_hist = train_pytorch_model(
    tc_swinv2, tc_swin_train, tc_swin_val, device,
    phase1_epochs=swinv2_tc_p1_epochs, phase2_epochs=swinv2_tc_p2_epochs,
    save_path=CYCLONE_SWINV2_PATH, phase2_batch_size=SWINV2_P2_BATCH)

# Inspect cyclone backbone quality before discarding the head
evaluate_pytorch_model(tc_swinv2, tc_swin_val, cyclone_classes, device)
del tc_swinv2; torch.cuda.empty_cache()

# Strip cyclone head, load backbone weights into fresh bug-bite model
cyclone_state  = torch.load(CYCLONE_SWINV2_PATH, map_location=device)
backbone_state = get_backbone_state_dict(cyclone_state)          # removes 'head.*' keys
tc_swinv2_bug  = timm.create_model('swinv2_large_window12to24_192to384', pretrained=False, num_classes=5)
missing, _     = tc_swinv2_bug.load_state_dict(backbone_state, strict=False)
# missing == ['head.weight', 'head.bias'] — expected, randomly initialised

# Phase 1+2: fine-tune on bug bites
tc_swinv2_bug_hist = train_pytorch_model(
    tc_swinv2_bug, pt_swin_train, pt_swin_val, device,
    phase1_epochs=swinv2_p1_epochs, phase2_epochs=swinv2_p2_epochs,
    save_path=TC_SWINV2_PATH, phase2_batch_size=SWINV2_P2_BATCH)
```

---

### 2d. Feature map inspection section (cells 21–25)

After all three models complete cyclone pre-training, a dedicated section visualises what the backbones learned before the cyclone head is discarded. Each model is loaded from its cyclone checkpoint, a forward hook is attached to an intermediate layer, and feature maps are displayed for both a cyclone image and a bug-bite image side by side.

| Model | Hook target | Token handling |
|---|---|---|
| SwinV2-Large | `layers[0].blocks[-1]` | Reshape H×W×C from flat sequence |
| DINOv2-Large | `blocks[-1]` | Skip CLS (idx 0) + 4 register tokens (idx 1–4); patch tokens idx 5+ reshaped to 37×37 |
| EfficientNetV2M | First conv layer with ≥32 output channels | Standard 4-D feature map |

**What to look for:** radial/ring activations on the cyclone image should also fire on the bug-bite region. Flat/uniform maps on the bug-bite image indicate the feature transfer hypothesis doesn't hold for that model.

---

### 2e. Bug fixes applied during the session

| Issue | Root cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pytorch_utils` | `sys.path.append(os.path.dirname(os.getcwd()))` fails when Jupyter is launched from project root, not `notebooks/` | Moved `sys.path.insert(0, os.path.join(PROJECT_ROOT, 'miscellaneous_code'))` to cell 7, immediately after `PROJECT_ROOT` is defined |
| `cyclone_classes` NameError in DINOv2 section | Cell 14 discarded classes with `_`; `cyclone_classes` only existed if SwinV2 section had run first | Changed to `tc_dino_train, tc_dino_val, cyclone_classes = get_pytorch_loaders(...)` — cell 14 now self-contained |
| SwinV2-Large VRAM overflow / kernel freeze | Phase 2 VRAM jumped from 5 GB → 23 GB + 10 GB system RAM; PCIe saturation caused a silent kernel hang rather than a clean OOM | Gradient checkpointing + `phase2_batch_size=4` (`SWINV2_P2_BATCH`) |

---

## Weight Files Produced

| File | Classification target | Status |
|---|---|---|
| `cyclone_swinv2_large.pt` | Cyclone (intermediate) | Backbone extracted; head discarded |
| `cyclone_dinov2_large.pt` | Cyclone (intermediate) | Backbone extracted; head discarded |
| `cyclone_efficientnetv2m_backbone.keras` | Cyclone (intermediate) | Backbone saved; cyclone head discarded |
| `multiclass_swinv2_large_tc_best.pt` | **Bug bites** | TC pre-trained ensemble member |
| `multiclass_dinov2_large_tc_best.pt` | **Bug bites** | TC pre-trained ensemble member |
| `multiclass_efficientnetv2m_tc_model.keras` | **Bug bites** | TC pre-trained ensemble member |

---

## Next Step

Run `full_metrics(compare_baseline=True)` in the ensemble cell (27) and compare Δ against:

| Metric | Baseline (Dataset B) |
|---|---|
| Accuracy | 0.930 |
| Balanced Accuracy | 0.9370 |
| Macro ROC-AUC | 0.9839 |
| Cohen's Kappa | 0.9122 |
| MCC | 0.9128 |

If TC pre-training improves the ensemble: update `Stacked_Model.ipynb` to load the `*_tc_*` weight files. If it does not improve: inspect feature maps (cells 22–25) to determine whether the morphological similarity hypothesis holds or whether the cyclone epoch budget needs adjustment.
