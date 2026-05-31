"""Shared imports, helpers, and path constants for all 4 pipeline scripts."""
import matplotlib
matplotlib.use('Agg')  # non-interactive — must be before pyplot import

import itertools
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as _T
from PIL import Image as _PILImage
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score, roc_curve, auc, silhouette_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'miscellaneous_code'))
from pytorch_utils import get_pytorch_loaders, get_pt_probs_preds  # noqa: E402

# ── dataset paths ─────────────────────────────────────────────────────────────
BUG_TRAIN   = '/home/test/bug_data/train'
BUG_VAL     = '/home/test/bug_data/val'
TC_TRAIN    = '/home/test/cyclone_data_split/train'
TC_VAL      = '/home/test/cyclone_data_split/val'

# ── model specs ───────────────────────────────────────────────────────────────
# (key, timm_id, img_size, cyc_batch, bug_batch)
MODEL_SPECS = [
    ('convnext',  'convnext_tiny.fb_in22k_ft_in1k', 256, 256, 16),
    ('densenet',  'densenet121.ra_in1k',             256, 256, 16),
    ('inception', 'inception_v3.tv_in1k',            299, 128, 16),
]
N_CLASSES = 5

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_MEAN_NP = np.array([0.485, 0.456, 0.406])
_STD_NP  = np.array([0.229, 0.224, 0.225])

# ── output dirs ───────────────────────────────────────────────────────────────
RESULTS_DIR = REPO_ROOT / 'results'

def out_dir(subdir: str, run_id: str) -> Path:
    p = RESULTS_DIR / subdir / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig(path: Path, dpi: int = 150):
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close('all')

# ── full_metrics ──────────────────────────────────────────────────────────────
def full_metrics(true_labels, preds, probs, class_names, title='', save_path=None):
    true_labels = list(true_labels)
    preds       = list(preds)
    probs_arr   = np.array(probs)

    acc   = accuracy_score(true_labels, preds)
    bacc  = balanced_accuracy_score(true_labels, preds)
    y_bin = label_binarize(true_labels, classes=list(range(len(class_names))))
    roc   = roc_auc_score(y_bin, probs_arr, average='macro', multi_class='ovr')
    kappa = cohen_kappa_score(true_labels, preds)
    mcc   = matthews_corrcoef(true_labels, preds)

    from sklearn.metrics import f1_score
    cm  = confusion_matrix(true_labels, preds)
    f1s = f1_score(true_labels, preds, average=None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if title:
        fig.suptitle(title, fontsize=13)

    axes[0].imshow(cm, cmap='viridis')
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_yticklabels(class_names)
    thresh = cm.max() * 0.7
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(j, i, cm[i, j], ha='center',
                     color='black' if cm[i, j] > thresh else 'white', fontsize=9)

    order = np.argsort(f1s)
    axes[1].barh([class_names[i] for i in order], f1s[order])
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Per-class F1')
    axes[1].axvline(np.mean(f1s), color='red', linestyle='--',
                    label=f'mean={np.mean(f1s):.3f}')
    axes[1].legend()

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_arr[:, i])
        axes[2].plot(fpr, tpr, label=f'{cls} (AUC={auc(fpr, tpr):.2f})')
    axes[2].plot([0, 1], [0, 1], 'k--')
    axes[2].set_title('ROC Curves OvR')
    axes[2].set_xlabel('FPR')
    axes[2].set_ylabel('TPR')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        savefig(save_path)
    else:
        plt.close('all')

    return dict(accuracy=acc, balanced_acc=bacc, macro_roc_auc=roc, kappa=kappa, mcc=mcc)


def save_history_plot(history: dict, title: str, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history['acc'],     label='Train Acc')
    ax1.plot(history['val_acc'], label='Val Acc')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax2.plot(history['loss'],     label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Loss')
    ax2.legend()
    fig.suptitle(title)
    plt.tight_layout()
    savefig(save_path)


# ── GradCAM ───────────────────────────────────────────────────────────────────
def gradcam(model, img_tensor, target_layer, n_tokens=None, spatial_hw=None):
    acts, grads = {}, {}
    fwd_h = target_layer.register_forward_hook(
        lambda m, i, o: acts.update({'a': o}))
    bwd_h = target_layer.register_full_backward_hook(
        lambda m, gi, go: grads.update({'g': go[0].detach()}))
    model.zero_grad()
    out = model(img_tensor)
    out[0, out[0].argmax()].backward()
    fwd_h.remove()
    bwd_h.remove()
    a = acts['a'].detach()
    g = grads['g']
    if a.dim() == 3:
        if n_tokens:
            a = a[:, n_tokens:, :]
            g = g[:, n_tokens:, :]
        N  = a.shape[1]
        hw = spatial_hw or (int(N ** 0.5), int(N ** 0.5))
        a = a[0].reshape(hw[0], hw[1], -1)
        g = g[0].reshape(hw[0], hw[1], -1)
    else:
        if a.shape[1] < a.shape[-1]:
            a = a[0].permute(1, 2, 0)
            g = g[0].permute(1, 2, 0)
        else:
            a = a[0]
            g = g[0]
    weights = g.mean(dim=(0, 1))
    cam = F.relu((a * weights).sum(dim=-1))
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                        size=(img_tensor.shape[-2], img_tensor.shape[-1]),
                        mode='bilinear', align_corners=False).squeeze().numpy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam


def gradcam_compare_save(ctrl_model, tc_model, img_path, img_size,
                         ctrl_layer, tc_layer, model_name, input_label,
                         save_path: Path, alpha=0.5):
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (img_size, img_size))
    t_base  = (torch.from_numpy(img_res).permute(2, 0, 1).float().div(255) - _MEAN) / _STD

    panels = []
    for model, layer in [(ctrl_model, ctrl_layer), (tc_model, tc_layer)]:
        t   = t_base.clone().unsqueeze(0).requires_grad_(True)
        cam = gradcam(model, t, layer)
        heatmap = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        overlay = np.clip(
            (1 - alpha) * img_res.astype(np.float32) / 255 + alpha * heatmap, 0, 1)
        panels.append((img_res, cam, overlay))

    col_labels = ['Control\n(ImageNet→Bug-bite)',
                  'TC Fine-tuned\n(ImageNet→Cyclone→Bug-bite)']
    row_labels  = ['Original', 'GradCAM', 'Overlay']

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(f'{model_name} GradCAM — {input_label}', fontsize=12, fontweight='bold')
    for col in range(2):
        axes[0, col].set_title(col_labels[col], fontsize=9, fontweight='bold')
        for row, row_label in enumerate(row_labels):
            ax = axes[row, col]
            ax.imshow(panels[col][row], cmap='jet' if row == 1 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight='bold')
    plt.tight_layout()
    savefig(save_path)


def load_model(timm_id: str, weight_path: str, device: torch.device,
               strict: bool = True) -> torch.nn.Module:
    m = timm.create_model(timm_id, pretrained=False, num_classes=N_CLASSES)
    m.load_state_dict(
        torch.load(weight_path, map_location='cpu', weights_only=True), strict=strict)
    return m.to(device).eval()
