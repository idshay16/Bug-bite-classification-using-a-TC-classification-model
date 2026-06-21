#!/usr/bin/env python3
"""
Standalone GradCAM re-run with fixed gradient computation.
Targets the 4 book configs. No retraining — uses existing checkpoints.

Fix: original code averaged gradients over wrong dims for channels-first (C,H,W)
tensors, producing horizontal stripe artifacts. This script detects layout and
computes weights correctly.

Usage (on server, from repo root):
    python scripts/run_gradcam_book.py

Outputs saved to: results/gradcam_fixed/<run_label>/
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'miscellaneous_code'))

from shared import _MEAN, _STD, N_CLASSES

BUG_VAL = '/home/test/bug_data/val'
TC_VAL  = '/home/test/cyclone_data_split/val'

CKPT = ROOT / 'results' / 'checkpoints'
OUT  = ROOT / 'results' / 'gradcam_fixed'

BOOK_RUNS = [
    {'label': 'control', 'seed': 2, 'cfg_id': 18,  'display': 'CONTROL  s2_ls0.00_lr1e-6'},
    {'label': 'best',    'seed': 6, 'cfg_id': 58,  'display': 'BEST     s6_ls0.10_lr5e-6'},
    {'label': 'worst',   'seed': 6, 'cfg_id': 59,  'display': 'WORST    s6_ls0.10_lr1e-5'},
    {'label': 'median',  'seed': 3, 'cfg_id': 35,  'display': 'MEDIAN   s3_ls0.20_lr1e-5'},
]

ARCHS = [
    ('convnext',  'convnext_tiny.fb_in22k_ft_in1k', 256),
    ('densenet',  'densenet121.ra_in1k',             256),
    ('inception', 'inception_v3.tv_in1k',            299),
]


def pick_sample_image(folder: str) -> str | None:
    p = Path(folder)
    if not p.exists():
        return None
    for cls_dir in sorted(p.iterdir()):
        if not cls_dir.is_dir():
            continue
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            imgs = sorted(cls_dir.glob(ext))
            if imgs:
                return str(imgs[0])
    return None


def load_img_tensor(img_path: str, img_size: int) -> torch.Tensor:
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    t = (torch.from_numpy(img).permute(2, 0, 1).float().div(255) - _MEAN) / _STD
    return t.unsqueeze(0)


def load_model(timm_id: str, weight_path: Path, device: torch.device,
               strict: bool = True) -> torch.nn.Module:
    m = timm.create_model(timm_id, pretrained=False, num_classes=N_CLASSES)
    m.load_state_dict(
        torch.load(str(weight_path), map_location='cpu', weights_only=True),
        strict=strict,
    )
    return m.to(device).eval()


def get_target_layer(model, arch: str):
    """Return best GradCAM target layer per architecture."""
    if arch == 'convnext':
        # stages[-2] gives 16x16 spatial (better resolution than stages[-1] at 8x8)
        return model.stages[-2]
    if arch == 'densenet':
        return model.features.denseblock4
    if arch == 'inception':
        return getattr(model, 'Mixed_7c', None)
    return None


def gradcam_fixed(model, img_tensor: torch.Tensor, target_layer) -> np.ndarray:
    """
    GradCAM with correct gradient averaging for any CNN tensor layout.

    Bug in original: for channels-first (B,C,H,W) activations, code did
    g.mean(dim=(0,1)) which averaged over C and H → shape (W,) instead of
    g.mean(dim=(1,2)) which averages over H and W → shape (C,).
    """
    activation = [None]

    def fwd_hook(m, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        activation[0] = t
        t.retain_grad()

    h = target_layer.register_forward_hook(fwd_hook)
    model.zero_grad()
    out = model(img_tensor)
    out[0, out[0].argmax()].backward()
    h.remove()

    a = activation[0]
    if a is None or a.grad is None:
        h_img, w_img = img_tensor.shape[-2], img_tensor.shape[-1]
        return np.zeros((h_img, w_img), dtype=np.float32)

    act = a.detach()
    grad = a.grad.detach()

    if act.dim() == 4:
        B, d1, d2, d3 = act.shape
        if d3 > d1 and d3 > d2:
            # channels-last (B, H, W, C)
            act  = act[0]          # (H, W, C)
            grad = grad[0]         # (H, W, C)
            weights = grad.mean(dim=(0, 1))           # (C,)
            cam = F.relu((act * weights).sum(dim=-1)) # (H, W)
        else:
            # channels-first (B, C, H, W)
            act  = act[0]          # (C, H, W)
            grad = grad[0]         # (C, H, W)
            weights = grad.mean(dim=(1, 2))                    # (C,)
            cam = F.relu((act * weights[:, None, None]).sum(dim=0))  # (H, W)
    elif act.dim() == 3:
        # transformer (B, N, C)
        act  = act[0]   # (N, C)
        grad = grad[0]  # (N, C)
        N = act.shape[0]
        hw = int(N ** 0.5)
        act  = act.reshape(hw, hw, -1)
        grad = grad.reshape(hw, hw, -1)
        weights = grad.mean(dim=(0, 1))
        cam = F.relu((act * weights).sum(dim=-1))
    else:
        h_img, w_img = img_tensor.shape[-2], img_tensor.shape[-1]
        return np.zeros((h_img, w_img), dtype=np.float32)

    h_img, w_img = img_tensor.shape[-2], img_tensor.shape[-1]
    cam_up = F.interpolate(
        cam.float().unsqueeze(0).unsqueeze(0),
        size=(h_img, w_img),
        mode='bilinear',
        align_corners=False,
    ).squeeze().cpu().numpy()
    cam_up -= cam_up.min()
    cam_up /= (cam_up.max() + 1e-8)
    return cam_up


def save_gradcam_comparison(ctrl_model, tc_model, arch: str,
                             img_path: str, img_size: int,
                             input_label: str, out_path: Path,
                             run_display: str, device: torch.device,
                             alpha: float = 0.5):
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (img_size, img_size))
    t_base  = ((torch.from_numpy(img_res).permute(2, 0, 1).float().div(255) - _MEAN) / _STD).to(device)

    panels = []
    for model in (ctrl_model, tc_model):
        layer = get_target_layer(model, arch)
        if layer is None:
            panels.append(None)
            continue
        t   = t_base.clone().unsqueeze(0)
        cam = gradcam_fixed(model, t, layer)
        heatmap = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32) / 255
        overlay = np.clip(
            (1 - alpha) * img_res.astype(np.float32) / 255 + alpha * heatmap, 0, 1
        )
        panels.append((img_res, cam, overlay))

    if any(p is None for p in panels):
        print(f'  [skip] {arch} — no target layer')
        return

    col_labels = ['Control\n(ImageNet→Bug-bite)',
                  'TC Fine-tuned\n(ImageNet→Cyclone→Bug-bite)']
    row_labels  = ['Original', 'GradCAM', 'Overlay']

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    arch_display = {'convnext': 'ConvNeXt-Tiny', 'densenet': 'DenseNet-121',
                    'inception': 'InceptionV3'}[arch]
    fig.suptitle(f'{arch_display} GradCAM — {input_label}\n{run_display}',
                 fontsize=11, fontweight='bold')
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f'  saved → {out_path.name}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    bug_sample = pick_sample_image(BUG_VAL)
    cyc_sample = pick_sample_image(TC_VAL)

    if not bug_sample:
        print(f'ERROR: no images found in {BUG_VAL}')
        sys.exit(1)

    inputs = [('bugbite', bug_sample)]
    if cyc_sample:
        inputs.append(('cyclone', cyc_sample))

    for run in BOOK_RUNS:
        label    = run['label']
        seed     = run['seed']
        cfg_id   = run['cfg_id']
        display  = run['display']
        out_dir  = OUT / label

        ctrl_ckpt_dir = CKPT / f'control_seed_{seed}'
        tc_ckpt_dir   = CKPT / f'config_{cfg_id:03d}'

        print(f'\n=== {display} ===')
        print(f'  ctrl: {ctrl_ckpt_dir.name}')
        print(f'  tc:   {tc_ckpt_dir.name}')

        for arch, timm_id, img_size in ARCHS:
            ctrl_pt = ctrl_ckpt_dir / f'control_{arch}.pt'
            tc_pt   = tc_ckpt_dir   / f'tc_bug_{arch}.pt'

            if not ctrl_pt.exists() or not tc_pt.exists():
                print(f'  [skip] {arch} — checkpoint missing')
                continue

            print(f'  loading {arch}...')
            ctrl_model = load_model(timm_id, ctrl_pt, device)
            tc_model   = load_model(timm_id, tc_pt,   device)

            for input_label, img_path in inputs:
                out_path = out_dir / f'gradcam_{arch}_{input_label}.png'
                save_gradcam_comparison(
                    ctrl_model, tc_model, arch,
                    img_path, img_size,
                    input_label, out_path, display, device,
                )

            del ctrl_model, tc_model

    print(f'\nDone. All outputs in: {OUT}')


if __name__ == '__main__':
    main()
