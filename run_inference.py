#!/usr/bin/env python3
"""
Single-image inference + XAI comparing Control vs TC models.
All results in one plot — no files saved.

Usage:
    python run_inference.py --image path/to/image.jpg
    python run_inference.py --image path/to/image.jpg --no-lime
    python run_inference.py --image path/to/image.jpg --lime-samples 100
"""
import argparse
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image as _PIL
from torchvision import transforms as _T

ROOT                = Path(__file__).parent
BOOK_RESULTS_ROOT   = ROOT / 'project_book_results'

# s3 = control run,  s6 = best TC run
CTRL_WEIGHTS_DIR    = BOOK_RESULTS_ROOT / 's3_ls0.20_lr1.0e-05'
TC_WEIGHTS_DIR      = BOOK_RESULTS_ROOT / 's6_ls0.10_lr5.0e-06'

CLASS_NAMES = ['ants', 'bed_bugs', 'mosquitos', 'spiders', 'ticks_fleas']
N_CLASSES   = 5

ARCHS = [
    ('convnext',  'convnext_tiny.fb_in22k_ft_in1k', 256),
    ('densenet',  'densenet121.ra_in1k',             256),
    ('inception', 'inception_v3.tv_in1k',            299),
]
ARCH_DISPLAY = {'convnext': 'ConvNeXt-Tiny', 'densenet': 'DenseNet-121', 'inception': 'InceptionV3'}

# filenames as saved by the training pipeline
_CTRL_NAMES = {
    'convnext':  'control_convnext.pt',
    'densenet':  'control_densenet.pt',
    'inception': 'control_inception.pt',
}
_TC_NAMES = {
    'convnext':  'tc_bug_convnext.pt',
    'densenet':  'tc_bug_densenet.pt',
    'inception': 'tc_bug_inception.pt',
}

# per-arch override — best config per model (by balanced_acc), mixed across sweep runs
_TC_PATHS = {
    'convnext':  ROOT / 'results/checkpoints/config_014/tc_bug_convnext.pt',
    'densenet':  ROOT / 'results/checkpoints/config_050/tc_bug_densenet.pt',
    'inception': ROOT / 'results/checkpoints/config_054/tc_bug_inception.pt',
}

_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_models(use_tc, device, best_per_arch=False):
    torch.backends.cudnn.benchmark = True
    models = {}
    for key, timm_id, _ in ARCHS:
        if use_tc:
            path = _TC_PATHS[key] if best_per_arch else TC_WEIGHTS_DIR / _TC_NAMES[key]
        else:
            path = CTRL_WEIGHTS_DIR / _CTRL_NAMES[key]
        if not path.exists():
            raise FileNotFoundError(f'Weight not found: {path}')
        m = timm.create_model(timm_id, pretrained=False, num_classes=N_CLASSES)
        m.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
        models[key] = m.to(device).eval()
    return models


def _base_tensor(img_np, device):
    """numpy HWC uint8 -> BCHW float32 on device (no resize yet)."""
    return torch.from_numpy(img_np).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)


def _resize_norm(t_base, size, device):
    """GPU resize + normalize. t_base is BCHW on device."""
    r = F.interpolate(t_base, size=(size, size), mode='bilinear', align_corners=False)
    return (r - _mean.to(device)) / _std.to(device)


def predict(img_np, models, device):
    avg      = np.zeros(N_CLASSES)
    per_arch = {}
    t_base   = _base_tensor(img_np, device)
    with torch.inference_mode():
        for key, _, size in ARCHS:
            t    = _resize_norm(t_base, size, device)
            prob = torch.softmax(models[key](t), dim=1).cpu().numpy()[0]
            per_arch[key] = prob
            avg += prob
    avg /= len(ARCHS)
    return CLASS_NAMES[int(np.argmax(avg))], avg, per_arch


def _get_target(model, key):
    if key == 'convnext':  return model.stages[-1]
    if key == 'densenet':  return model.features.norm5
    if key == 'inception': return getattr(model, 'Mixed_7c', None)


def gradcam(model, img_tensor, target_layer):
    saved = {}

    def fwd_hook(m, inp, out):
        x = out if isinstance(out, torch.Tensor) else out[0]
        saved['act'] = x.detach().clone()

    def bwd_hook(m, grad_inp, grad_out):
        saved['grad'] = grad_out[0].detach().clone()

    fwd_h = target_layer.register_forward_hook(fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(bwd_hook)
    model.zero_grad()
    out = model(img_tensor)
    out[0, out[0].argmax()].backward()
    fwd_h.remove()
    bwd_h.remove()

    act  = saved.get('act')
    grad = saved.get('grad')
    if act is None or grad is None:
        return np.zeros(img_tensor.shape[-2:], dtype=np.float32)

    act, grad = act[0], grad[0]
    cam    = F.relu((act * grad.mean(dim=(1, 2))[:, None, None]).sum(0))
    # smooth before upsampling to reduce high-freq noise
    cam_s  = F.avg_pool2d(cam.float()[None, None], kernel_size=3, stride=1, padding=1).squeeze()
    H, W   = img_tensor.shape[-2], img_tensor.shape[-1]
    cam_up = F.interpolate(cam_s[None, None], size=(H, W),
                           mode='bilinear', align_corners=False).squeeze().cpu().numpy()
    cam_up -= cam_up.min()
    cam_up /= cam_up.max() + 1e-8
    return cam_up


def overlay(img_np, cam, alpha=0.30):
    hm = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
                      cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip((1 - alpha) * img_np.astype(np.float32) / 255.0 + alpha * hm, 0, 1)


def all_gradcams(img_np, models, device):
    result  = {}
    t_base  = _base_tensor(img_np, device)
    for key, _, size in ARCHS:
        layer = _get_target(models[key], key)
        if layer is None:
            continue
        t           = _resize_norm(t_base, size, device)
        cam         = gradcam(models[key], t, layer)
        orig        = cv2.resize(img_np, (size, size))
        result[key] = overlay(orig, cam)
    return result


def lime_explain(img_np, models, device, n_samples):
    from lime.lime_image import LimeImageExplainer

    _m = _mean.to(device)
    _s = _std.to(device)

    def _predict(images):
        avg    = np.zeros((len(images), N_CLASSES), dtype=np.float32)
        t_base = torch.from_numpy(
            np.stack(images).astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2).to(device)          # (N,3,H,W) — one CPU->GPU transfer
        with torch.inference_mode():
            for key, _, size in ARCHS:
                batch = (F.interpolate(t_base, size=(size, size),
                                       mode='bilinear', align_corners=False) - _m) / _s
                avg += torch.softmax(models[key](batch), dim=1).cpu().numpy()
        return avg / len(ARCHS)

    print(f'    Running LIME ({n_samples} samples)...')
    exp      = LimeImageExplainer().explain_instance(
        img_np, _predict, top_labels=1, num_samples=n_samples, batch_size=32)
    top      = exp.top_labels[0]
    segments = exp.segments
    weights  = dict(exp.local_exp[top])
    heatmap  = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, w in weights.items():
        heatmap[segments == seg_id] = w
    return exp, heatmap, segments


def show(img_np, ctrl_label, ctrl_probs, tc_label, tc_probs,
         ctrl_cams, tc_cams, ctrl_lime, tc_lime):
    """
    Layout — 2 cols: Control | TC
      row 0      : input image (full width)
      row 1      : ctrl confidence bar   | tc confidence bar
      row 2+     : ctrl gradcam overlay  | tc gradcam overlay  (one per arch)
      last (opt) : ctrl lime             | tc lime
    """
    arch_keys = list(ctrl_cams.keys())
    has_lime  = ctrl_lime is not None and tc_lime is not None
    n_rows    = 2 + len(arch_keys) + (1 if has_lime else 0)

    fig = plt.figure(figsize=(12, 4 * n_rows), constrained_layout=True)
    fig.suptitle(f'Control -> {ctrl_label}   |   TC -> {tc_label}',
                 fontsize=13, fontweight='bold')
    gs = fig.add_gridspec(n_rows, 2)

    # ── row 0: input (full width) ─────────────────────────────────────────────
    ax_in = fig.add_subplot(gs[0, :])
    ax_in.imshow(img_np); ax_in.axis('off'); ax_in.set_title('Input', fontweight='bold')

    # ── row 1: confidence bars ────────────────────────────────────────────────
    colors_ctrl = ['#d9534f' if c == ctrl_label else '#aec6cf' for c in CLASS_NAMES]
    colors_tc   = ['#d9534f' if c == tc_label   else '#90ee90' for c in CLASS_NAMES]
    for col, (probs, label, colors, title) in enumerate([
        (ctrl_probs, ctrl_label, colors_ctrl, f'Control -> {ctrl_label}'),
        (tc_probs,   tc_label,   colors_tc,   f'TC      -> {tc_label}'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        bars = ax.barh(CLASS_NAMES, probs, color=colors)
        ax.set_xlim(0, 1); ax.set_xlabel('Confidence')
        ax.set_title(title, fontweight='bold')
        for bar, p in zip(bars, probs):
            ax.text(min(p + 0.01, 0.88), bar.get_y() + bar.get_height() / 2,
                    f'{p:.3f}', va='center', fontsize=9)

    # ── GradCAM rows ──────────────────────────────────────────────────────────
    for i, key in enumerate(arch_keys):
        row  = 2 + i
        ax_c = fig.add_subplot(gs[row, 0])
        ax_t = fig.add_subplot(gs[row, 1])
        ax_c.imshow(ctrl_cams[key]); ax_c.axis('off')
        ax_t.imshow(tc_cams[key]);   ax_t.axis('off')
        ax_c.set_title(f'{ARCH_DISPLAY[key]} — Control', fontweight='bold')
        ax_t.set_title(f'{ARCH_DISPLAY[key]} — TC',      fontweight='bold')

    # ── LIME row ──────────────────────────────────────────────────────────────
    if has_lime:
        import skimage.segmentation
        from matplotlib.patches import Patch
        r     = 2 + len(arch_keys)
        TOP_N = 12  # only show top N segments by |weight|

        import matplotlib.cm as _cm
        ctrl_exp_obj, ctrl_hm, ctrl_segs = ctrl_lime
        tc_exp_obj,   tc_hm,   tc_segs   = tc_lime
        for col, (hm, segs, title) in enumerate([
            (ctrl_hm, ctrl_segs, f'LIME — Control  →  {ctrl_label}'),
            (tc_hm,   tc_segs,   f'LIME — TC  →  {tc_label}'),
        ]):
            seg_ids = np.unique(segs)
            seg_w   = {s: float(hm[segs == s].mean()) for s in seg_ids}
            top_ids = sorted(seg_w, key=lambda s: abs(seg_w[s]), reverse=True)[:TOP_N]
            vmax    = max(abs(seg_w[s]) for s in top_ids) + 1e-8
            img_f   = img_np.astype(np.float32) / 255.0
            colored = img_f.copy()
            for s in top_ids:
                mask = segs == s
                norm = (np.clip(seg_w[s], -vmax, vmax) + vmax) / (2 * vmax)
                tint = np.array(_cm.RdBu_r(norm)[:3], dtype=np.float32)
                colored[mask] = np.clip(0.60 * img_f[mask] + 0.40 * tint, 0, 1)
            top_mask = np.isin(segs, top_ids).astype(int)
            vis = skimage.segmentation.mark_boundaries(
                colored, top_mask, color=(1, 1, 0), mode='outer')
            ax = fig.add_subplot(gs[r, col])
            ax.imshow(vis); ax.axis('off'); ax.set_title(title, fontweight='bold')
            legend = [Patch(facecolor='#d73027', label='Supports prediction'),
                      Patch(facecolor='#4575b4', label='Contradicts prediction'),
                      Patch(facecolor='yellow',  label='Segment boundary')]
            ax.legend(handles=legend, loc='lower left', fontsize=7,
                      framealpha=0.8, handlelength=1.2)

    out_path = ROOT / 'inference_result.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',        required=True)
    parser.add_argument('--lime-samples', type=int, default=50)
    parser.add_argument('--no-lime',      action='store_true')
    parser.add_argument('--no-gradcam',   action='store_true')
    parser.add_argument('--best-per-arch', action='store_true',
                         help='use best config per arch (config_014/050/054) instead of the single s6 TC run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    img_np = np.array(_PIL.open(args.image).convert('RGB'))

    print('Loading control models...')
    ctrl_models = load_models(use_tc=False, device=device)
    print('Loading TC models...')
    tc_models   = load_models(use_tc=True,  device=device, best_per_arch=args.best_per_arch)

    ctrl_label, ctrl_probs, ctrl_per_arch = predict(img_np, ctrl_models, device)
    tc_label,   tc_probs,   tc_per_arch   = predict(img_np, tc_models,   device)
    print(f'\nControl → {ctrl_label}')
    for cls, p in sorted(zip(CLASS_NAMES, ctrl_probs), key=lambda x: -x[1]):
        print(f'  {cls:<15} {p:.4f}', '<' if cls == ctrl_label else '')
    for key, _, _ in ARCHS:
        pred = CLASS_NAMES[int(np.argmax(ctrl_per_arch[key]))]
        print(f'  [{ARCH_DISPLAY[key]}] → {pred}  ({ctrl_per_arch[key].max():.4f})')

    print(f'\nTC      → {tc_label}')
    for cls, p in sorted(zip(CLASS_NAMES, tc_probs), key=lambda x: -x[1]):
        print(f'  {cls:<15} {p:.4f}', '<' if cls == tc_label else '')
    for key, _, _ in ARCHS:
        pred = CLASS_NAMES[int(np.argmax(tc_per_arch[key]))]
        print(f'  [{ARCH_DISPLAY[key]}] → {pred}  ({tc_per_arch[key].max():.4f})')

    ctrl_cams = all_gradcams(img_np, ctrl_models, device) if not args.no_gradcam else {}
    tc_cams   = all_gradcams(img_np, tc_models,   device) if not args.no_gradcam else {}

    ctrl_lime = tc_lime = None
    if not args.no_lime:
        try:
            print('LIME — Control:')
            ctrl_exp_obj, ctrl_hm, ctrl_segs = lime_explain(img_np, ctrl_models, device, args.lime_samples)
            ctrl_lime = (ctrl_exp_obj, ctrl_hm, ctrl_segs)
            print('LIME — TC:')
            tc_exp_obj, tc_hm, tc_segs = lime_explain(img_np, tc_models, device, args.lime_samples)
            tc_lime = (tc_exp_obj, tc_hm, tc_segs)
        except ImportError:
            print('LIME skipped — pip install lime scikit-image')

    show(img_np, ctrl_label, ctrl_probs, tc_label, tc_probs,
         ctrl_cams, tc_cams, ctrl_lime, tc_lime)


if __name__ == '__main__':
    main()
