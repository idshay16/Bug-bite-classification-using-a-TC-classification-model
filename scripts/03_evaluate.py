"""
Section 3 — Feature map and GradCAM visualization.

Produces feature map grids (Control | TC Backbone | TC Fine-tuned) and
GradCAM comparisons (Control vs TC Fine-tuned) for cyclone and bug-bite inputs.

Usage (called by run_experiments.py):
    python scripts/03_evaluate.py \\
        --run-id s0_ls0.10_lr5.0e-06 \\
        --ctrl-convnext  <path> --ctrl-densenet  <path> --ctrl-inception  <path> \\
        --cyc-convnext   <path> --cyc-densenet   <path> --cyc-inception   <path> \\
        --tc-convnext    <path> --tc-densenet    <path> --tc-inception    <path>
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as _T
from PIL import Image as _PILImage

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'miscellaneous_code'))

from shared import (
    BUG_VAL, MODEL_SPECS, N_CLASSES, TC_VAL,
    _MEAN, _STD, gradcam_compare_save, load_model, out_dir, savefig,
)

N_FEAT_DISPLAY = 8
_MEAN_T = _MEAN
_STD_T  = _STD


def _pick_sample_image(folder: str) -> str | None:
    p = Path(folder)
    for cls_dir in sorted(p.iterdir()):
        if not cls_dir.is_dir():
            continue
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            imgs = sorted(cls_dir.glob(ext))
            if imgs:
                return str(imgs[0])
    return None


def _feat_to_array(feat_tensor):
    arr = feat_tensor.permute(1, 2, 0).numpy()
    lo = arr.min(axis=(0, 1))
    hi = arr.max(axis=(0, 1))
    return (arr - lo) / (hi - lo + 1e-8)


def _load_img_tensor(img_path: str, img_size: int) -> torch.Tensor:
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    t = (torch.from_numpy(img).permute(2, 0, 1).float().div(255) - _MEAN_T) / _STD_T
    return t.unsqueeze(0)


def save_feature_maps(model, img_path, img_size, hook_fn, model_label,
                      input_label, out_path, n=N_FEAT_DISPLAY):
    cap = {}
    h = hook_fn(model, cap)
    with torch.no_grad():
        model(_load_img_tensor(img_path, img_size))
    h.remove()
    feats = _feat_to_array(cap['feat'][0])

    fig, axes = plt.subplots(n, 1, figsize=(3, n * 2.4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f'{model_label}\n{input_label}', fontsize=9, fontweight='bold')
    for i, ax in enumerate(axes):
        ax.imshow(feats[:, :, i], cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    savefig(out_path)


def save_feature_maps_compare(variants, img_path, img_size, hook_fn,
                               arch_name, input_label, out_path,
                               n=N_FEAT_DISPLAY):
    import matplotlib
    import matplotlib.pyplot as plt

    all_feats = []
    for label, model in variants:
        cap = {}
        h = hook_fn(model, cap)
        with torch.no_grad():
            model(_load_img_tensor(img_path, img_size))
        h.remove()
        all_feats.append((label, _feat_to_array(cap['feat'][0])))

    ncols = len(variants)
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.2, n * 2.4))
    fig.suptitle(f'{arch_name} Feature Maps — {input_label}',
                 fontsize=12, fontweight='bold')
    for col, (label, feats) in enumerate(all_feats):
        axes[0, col].set_title(label, fontsize=8, fontweight='bold')
        for row in range(n):
            ax = axes[row, col]
            ax.imshow(feats[:, :, row], cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    savefig(out_path)


def main():
    import matplotlib.pyplot as plt  # imported after Agg backend set in shared

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',          required=True)
    parser.add_argument('--ctrl-convnext',   required=True)
    parser.add_argument('--ctrl-densenet',   required=True)
    parser.add_argument('--ctrl-inception',  required=True)
    parser.add_argument('--cyc-convnext',    required=True)
    parser.add_argument('--cyc-densenet',    required=True)
    parser.add_argument('--cyc-inception',   required=True)
    parser.add_argument('--tc-convnext',     required=True)
    parser.add_argument('--tc-densenet',     required=True)
    parser.add_argument('--tc-inception',    required=True)
    args = parser.parse_args()

    out = out_dir('feature_maps', args.run_id)
    cyc_sample  = _pick_sample_image(str(Path(TC_VAL)))
    bug_sample  = _pick_sample_image(BUG_VAL)

    inputs = []
    if cyc_sample:
        inputs.append((cyc_sample, 'cyclone'))
    if bug_sample:
        inputs.append((bug_sample, 'bugbite'))

    path_map = {
        'convnext':  {'ctrl': args.ctrl_convnext, 'cyc': args.cyc_convnext, 'tc': args.tc_convnext},
        'densenet':  {'ctrl': args.ctrl_densenet, 'cyc': args.cyc_densenet, 'tc': args.tc_densenet},
        'inception': {'ctrl': args.ctrl_inception,'cyc': args.cyc_inception,'tc': args.tc_inception},
    }

    for img_path, input_label in inputs:
        # ── ConvNeXt feature maps ─────────────────────────────────────────────
        def _cnxt_hook(m, cap):
            return m.stages[-1].register_forward_hook(
                lambda mod, i, o: cap.update({'feat': o.detach().cpu()}))

        _, timm_cnxt, size_cnxt, _, _ = MODEL_SPECS[0]
        variants_cnxt = [
            ('Control\n(ImageNet→Bug-bite)',
             load_model(timm_cnxt, path_map['convnext']['ctrl'], torch.device('cpu'))),
            ('TC Backbone\n(ImageNet→Cyclone)',
             load_model(timm_cnxt, path_map['convnext']['cyc'], torch.device('cpu'), strict=False)),
            ('TC Fine-tuned\n(ImageNet→Cyclone→Bug-bite)',
             load_model(timm_cnxt, path_map['convnext']['tc'], torch.device('cpu'))),
        ]
        save_feature_maps_compare(
            variants_cnxt, img_path, size_cnxt, _cnxt_hook,
            'ConvNeXt-Tiny', input_label,
            out / f'featmaps_convnext_{input_label}.png',
        )
        for _, m in variants_cnxt:
            del m

        # ── DenseNet feature maps ─────────────────────────────────────────────
        def _dens_hook(m, cap):
            return m.features.denseblock4.register_forward_hook(
                lambda mod, i, o: cap.update({'feat': o.detach().cpu()}))

        _, timm_dens, size_dens, _, _ = MODEL_SPECS[1]
        variants_dens = [
            ('Control\n(ImageNet→Bug-bite)',
             load_model(timm_dens, path_map['densenet']['ctrl'], torch.device('cpu'))),
            ('TC Backbone\n(ImageNet→Cyclone)',
             load_model(timm_dens, path_map['densenet']['cyc'], torch.device('cpu'), strict=False)),
            ('TC Fine-tuned\n(ImageNet→Cyclone→Bug-bite)',
             load_model(timm_dens, path_map['densenet']['tc'], torch.device('cpu'))),
        ]
        save_feature_maps_compare(
            variants_dens, img_path, size_dens, _dens_hook,
            'DenseNet-121', input_label,
            out / f'featmaps_densenet_{input_label}.png',
        )
        for _, m in variants_dens:
            del m

        # ── InceptionV3 feature maps ──────────────────────────────────────────
        def _inc_hook(m, cap):
            tgt = getattr(m, 'Mixed_7c', None)
            if tgt is None:
                return None
            return tgt.register_forward_hook(
                lambda mod, i, o: cap.update({'feat': o.detach().cpu()}))

        _, timm_inc, size_inc, _, _ = MODEL_SPECS[2]
        variants_inc = [
            ('Control\n(ImageNet→Bug-bite)',
             load_model(timm_inc, path_map['inception']['ctrl'], torch.device('cpu'))),
            ('TC Backbone\n(ImageNet→Cyclone)',
             load_model(timm_inc, path_map['inception']['cyc'], torch.device('cpu'), strict=False)),
            ('TC Fine-tuned\n(ImageNet→Cyclone→Bug-bite)',
             load_model(timm_inc, path_map['inception']['tc'], torch.device('cpu'))),
        ]
        save_feature_maps_compare(
            variants_inc, img_path, size_inc, _inc_hook,
            'InceptionV3', input_label,
            out / f'featmaps_inception_{input_label}.png',
        )
        for _, m in variants_inc:
            del m

    # ── GradCAM ───────────────────────────────────────────────────────────────
    for img_path, input_label in inputs:
        # ConvNeXt
        _, timm_cnxt, size_cnxt, _, _ = MODEL_SPECS[0]
        ctrl = load_model(timm_cnxt, path_map['convnext']['ctrl'], torch.device('cpu'))
        tc   = load_model(timm_cnxt, path_map['convnext']['tc'],   torch.device('cpu'))
        gradcam_compare_save(
            ctrl, tc, img_path, size_cnxt,
            ctrl.stages[-1], tc.stages[-1],
            'ConvNeXt-Tiny', input_label,
            out / f'gradcam_convnext_{input_label}.png',
        )
        del ctrl, tc

        # DenseNet
        _, timm_dens, size_dens, _, _ = MODEL_SPECS[1]
        ctrl = load_model(timm_dens, path_map['densenet']['ctrl'], torch.device('cpu'))
        tc   = load_model(timm_dens, path_map['densenet']['tc'],   torch.device('cpu'))
        gradcam_compare_save(
            ctrl, tc, img_path, size_dens,
            ctrl.features.denseblock4, tc.features.denseblock4,
            'DenseNet-121', input_label,
            out / f'gradcam_densenet_{input_label}.png',
        )
        del ctrl, tc

        # InceptionV3
        _, timm_inc, size_inc, _, _ = MODEL_SPECS[2]
        ctrl = load_model(timm_inc, path_map['inception']['ctrl'], torch.device('cpu'))
        tc   = load_model(timm_inc, path_map['inception']['tc'],   torch.device('cpu'))
        ctrl_tgt = getattr(ctrl, 'Mixed_7c', None)
        tc_tgt   = getattr(tc,   'Mixed_7c', None)
        if ctrl_tgt and tc_tgt:
            gradcam_compare_save(
                ctrl, tc, img_path, size_inc,
                ctrl_tgt, tc_tgt,
                'InceptionV3', input_label,
                out / f'gradcam_inception_{input_label}.png',
            )
        del ctrl, tc

    print(f'[03_evaluate] saved → {out}')


if __name__ == '__main__':
    main()
