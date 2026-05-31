"""
Section 1 — Control training (ImageNet → Bug-Bite).

Trains ConvNeXt-Tiny, DenseNet-121, InceptionV3 from ImageNet pretrained weights.
Saves training curves and per-model metric plots. Metrics written to results/metrics/.

Usage (called by run_experiments.py):
    python scripts/01_train_control.py \\
        --run-id s0_ls0.10_lr5.0e-06 \\
        --seed 0 \\
        --ctrl-convnext results/checkpoints/control_seed_0/control_convnext.pt \\
        --ctrl-densenet results/checkpoints/control_seed_0/control_densenet.pt \\
        --ctrl-inception results/checkpoints/control_seed_0/control_inception.pt
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'miscellaneous_code'))

from shared import (
    BUG_VAL, MODEL_SPECS, N_CLASSES, RESULTS_DIR,
    full_metrics, get_pytorch_loaders, get_pt_probs_preds,
    load_model, out_dir, save_history_plot,
)
from pytorch_utils import train_pytorch_model


PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20
PATIENCE      = 2
BUG_TRAIN     = '/home/test/bug_data/train'


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',         required=True)
    parser.add_argument('--seed',           type=int,   required=True)
    parser.add_argument('--ctrl-convnext',  required=True)
    parser.add_argument('--ctrl-densenet',  required=True)
    parser.add_argument('--ctrl-inception', required=True)
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plots_d  = out_dir('plots',   args.run_id)
    metrics_d = RESULTS_DIR / 'metrics'
    metrics_d.mkdir(parents=True, exist_ok=True)

    ckpt_map = {
        'convnext':  args.ctrl_convnext,
        'densenet':  args.ctrl_densenet,
        'inception': args.ctrl_inception,
    }

    all_metrics = {}

    for model_key, timm_id, img_size, _, bug_batch in MODEL_SPECS:
        ckpt_path = ckpt_map[model_key]
        seed_everything(args.seed)

        train_loader, val_loader, classes = get_pytorch_loaders(
            BUG_TRAIN, BUG_VAL, img_size=img_size,
            batch_size=bug_batch, augment=False)

        if not Path(ckpt_path).exists():
            import timm as _timm
            model = _timm.create_model(timm_id, pretrained=True, num_classes=N_CLASSES)
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            history = train_pytorch_model(
                model, train_loader, val_loader, device,
                phase1_epochs=PHASE1_EPOCHS, phase2_epochs=PHASE2_EPOCHS,
                patience=PATIENCE, save_path=ckpt_path,
            )
            save_history_plot(
                history,
                title=f'Control {model_key} — {args.run_id}',
                save_path=plots_d / f'training_curves_control_{model_key}.png',
            )
            del model
            torch.cuda.empty_cache()

        model = load_model(timm_id, ckpt_path, device)
        labels, preds, probs = get_pt_probs_preds(model, val_loader, device)
        m = full_metrics(
            labels, preds, probs, classes,
            title=f'Control {model_key} — {args.run_id}',
            save_path=plots_d / f'metrics_control_{model_key}.png',
        )
        all_metrics[f'control_{model_key}'] = m
        del model
        torch.cuda.empty_cache()

    metrics_file = metrics_d / f'{args.run_id}_control.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'[01_train_control] metrics saved → {metrics_file}')


if __name__ == '__main__':
    main()
