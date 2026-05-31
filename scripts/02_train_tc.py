"""
Section 2 — TC pre-training (ImageNet → Cyclone → Bug-Bite).

Trains 3 models through cyclone pre-training then bug-bite fine-tuning.
Saves training curves and per-model metric plots.

Usage (called by run_experiments.py):
    python scripts/02_train_tc.py \\
        --run-id s0_ls0.10_lr5.0e-06 \\
        --seed 0 \\
        --label-smoothing 0.1 \\
        --phase2-lr 5e-6 \\
        --cyc-convnext  results/checkpoints/config_000/cyclone_convnext.pt \\
        --cyc-densenet  results/checkpoints/config_000/cyclone_densenet.pt \\
        --cyc-inception results/checkpoints/config_000/cyclone_inception.pt \\
        --tc-convnext   results/checkpoints/config_000/tc_bug_convnext.pt \\
        --tc-densenet   results/checkpoints/config_000/tc_bug_densenet.pt \\
        --tc-inception  results/checkpoints/config_000/tc_bug_inception.pt
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'miscellaneous_code'))

from shared import (
    BUG_VAL, MODEL_SPECS, N_CLASSES, RESULTS_DIR, TC_TRAIN, TC_VAL,
    full_metrics, get_pytorch_loaders, get_pt_probs_preds,
    load_model, out_dir, save_history_plot,
)
from pytorch_utils import get_backbone_state_dict, train_pytorch_model

PHASE1_EPOCHS     = 5
CYC_PHASE2_EPOCHS = 30
BUG_PHASE2_EPOCHS = 20
PATIENCE          = 2
BUG_TRAIN         = '/home/test/bug_data/train'
BUG_PHASE2_LR     = 5e-6
BUG_LS            = 0.0


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',          required=True)
    parser.add_argument('--seed',            type=int,   required=True)
    parser.add_argument('--label-smoothing', type=float, required=True)
    parser.add_argument('--phase2-lr',       type=float, required=True)
    parser.add_argument('--cyc-convnext',    required=True)
    parser.add_argument('--cyc-densenet',    required=True)
    parser.add_argument('--cyc-inception',   required=True)
    parser.add_argument('--tc-convnext',     required=True)
    parser.add_argument('--tc-densenet',     required=True)
    parser.add_argument('--tc-inception',    required=True)
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plots_d   = out_dir('plots',   args.run_id)
    metrics_d = RESULTS_DIR / 'metrics'
    metrics_d.mkdir(parents=True, exist_ok=True)

    cyc_map = {
        'convnext':  args.cyc_convnext,
        'densenet':  args.cyc_densenet,
        'inception': args.cyc_inception,
    }
    tc_map = {
        'convnext':  args.tc_convnext,
        'densenet':  args.tc_densenet,
        'inception': args.tc_inception,
    }

    all_metrics = {}

    for model_key, timm_id, img_size, cyc_batch, bug_batch in MODEL_SPECS:
        cyc_path = cyc_map[model_key]
        tc_path  = tc_map[model_key]
        seed_everything(args.seed)

        # ── cyclone pre-training ──────────────────────────────────────────────
        if not Path(cyc_path).exists():
            Path(cyc_path).parent.mkdir(parents=True, exist_ok=True)
            tc_train, tc_val, _ = get_pytorch_loaders(
                TC_TRAIN, TC_VAL, img_size=img_size,
                batch_size=cyc_batch, augment='strong')
            cyc_model = timm.create_model(timm_id, pretrained=True, num_classes=N_CLASSES)
            cyc_hist  = train_pytorch_model(
                cyc_model, tc_train, tc_val, device,
                phase1_epochs=PHASE1_EPOCHS, phase2_epochs=CYC_PHASE2_EPOCHS,
                patience=PATIENCE, save_path=cyc_path,
                label_smoothing=args.label_smoothing, phase2_lr=args.phase2_lr,
            )
            save_history_plot(
                cyc_hist,
                title=f'Cyclone pre-train {model_key} — {args.run_id}',
                save_path=plots_d / f'training_curves_cyclone_{model_key}.png',
            )
            del cyc_model, tc_train, tc_val
            torch.cuda.empty_cache()

        # ── TC bug-bite fine-tuning ───────────────────────────────────────────
        seed_everything(args.seed)
        bug_train, bug_val, classes = get_pytorch_loaders(
            BUG_TRAIN, BUG_VAL, img_size=img_size,
            batch_size=bug_batch, augment=False)

        if not Path(tc_path).exists():
            Path(tc_path).parent.mkdir(parents=True, exist_ok=True)
            backbone_sd = get_backbone_state_dict(
                torch.load(cyc_path, map_location='cpu', weights_only=True))
            tc_model = timm.create_model(timm_id, pretrained=False, num_classes=N_CLASSES)
            tc_model.load_state_dict(backbone_sd, strict=False)
            tc_hist = train_pytorch_model(
                tc_model, bug_train, bug_val, device,
                phase1_epochs=PHASE1_EPOCHS, phase2_epochs=BUG_PHASE2_EPOCHS,
                patience=PATIENCE, save_path=tc_path,
                label_smoothing=BUG_LS, phase2_lr=BUG_PHASE2_LR,
            )
            save_history_plot(
                tc_hist,
                title=f'TC bug-bite {model_key} — {args.run_id}',
                save_path=plots_d / f'training_curves_tc_{model_key}.png',
            )
            del tc_model
            torch.cuda.empty_cache()

        model = load_model(timm_id, tc_path, device)
        labels, preds, probs = get_pt_probs_preds(model, bug_val, device)
        m = full_metrics(
            labels, preds, probs, classes,
            title=f'TC {model_key} — {args.run_id}',
            save_path=plots_d / f'metrics_tc_{model_key}.png',
        )
        all_metrics[f'tc_{model_key}'] = m
        del model, bug_train, bug_val
        torch.cuda.empty_cache()

    metrics_file = metrics_d / f'{args.run_id}_tc.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'[02_train_tc] metrics saved → {metrics_file}')


if __name__ == '__main__':
    main()
