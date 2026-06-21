#!/usr/bin/env python3
"""
Classification evaluation for the two project-book configs:
  - Control : results/checkpoints/control_seed_3/   (s3_ls0.20_lr1.0e-05)
  - Best TC  : results/checkpoints/config_058/       (s6_ls0.10_lr5.0e-06)

Outputs per-arch metrics + ensemble metrics to:
  results/classification_book/<control|best_tc>/

Usage (from repo root, with tc-env active):
    python scripts/run_classification_book.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'miscellaneous_code'))

from shared import MODEL_SPECS, N_CLASSES, full_metrics, out_dir
from pytorch_utils import get_pytorch_loaders, get_pt_probs_preds

import numpy as np
import timm

BUG_VAL = '/home/test/bug_data/val'

CKPT = ROOT / 'results' / 'checkpoints'
OUT  = ROOT / 'results' / 'classification_book'

RUNS = [
    {
        'label':    'control',
        'display':  'Control  s3_ls0.20_lr1.0e-05',
        'ckpt_dir': CKPT / 'control_seed_3',
        'pt_prefix': 'control',   # control_convnext.pt etc.
    },
    {
        'label':    'best_tc',
        'display':  'Best TC  s6_ls0.10_lr5.0e-06',
        'ckpt_dir': CKPT / 'config_058',
        'pt_prefix': 'tc_bug',    # tc_bug_convnext.pt etc.
    },
]

ARCH_KEY = {'convnext': 'convnext', 'densenet': 'densenet', 'inception': 'inception'}


def load_model(timm_id: str, weight_path: Path, device: torch.device) -> torch.nn.Module:
    m = timm.create_model(timm_id, pretrained=False, num_classes=N_CLASSES)
    m.load_state_dict(
        torch.load(str(weight_path), map_location='cpu', weights_only=True)
    )
    return m.to(device).eval()


def ensemble_probs(probs_list):
    return np.mean(probs_list, axis=0).tolist()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    for run in RUNS:
        label    = run['label']
        display  = run['display']
        ckpt_dir = run['ckpt_dir']
        prefix   = run['pt_prefix']

        print(f'\n{"="*60}')
        print(f'{display}')
        print(f'Checkpoints: {ckpt_dir.name}')

        save_dir = OUT / label
        save_dir.mkdir(parents=True, exist_ok=True)

        all_true, all_preds, all_probs_ensemble = None, None, []
        class_names = None

        for arch_key, timm_id, img_size, _, bug_batch in MODEL_SPECS:
            pt_name = f'{prefix}_{arch_key}.pt'
            pt_path = ckpt_dir / pt_name

            if not pt_path.exists():
                print(f'  [skip] {arch_key} — {pt_path.name} not found')
                continue

            print(f'  [{arch_key}] loading {pt_name}...')
            model = load_model(timm_id, pt_path, device)

            _, val_loader = get_pytorch_loaders(
                BUG_VAL, BUG_VAL,
                img_size=img_size,
                batch_size=bug_batch,
                augment=False,
            )

            if class_names is None:
                class_names = val_loader.dataset.classes

            true_labels, preds, probs = get_pt_probs_preds(model, val_loader, device)
            del model

            arch_metrics = full_metrics(
                true_labels, preds, probs, class_names,
                title=f'{display} — {arch_key}',
                save_path=save_dir / f'metrics_{arch_key}.png',
            )
            print(f'    acc={arch_metrics["accuracy"]:.4f}  '
                  f'bacc={arch_metrics["balanced_acc"]:.4f}  '
                  f'auc={arch_metrics["macro_roc_auc"]:.4f}')

            with open(save_dir / f'metrics_{arch_key}.json', 'w') as f:
                json.dump(arch_metrics, f, indent=2)

            if all_true is None:
                all_true = true_labels
                all_preds = preds
            all_probs_ensemble.append(np.array(probs))

        if not all_probs_ensemble:
            print('  No checkpoints loaded — skipping ensemble.')
            continue

        ens_probs = np.mean(all_probs_ensemble, axis=0).tolist()
        ens_preds = np.argmax(ens_probs, axis=1).tolist()

        ens_metrics = full_metrics(
            all_true, ens_preds, ens_probs, class_names,
            title=f'{display} — Ensemble',
            save_path=save_dir / 'metrics_ensemble.png',
        )
        print(f'  [ensemble] acc={ens_metrics["accuracy"]:.4f}  '
              f'bacc={ens_metrics["balanced_acc"]:.4f}  '
              f'auc={ens_metrics["macro_roc_auc"]:.4f}')

        with open(save_dir / 'metrics_ensemble.json', 'w') as f:
            json.dump(ens_metrics, f, indent=2)

        print(f'  Saved → {save_dir}')

    print(f'\nDone. All outputs in: {OUT}')


if __name__ == '__main__':
    main()
