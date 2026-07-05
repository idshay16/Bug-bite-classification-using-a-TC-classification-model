#!/usr/bin/env python3
"""
Section 5 — Per-model (non-ensemble) performance metrics.

04_xai.py only ever wrote ensemble-level metrics (control_ensemble / tc_ensemble)
in results/xai/<run_id>/ensemble_metrics.json — the individual per-architecture
control/tc models it loads along the way were never evaluated on their own.
This script closes that gap using every checkpoint already on disk, without
retraining anything.

Fully standalone — run directly, no dependency on run_experiments.py:
    python scripts/05_per_model_metrics.py
(only shared.py / pytorch_utils.py are imported, both resolved relative to
this file, so cwd doesn't matter)

Also called once per config from run_experiments.py's run_eval_and_xai(), right
after 03_evaluate.py / 04_xai.py, so per-model scores land incrementally as the
sweep progresses instead of only at the very end.

Incremental / resumable: on every run it loads the existing scores.json,
skips any (checkpoint set, architecture) pair already scored, and only
evaluates what's missing — new configs, or configs left half-done by a
previous crash. scores.json is re-saved after each checkpoint set (not just
once at the end), so a crash mid-run only loses the in-progress set, not
everything.

Evaluates every checkpoint found under results/checkpoints/:
  - control_seed_<N>/control_<model>.pt   (all control seeds)
  - config_NNN/tc_bug_<model>.pt          (all TC sweep configs)

Output: results/per_model_score/
  - scores.json        — one entry per checkpoint set, each tagged
                          is_control plus seed / label_smoothing / phase2_lr,
                          with per-architecture metrics nested inside
  - metrics_<tag>.png   — confusion matrix / per-class F1 / ROC per model
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'miscellaneous_code'))

from shared import (
    BUG_TRAIN, BUG_VAL, MODEL_SPECS, RESULTS_DIR,
    full_metrics, get_pt_probs_preds, get_pytorch_loaders, load_model,
)

CKPT_DIR = RESULTS_DIR / 'checkpoints'
OUT_DIR  = RESULTS_DIR / 'per_model_score'
OUT_JSON = OUT_DIR / 'scores.json'

# Mirrors miscellaneous_code/run_experiments.py's config grid ordering
# (seed outer loop, then label_smoothing, then phase2_lr) so a config_id
# can be mapped back to (seed, ls, lr) metadata.
LABEL_SMOOTHING_VALUES = [0.0, 0.1, 0.2]
PHASE2_LR_VALUES       = [1e-6, 5e-6, 1e-5]
_N_PER_SEED = len(LABEL_SMOOTHING_VALUES) * len(PHASE2_LR_VALUES)


def _config_meta(config_id: int) -> dict:
    seed = config_id // _N_PER_SEED
    ls_i, lr_i = divmod(config_id % _N_PER_SEED, len(PHASE2_LR_VALUES))
    return {'seed': seed, 'label_smoothing': LABEL_SMOOTHING_VALUES[ls_i],
            'phase2_lr': PHASE2_LR_VALUES[lr_i]}


def _entry_key(entry: dict):
    return ('control', entry['seed']) if entry['is_control'] else ('config', entry['config_id'])


def _load_scores() -> dict:
    """Returns {key: entry} keyed by _entry_key, empty if no prior run."""
    if not OUT_JSON.exists():
        return {}
    with open(OUT_JSON) as f:
        entries = json.load(f)
    return {_entry_key(e): e for e in entries}


def _save_scores(by_key: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(list(by_key.values()), f, indent=2)
    tmp.replace(OUT_JSON)


def _eval_one(timm_id, img_size, bug_batch, weight_path, tag):
    _, val_loader, classes = get_pytorch_loaders(
        BUG_TRAIN, BUG_VAL, img_size=img_size, batch_size=bug_batch, augment=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(timm_id, str(weight_path), device)
    labels, preds, probs = get_pt_probs_preds(model, val_loader, device)
    metrics = full_metrics(
        labels, preds, probs, classes,
        title=tag, save_path=OUT_DIR / f'metrics_{tag}.png')
    del model
    torch.cuda.empty_cache()
    return metrics


def _fill_missing_models(entry: dict, ckpt_dir: Path, weight_prefix: str, tag_prefix: str) -> bool:
    """Evaluates only architectures not already present in entry['models'].
    Returns True if anything changed."""
    changed = False
    for model_key, timm_id, img_size, _, bug_batch in MODEL_SPECS:
        if model_key in entry['models']:
            continue
        wpath = ckpt_dir / f'{weight_prefix}_{model_key}.pt'
        if not wpath.exists():
            print(f'  [skip] missing {wpath}')
            continue
        print(f'  evaluating {model_key} ...')
        m = _eval_one(timm_id, img_size, bug_batch, wpath, f'{tag_prefix}_{model_key}')
        entry['models'][model_key] = m
        changed = True
        print(f'    acc={m["accuracy"]:.4f}  bal_acc={m["balanced_acc"]:.4f}  '
              f'roc_auc={m["macro_roc_auc"]:.4f}  kappa={m["kappa"]:.4f}  mcc={m["mcc"]:.4f}')
    return changed


def main():
    by_key = _load_scores()

    control_dirs = sorted(
        (d for d in CKPT_DIR.glob('control_seed_*') if d.is_dir()),
        key=lambda d: int(d.name.split('_')[-1]))
    config_dirs = sorted(
        (d for d in CKPT_DIR.glob('config_*') if d.is_dir()),
        key=lambda d: int(d.name.split('_')[-1]))

    print(f'[05_per_model_metrics] {len(control_dirs)} control seed(s), '
          f'{len(config_dirs)} config(s) found on disk; '
          f'{len(by_key)} checkpoint set(s) already in scores.json\n')

    # ── every control seed ────────────────────────────────────────────────
    for ctrl_dir in control_dirs:
        seed = int(ctrl_dir.name.split('_')[-1])
        key = ('control', seed)
        entry = by_key.setdefault(key, {'is_control': True, 'seed': seed, 'models': {}})
        if len(entry['models']) == len(MODEL_SPECS):
            continue
        print(f'── control seed={seed} ──')
        if _fill_missing_models(entry, ctrl_dir, 'control', f'control_seed{seed}'):
            _save_scores(by_key)

    # ── every TC sweep config ──────────────────────────────────────────────
    for cfg_dir in config_dirs:
        cfg_id = int(cfg_dir.name.split('_')[-1])
        key = ('config', cfg_id)
        entry = by_key.setdefault(
            key, {'is_control': False, 'config_id': cfg_id, **_config_meta(cfg_id), 'models': {}})
        if len(entry['models']) == len(MODEL_SPECS):
            continue
        print(f"\n── config_{cfg_id:03d}  seed={entry['seed']}  "
              f"ls={entry['label_smoothing']}  phase2_lr={entry['phase2_lr']:.0e} ──")
        if _fill_missing_models(entry, cfg_dir, 'tc_bug', f'config{cfg_id:03d}'):
            _save_scores(by_key)

    print(f'\n[05_per_model_metrics] saved → {OUT_JSON}')

    # ── summary: mean Δaccuracy per architecture, TC configs vs mean control ──
    entries = list(by_key.values())
    control_acc = {}
    for entry in entries:
        if entry['is_control']:
            for model_key, m in entry['models'].items():
                control_acc.setdefault(model_key, []).append(m['accuracy'])
    control_mean_acc = {k: sum(v) / len(v) for k, v in control_acc.items()}

    print('\n── mean Δaccuracy (TC − mean control) by architecture, across all configs ──')
    for model_key, *_ in MODEL_SPECS:
        base = control_mean_acc.get(model_key)
        if base is None:
            continue
        deltas = [e['models'][model_key]['accuracy'] - base
                  for e in entries if not e['is_control'] and model_key in e['models']]
        if deltas:
            print(f'  {model_key:<10}  n={len(deltas)}  '
                  f'mean_delta={sum(deltas)/len(deltas):+.4f}  '
                  f'min={min(deltas):+.4f}  max={max(deltas):+.4f}')


if __name__ == '__main__':
    main()
