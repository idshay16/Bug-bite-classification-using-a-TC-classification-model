#!/usr/bin/env python3
"""
run_experiments.py — Hyperparameter sweep for TC vs Control classification.

Usage:
    python run_experiments.py            # run all pending configs
    python run_experiments.py --status   # show progress summary
    python run_experiments.py --reset    # wipe state and results (prompts)

Pause / resume:
    touch results/PAUSE   → pauses after the current model finishes
    Ctrl-C                → same effect

Crash recovery:
    Re-run the script. Configs marked "running" (interrupted) restart from
    the last completed model checkpoint. Already-completed model checkpoints
    are reused and only re-evaluated (fast).

Config grid lives at the top of this file — add values to
LABEL_SMOOTHING_VALUES or PHASE2_LR_VALUES and re-run; the new configs
are appended to the existing state file automatically.

Results → results/experiment_results.json
State   → results/experiment_state.json
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import classification_report as skl_report

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_utils import (
    get_backbone_state_dict,
    get_pytorch_loaders,
    get_pt_probs_preds,
    train_pytorch_model,
)

# ── dataset paths ─────────────────────────────────────────────────────────────
BUG_TRAIN = '/home/test/bug_data/train'
BUG_VAL   = '/home/test/bug_data/val'
TC_TRAIN  = '/home/test/cyclone_data_split/train'
TC_VAL    = '/home/test/cyclone_data_split/val'

# ── output paths ──────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent
RESULTS_DIR  = REPO_ROOT / 'results'
STATE_FILE   = RESULTS_DIR / 'experiment_state.json'
RESULTS_FILE = RESULTS_DIR / 'experiment_results.json'
CKPT_DIR     = RESULTS_DIR / 'checkpoints'
PAUSE_FILE   = RESULTS_DIR / 'PAUSE'

# ── config grid ───────────────────────────────────────────────────────────────
# These params control CYCLONE PRE-TRAINING only.
# Bug-bite fine-tuning always uses BUG_LABEL_SMOOTHING / BUG_PHASE2_LR below.
LABEL_SMOOTHING_VALUES = [0.0, 0.1, 0.2]
PHASE2_LR_VALUES       = [1e-6, 5e-6, 1e-5]
N_SEEDS                = 7

# Fixed bug-bite fine-tuning params
BUG_LABEL_SMOOTHING = 0.0
BUG_PHASE2_LR       = 5e-6

# (key, timm_id, img_size, cyc_p1_bs, bug_bs, cyc_p2_epochs, bug_p2_epochs)
MODELS = [
    ('convnext',  'convnext_tiny.fb_in22k_ft_in1k', 256, 256, 16, 30, 20),
    ('densenet',  'densenet121.ra_in1k',             256, 256, 16, 30, 20),
    ('inception', 'inception_v3.tv_in1k',            299, 128, 16, 30, 20),
]
PHASE1_EPOCHS = 5
PATIENCE      = 2
N_CLASSES     = 5

# ── interrupt flag ────────────────────────────────────────────────────────────
_stop_requested = False

def _handle_sigint(sig, frame):
    global _stop_requested
    _stop_requested = True
    print('\n[runner] Ctrl-C — will stop after current model. '
          'Run again to resume. (touch results/PAUSE for same effect)')

signal.signal(signal.SIGINT, _handle_sigint)

# ── utils ─────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _atomic_write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)

def load_json(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default

def is_paused() -> bool:
    return _stop_requested or PAUSE_FILE.exists()

def clear_pause():
    if PAUSE_FILE.exists():
        PAUSE_FILE.unlink()

# ── state management ──────────────────────────────────────────────────────────
def _make_config(cfg_id, seed, ls, p2lr):
    return {
        'id': cfg_id,
        'seed': seed,
        'label_smoothing': ls,
        'phase2_lr': p2lr,
        'status': 'pending',   # pending | completed
        'completed_at': None,
        'elapsed_h': None,
    }

def build_config_grid():
    configs, cfg_id = [], 0
    for seed in range(N_SEEDS):
        for ls in LABEL_SMOOTHING_VALUES:
            for p2lr in PHASE2_LR_VALUES:
                configs.append(_make_config(cfg_id, seed, ls, p2lr))
                cfg_id += 1
    return configs

def init_state():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        _atomic_write(RESULTS_FILE, [])

    existing = load_json(STATE_FILE, None)
    if existing is None:
        _atomic_write(STATE_FILE, {'configs': build_config_grid()})
        return

    # Merge: add any new configs not already present (new grid values added later)
    known_keys = {(c['seed'], c['label_smoothing'], c['phase2_lr'])
                  for c in existing['configs']}
    new_grid   = build_config_grid()
    next_id    = max(c['id'] for c in existing['configs']) + 1
    added = 0
    for cfg in new_grid:
        key = (cfg['seed'], cfg['label_smoothing'], cfg['phase2_lr'])
        if key not in known_keys:
            cfg['id'] = next_id
            existing['configs'].append(cfg)
            next_id += 1
            added += 1

    # Any config stuck in "running" state means previous run crashed — reset to pending
    for cfg in existing['configs']:
        if cfg['status'] == 'running':
            cfg['status'] = 'pending'

    if added:
        print(f'[runner] Added {added} new config(s) from updated grid.')
    _atomic_write(STATE_FILE, existing)

def load_state():
    return load_json(STATE_FILE, {'configs': []})

def save_state(state):
    _atomic_write(STATE_FILE, state)

def result_exists(config_id: int, model_key: str, model_type: str) -> bool:
    results = load_json(RESULTS_FILE, [])
    return any(
        r['config_id'] == config_id
        and r['model'] == model_key
        and r['type'] == model_type
        for r in results
    )

def append_result(result: dict):
    results = load_json(RESULTS_FILE, [])
    # Deduplicate — safe to call even on re-run after crash
    key = (result['config_id'], result['model'], result['type'])
    results = [r for r in results
               if (r['config_id'], r['model'], r['type']) != key]
    results.append(result)
    _atomic_write(RESULTS_FILE, results)

# ── run id ───────────────────────────────────────────────────────────────────
def make_run_id(cfg: dict) -> str:
    return f's{cfg["seed"]}_ls{cfg["label_smoothing"]:.2f}_lr{cfg["phase2_lr"]:.1e}'

# ── post-training evaluation scripts ─────────────────────────────────────────
def run_eval_and_xai(cfg: dict):
    import subprocess
    run_id   = make_run_id(cfg)
    cfg_id   = cfg['id']
    seed     = cfg['seed']
    scripts  = REPO_ROOT / 'scripts'

    base = [
        '--run-id',         run_id,
        '--ctrl-convnext',  str(ctrl_ckpt_path(seed, 'convnext')),
        '--ctrl-densenet',  str(ctrl_ckpt_path(seed, 'densenet')),
        '--ctrl-inception', str(ctrl_ckpt_path(seed, 'inception')),
        '--tc-convnext',    str(tc_ckpt_path(cfg_id, 'convnext')),
        '--tc-densenet',    str(tc_ckpt_path(cfg_id, 'densenet')),
        '--tc-inception',   str(tc_ckpt_path(cfg_id, 'inception')),
    ]
    cyc_args = [
        '--cyc-convnext',   str(cyc_ckpt_path(cfg_id, 'convnext')),
        '--cyc-densenet',   str(cyc_ckpt_path(cfg_id, 'densenet')),
        '--cyc-inception',  str(cyc_ckpt_path(cfg_id, 'inception')),
    ]

    for script, extra in [
        ('03_evaluate.py', cyc_args),
        ('04_xai.py',      []),
    ]:
        print(f'[runner] running {script} for {run_id}')
        result = subprocess.run(
            [sys.executable, str(scripts / script)] + base + extra,
            check=False,
        )
        if result.returncode != 0:
            print(f'[runner] WARNING: {script} exited with code {result.returncode}')

    # 05_per_model_metrics.py is incremental/resumable (skips anything already
    # in results/per_model_score/scores.json), so calling it after every config
    # only costs the newly-finished config's 3 models, and self-heals any
    # config a previous crash left un-scored.
    print(f'[runner] running 05_per_model_metrics.py for {run_id}')
    result = subprocess.run(
        [sys.executable, str(scripts / '05_per_model_metrics.py')], check=False)
    if result.returncode != 0:
        print(f'[runner] WARNING: 05_per_model_metrics.py exited with code {result.returncode}')

# ── checkpoint path helpers ───────────────────────────────────────────────────
def cyc_ckpt_path(cfg_id: int, model_key: str) -> Path:
    return CKPT_DIR / f'config_{cfg_id:03d}' / f'cyclone_{model_key}.pt'

def tc_ckpt_path(cfg_id: int, model_key: str) -> Path:
    return CKPT_DIR / f'config_{cfg_id:03d}' / f'tc_bug_{model_key}.pt'

def ctrl_ckpt_path(seed: int, model_key: str) -> Path:
    # Control doesn't depend on ls/p2lr — key by seed only to avoid redundant training
    return CKPT_DIR / f'control_seed_{seed}' / f'control_{model_key}.pt'

# ── training ──────────────────────────────────────────────────────────────────
def _evaluate(model, loader, classes, device, config_id, model_key, model_type,
              seed, ls, p2lr, history=None):
    labels, preds, _ = get_pt_probs_preds(model, loader, device)
    report = skl_report(labels, preds, target_names=classes,
                        output_dict=True, zero_division=0)
    result = {
        'config_id': config_id,
        'seed': seed,
        'label_smoothing': ls,
        'phase2_lr': p2lr,
        'model': model_key,
        'type': model_type,
        'val_acc': report['accuracy'],
        'val_macro_f1': report['macro avg']['f1-score'],
        'val_macro_precision': report['macro avg']['precision'],
        'val_macro_recall': report['macro avg']['recall'],
        'classification_report': report,
        'completed_at': now_iso(),
    }
    if history is not None:
        result['final_val_loss'] = min(history['val_loss'])
        result['history'] = history
    return result


def train_one_config(cfg: dict, device: torch.device):
    cfg_id = cfg['id']
    seed   = cfg['seed']
    ls     = cfg['label_smoothing']
    p2lr   = cfg['phase2_lr']

    for model_key, timm_id, img_size, cyc_p1_bs, bug_bs, cyc_p2_ep, bug_p2_ep in MODELS:
        print(f'\n  ── {model_key} ──────────────────────────────────────────')

        # ── phase 0: cyclone pre-training ─────────────────────────────────────
        cyc_path = cyc_ckpt_path(cfg_id, model_key)
        if not cyc_path.exists():
            print(f'  [phase 0] cyclone pre-training  ls={ls}  p2lr={p2lr:.0e}')
            seed_everything(seed)
            cyc_path.parent.mkdir(parents=True, exist_ok=True)
            tc_train, tc_val, _ = get_pytorch_loaders(
                TC_TRAIN, TC_VAL, img_size=img_size,
                batch_size=cyc_p1_bs, augment='strong')
            cyc_model = timm.create_model(timm_id, pretrained=True, num_classes=N_CLASSES)
            train_pytorch_model(
                cyc_model, tc_train, tc_val, device,
                phase1_epochs=PHASE1_EPOCHS, phase2_epochs=cyc_p2_ep,
                patience=PATIENCE, save_path=str(cyc_path),
                label_smoothing=ls, phase2_lr=p2lr,
            )
            del cyc_model, tc_train, tc_val
            torch.cuda.empty_cache()
        else:
            print(f'  [phase 0] cyclone checkpoint found — skipping')

        if is_paused():
            return False   # signal caller to stop

        # ── phase 1+2: TC transfer → bug-bite ─────────────────────────────────
        tc_path  = tc_ckpt_path(cfg_id, model_key)
        need_tc  = not result_exists(cfg_id, model_key, 'tc')
        seed_everything(seed)
        bug_train, bug_val, bug_classes = get_pytorch_loaders(
            BUG_TRAIN, BUG_VAL, img_size=img_size,
            batch_size=bug_bs, augment=False)

        if not tc_path.exists():
            print(f'  [phase 1+2] TC fine-tune on bug-bite')
            tc_path.parent.mkdir(parents=True, exist_ok=True)
            tc_model = timm.create_model(timm_id, pretrained=False, num_classes=N_CLASSES)
            backbone_sd = get_backbone_state_dict(
                torch.load(str(cyc_path), map_location='cpu', weights_only=True))
            tc_model.load_state_dict(backbone_sd, strict=False)
            history = train_pytorch_model(
                tc_model, bug_train, bug_val, device,
                phase1_epochs=PHASE1_EPOCHS, phase2_epochs=bug_p2_ep,
                patience=PATIENCE, save_path=str(tc_path),
                label_smoothing=BUG_LABEL_SMOOTHING, phase2_lr=BUG_PHASE2_LR,
            )
            tc_model.load_state_dict(
                torch.load(str(tc_path), map_location='cpu', weights_only=True))
        elif need_tc:
            print(f'  [phase 1+2] TC checkpoint found — re-evaluating')
            tc_model = timm.create_model(timm_id, pretrained=False, num_classes=N_CLASSES)
            tc_model.load_state_dict(
                torch.load(str(tc_path), map_location='cpu', weights_only=True))
            history = None
        else:
            tc_model = None

        if need_tc and tc_model is not None:
            tc_model = tc_model.to(device)
            append_result(_evaluate(tc_model, bug_val, bug_classes, device,
                                    cfg_id, model_key, 'tc', seed, ls, p2lr, history))
            del tc_model
            torch.cuda.empty_cache()

        if is_paused():
            del bug_train, bug_val
            return False

        # ── phase 1+2: control ImageNet → bug-bite ───────────────────────────
        ctrl_path  = ctrl_ckpt_path(seed, model_key)
        need_ctrl  = not result_exists(cfg_id, model_key, 'control')

        if not ctrl_path.exists():
            print(f'  [phase 1+2] Control fine-tune on bug-bite  seed={seed}')
            seed_everything(seed)
            ctrl_path.parent.mkdir(parents=True, exist_ok=True)
            ctrl_model = timm.create_model(timm_id, pretrained=True, num_classes=N_CLASSES)
            ctrl_history = train_pytorch_model(
                ctrl_model, bug_train, bug_val, device,
                phase1_epochs=PHASE1_EPOCHS, phase2_epochs=bug_p2_ep,
                patience=PATIENCE, save_path=str(ctrl_path),
                label_smoothing=BUG_LABEL_SMOOTHING, phase2_lr=BUG_PHASE2_LR,
            )
            ctrl_model.load_state_dict(
                torch.load(str(ctrl_path), map_location='cpu', weights_only=True))
        elif need_ctrl:
            print(f'  [phase 1+2] Control checkpoint found — re-evaluating')
            ctrl_model = timm.create_model(timm_id, pretrained=False, num_classes=N_CLASSES)
            ctrl_model.load_state_dict(
                torch.load(str(ctrl_path), map_location='cpu', weights_only=True))
            ctrl_history = None
        else:
            ctrl_model = None

        if need_ctrl and ctrl_model is not None:
            ctrl_model = ctrl_model.to(device)
            append_result(_evaluate(ctrl_model, bug_val, bug_classes, device,
                                    cfg_id, model_key, 'control', seed, ls, p2lr, ctrl_history))
            del ctrl_model
            torch.cuda.empty_cache()

        del bug_train, bug_val
        torch.cuda.empty_cache()

        if is_paused():
            return False

    return True   # config fully complete


# ── commands ──────────────────────────────────────────────────────────────────
def cmd_status():
    if not STATE_FILE.exists():
        print('No state file. Run without --status to initialize.')
        return
    state   = load_state()
    configs = state['configs']
    done    = [c for c in configs if c['status'] == 'completed']
    pending = [c for c in configs if c['status'] != 'completed']
    print(f'Configs: {len(configs)} total | {len(done)} completed | {len(pending)} pending')
    remaining_h_lo = len(pending) * 1.5
    remaining_h_hi = len(pending) * 2.0
    print(f'Estimated remaining: {remaining_h_lo:.0f}–{remaining_h_hi:.0f}h')

    results = load_json(RESULTS_FILE, [])
    if results:
        print('\nVal acc by label_smoothing (all completed TC results):')
        by_ls = defaultdict(list)
        for r in results:
            if r['type'] == 'tc':
                by_ls[r['label_smoothing']].append(r['val_acc'])
        for ls in sorted(by_ls):
            accs = by_ls[ls]
            print(f'  ls={ls:<4}  n={len(accs):>2}  '
                  f'mean={np.mean(accs):.4f}  std={np.std(accs):.4f}  '
                  f'min={np.min(accs):.4f}  max={np.max(accs):.4f}')

        print('\nVal acc by model (TC, all ls):')
        by_model = defaultdict(list)
        for r in results:
            if r['type'] == 'tc':
                by_model[r['model']].append(r['val_acc'])
        for m in sorted(by_model):
            accs = by_model[m]
            print(f'  {m:<10}  n={len(accs):>2}  mean={np.mean(accs):.4f}  std={np.std(accs):.4f}')


def cmd_run(device: torch.device):
    init_state()
    state   = load_state()
    configs = state['configs']
    pending = [c for c in configs if c['status'] != 'completed']

    if not pending:
        print('[runner] All configs already completed.')
        cmd_status()
        return

    print(f'[runner] {len(pending)} configs pending. Device: {device}')
    print(f'[runner] Touch results/PAUSE or press Ctrl-C to pause between models.\n')

    for cfg in configs:
        if cfg['status'] == 'completed':
            continue
        if is_paused():
            clear_pause()
            print('[runner] Paused. Run again to resume.')
            break

        cfg['status'] = 'running'
        save_state(state)
        t0 = time.time()
        done_idx = sum(1 for c in configs if c['status'] == 'completed') + 1

        print(f'\n{"═"*62}')
        print(f'[runner] Config {done_idx}/{len(configs)}: '
              f'id={cfg["id"]}  seed={cfg["seed"]}  '
              f'ls={cfg["label_smoothing"]}  p2lr={cfg["phase2_lr"]:.0e}')
        print(f'{"═"*62}')

        try:
            completed = train_one_config(cfg, device)
        except KeyboardInterrupt:
            cfg['status'] = 'pending'
            save_state(state)
            print('\n[runner] Interrupted mid-model. State saved. Run again to resume.')
            sys.exit(0)
        except Exception as exc:
            cfg['status'] = 'pending'
            save_state(state)
            print(f'[runner] Config {cfg["id"]} failed: {exc}')
            raise

        elapsed = (time.time() - t0) / 3600
        if completed:
            run_eval_and_xai(cfg)
            cfg['status'] = 'completed'
            cfg['completed_at'] = now_iso()
            cfg['elapsed_h']    = round((time.time() - t0) / 3600, 2)
            print(f'\n[runner] Config {cfg["id"]} done in {cfg["elapsed_h"]:.2f}h')
        else:
            # Paused mid-config — leave as pending so it resumes from checkpoints
            cfg['status'] = 'pending'
            print(f'\n[runner] Config {cfg["id"]} paused at {elapsed:.2f}h. Checkpoints saved.')
        save_state(state)

        if is_paused():
            clear_pause()
            print('[runner] Paused. Run again to resume.')
            break

    print()
    cmd_status()


def cmd_reset():
    confirm = input(
        'This will delete experiment_state.json and experiment_results.json.\n'
        'Checkpoint .pt files are NOT deleted (re-use them by re-running).\n'
        'Type YES to confirm: ')
    if confirm.strip() == 'YES':
        for f in (STATE_FILE, RESULTS_FILE):
            if f.exists():
                f.unlink()
        print('State and results cleared.')
    else:
        print('Aborted.')


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='TC hyperparameter sweep runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--status', action='store_true',
                        help='Show progress summary and exit')
    parser.add_argument('--reset',  action='store_true',
                        help='Clear state/results files (prompts for confirmation)')
    args = parser.parse_args()

    if args.reset:
        cmd_reset()
        return
    if args.status:
        cmd_status()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cmd_run(device)


if __name__ == '__main__':
    main()
