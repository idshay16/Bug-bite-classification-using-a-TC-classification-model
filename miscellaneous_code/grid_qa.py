"""Grid-line QA script — runs autonomously, outputs /tmp/grid_qa.png.

Usage:  python miscellaneous_code/grid_qa.py
"""

import os
import sys
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from cyclone_preprocessing import (build_grid_masks, apply_grid_mask,
                                   _flag_ranges, _make_grid_mask,
                                   _grid_position_seams)

SRC_DIR   = os.path.join(PROJECT_ROOT, 'Cyclone-Data',
                         'data_categorised_rgb', 'data_categorised_rgb')
OUT_DIR   = os.path.join(PROJECT_ROOT, 'tmp')
OUT_PNG   = os.path.join(OUT_DIR, 'grid_qa.png')

N_SAMPLES      = 20   # total images to process
SHOW_N         = 10   # rows in the PNG
RESIDUAL_SIGMA = 1.2  # match per-image fallback so QA accurately reports what survives
RESIDUAL_MAX_W = 8    # match per-image fallback width


# ── helpers ───────────────────────────────────────────────────────────────────

def residual_check(clean_bgr, canonical_rows, canonical_cols):
    """Run position-constrained residual seam detector on a cleaned image.

    Only checks positions corresponding to valid grid spacings (divisors 3/4
    of each image dimension).  Geographic coastlines — which appear at
    arbitrary positions — are therefore excluded from the residual count.

    Canonical rows/cols already handled are excluded so we only count
    lines that the cleaning step MISSED.
    """
    gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    rows = _grid_position_seams(gray.mean(axis=1), h,
                                canonical_rows, RESIDUAL_SIGMA, RESIDUAL_MAX_W)
    cols = _grid_position_seams(gray.mean(axis=0), w,
                                canonical_cols, RESIDUAL_SIGMA, RESIDUAL_MAX_W)
    ann  = clean_bgr.copy()
    for s, e in rows:
        ann[s:e, :] = (0, 0, 255)
    for s, e in cols:
        ann[:, s:e] = (0, 0, 255)
    return ann, len(rows) + len(cols)


def lap_ratio_unmasked(raw_bgr, clean_bgr, mask_2d):
    """Laplacian variance ratio computed only on unmasked (untouched) pixels.

    A ratio near 1.0 means cloud structure sharpness is preserved.
    Computing over the full image would penalise legitimate seam removal.
    """
    gray_r = cv2.cvtColor(raw_bgr,   cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_c = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap_r  = cv2.Laplacian(gray_r, cv2.CV_32F)
    lap_c  = cv2.Laplacian(gray_c, cv2.CV_32F)
    unmasked = (mask_2d == 0)
    if unmasked.sum() < 100:
        return 1.0
    var_r = float(lap_r[unmasked].var())
    var_c = float(lap_c[unmasked].var())
    return var_c / var_r if var_r > 0 else 1.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    # Collect all paths, stratified sample
    by_cls = {}
    for cls in sorted(os.listdir(SRC_DIR)):
        cls_path = os.path.join(SRC_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        by_cls[cls] = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    print(f'Classes: {list(by_cls.keys())}')
    per_class = max(1, N_SAMPLES // len(by_cls))
    sampled   = []
    for cls, paths in by_cls.items():
        sampled.extend(random.sample(paths, min(per_class, len(paths))))
    random.shuffle(sampled)
    sampled = sampled[:N_SAMPLES]
    print(f'Processing {len(sampled)} images...\n')

    # Build masks once
    masks, _, _ = build_grid_masks(SRC_DIR)
    print()

    # Process each image
    results = []
    h_key, w_key = max(masks.keys(), key=lambda k: k[0] * k[1])  # dominant size
    dom_mask = masks[(h_key, w_key)]

    for p in sampled:
        cls = os.path.basename(os.path.dirname(p))
        raw = cv2.imread(p)
        if raw is None:
            continue
        clean = apply_grid_mask(raw, masks)

        # Canonical mask for this image — scale spans to match actual image size
        h_img, w_img = raw.shape[:2]
        if (h_img, w_img) in masks:
            cm = masks[(h_img, w_img)]
            can_rows = cm['grid_rows']
            can_cols = cm['grid_cols']
        else:
            nearest_k = min(masks.keys(), key=lambda k: abs(k[0]-h_img)+abs(k[1]-w_img))
            cm = masks[nearest_k]
            sh = h_img / nearest_k[0]
            sw = w_img / nearest_k[1]
            can_rows = [(int(s*sh), int(e*sh)) for s, e in cm['grid_rows']]
            can_cols = [(int(s*sw), int(e*sw)) for s, e in cm['grid_cols']]

        ann, n_res = residual_check(clean, can_rows, can_cols)

        mask2d = cm['visual']
        if mask2d.shape != (h_img, w_img):
            mask2d = cv2.resize(mask2d, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        lr = lap_ratio_unmasked(raw, clean, mask2d)

        results.append((cls, os.path.basename(p), raw, clean, ann, n_res, lr))
        flag = ' *** RESIDUAL ***' if n_res > 0 else ''
        print(f'  {cls}/{os.path.basename(p)}: '
              f'residual={n_res}  lap_unmasked={lr:.3f}{flag}')

    total_res  = sum(r[5] for r in results)
    mean_lap   = float(np.mean([r[6] for r in results])) if results else 0.0
    print(f'\n=== SUMMARY ===')
    print(f'  total_residual_lines   : {total_res}')
    print(f'  mean_lap_unmasked_ratio: {mean_lap:.3f}')
    print(f'  (1.0 = cloud sharpness unchanged; <0.95 = blur introduced)')

    # Generate PNG: SHOW_N rows x 3 cols [Raw | Cleaned | Residual]
    show = results[:SHOW_N]
    fig, axes = plt.subplots(len(show), 3,
                             figsize=(15, 4 * len(show)),
                             squeeze=False)
    for i, (cls, fname, raw, clean, ann, n_res, lr) in enumerate(show):
        axes[i][0].imshow(cv2.cvtColor(raw,   cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f'Raw  [{cls}]\n{fname}', fontsize=7)
        axes[i][0].axis('off')

        axes[i][1].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        axes[i][1].set_title(f'Cleaned  lap_unmask={lr:.3f}', fontsize=7)
        axes[i][1].axis('off')

        axes[i][2].imshow(cv2.cvtColor(ann,   cv2.COLOR_BGR2RGB))
        col = 'red' if n_res > 0 else 'green'
        axes[i][2].set_title(f'Residual={n_res}  (red=detected)', fontsize=7, color=col)
        axes[i][2].axis('off')

    plt.suptitle(
        f'Grid QA  |  total_residual={total_res}  mean_lap_unmasked={mean_lap:.3f}',
        fontsize=11, y=1.002)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=90, bbox_inches='tight')
    plt.close()
    print(f'\nSaved -> {OUT_PNG}')


if __name__ == '__main__':
    main()
