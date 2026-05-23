"""Grid-line QA script — focused single-image debug mode.

Usage:  python miscellaneous_code/grid_qa.py [path/to/image.jpg]
Default target: DD(30-35)/20030517.00-30.jpg

Output: tmp/grid_qa.png  —  4 panels:
  Raw | Row profile (raw vs clean overlaid) | Mask | Cleaned
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from cyclone_preprocessing import (build_grid_masks, apply_grid_mask,
                                   _make_grid_mask, _highpass,
                                   _grid_position_seams)

SRC_DIR  = os.path.join(PROJECT_ROOT, 'Cyclone-Data',
                        'data_categorised_rgb', 'data_categorised_rgb')
OUT_DIR  = os.path.join(PROJECT_ROOT, 'tmp')
OUT_PNG  = os.path.join(OUT_DIR, 'grid_qa.png')

DEFAULT_IMG = os.path.join(SRC_DIR, 'DD(30-35)', '20030517.00-30.jpg')


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    print(f'Target: {img_path}')

    masks, _, _ = build_grid_masks(SRC_DIR)

    raw   = cv2.imread(img_path)
    if raw is None:
        print(f'ERROR: cannot read {img_path}')
        sys.exit(1)

    clean = apply_grid_mask(raw, masks)
    h, w  = raw.shape[:2]

    # Canonical mask for this image
    if (h, w) in masks:
        cm       = masks[(h, w)]
        can_rows = cm['grid_rows']
        can_cols = cm['grid_cols']
    else:
        nearest_k = min(masks.keys(), key=lambda k: abs(k[0]-h)+abs(k[1]-w))
        cm  = masks[nearest_k]
        sh  = h / nearest_k[0]
        sw  = w / nearest_k[1]
        can_rows = [(int(s*sh), int(e*sh)) for s, e in cm['grid_rows']]
        can_cols = [(int(s*sw), int(e*sw)) for s, e in cm['grid_cols']]

    # Build visual mask (canonical + any per-image extra would need apply internals)
    mask2d = _make_grid_mask(can_rows, can_cols, h, w)

    # Row profiles
    gray_r  = cv2.cvtColor(raw,   cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_c  = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prof_r  = gray_r.mean(axis=1)
    prof_c  = gray_c.mean(axis=1)
    hp_r    = _highpass(prof_r)
    hp_c    = _highpass(prof_c)

    # Detected extra rows (per-image fallback candidates)
    PER_IMG_ROW_SIGMA = 0.6
    PER_IMG_MW        = 8
    extra_rows = _grid_position_seams(
        cv2.cvtColor(apply_grid_mask(raw, masks), cv2.COLOR_BGR2GRAY).astype(np.float32).mean(axis=1),
        h, can_rows, PER_IMG_ROW_SIGMA, PER_IMG_MW)

    print(f'Image size: {h}x{w}')
    print(f'Canonical rows: {can_rows}  cols: {can_cols}')
    print(f'Extra rows detected by fallback: {extra_rows}')

    # ── Plot ────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 8))
    gs  = fig.add_gridspec(1, 4, wspace=0.08)

    ax_raw   = fig.add_subplot(gs[0])
    ax_prof  = fig.add_subplot(gs[1])
    ax_mask  = fig.add_subplot(gs[2])
    ax_clean = fig.add_subplot(gs[3])

    # Raw
    ax_raw.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    ax_raw.set_title(f'Raw\n{os.path.basename(img_path)}', fontsize=8)
    ax_raw.axis('off')

    # Row profile graph — raw (blue) and clean (orange), high-pass residual (green)
    rows_idx = np.arange(h)
    ax_prof.plot(prof_r, rows_idx, color='steelblue',  lw=1,   label='raw mean')
    ax_prof.plot(prof_c, rows_idx, color='darkorange',  lw=1,   label='clean mean')
    ax_prof.plot(hp_r,   rows_idx, color='green',       lw=0.7, alpha=0.6, label='raw HP')
    # Mark candidate seam rows
    for divisor in (3, 4):
        spacing = h // divisor
        for i in range(1, divisor):
            pos = i * spacing
            ax_prof.axhline(pos, color='red', lw=0.5, alpha=0.5, ls='--')
    # Mark canonical rows
    for s, e in can_rows:
        ax_prof.axhspan(s, e, color='blue', alpha=0.15)
    # Mark extra rows
    for s, e in extra_rows:
        ax_prof.axhspan(s, e, color='orange', alpha=0.3)
    ax_prof.set_title('Row profile\n(red=grid candidates, blue=canonical, orange=extra)', fontsize=7)
    ax_prof.set_xlabel('brightness / HP value')
    ax_prof.set_ylabel('row index')
    ax_prof.invert_yaxis()
    ax_prof.legend(fontsize=6)
    ax_prof.grid(True, lw=0.3)

    # Mask
    ax_mask.imshow(mask2d, cmap='gray')
    ax_mask.set_title('Canonical mask', fontsize=8)
    ax_mask.axis('off')

    # Cleaned
    ax_clean.imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    ax_clean.set_title('Cleaned', fontsize=8)
    ax_clean.axis('off')

    plt.suptitle(
        f'Grid debug  |  {os.path.basename(img_path)}  |  {h}x{w}',
        fontsize=10)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {OUT_PNG}')


if __name__ == '__main__':
    main()
