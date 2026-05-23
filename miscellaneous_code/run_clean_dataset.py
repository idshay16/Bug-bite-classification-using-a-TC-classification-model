"""Apply grid-line cleaning to all TC images and write to data_categorised_rgb_clean/.

Usage:  python miscellaneous_code/run_clean_dataset.py
"""

import os
import sys
import shutil
import cv2

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from cyclone_preprocessing import build_grid_masks, apply_grid_mask

SRC_DIR   = os.path.join(PROJECT_ROOT, 'Cyclone-Data',
                         'data_categorised_rgb', 'data_categorised_rgb')
CLEAN_DIR = os.path.join(PROJECT_ROOT, 'Cyclone-Data', 'data_categorised_rgb_clean')


def main():
    print('Building per-size grid masks...')
    masks, _, _ = build_grid_masks(SRC_DIR)
    print()

    if os.path.exists(CLEAN_DIR):
        shutil.rmtree(CLEAN_DIR)
        print(f'Removed existing {CLEAN_DIR}')

    total_written = 0
    classes = sorted(c for c in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, c)))
    print(f'Classes: {classes}\n')

    for cls in classes:
        cls_in  = os.path.join(SRC_DIR, cls)
        cls_out = os.path.join(CLEAN_DIR, cls)
        os.makedirs(cls_out, exist_ok=True)
        files = [f for f in os.listdir(cls_in)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for i, fname in enumerate(files):
            img = cv2.imread(os.path.join(cls_in, fname))
            if img is None:
                continue
            cleaned = apply_grid_mask(img, masks)
            cv2.imwrite(os.path.join(cls_out, fname), cleaned)
            total_written += 1
        print(f'  {cls}: {len(files)} images written')

    print(f'\nDone. Total images written: {total_written}')
    print(f'Output dir: {CLEAN_DIR}')


if __name__ == '__main__':
    main()
