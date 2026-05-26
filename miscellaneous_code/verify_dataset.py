#!/usr/bin/env python3
"""
verify_dataset.py — Spot-check the downloaded cyclone dataset.

Run this after fetch_cyclone_dataset.py to:
  • Print a per-category image count
  • Check a sample of images for corruption
  • Generate a sample_grid.png so you can visually confirm the imagery
  • Print ready-to-use PyTorch / Keras loading snippets

Usage:
  python verify_dataset.py
  python verify_dataset.py --dir path/to/cyclone_dataset
"""

import sys
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# ─────────────────────────────────────────────────────────────────────

CATEGORIES = [
    "cat1_tropical_depression",
    "cat2_tropical_storm",
    "cat3_hurricane_cat12",
    "cat4_hurricane_cat34",
    "cat5_hurricane_cat5",
]

LABELS = {
    "cat1_tropical_depression": "Tropical Depression  (< 34 kt)",
    "cat2_tropical_storm":      "Tropical Storm       (34–63 kt)",
    "cat3_hurricane_cat12":     "Hurricane Cat 1–2    (64–95 kt)",
    "cat4_hurricane_cat34":     "Hurricane Cat 3–4    (96–136 kt)",
    "cat5_hurricane_cat5":      "Major Hurricane      (≥ 137 kt)",
}

# How many images to spot-check per category for corruption.
# Checking all 4,000 would take minutes — 50 is enough to catch systemic issues.
CORRUPTION_SAMPLE_SIZE = 50

# How many sample images to show per category in the grid
GRID_SAMPLES = 4

# ─────────────────────────────────────────────────────────────────────

def sep(title=""):
    print()
    print("─" * 60)
    if title:
        print(f"  {title}")
        print("─" * 60)

def check_image(path):
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)  # reopen after verify (verify closes it)
        return True, img.width, img.height
    except Exception:
        return False, 0, 0

def load_rgb(path):
    try:
        return np.array(Image.open(path).convert("L"))
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────

def verify(dataset_dir):
    base = Path(dataset_dir)
    if not base.exists():
        print(f"[ERROR] Directory not found: {base.resolve()}")
        sys.exit(1)

    sep("Dataset Verification Report")
    print(f"  Path: {base.resolve()}")

    counts       = {}
    corrupt_hits = {}
    sample_imgs  = {}
    all_sizes    = {}

    # ── Per-category scan ─────────────────────────────────────────────
    sep("Image Counts")
    total = 0
    for cat in CATEGORIES:
        cat_dir = base / cat
        if not cat_dir.exists():
            print(f"  ✗  MISSING folder: {cat}/")
            counts[cat] = 0
            continue

        images = sorted(cat_dir.glob("*.png")) + sorted(cat_dir.glob("*.jpg"))
        n = len(images)
        counts[cat] = n
        total += n

        # Corruption spot-check on a random sample
        sample = random.sample(images, min(CORRUPTION_SAMPLE_SIZE, n))
        corrupt = 0
        sizes   = set()
        ok_imgs = []
        for p in sample:
            ok, w, h = check_image(p)
            if ok:
                sizes.add((w, h))
                if len(ok_imgs) < GRID_SAMPLES:
                    ok_imgs.append(p)
            else:
                corrupt += 1

        corrupt_hits[cat] = corrupt
        all_sizes[cat]    = sizes
        sample_imgs[cat]  = ok_imgs

        size_str   = ", ".join(f"{w}×{h}" for w, h in sizes) or "—"
        corrupt_str = f"  ⚠ {corrupt} corrupt in sample" if corrupt else ""
        print(f"  {'✓' if n > 0 else '✗'}  {LABELS[cat]}")
        print(f"       {n:>5,} images  |  size: {size_str}{corrupt_str}")

    print(f"\n  Total: {total:,} images across {len(CATEGORIES)} categories")

    # ── Balance check ─────────────────────────────────────────────────
    sep("Balance")
    vals = [counts[c] for c in CATEGORIES]
    mn, mx = min(vals), max(vals)
    if mx > 0:
        ratio = mn / mx
        bar_max = 30
        for cat in CATEGORIES:
            n   = counts[cat]
            bar = "█" * int((n / mx) * bar_max)
            print(f"  {bar:<{bar_max}}  {n:>5,}  {LABELS[cat][:25]}")
        print()
        status = "✓ Well balanced" if ratio >= 0.9 else \
                 "~ Acceptable"    if ratio >= 0.7 else \
                 "⚠ Consider augmenting the smaller categories"
        print(f"  Balance ratio (min/max): {ratio:.2f}  —  {status}")

    # ── Sample grid ───────────────────────────────────────────────────
    sep(f"Sample Image Grid  →  sample_grid.png")
    n_rows = len(CATEGORIES)
    n_cols = GRID_SAMPLES
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 3.0))
    fig.patch.set_facecolor("#0d0d0d")

    for row, cat in enumerate(CATEGORIES):
        samples = sample_imgs.get(cat, [])
        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_facecolor("#0d0d0d")
            ax.axis("off")
            if col < len(samples):
                arr = load_rgb(samples[col])
                if arr is not None:
                    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        color="#333", fontsize=14, transform=ax.transAxes)
            if col == 0:
                ax.set_ylabel(
                    LABELS[cat], color="white", fontsize=7,
                    rotation=0, labelpad=110, va="center",
                    fontfamily="monospace"
                )

    plt.suptitle("Cyclone Dataset — Sample Images by Intensity Category",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    grid_path = base / "sample_grid.png"
    plt.savefig(grid_path, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved: {grid_path}")

    # ── Corrupt summary ───────────────────────────────────────────────
    total_corrupt = sum(corrupt_hits.values())
    if total_corrupt > 0:
        sep(f"⚠ Corruption Detected (in {CORRUPTION_SAMPLE_SIZE}-image samples)")
        for cat, n in corrupt_hits.items():
            if n > 0:
                print(f"  {cat}: {n} corrupt in sample — consider re-running fetch")
    else:
        sep("Corruption Check")
        print(f"  ✓  No corrupt images found in spot-check "
              f"({CORRUPTION_SAMPLE_SIZE} sampled per category)")

    # ── Loading snippets ──────────────────────────────────────────────
    sep("Ready-to-Use Loading Snippets")
    p = str(base.resolve()).replace("\\", "/")

    print("  ── PyTorch ───────────────────────────────────────────────")
    print(f"""
  from torchvision import datasets, transforms
  from torch.utils.data import DataLoader, random_split

  transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=3),
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ])

  full_ds    = datasets.ImageFolder(r"{p}", transform=transform)
  train_size = int(0.8 * len(full_ds))
  val_size   = len(full_ds) - train_size
  train_ds, val_ds = random_split(full_ds, [train_size, val_size])

  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
  val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
  # full_ds.class_to_idx → maps folder name to class index
""")

    print("  ── Keras / TensorFlow ────────────────────────────────────")
    print(f"""
  from tensorflow.keras.utils import image_dataset_from_directory

  train_ds = image_dataset_from_directory(
      r"{p}",
      validation_split=0.2,
      subset="training",
      seed=42,
      image_size=(224, 224),
      batch_size=32,
      color_mode="grayscale",
  )
  val_ds = image_dataset_from_directory(
      r"{p}",
      validation_split=0.2,
      subset="validation",
      seed=42,
      image_size=(224, 224),
      batch_size=32,
      color_mode="grayscale",
  )
""")

    sep("Done ✓")

# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify cyclone image dataset.")
    parser.add_argument("--dir", type=str, default="cyclone_dataset",
                        help="Dataset root directory (default: ./cyclone_dataset)")
    args = parser.parse_args()
    verify(args.dir)