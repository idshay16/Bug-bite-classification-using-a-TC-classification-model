#!/usr/bin/env python3
"""
verify_dataset.py — Inspect dataset quality after filtering.

Works with any directory that has category subdirectories (cyclone_dataset,
cyclone_data_clean/KEEP, cnn_labels/positive, etc.).

Usage:
  python miscellaneous_code/verify_dataset.py
  python miscellaneous_code/verify_dataset.py --dir cyclone_data_clean/KEEP
  python miscellaneous_code/verify_dataset.py --dir cyclone_data_clean/KEEP --samples 10
  python miscellaneous_code/verify_dataset.py --dir cyclone_data_clean/KEEP --csv cyclone_data_clean/filter_report.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
CORRUPTION_SAMPLE = 50


def sep(title=""):
    print()
    print("─" * 64)
    if title:
        print(f"  {title}")
        print("─" * 64)


def check_image(path):
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        return True, img.width, img.height
    except Exception:
        return False, 0, 0


def load_gray(path):
    try:
        return np.array(Image.open(path).convert("L"))
    except Exception:
        return None


def collect_categories(base: Path):
    """Return sorted list of (cat_name, [image_paths]) for all non-empty subdirs."""
    cats = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        imgs = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            cats.append((d.name, sorted(imgs)))
    return cats


def load_filter_csv(csv_path: Path):
    """Return {path_str: row_dict} from filter_report.csv."""
    data = {}
    if not csv_path or not csv_path.exists():
        return data
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["path"]] = row
    return data


def print_filter_stats(cats, filter_data):
    if not filter_data:
        return
    sep("Filter Statistics (from CSV)")
    total_keep = total_review = 0
    flag_counts = {}
    for cat, imgs in cats:
        keep = review = 0
        for p in imgs:
            row = filter_data.get(str(p))
            if row is None:
                continue
            if row.get("decision") == "keep":
                keep += 1
                for flag in ("profile_range_ok", "streak_ok", "bright_fill_ok", "edge_ok"):
                    if row.get(flag) == "0":
                        flag_counts[flag] = flag_counts.get(flag, 0) + 1
            else:
                review += 1
        total_keep += keep
        total_review += review
        if keep + review > 0:
            pct = 100 * keep / (keep + review)
            print(f"  {cat:<40}  keep={keep:>5}  review={review:>5}  ({pct:.0f}% pass)")

    print(f"\n  Total keep: {total_keep}  review: {total_review}")

    if flag_counts:
        print("\n  Hard-reject breakdown (images failing each check):")
        for flag, n in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"    {flag:<20}  {n:>5} rejected")


def build_grid(cats, n_samples: int, out_path: Path, title: str):
    n_rows = len(cats)
    n_cols = n_samples

    fig_w = min(n_cols * 2.5, 40)
    fig_h = min(n_rows * 2.8, 60)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              squeeze=False)
    fig.patch.set_facecolor("#0d0d0d")

    for row, (cat, imgs) in enumerate(cats):
        sample = random.sample(imgs, min(n_samples, len(imgs)))
        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_facecolor("#0d0d0d")
            ax.axis("off")
            if col < len(sample):
                arr = load_gray(sample[col])
                if arr is not None:
                    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
                    ax.set_title(sample[col].stem[:18], color="#888", fontsize=5)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        color="#444", fontsize=12, transform=ax.transAxes)
        axes[row][0].set_ylabel(
            cat, color="white", fontsize=7,
            rotation=0, labelpad=120, va="center",
            fontfamily="monospace",
        )

    plt.suptitle(title, color="white", fontsize=10, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Saved: {out_path.resolve()}")


def verify(dataset_dir: str, csv_path: Path | None, n_samples: int, seed: int):
    random.seed(seed)
    base = Path(dataset_dir)
    if not base.exists():
        print(f"[ERROR] Not found: {base.resolve()}")
        sys.exit(1)

    sep("Dataset Verification Report")
    print(f"  Path: {base.resolve()}")

    cats = collect_categories(base)
    if not cats:
        print("  No category subdirectories with images found.")
        sys.exit(1)

    filter_data = load_filter_csv(csv_path)

    # ── Per-category counts ───────────────────────────────────────────
    sep("Image Counts")
    total = 0
    corrupt_total = 0
    for cat, imgs in cats:
        n = len(imgs)
        total += n
        sample = random.sample(imgs, min(CORRUPTION_SAMPLE, n))
        corrupt = sum(1 for p in sample if not check_image(p)[0])
        corrupt_total += corrupt
        corrupt_str = f"  !! {corrupt} corrupt in sample" if corrupt else ""
        print(f"  {cat:<40}  {n:>6,} images{corrupt_str}")

    print(f"\n  Total: {total:,} images in {len(cats)} categories")

    # ── Balance ───────────────────────────────────────────────────────
    sep("Balance")
    max_n = max(len(imgs) for _, imgs in cats)
    bar_w = 30
    for cat, imgs in cats:
        n   = len(imgs)
        bar = "█" * int((n / max_n) * bar_w)
        print(f"  {bar:<{bar_w}}  {n:>6,}  {cat}")
    vals = [len(imgs) for _, imgs in cats]
    ratio = min(vals) / max(vals) if max(vals) else 0
    status = ("OK" if ratio >= 0.9 else
              "acceptable" if ratio >= 0.7 else
              "consider augmenting smaller categories")
    print(f"\n  Balance ratio (min/max): {ratio:.2f}  —  {status}")

    # ── Filter stats ──────────────────────────────────────────────────
    print_filter_stats(cats, filter_data)

    # ── Corruption summary ────────────────────────────────────────────
    sep("Corruption Check")
    if corrupt_total:
        print(f"  !! {corrupt_total} corrupt images found in spot-checks — re-run extraction")
    else:
        print(f"  OK  No corrupt images in spot-check ({CORRUPTION_SAMPLE} sampled per category)")

    # ── Sample grid ───────────────────────────────────────────────────
    sep(f"Sample Grid  ({n_samples} per category)")
    grid_path = base / "sample_grid.png"
    build_grid(cats, n_samples, grid_path,
               f"Dataset Quality Check — {base.name}  (seed={seed})")

    sep("Done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",     default="cyclone_data_clean/KEEP",
                    help="Directory to verify (default: cyclone_data_clean/KEEP)")
    ap.add_argument("--csv",     default=None,
                    help="filter_report.csv path for filter stats (optional)")
    ap.add_argument("--samples", type=int, default=8,
                    help="Images per category in sample grid (default: 8)")
    ap.add_argument("--seed",    type=int, default=42,
                    help="Random seed for reproducible sampling")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    if csv_path is None:
        auto = Path(args.dir).parent / "filter_report.csv"
        if auto.exists():
            csv_path = auto
            print(f"Auto-detected CSV: {auto}")

    verify(args.dir, csv_path, args.samples, args.seed)
