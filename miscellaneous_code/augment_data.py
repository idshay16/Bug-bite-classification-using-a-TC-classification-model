#!/usr/bin/env python3
"""
Data augmentation utility for bug bite classification dataset.

Usage:
    python augment_data.py                          # interactive menu
    python augment_data.py --data-dir /path/to/data # specify data directory
    python augment_data.py --option 1               # run option directly
"""

import os
import sys
import random
import argparse
import math
import shutil
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _longpath(p: Path) -> str:
    """Prepend the Windows extended-length prefix to bypass MAX_PATH (260 chars)."""
    if sys.platform != "win32":
        return str(p)
    resolved = str(p.resolve())
    return resolved if resolved.startswith("\\\\") else "\\\\?\\" + resolved


def find_splits(data_dir: Path) -> dict[str, Path]:
    """
    Return a dict of {split_name: Path} for any train/val/test sub-folders.
    If none are found, treat data_dir itself as one split named 'all'.
    """
    splits = {}
    for name in ("train", "val", "test"):
        p = data_dir / name
        if p.is_dir():
            splits[name] = p
    if not splits:
        splits["all"] = data_dir
    return splits


def collect_images(split_dir: Path) -> dict[str, list[Path]]:
    """Return {class_name: [image_paths]} for every class folder inside split_dir."""
    classes = {}
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if imgs:
            classes[cls_dir.name] = imgs
    return classes


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation transforms
# ──────────────────────────────────────────────────────────────────────────────

def _random_flip(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.3:
        img = ImageOps.flip(img)
    return img


def _random_rotate(img: Image.Image, max_angle: float = 30.0) -> Image.Image:
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)


def _random_brightness(img: Image.Image, factor_range=(0.6, 1.4)) -> Image.Image:
    factor = random.uniform(*factor_range)
    return ImageEnhance.Brightness(img).enhance(factor)


def _random_contrast(img: Image.Image, factor_range=(0.7, 1.3)) -> Image.Image:
    factor = random.uniform(*factor_range)
    return ImageEnhance.Contrast(img).enhance(factor)


def _random_saturation(img: Image.Image, factor_range=(0.7, 1.3)) -> Image.Image:
    factor = random.uniform(*factor_range)
    return ImageEnhance.Color(img).enhance(factor)


def _random_crop_zoom(img: Image.Image, zoom_range=(0.80, 1.00)) -> Image.Image:
    """Crop a random sub-region and resize back to original size."""
    w, h = img.size
    scale = random.uniform(*zoom_range)
    new_w, new_h = int(w * scale), int(h * scale)
    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)
    return img.crop((x0, y0, x0 + new_w, y0 + new_h)).resize((w, h), Image.BILINEAR)


TRANSFORM_NAMES = [
    "flip",
    "rotate",
    "brightness",
    "contrast",
    "saturation",
    "crop_zoom",
]


def augment_image(img: Image.Image) -> tuple[Image.Image, list[str]]:
    """
    Apply a random combination of transforms and return (augmented_img, applied_names).
    At least 2 transforms are always applied.
    """
    transforms = {
        "flip":       _random_flip,
        "rotate":     _random_rotate,
        "brightness": _random_brightness,
        "contrast":   _random_contrast,
        "saturation": _random_saturation,
        "crop_zoom":  _random_crop_zoom,
    }

    # Always apply flip + rotate; pick 1-3 more at random
    mandatory = ["flip", "rotate"]
    optional  = [t for t in TRANSFORM_NAMES if t not in mandatory]
    selected  = mandatory + random.sample(optional, k=random.randint(1, 3))

    applied = []
    for name in selected:
        img = transforms[name](img)
        applied.append(name)
    return img, applied


# ──────────────────────────────────────────────────────────────────────────────
# Option 1 – Distribution
# ──────────────────────────────────────────────────────────────────────────────

def show_distribution(data_dir: Path) -> None:
    splits = find_splits(data_dir)
    all_data: dict[str, dict[str, int]] = {}

    for split_name, split_path in splits.items():
        classes = collect_images(split_path)
        all_data[split_name] = {cls: len(imgs) for cls, imgs in classes.items()}

    if not all_data:
        print("No image data found in", data_dir)
        return

    # Console summary
    print("\n-- Data Distribution ---------------------------------------")
    for split_name, counts in all_data.items():
        total = sum(counts.values())
        print(f"\n  [{split_name}]  total: {total}")
        max_count = max(counts.values()) if counts else 1
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "#" * int(30 * n / max_count)
            print(f"    {cls:<20s} {n:>4d}  {bar}")

    # Bar chart
    n_splits = len(all_data)
    fig, axes = plt.subplots(1, n_splits, figsize=(7 * n_splits, 5), squeeze=False)

    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    for ax, (split_name, counts) in zip(axes[0], all_data.items()):
        classes = list(counts.keys())
        values  = list(counts.values())
        bars = ax.bar(classes, values, color=colors[: len(classes)])
        ax.set_title(f"Distribution – {split_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("Image count")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val),
                ha="center", va="bottom", fontsize=9,
            )

    plt.suptitle("Bug Bite Dataset – Class Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Option 2 – Augmentation preview
# ──────────────────────────────────────────────────────────────────────────────

def show_augmentation_samples(data_dir: Path, n_augmented: int = 4) -> None:
    splits = find_splits(data_dir)
    split_name = "train" if "train" in splits else list(splits.keys())[0]
    classes = collect_images(splits[split_name])

    if not classes:
        print("No classes found in", splits[split_name])
        return

    n_classes = len(classes)
    cols = 1 + n_augmented          # original + augmented versions
    fig, axes = plt.subplots(n_classes, cols, figsize=(cols * 2.5, n_classes * 2.8))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    print(f"\n-- Augmentation Preview  (split: {split_name}) -------------")
    for row, (cls_name, img_paths) in enumerate(sorted(classes.items())):
        src_path = random.choice(img_paths)
        original = Image.open(src_path).convert("RGB")

        axes[row, 0].imshow(original)
        axes[row, 0].set_title("original", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(cls_name, fontsize=10, rotation=0, labelpad=55, va="center")

        for col in range(1, cols):
            aug_img, applied = augment_image(original)
            axes[row, col].imshow(aug_img)
            axes[row, col].set_title("\n".join(applied), fontsize=6)
            axes[row, col].axis("off")

        print(f"  {cls_name:<20s}  source: {src_path.name}")

    fig.suptitle(
        f"Augmentation Preview – {n_augmented} variants per class",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Option 3 – Balance dataset
# ──────────────────────────────────────────────────────────────────────────────

def balance_dataset(
    data_dir: Path,
    output_dir: Path | None = None,
    split: str = "train",
    target_count: int | None = None,
    dry_run: bool = False,
) -> None:
    splits = find_splits(data_dir)
    if split not in splits:
        available = list(splits.keys())
        print(f"Split '{split}' not found. Available: {available}")
        split = available[0]
        print(f"Using '{split}' instead.")

    split_path = splits[split]
    classes    = collect_images(split_path)

    if not classes:
        print("No classes found.")
        return

    counts      = {cls: len(imgs) for cls, imgs in classes.items()}
    max_count   = target_count or max(counts.values())
    total_new   = sum(max(0, max_count - n) for n in counts.values())

    in_place = output_dir is None

    print("\n-- Balancing Plan ------------------------------------------")
    print(f"  Target count per class : {max_count}")
    print(f"  Total new images       : {total_new}")
    if in_place:
        print(f"  Output                 : in-place (same class folders)")
    else:
        print(f"  Output directory       : {output_dir}")
    print()

    for cls_name, imgs in sorted(classes.items()):
        need = max(0, max_count - len(imgs))
        print(f"  {cls_name:<20s}  have {len(imgs):>4d}  →  need {need:>4d} new")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    if total_new == 0:
        print("\nDataset is already balanced.")
        return

    confirm = input("\nProceed with augmentation? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    if not in_place:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0

    for cls_name, src_imgs in sorted(classes.items()):
        need = max(0, max_count - len(src_imgs))

        if in_place:
            cls_dest = split_path / cls_name
        else:
            cls_dest = output_dir / cls_name
            cls_dest.mkdir(parents=True, exist_ok=True)
            for src in src_imgs:
                dst = cls_dest / src.name
                if not dst.exists():
                    shutil.copy2(_longpath(src), _longpath(dst))

        if need == 0:
            print(f"  {cls_name:<20s}  [ok] no augmentation needed")
            continue

        # Only augment originals, never re-augment already-augmented files
        pool = [p for p in src_imgs if not p.name.startswith("augmented_")]
        if not pool:
            print(f"  {cls_name:<20s}  [skip] no original images found")
            continue

        # Start counter after any existing augmented files to avoid name collisions on re-runs
        aug_counter = len(src_imgs) - len(pool)

        written = 0
        src_idx = 0
        while written < need:
            src_path = pool[src_idx % len(pool)]
            src_idx += 1
            out_name = f"augmented_{aug_counter:04d}_{src_path.name}"
            aug_counter += 1
            img = Image.open(src_path).convert("RGB")
            aug, _ = augment_image(img)
            aug.save(_longpath(cls_dest / out_name))
            written += 1

        total_written += written
        print(f"  {cls_name:<20s}  [ok] wrote {written} augmented images")

    print(f"\nDone. Total new images written: {total_written}")
    if in_place:
        print(f"Output: in-place within {split_path}")
    else:
        print(f"Output: {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Option 4 – Remove augmented data
# ──────────────────────────────────────────────────────────────────────────────

def remove_augmented_data(
    data_dir: Path,
    output_dir: Path | None = None,
    split: str = "train",
    dry_run: bool = False,
) -> None:
    if output_dir is not None:
        # Dedicated output dir was used — offer to delete the whole directory
        if not output_dir.exists():
            print(f"\nNothing to remove — directory not found: {output_dir}")
            return
        aug_files = [p for p in output_dir.rglob("*") if p.is_file()]
        print("\n-- Cleanup Plan --------------------------------------------")
        print(f"  Directory : {output_dir}")
        print(f"  Files     : {len(aug_files)}")
        if dry_run:
            print("\n[dry-run] No files removed.")
            return
        confirm = input(f"\nDelete entire directory '{output_dir}'? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        shutil.rmtree(output_dir)
        print(f"\nDone. Removed directory: {output_dir}")
    else:
        # In-place mode — find augmented_* files within the split's class folders
        splits = find_splits(data_dir)
        if split not in splits:
            available = list(splits.keys())
            split = available[0]
        split_path = splits[split]
        aug_files = [p for p in split_path.rglob("augmented_*") if p.is_file()]
        print("\n-- Cleanup Plan --------------------------------------------")
        print(f"  Split directory: {split_path}")
        print(f"  Augmented files: {len(aug_files)}")
        if dry_run:
            print("\n[dry-run] No files removed.")
            return
        if not aug_files:
            print("\nNo augmented files found.")
            return
        confirm = input(f"\nRemove {len(aug_files)} augmented files? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        for f in aug_files:
            f.unlink()
        print(f"\nDone. Removed {len(aug_files)} augmented files.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

MENU = """
+--------------------------------------------------+
|   Bug Bite Dataset - Augmentation Utility        |
+--------------------------------------------------+
|  1)  Show class distribution                     |
|  2)  Preview augmented samples                   |
|  3)  Augment data to balance the dataset         |
|  4)  Remove augmented data                       |
|  q)  Quit                                        |
+--------------------------------------------------+
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bug bite dataset augmentation utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", "-d",
        default=None,
        help="Path to dataset root (should contain train/ and/or val/ folders).",
    )
    parser.add_argument(
        "--option", "-o",
        choices=["1", "2", "3", "4"],
        default=None,
        help="Run a specific option directly (1, 2, or 3).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Which split to use for balancing (default: train).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target image count per class (default: max existing class size).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write balanced data (default: <data-dir>/<split>_augmented/).",
    )
    parser.add_argument(
        "--n-augmented",
        type=int,
        default=4,
        help="Number of augmented variants to show per class in preview (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show balancing plan without writing any files.",
    )
    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
    else:
        # Try common relative locations
        script_dir = Path(__file__).resolve().parent
        repo_root  = script_dir.parent
        candidates = [
            repo_root / "Bug-Data" / "Multiclass_Bug_data",
            repo_root / "Bug-Data" / "Multiclass_Bug_Data",
            Path("Bug-Data/Multiclass_Bug_data"),
            Path("../Bug-Data/Multiclass_Bug_data"),
            Path.cwd(),
        ]
        data_dir = next((p for p in candidates if p.is_dir()), Path.cwd())
        print(f"No --data-dir given. Using: {data_dir}")
        print("  (pass --data-dir /your/path to override)\n")

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    def run_option(choice: str) -> bool:
        match choice:
            case "1":
                show_distribution(data_dir)
            case "2":
                show_augmentation_samples(data_dir, n_augmented=args.n_augmented)
            case "3":
                balance_dataset(
                    data_dir,
                    output_dir=output_dir,
                    split=args.split,
                    target_count=args.target_count,
                    dry_run=args.dry_run,
                )
            case "4":
                remove_augmented_data(
                    data_dir,
                    output_dir=output_dir,
                    split=args.split,
                    dry_run=args.dry_run,
                )
            case _:
                return False
        return True

    # Direct option via CLI flag
    if args.option:
        run_option(args.option)
        return

    # Interactive menu
    while True:
        print(MENU)
        choice = input("Select option: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if not run_option(choice):
            print("Invalid choice. Enter 1, 2, 3, or q.")


if __name__ == "__main__":
    main()
