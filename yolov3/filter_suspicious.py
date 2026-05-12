"""Copy images with suspicious YOLO annotations into a review folder.

Heuristics:
  - Box area > 85% of image (too large)
  - Box area < 0.2% of image (too small / noise)
  - Extreme aspect ratio (max/min side > 8)
  - Box goes out of image bounds
  - Empty or malformed annotation file

Usage:
    python yolov3/filter_suspicious.py \
        --images-dir Yolo_Bug_Data/images \
        --out-dir Yolo_Bug_Data/suspicious_review
"""

from __future__ import annotations

import argparse
import os
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
AREA_MAX = 0.85
AREA_MIN = 0.002
ASPECT_MAX = 8.0


def check_annotation(txt_path: str) -> list[str]:
    reasons: list[str] = []
    try:
        with open(txt_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception:
        return ["unreadable file"]

    if not lines:
        return ["empty annotation"]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            reasons.append(f"malformed line: {line!r}")
            continue
        try:
            _, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            reasons.append(f"non-numeric values: {line!r}")
            continue

        if cx - w / 2 < -0.01 or cx + w / 2 > 1.01:
            reasons.append(f"out-of-bounds X: cx={cx:.3f} w={w:.3f}")
        if cy - h / 2 < -0.01 or cy + h / 2 > 1.01:
            reasons.append(f"out-of-bounds Y: cy={cy:.3f} h={h:.3f}")

        area = w * h
        if area > AREA_MAX:
            reasons.append(f"box too large: area={area:.1%}")
        if area < AREA_MIN:
            reasons.append(f"box too small: area={area:.3%}")

        min_side = min(w, h)
        if min_side > 0 and max(w, h) / min_side > ASPECT_MAX:
            reasons.append(f"extreme aspect ratio: {w:.3f}x{h:.3f}")

    return reasons


def find_image(images_dir: str, stem: str) -> str | None:
    for ext in IMAGE_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-dir", default="Yolo_Bug_Data/suspicious_review")
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Copy classes.txt into review folder so LabelImg can read it
    classes_src = os.path.join(images_dir, "classes.txt")
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        if os.path.exists(classes_src):
            f.write(open(classes_src, encoding="utf-8").read().strip() + "\n")
        else:
            f.write("bite\n")

    flagged: list[tuple[str, list[str]]] = []

    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(".txt") or fname == "classes.txt":
            continue
        stem = os.path.splitext(fname)[0]
        txt_path = os.path.join(images_dir, fname)
        reasons = check_annotation(txt_path)
        if not reasons:
            continue
        img_path = find_image(images_dir, stem)
        if img_path is None:
            continue
        shutil.copy2(img_path, out_dir)
        shutil.copy2(txt_path, out_dir)
        flagged.append((fname, reasons))

    print(f"\n{len(flagged)} suspicious annotation(s) copied to: {out_dir}\n")
    for fname, reasons in flagged:
        print(f"  {fname}")
        for r in reasons:
            print(f"    - {r}")


if __name__ == "__main__":
    main()
