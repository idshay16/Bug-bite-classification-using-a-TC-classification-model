#!/usr/bin/env python3
"""
Bug bite morphology filter: scores circularity and spot patterns.
"""

import argparse
import csv
import glob
import math
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CIRCULARITY_THRESHOLDS = {
    "strict": 0.70,
    "medium": 0.55,
    "loose": 0.40,
}

SCORE_THRESHOLDS = {
    "strict": 0.70,
    "medium": 0.60,
    "loose": 0.50,
}


def otsu_threshold(gray):
    vals = np.clip((gray * 255).astype(np.uint8), 0, 255)
    hist = np.bincount(vals.ravel(), minlength=256).astype(np.float64)
    total = vals.size
    if total == 0:
        return 0.5

    sum_all = (np.arange(256) * hist).sum()
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 128

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold / 255.0


def glob_base(pattern):
    parts = Path(pattern).parts
    acc = []
    for part in parts:
        if part == "**" or any(ch in part for ch in "*?["):
            break
        acc.append(part)
    if not acc:
        return Path(".").resolve()
    return Path(*acc).resolve()


def iter_images(input_specs):
    specs = input_specs or []
    bases = [glob_base(s) for s in specs]
    seen = set()

    for spec in specs:
        matches = glob.glob(spec, recursive=True)
        for m in matches:
            p = Path(m)
            if p.is_dir():
                for root, _, files in os.walk(p):
                    for name in files:
                        f = Path(root) / name
                        if f.suffix.lower() in IMAGE_EXTS:
                            if f not in seen:
                                seen.add(f)
                                yield f, bases
            elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                if p not in seen:
                    seen.add(p)
                    yield p, bases


def find_components(mask):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            perim = 0
            sum_y = 0.0
            sum_x = 0.0

            while stack:
                cy, cx = stack.pop()
                area += 1
                sum_y += cy
                sum_x += cx

                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        perim += 1
                        continue
                    if not mask[ny, nx]:
                        perim += 1
                        continue
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            centroid = (sum_y / area, sum_x / area)
            comps.append({"area": area, "perim": perim, "centroid": centroid})

    return comps


def analyze_image(path, allow_single_spot, min_area_ratio, max_area_ratio, circ_level):
    img = Image.open(path).convert("L")
    gray = np.asarray(img, dtype=np.float32) / 255.0
    if gray.size == 0:
        return None

    thresh = otsu_threshold(gray)
    below = gray <= thresh
    above = ~below

    if below.sum() == 0 or above.sum() == 0:
        return None

    mean_below = gray[below].mean()
    mean_above = gray[above].mean()
    if mean_below < mean_above:
        mask = below
    else:
        mask = above

    h, w = mask.shape
    area_min = max(1, int(min_area_ratio * h * w))
    area_max = max(area_min, int(max_area_ratio * h * w))

    comps = find_components(mask)
    if not comps:
        return None

    filtered = [c for c in comps if area_min <= c["area"] <= area_max]
    if not filtered:
        filtered = comps

    filtered.sort(key=lambda c: c["area"], reverse=True)
    largest = filtered[0]

    area = largest["area"]
    perim = largest["perim"]
    circularity = 0.0
    if perim > 0:
        circularity = (4.0 * math.pi * area) / (perim ** 2)
    circularity = max(0.0, min(1.0, circularity))

    cy, cx = largest["centroid"]
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0
    dist = math.hypot(cy - center_y, cx - center_x)
    diag = math.hypot(center_y, center_x)
    center_dist = dist / diag if diag > 0 else 0.0
    center_score = max(0.0, 1.0 - center_dist)

    spot_count = len(filtered)
    if allow_single_spot:
        spot_score = min(1.0, spot_count / 2.0)
    else:
        spot_score = 1.0 if spot_count >= 2 else 0.0

    score = 0.5 * circularity + 0.3 * center_score + 0.2 * spot_score
    score = max(0.0, min(1.0, score))

    keep = (circularity >= CIRCULARITY_THRESHOLDS[circ_level]
            and score >= SCORE_THRESHOLDS[circ_level]
            and (allow_single_spot or spot_count >= 2))

    return {
        "circularity": circularity,
        "score": score,
        "spot_count": spot_count,
        "center_dist": center_dist,
        "keep": keep,
    }


def pick_base(path, bases):
    path = path.resolve()
    candidates = []
    for base in bases:
        base = base.resolve()
        if base in [path] + list(path.parents):
            candidates.append(base)
    if candidates:
        return max(candidates, key=lambda p: len(str(p)))
    return path.parent


def run_filter(input_specs, out, mode="copy", circularity="loose",
               allow_single_spot=True, require_multiple_spots=False,
               min_area_ratio=0.001, max_area_ratio=0.30, write_csv=True,
               dry_run=False, max_files=0):
    allow_single = allow_single_spot and not require_multiple_spots

    out_base = Path(out)
    keep_dir = out_base / "KEEP"
    review_dir = out_base / "REVIEW"
    if not dry_run:
        keep_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

    bases = [glob_base(s) for s in input_specs]

    csv_path = out_base / "filter_report.csv"
    csv_file = None
    writer = None
    if write_csv and not dry_run:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=[
            "path", "decision", "score", "circularity",
            "spot_count", "center_dist"
        ])
        writer.writeheader()

    processed = 0
    kept = 0
    reviewed = 0
    for path, _ in iter_images(input_specs):
        if max_files and processed >= max_files:
            break

        if out_base.resolve() in [path.resolve()] + list(path.resolve().parents):
            continue

        metrics = analyze_image(path, allow_single,
                    min_area_ratio, max_area_ratio,
                    circularity)
        if metrics is None:
            decision = "review"
        else:
            decision = "keep" if metrics["keep"] else "review"

        base = pick_base(path, bases)
        rel = path.resolve().relative_to(base)
        dest_dir = keep_dir if decision == "keep" else review_dir
        dest = dest_dir / rel

        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if mode == "copy":
                shutil.copy2(path, dest)
            else:
                shutil.move(path, dest)

        if writer is not None:
            writer.writerow({
                "path": str(path),
                "decision": decision,
                "score": "" if metrics is None else f"{metrics['score']:.4f}",
                "circularity": "" if metrics is None else f"{metrics['circularity']:.4f}",
                "spot_count": "" if metrics is None else metrics["spot_count"],
                "center_dist": "" if metrics is None else f"{metrics['center_dist']:.4f}",
            })

        processed += 1
        if decision == "keep":
            kept += 1
        else:
            reviewed += 1

    if csv_file is not None:
        csv_file.close()

    print(f"Processed: {processed}")
    print(f"Keep: {kept}")
    print(f"Review: {reviewed}")
    print(f"Output: {out_base.resolve()}")

    return {
        "processed": processed,
        "keep": kept,
        "review": reviewed,
        "output": str(out_base.resolve()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True,
                   help="Input files/folders/globs to scan")
    p.add_argument("--out", type=str, default="bugbite_filter_output",
                   help="Output base folder")
    p.add_argument("--mode", choices=["copy", "move"], default="copy")
    p.add_argument("--circularity", choices=["strict", "medium", "loose"],
                   default="loose")
    p.add_argument("--allow-single-spot", action="store_true", default=True)
    p.add_argument("--require-multiple-spots", action="store_true")
    p.add_argument("--min-area-ratio", type=float, default=0.001)
    p.add_argument("--max-area-ratio", type=float, default=0.30)
    p.add_argument("--write-csv", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-files", type=int, default=0,
                   help="Stop after processing N files (0=all)")
    args = p.parse_args()

    run_filter(
        input_specs=args.input,
        out=args.out,
        mode=args.mode,
        circularity=args.circularity,
        allow_single_spot=args.allow_single_spot,
        require_multiple_spots=args.require_multiple_spots,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        write_csv=args.write_csv,
        dry_run=args.dry_run,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
