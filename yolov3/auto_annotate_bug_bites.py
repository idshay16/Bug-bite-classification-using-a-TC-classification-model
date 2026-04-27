"""Auto-generate YOLO bounding boxes for bug-bite images.

This tool uses a classical computer-vision pipeline (redness + texture) to
propose bite regions and writes YOLO-format .txt files next to each image.
It can write multiple bounding boxes per image when multiple candidates exist.

Usage example:
    python yolov3/auto_annotate_bug_bites.py \
        --images-dir yolov3/custom_data \
        --class-id 0 \
        --preview-dir yolov3/preview_labels
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


@dataclass
class Config:
    class_id: int
    min_area_ratio: float
    max_area_ratio: float
    max_boxes: int
    overwrite: bool
    fallback_center_box: bool
    fallback_size_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically generate YOLO labels for bug-bite images."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class index to write for detected bite regions.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.002,
        help="Ignore components smaller than this ratio of image area.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.75,
        help="Ignore components larger than this ratio of image area.",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=5,
        help="Maximum number of boxes to keep per image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt labels if they already exist.",
    )
    parser.add_argument(
        "--preview-dir",
        default="",
        help="Optional output directory for annotated preview images.",
    )
    parser.add_argument(
        "--fallback-center-box",
        action="store_true",
        help=(
            "If nothing is detected, write one centered fallback box instead "
            "of leaving the label file empty."
        ),
    )
    parser.add_argument(
        "--fallback-size-ratio",
        type=float,
        default=0.4,
        help="Relative width/height of fallback centered box (0 to 1).",
    )
    return parser.parse_args()


def list_images(images_dir: str) -> List[str]:
    image_paths: List[str] = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(images_dir, pattern)))
    image_paths.sort()
    return image_paths


def get_candidate_mask(img_bgr: np.ndarray) -> np.ndarray:
    # Lab a-channel highlights red/magenta areas often present around irritation.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    a_blur = cv2.GaussianBlur(a_channel, (0, 0), 5)
    red_boost = cv2.subtract(a_channel, a_blur)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Merge redness and saturation while down-weighting very dark regions.
    redness_map = cv2.addWeighted(red_boost, 1.3, sat, 0.7, 0)
    redness_map = cv2.normalize(redness_map, None, 0, 255, cv2.NORM_MINMAX)
    darkness_mask = (val < 25).astype(np.uint8) * 255

    _, mask = cv2.threshold(
        redness_map.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(darkness_mask))

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_boxes(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
    max_boxes: int,
) -> List[Tuple[int, int, int, int]]:
    h, w = img_bgr.shape[:2]
    image_area = float(w * h)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = float(bw * bh)
        area_ratio = area / image_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        aspect = bw / max(1.0, float(bh))
        if aspect < 0.25 or aspect > 4.0:
            continue

        roi = mask[y : y + bh, x : x + bw]
        nonzero = float(cv2.countNonZero(roi))
        fill_ratio = nonzero / max(1.0, area)
        if fill_ratio < 0.1:
            continue

        score = area * fill_ratio
        candidates.append((score, x, y, bw, bh))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [(x, y, bw, bh) for _, x, y, bw, bh in candidates[:max_boxes]]
    return selected


def centered_fallback_box(img_shape: Sequence[int], size_ratio: float) -> Tuple[int, int, int, int]:
    h, w = img_shape[:2]
    size_ratio = float(np.clip(size_ratio, 0.05, 0.95))
    bw = int(w * size_ratio)
    bh = int(h * size_ratio)
    x = max(0, (w - bw) // 2)
    y = max(0, (h - bh) // 2)
    return (x, y, bw, bh)


def to_yolo_line(class_id: int, box: Tuple[int, int, int, int], w: int, h: int) -> str:
    x, y, bw, bh = box
    x_center = (x + bw / 2.0) / w
    y_center = (y + bh / 2.0) / h
    width = bw / w
    height = bh / h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def save_preview(
    preview_dir: str,
    image_path: str,
    boxes: Sequence[Tuple[int, int, int, int]],
) -> None:
    os.makedirs(preview_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        return

    for x, y, bw, bh in boxes:
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    filename = os.path.basename(image_path)
    output_path = os.path.join(preview_dir, filename)
    cv2.imwrite(output_path, image)


def process_image(image_path: str, cfg: Config, preview_dir: str = "") -> str:
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    if os.path.exists(txt_path) and not cfg.overwrite:
        return "skipped_existing"

    image = cv2.imread(image_path)
    if image is None:
        return "read_error"

    h, w = image.shape[:2]
    mask = get_candidate_mask(image)
    boxes = find_boxes(
        image,
        mask,
        min_area_ratio=cfg.min_area_ratio,
        max_area_ratio=cfg.max_area_ratio,
        max_boxes=cfg.max_boxes,
    )

    if not boxes and cfg.fallback_center_box:
        boxes = [centered_fallback_box(image.shape, cfg.fallback_size_ratio)]

    lines = [to_yolo_line(cfg.class_id, box, w, h) for box in boxes]
    with open(txt_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")

    if preview_dir:
        save_preview(preview_dir, image_path, boxes)

    return "labeled" if boxes else "empty"


def main() -> None:
    args = parse_args()
    cfg = Config(
        class_id=args.class_id,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        max_boxes=max(1, args.max_boxes),
        overwrite=args.overwrite,
        fallback_center_box=args.fallback_center_box,
        fallback_size_ratio=args.fallback_size_ratio,
    )

    images_dir = os.path.abspath(args.images_dir)
    preview_dir = os.path.abspath(args.preview_dir) if args.preview_dir else ""

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = list_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No supported image files found in: {images_dir}")

    stats = {
        "labeled": 0,
        "empty": 0,
        "skipped_existing": 0,
        "read_error": 0,
    }

    for image_path in image_paths:
        status = process_image(image_path, cfg, preview_dir)
        stats[status] += 1

    print("Auto-annotation finished.")
    print(f"Images processed: {len(image_paths)}")
    print(f"Labeled: {stats['labeled']}")
    print(f"Empty labels: {stats['empty']}")
    print(f"Skipped existing labels: {stats['skipped_existing']}")
    print(f"Read errors: {stats['read_error']}")
    if preview_dir:
        print(f"Preview images saved to: {preview_dir}")


if __name__ == "__main__":
    main()