"""Manual YOLO annotation editor using OpenCV.

Features:
- Draw multiple boxes per image (left-click and drag)
- Save labels in YOLO format next to each image
- Load existing labels for correction

Controls:
- Left mouse drag: draw new box
- d: delete last box
- c: clear all boxes
- s: save current image labels
- n: save and next image
- p: previous image
- q: quit
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Sequence, Tuple

import cv2


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual YOLO annotation editor.")
    parser.add_argument("--images-dir", required=True, help="Folder containing images.")
    parser.add_argument("--class-id", type=int, default=0, help="Class id for drawn boxes.")
    return parser.parse_args()


def list_images(images_dir: str) -> List[str]:
    images: List[str] = []
    for pattern in IMAGE_PATTERNS:
        images.extend(glob.glob(os.path.join(images_dir, pattern)))
    images.sort()
    return images


def to_yolo(box: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def from_yolo(xc: float, yc: float, bw: float, bh: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x1 = int((xc - bw / 2.0) * img_w)
    y1 = int((yc - bh / 2.0) * img_h)
    x2 = int((xc + bw / 2.0) * img_w)
    y2 = int((yc + bh / 2.0) * img_h)
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def read_labels(txt_path: str, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    if not os.path.exists(txt_path):
        return []
    boxes: List[Tuple[int, int, int, int]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                continue
            boxes.append(from_yolo(xc, yc, bw, bh, img_w, img_h))
    return boxes


def write_labels(
    txt_path: str,
    boxes: Sequence[Tuple[int, int, int, int]],
    class_id: int,
    img_w: int,
    img_h: int,
) -> None:
    lines = []
    for box in boxes:
        xc, yc, bw, bh = to_yolo(box, img_w, img_h)
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(txt_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def draw_overlay(
    image,
    boxes: Sequence[Tuple[int, int, int, int]],
    current_box: Tuple[int, int, int, int] | None,
    index: int,
    total: int,
):
    canvas = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            str(i),
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if current_box is not None:
        x1, y1, x2, y2 = current_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)

    help_text = "drag=box | d=undo | c=clear | s=save | n=next | p=prev | q=quit"
    cv2.putText(canvas, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    progress = f"image {index + 1}/{total} | boxes: {len(boxes)}"
    cv2.putText(canvas, progress, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, progress, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def normalize_box(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    images_dir = os.path.abspath(args.images_dir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = list_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No supported image files found in: {images_dir}")

    win_name = "manual-yolo-annotator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    index = 0
    while 0 <= index < len(image_paths):
        image_path = image_paths[index]
        txt_path = os.path.splitext(image_path)[0] + ".txt"

        image = cv2.imread(image_path)
        if image is None:
            index += 1
            continue

        img_h, img_w = image.shape[:2]
        boxes = read_labels(txt_path, img_w, img_h)

        drawing = {"active": False, "x": 0, "y": 0}
        current_box = None

        def on_mouse(event, x, y, _flags, _param):
            nonlocal current_box
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing["active"] = True
                drawing["x"] = x
                drawing["y"] = y
                current_box = (x, y, x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
                current_box = (drawing["x"], drawing["y"], x, y)
            elif event == cv2.EVENT_LBUTTONUP and drawing["active"]:
                drawing["active"] = False
                x1, y1, x2, y2 = normalize_box(drawing["x"], drawing["y"], x, y, img_w, img_h)
                if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                    boxes.append((x1, y1, x2, y2))
                current_box = None

        cv2.setMouseCallback(win_name, on_mouse)

        while True:
            frame = draw_overlay(image, boxes, current_box, index, len(image_paths))
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("d"):
                if boxes:
                    boxes.pop()
            elif key == ord("c"):
                boxes.clear()
            elif key == ord("s"):
                write_labels(txt_path, boxes, args.class_id, img_w, img_h)
            elif key == ord("n"):
                write_labels(txt_path, boxes, args.class_id, img_w, img_h)
                index += 1
                break
            elif key == ord("p"):
                write_labels(txt_path, boxes, args.class_id, img_w, img_h)
                index = max(0, index - 1)
                break
            elif key == ord("q"):
                write_labels(txt_path, boxes, args.class_id, img_w, img_h)
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
