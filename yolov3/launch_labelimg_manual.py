"""Launch labelme for manual annotation and convert output to YOLO .txt format.

Usage example:
    python yolov3/launch_labelimg_manual.py \
        --images-dir Yolo_Bug_Data/suspicious_review \
        --classes-file Yolo_Bug_Data/suspicious_review/classes.txt \
        --save-dir Yolo_Bug_Data/suspicious_review
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch labelme and convert to YOLO .txt.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--classes-file", default="")
    parser.add_argument("--save-dir", default="")
    return parser.parse_args()


def read_labels(classes_file: str) -> list[str]:
    with open(classes_file, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def convert_json_to_yolo(json_path: str, class_names: list[str]) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines: list[str] = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape["label"]
        if label not in class_names:
            continue
        class_id = class_names.index(label)

        (x1, y1), (x2, y2) = shape["points"]
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    txt_path = os.path.splitext(json_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def convert_all(save_dir: str, class_names: list[str]) -> int:
    count = 0
    for fname in os.listdir(save_dir):
        if not fname.lower().endswith(".json"):
            continue
        convert_json_to_yolo(os.path.join(save_dir, fname), class_names)
        count += 1
    return count


def main() -> None:
    args = parse_args()

    images_dir = os.path.abspath(args.images_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else images_dir

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    os.makedirs(save_dir, exist_ok=True)

    class_names: list[str] = ["bite"]
    if args.classes_file:
        classes_file = os.path.abspath(args.classes_file)
        if os.path.exists(classes_file):
            class_names = read_labels(classes_file)

    labelme_exe = shutil.which("labelme")
    if labelme_exe is None:
        raise RuntimeError("labelme is not installed. Install with: pip install labelme")

    command = [labelme_exe, images_dir, "--output", save_dir, "--autosave"]
    for label in class_names:
        command += ["--labels", label]

    subprocess.run(command, check=True)

    count = convert_all(save_dir, class_names)
    print(f"Converted {count} JSON annotation(s) to YOLO .txt format.")


if __name__ == "__main__":
    main()
