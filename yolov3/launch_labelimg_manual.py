"""Launch LabelImg for manual YOLO annotation/fixes.

This helper starts LabelImg with your dataset folder, classes file, and
save directory so you can quickly fix auto-generated labels.

Usage example:
    python yolov3/launch_labelimg_manual.py \
        --images-dir Yolo_Bug_Data/bites \
        --classes-file Yolo_Bug_Data/classes.txt \
        --save-dir Yolo_Bug_Data/bites
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch LabelImg for manual YOLO annotation updates."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing images to annotate.",
    )
    parser.add_argument(
        "--classes-file",
        default="",
        help="Path to class names file (one class per line).",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Directory where YOLO .txt labels should be saved.",
    )
    parser.add_argument(
        "--default-class-name",
        default="bite",
        help="Class name to write when auto-creating a classes file.",
    )
    return parser.parse_args()


def ensure_classes_file(path: str, default_class_name: str) -> str:
    if path:
        classes_path = os.path.abspath(path)
        if not os.path.exists(classes_path):
            os.makedirs(os.path.dirname(classes_path), exist_ok=True)
            with open(classes_path, "w", encoding="utf-8") as f:
                f.write(default_class_name.strip() + "\n")
        return classes_path

    classes_path = os.path.abspath("Yolo_Bug_Data/classes.txt")
    os.makedirs(os.path.dirname(classes_path), exist_ok=True)
    if not os.path.exists(classes_path):
        with open(classes_path, "w", encoding="utf-8") as f:
            f.write(default_class_name.strip() + "\n")
    return classes_path


def main() -> None:
    args = parse_args()

    images_dir = os.path.abspath(args.images_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else images_dir
    classes_file = ensure_classes_file(args.classes_file, args.default_class_name)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    os.makedirs(save_dir, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "labelImg",
        images_dir,
        classes_file,
        save_dir,
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "LabelImg is not installed. Install with: python -m pip install labelImg"
        ) from exc


if __name__ == "__main__":
    main()
