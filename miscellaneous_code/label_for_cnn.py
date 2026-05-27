#!/usr/bin/env python3
"""
Strategic labeling tool for CNN training data.

Keeps feeding images until --target labels per class are collected.
Biases refill batches toward whichever class is behind.

Saves copies (never moves) to:
  cnn_labels/positive/  — y key  (organized, good training example)
  cnn_labels/negative/  — n key  (disorganized, bad training example)

Usage:
  python miscellaneous_code/label_for_cnn.py
  python miscellaneous_code/label_for_cnn.py --target 350
  python miscellaneous_code/label_for_cnn.py --resume
"""

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
REFILL_BATCH = 200   # images to add each time queue runs dry


def load_csv_scores(csv_path: Path) -> dict:
    scores = {}
    if not csv_path.exists():
        return scores
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("decision") == "keep" and row.get("cyclone_score"):
                try:
                    scores[row["path"]] = float(row["cyclone_score"])
                except ValueError:
                    pass
    return scores


def already_labeled(p: Path, out_dir: Path) -> bool:
    for bucket in ("positive", "negative"):
        dest = out_dir / bucket / p.parent.name / p.name
        if dest.exists():
            return True
    return False


def _bucket_images(keep_dir: Path, scores: dict, out_dir: Path):
    """Return (low, mid, high, unscored) lists of unlabeled images."""
    def find_score(p: Path):
        key = str(p)
        if key in scores:
            return scores[key]
        tail = str(Path(p.parent.name) / p.name)
        for k, v in scores.items():
            if k.endswith(tail) or k.replace("\\", "/").endswith(tail):
                return v
        return None

    low, mid, high, unscored = [], [], [], []
    for p in keep_dir.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            continue
        if already_labeled(p, out_dir):
            continue
        s = find_score(p)
        if s is None:
            unscored.append(p)
        elif s < 0.55:
            low.append(p)
        elif s < 0.65:
            mid.append(p)
        else:
            high.append(p)
    return low, mid, high, unscored


def sample_batch(keep_dir: Path, scores: dict, out_dir: Path,
                 n_low: int, n_mid: int, n_high: int) -> list:
    """Draw a fresh batch of unlabeled images, skipping already-labeled ones."""
    if not scores:
        unlabeled = [
            p for p in keep_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            and not already_labeled(p, out_dir)
        ]
        batch = random.sample(unlabeled, min(n_low + n_mid + n_high, len(unlabeled)))
        random.shuffle(batch)
        return batch

    low, mid, high, unscored = _bucket_images(keep_dir, scores, out_dir)
    batch = (
        random.sample(low,      min(n_low,  len(low)))
        + random.sample(mid,    min(n_mid,  len(mid)))
        + random.sample(high,   min(n_high, len(high)))
        + random.sample(unscored, min(10,   len(unscored)))
    )
    random.shuffle(batch)
    print(f"  Refill: {min(n_low,len(low))} low  "
          f"{min(n_mid,len(mid))} mid  "
          f"{min(n_high,len(high))} high  "
          f"→ {len(batch)} new images queued")
    return batch


def load_session(session_path: Path):
    if session_path.exists():
        return json.loads(session_path.read_text())
    return {"decisions": {}}


def save_session(session_path: Path, decisions: dict):
    session_path.write_text(json.dumps({"decisions": decisions}, indent=2))


def run_labeler(keep_dir: Path, scores: dict, out_dir: Path,
                session_path: Path, target: int,
                n_low: int, n_mid: int, n_high: int):
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError:
        sys.exit("tkinter or Pillow not available.")

    pos_dir = out_dir / "positive"
    neg_dir = out_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    session   = load_session(session_path)
    decisions = session.get("decisions", {})

    def pos_count():
        return sum(1 for v in decisions.values() if v == "positive")

    def neg_count():
        return sum(1 for v in decisions.values() if v == "negative")

    def targets_met():
        return pos_count() >= target and neg_count() >= target

    # Initial queue — bias toward class behind target
    queue = sample_batch(keep_dir, scores, out_dir, n_low, n_mid, n_high)
    if not queue:
        print("No unlabeled images available.")
        return

    idx       = [0]
    photo_ref = [None]

    root = tk.Tk()
    root.title("CNN Label Tool")
    root.configure(bg="#111111")

    status_var = tk.StringVar()
    tk.Label(root, textvariable=status_var, bg="#111111", fg="white",
             font=("Helvetica", 11)).pack(side=tk.TOP, pady=(8, 0))

    score_var = tk.StringVar()
    tk.Label(root, textvariable=score_var, bg="#111111", fg="#aaaaaa",
             font=("Helvetica", 9)).pack(side=tk.TOP)

    img_label = tk.Label(root, bg="#111111")
    img_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=8)

    tk.Label(
        root,
        text="y = Organized (positive)   n = Disorganized (negative)   s = Skip   <- -> = Navigate   q = Quit",
        bg="#111111", fg="#666666", font=("Helvetica", 9),
    ).pack(side=tk.BOTTOM, pady=(0, 8))

    progress_var = tk.StringVar()
    tk.Label(root, textvariable=progress_var, bg="#111111", fg="#55cc55",
             font=("Helvetica", 10)).pack(side=tk.BOTTOM)

    def refresh_ui():
        p   = neg_count()
        pos = pos_count()
        progress_var.set(
            f"positive: {pos}/{target}  negative: {p}/{target}"
            + ("  ✓ DONE" if targets_met() else "")
        )

    def load_current():
        if idx[0] >= len(queue):
            return
        p = queue[idx[0]]
        img = Image.open(p)
        img.thumbnail((860, 680), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        photo_ref[0] = photo
        img_label.config(image=photo)
        dec = decisions.get(str(p), "-")
        status_var.set(f"[{idx[0]+1}/{len(queue)}]  {p.parent.name}/{p.name}  -> {dec}")
        score_var.set(f"category: {p.parent.name}")
        refresh_ui()

    def refill_queue():
        """Add more images biased toward whichever class is behind."""
        pos = pos_count()
        neg = neg_count()
        need_pos = max(0, target - pos)
        need_neg = max(0, target - neg)
        total_need = need_pos + need_neg
        if total_need == 0:
            return False

        # Scale bucket sizes toward deficit class
        # More high-score → more positive candidates; more low-score → more negative
        ratio_high = need_pos / total_need
        ratio_low  = need_neg / total_need
        batch_size = min(REFILL_BATCH, total_need * 2)
        n_h = max(1, int(batch_size * ratio_high))
        n_l = max(1, int(batch_size * ratio_low))
        n_m = max(1, batch_size // 4)

        new_imgs = sample_batch(keep_dir, scores, out_dir, n_l, n_m, n_h)
        if not new_imgs:
            return False
        queue.extend(new_imgs)
        return True

    def apply_decision(decision):
        if idx[0] >= len(queue):
            return
        p = queue[idx[0]]
        decisions[str(p)] = decision

        rel      = p.relative_to(keep_dir)
        dest_dir = pos_dir if decision == "positive" else neg_dir
        dest     = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

        save_session(session_path, decisions)

        if targets_met():
            finish()
            return

        next_idx = idx[0] + 1
        if next_idx >= len(queue):
            if not refill_queue():
                finish()
                return
        idx[0] = next_idx
        load_current()

    def skip_image():
        next_idx = idx[0] + 1
        if next_idx >= len(queue):
            refill_queue()
        if next_idx < len(queue):
            idx[0] = next_idx
            load_current()

    def nav(delta):
        new = idx[0] + delta
        if 0 <= new < len(queue):
            idx[0] = new
            load_current()

    def finish():
        save_session(session_path, decisions)
        root.destroy()
        pos = pos_count()
        neg = neg_count()
        print(f"\nLabeling done: {pos} positive, {neg} negative  (target: {target} each)")
        if pos < target:
            print(f"  Need {target - pos} more positive — run again to continue")
        if neg < target:
            print(f"  Need {target - neg} more negative — run again to continue")
        if pos >= target and neg >= target:
            print("  Both targets met. Ready to train.")
        print(f"Saved to: {out_dir.resolve()}")

    root.bind("y", lambda e: apply_decision("positive"))
    root.bind("n", lambda e: apply_decision("negative"))
    root.bind("s", lambda e: skip_image())
    root.bind("<Right>", lambda e: nav(1))
    root.bind("<Left>",  lambda e: nav(-1))
    root.bind("q", lambda e: finish())

    load_current()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep",   default="cyclone_data_clean/KEEP")
    ap.add_argument("--csv",    default="cyclone_data_clean/filter_report.csv")
    ap.add_argument("--out",    default="cnn_labels")
    ap.add_argument("--target", type=int, default=350,
                    help="Labels to collect per class before stopping (default: 350)")
    ap.add_argument("--low",    type=int, default=80,
                    help="Initial low-score samples (0.45-0.55)")
    ap.add_argument("--mid",    type=int, default=120,
                    help="Initial mid-score samples (0.55-0.65)")
    ap.add_argument("--high",   type=int, default=300,
                    help="Initial high-score samples (>0.65)")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from saved session (keeps existing labels)")
    args = ap.parse_args()

    keep_dir     = Path(args.keep)
    csv_path     = Path(args.csv)
    out_dir      = Path(args.out)
    session_path = out_dir / ".session.json"

    if not keep_dir.exists():
        sys.exit(f"KEEP dir not found: {keep_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Count already-labeled from disk (survives session file deletion)
    existing_pos = sum(1 for _ in (out_dir / "positive").rglob("*")
                       if _.is_file() and _.suffix.lower() in IMAGE_EXTS) \
                   if (out_dir / "positive").exists() else 0
    existing_neg = sum(1 for _ in (out_dir / "negative").rglob("*")
                       if _.is_file() and _.suffix.lower() in IMAGE_EXTS) \
                   if (out_dir / "negative").exists() else 0
    print(f"Existing labels: {existing_pos} positive, {existing_neg} negative  "
          f"(target: {args.target} each)")

    if existing_pos >= args.target and existing_neg >= args.target:
        print("Both targets already met. Ready to train.")
        sys.exit(0)

    print("Loading CSV scores...")
    scores = load_csv_scores(csv_path)
    print(f"  {len(scores)} scored KEEP images in CSV")

    run_labeler(keep_dir, scores, out_dir, session_path,
                args.target, args.low, args.mid, args.high)


if __name__ == "__main__":
    main()
