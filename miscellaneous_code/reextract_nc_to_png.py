#!/usr/bin/env python3
"""
Re-extract PNG frames from existing .nc files in cyclone_dataset/_tmp_nc.

Usage:
  python reextract_nc_to_png.py                   # all files
  python reextract_nc_to_png.py --sample 50       # quick test on 50 files
  python reextract_nc_to_png.py --frames 2        # 2 frames per file
"""
import argparse
import importlib.util
import json
import random
import re
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

ROOT = Path("cyclone_dataset").resolve()
TMP = ROOT / "_tmp_nc"

# Load fetch module by path to reuse extraction logic
spec = importlib.util.spec_from_file_location(
    "fetch_mod",
    str(Path(__file__).resolve().parent / "fetch_cyclone_dataset.py"),
)
fetch_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_mod)

BT_MIN = fetch_mod.BT_MIN
BT_MAX = fetch_mod.BT_MAX

SID_PATTERN = re.compile(r"([A-Z0-9]+)(?:_[0-9]+)?\.nc", re.IGNORECASE)


def build_sid_to_cat():
    df = fetch_mod.load_ibtracs(ROOT)
    # Use peak intensity (max wind_kt) per storm — same logic as fetch_cyclone_dataset.py
    peak = (
        df.sort_values("wind_kt", ascending=False)
          .drop_duplicates("SID")
    )
    sid_to_cat = {}
    for _, r in peak.iterrows():
        cat = r.get("category")
        if cat is not None:
            sid_to_cat[str(r["SID"])] = cat
    return sid_to_cat


def extract_one(nc_path: Path, out_dir: Path, key: str, frames_per_file: int):
    """
    Extract up to `frames_per_file` PNG frames from a single .nc file.
    Output filename includes the nc file's own stem for uniqueness across passes:
      {SID}_{nc_suffix}_f{frame_idx:04d}.png
    Skips a frame only if the output file already exists on disk.
    Returns (saved, skipped, error_msg).
    """
    import netCDF4 as nc_lib
    import numpy as np
    from PIL import Image

    # Use the full nc stem (e.g. "1995193N17305_100") for unique filenames
    nc_stem = nc_path.stem   # "1995193N17305_100"

    saved = skipped = 0
    try:
        ds = nc_lib.Dataset(str(nc_path), "r")
        bt_var = fetch_mod.find_bt_var(ds)
        if bt_var is None:
            ds.close()
            return 0, 0, f"no image variable in {nc_path.name}"

        data = np.array(ds.variables[bt_var][:])
        ds.close()

        if data.ndim == 2:
            data = data[np.newaxis]

        n_frames = data.shape[0]
        indices = np.linspace(0, n_frames - 1, min(frames_per_file, n_frames), dtype=int)

        for idx in indices:
            out_path = out_dir / f"{nc_stem}_f{idx:04d}.png"

            if out_path.exists():
                skipped += 1
                continue

            frame = data[idx]
            if hasattr(frame, "filled"):
                frame = frame.filled(np.nan)

            finite_count = int(np.isfinite(frame).sum())
            if finite_count < frame.size * 0.3:
                continue

            clipped = np.clip(frame, BT_MIN, BT_MAX)
            normed = 1.0 - (clipped - BT_MIN) / (BT_MAX - BT_MIN)
            img = Image.fromarray((normed * 255).astype(np.uint8))
            img = img.resize((fetch_mod.IMAGE_SIZE, fetch_mod.IMAGE_SIZE), Image.Resampling.LANCZOS)
            img.save(str(out_path))
            saved += 1

    except Exception:
        return saved, skipped, traceback.format_exc()

    return saved, skipped, None


def main(sample, frames_per_file, dry_run):
    if not TMP.exists():
        print("_tmp_nc folder not found — nothing to do.")
        return 1

    print("Building SID -> category map from IBTrACS...")
    sid_to_cat = build_sid_to_cat()
    print(f"  {len(sid_to_cat):,} SIDs mapped.")

    all_nc = sorted(TMP.rglob("*.nc"))
    if not all_nc:
        print("No .nc files found in _tmp_nc.")
        return 1

    files = random.sample(all_nc, min(sample, len(all_nc))) if sample else all_nc
    print(f"Files to process: {len(files):,}  |  frames_per_file: {frames_per_file}  |  dry_run: {dry_run}")

    saved_total = skipped_total = error_total = 0
    error_log = []

    for p in tqdm(files, desc="re-extract", unit="file"):
        m = SID_PATTERN.search(p.name)
        key = m.group(1) if m else p.stem
        cat = sid_to_cat.get(key, "unknown")
        out_dir = ROOT / cat

        if dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        saved, skipped, err = extract_one(p, out_dir, key, frames_per_file)
        saved_total += saved
        skipped_total += skipped

        if err:
            error_total += 1
            error_log.append({"file": str(p), "error": err})

    print(f"\nSaved  : {saved_total:,}")
    print(f"Skipped: {skipped_total:,}  (already existed on disk)")
    print(f"Errors : {error_total:,}")

    if error_log:
        log_path = ROOT / "reextract_errors.json"
        log_path.write_text(json.dumps(error_log, indent=2))
        print(f"Error details → {log_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Process N random files only (default: all)")
    parser.add_argument("--frames", type=int, default=1, dest="frames",
                        help="Frames per .nc file (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing files")
    args = parser.parse_args()

    sys.exit(main(args.sample, args.frames, args.dry_run))
