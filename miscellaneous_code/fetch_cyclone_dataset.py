#!/usr/bin/env python3
"""
HURSAT-B1 + IBTrACS Cyclone Dataset Fetcher
Braude College of Engineering — Capstone Phase B

Usage:
  python fetch_cyclone_dataset.py --trial      # 5 images per category, quick test
  python fetch_cyclone_dataset.py              # 4,000 per category
  python fetch_cyclone_dataset.py --target 2000
  python fetch_cyclone_dataset.py --workers 8
"""

import re, sys, json, math, time, argparse, tarfile, warnings, random, threading, shutil
import requests, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from contextlib import nullcontext

warnings.filterwarnings("ignore")

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("netCDF4 not installed — run: pip install netCDF4")
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed — run: pip install Pillow")

# ── CONFIG ────────────────────────────────────────────────────────────

OUTPUT_DIR   = "cyclone_dataset"
YEAR_START   = 1995
YEAR_END     = 2016
TARGET       = 4000
TRIAL_TARGET = 5
WORKERS      = 4
MAX_PARALLEL_DOWNLOADS = 6
RETRY_TOTAL = 5
RETRY_BACKOFF = 1.0
RATE_JITTER_SEC = 0.2
INDEX_WORKERS = 6
EXTRACT_WORKERS = 1

CATEGORIES = {
    "cat1_tropical_depression": (0,    33),
    "cat2_tropical_storm":      (34,   63),
    "cat3_hurricane_cat12":     (64,   95),
    "cat4_hurricane_cat34":     (96,  136),
    "cat5_hurricane_cat5":      (137, 9999),
}

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r00/access/csv/ibtracs.ALL.list.v04r00.csv"
)
HURSAT_BASE = (
    "https://www.ncei.noaa.gov/data/"
    "hurricane-satellite-hursat-b1/archive/v06/"
)

IMAGE_SIZE = 224
BT_MIN, BT_MAX = 180, 300   # Kelvin clipping range

_session_local = threading.local()
DOWNLOAD_SEMAPHORE = None

# ── HELPERS ───────────────────────────────────────────────────────────

def log(msg, warn=False):
    print(f"  {'!' if warn else '+'}  {msg}")

def get_session():
    session = getattr(_session_local, "session", None)
    if session is None:
        retry = Retry(
            total=RETRY_TOTAL,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=MAX_PARALLEL_DOWNLOADS,
            pool_maxsize=MAX_PARALLEL_DOWNLOADS,
        )
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _session_local.session = session
    return session

def make_dirs():
    base = Path(OUTPUT_DIR)
    base.mkdir(exist_ok=True)
    for cat in CATEGORIES:
        (base / cat).mkdir(exist_ok=True)
    (base / "_tmp_nc").mkdir(exist_ok=True)
    return base

def load_resume_log(base):
    p = base / "download_log.json"
    if p.exists():
        return set(json.loads(p.read_text()).get("done", []))
    return set()

def save_resume_log(base, done_set):
    (base / "download_log.json").write_text(
        json.dumps({"done": list(done_set)}, indent=2)
    )

def cleanup_category(base, done_set, category):
    cat_dir = base / category
    if not cat_dir.exists():
        return done_set
    cat_prefix = str(cat_dir.resolve())
    done_set = {p for p in done_set if not str(p).startswith(cat_prefix)}
    shutil.rmtree(cat_dir)
    cat_dir.mkdir(exist_ok=True)
    return done_set

def categorise(wind_kt):
    for name, (lo, hi) in CATEGORIES.items():
        if lo <= float(wind_kt) <= hi:
            return name
    return None

# ── STEP 1: IBTrACS ───────────────────────────────────────────────────

def load_ibtracs(base):
    csv = base / "ibtracs_raw.csv"
    if not csv.exists():
        print("  Downloading IBTrACS CSV (~50 MB)...")
        r = requests.get(IBTRACS_URL, timeout=120, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(csv, "wb") as f, tqdm(total=total, unit="B",
                unit_scale=True, desc="  IBTrACS") as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk); bar.update(len(chunk))
    else:
        log("IBTrACS CSV cached.")

    # keep_default_na=False is CRITICAL — pandas treats "NA" as NaN by default,
    # which silently drops all North Atlantic (BASIN="NA") storms.
    df = pd.read_csv(csv, skiprows=[1], low_memory=False,
                     na_values=["", " "], keep_default_na=False)

    df.columns = [c.strip().upper() for c in df.columns]
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    preserve = {"BASIN", "USA_ATCF_ID", "SID", "NAME"}
    for col in df.columns:
        if col not in preserve and df[col].dtype == object:
            df[col] = df[col].replace("NA", np.nan)

    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df = df[(df["SEASON"] >= YEAR_START) & (df["SEASON"] <= YEAR_END)]

    wind_col = "USA_WIND" if "USA_WIND" in df.columns else "WIND"
    df["wind_kt"] = pd.to_numeric(df[wind_col], errors="coerce")
    df = df.dropna(subset=["wind_kt", "SID"])
    df["category"] = df["wind_kt"].apply(categorise)
    df = df.dropna(subset=["category"])

    log(f"IBTrACS: {df['SID'].nunique():,} unique storms loaded.")
    return df

# ── STEP 2: Build storm list — match SID to HURSAT URL ───────────────

def fetch_year_index(base, year):
    """Fetch HURSAT directory listing for one year. Returns {SID: full_url}. Cached."""
    cache = base / f"_idx_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    url = f"{HURSAT_BASE}{int(year)}/"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        log(f"Index fetch failed for {year}: {e}", warn=True)
        return {}

    # Filenames: HURSAT_b1_v06_{SID}_{NAME}_c{DATE}.tar.gz
    pat = re.compile(r'(HURSAT_b1_v06_([A-Z0-9]+)_[A-Z0-9]+_c\d+\.tar\.gz)')
    sid_map = {}
    for m in pat.finditer(r.text):
        fname, sid = m.group(1), m.group(2)
        sid_map[sid] = f"{HURSAT_BASE}{int(year)}/{fname}"

    cache.write_text(json.dumps(sid_map, indent=2))
    return sid_map

def build_storm_list(df, base, index_workers):
    storms = (
        df.sort_values("wind_kt", ascending=False)
          .drop_duplicates("SID")
          .reset_index(drop=True)
    )

    years = storms["SEASON"].dropna().unique().astype(int)
    print(f"  Fetching HURSAT directory indexes for {len(years)} years...")
    year_indexes = {}
    with ThreadPoolExecutor(max_workers=index_workers) as pool:
        futs = {pool.submit(fetch_year_index, base, year): year
                for year in sorted(years)}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="  Year indexes", unit="yr"):
            year = futs[fut]
            try:
                year_indexes[year] = fut.result()
            except Exception as e:
                log(f"Index fetch failed for {year}: {e}", warn=True)
                year_indexes[year] = {}

    def lookup_url(row):
        year = int(row["SEASON"]) if pd.notna(row["SEASON"]) else None
        if year is None:
            return None
        return year_indexes.get(year, {}).get(row["SID"])

    storms["hursat_url"] = storms.apply(lookup_url, axis=1)
    storms["hursat_key"] = storms["SID"]

    matched = storms["hursat_url"].notna().sum()
    log(f"{matched}/{len(storms)} storms matched to HURSAT files")
    if matched > 0:
        ex = storms[storms["hursat_url"].notna()].iloc[0]
        log(f"Example URL: {ex['hursat_url']}")

    storms = storms.dropna(subset=["hursat_url"])

    print()
    for cat in CATEGORIES:
        n = (storms["category"] == cat).sum()
        log(f"  {cat}: {n} storms")

    return storms

# ── STEP 3: Parallel download ─────────────────────────────────────────

def download_file(key, url, nc_dest):
    """Download HURSAT tar.gz, extract all .nc inside, delete the tar.gz."""
    dest_path = Path(nc_dest)
    tmp_dir = dest_path.parent
    existing = list(tmp_dir.glob(f"{key}*.nc"))
    if existing:
        return [str(p) for p in existing]

    tar_dest = str(dest_path) + ".tar.gz"

    sema = DOWNLOAD_SEMAPHORE

    for attempt in range(3):
        try:
            if sema is None:
                sema_ctx = nullcontext()
            else:
                sema_ctx = sema

            with sema_ctx:
                if RATE_JITTER_SEC > 0:
                    time.sleep(random.uniform(0, RATE_JITTER_SEC))
                session = get_session()
                with session.get(url, timeout=120, stream=True) as r:
                    if r.status_code == 404:
                        return False
                    r.raise_for_status()

                    with open(tar_dest, "wb") as f:
                        for chunk in r.iter_content(65536):
                            f.write(chunk)

            with tarfile.open(tar_dest) as tar:
                nc_members = [m for m in tar.getmembers()
                              if m.name.endswith(".nc")]
                if not nc_members:
                    Path(tar_dest).unlink(missing_ok=True)
                    return []

                out_paths = []
                for idx, member in enumerate(nc_members):
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    suffix = "" if len(nc_members) == 1 else f"_{idx:02d}"
                    out_path = tmp_dir / f"{key}{suffix}.nc"
                    out_path.write_bytes(extracted.read())
                    out_paths.append(str(out_path))

            Path(tar_dest).unlink(missing_ok=True)
            return out_paths

        except Exception:
            Path(tar_dest).unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(3)

    return []

def parallel_download(storm_rows, tmp_dir, n_workers):
    tasks   = [(r["hursat_key"], r["hursat_url"],
                str(tmp_dir / f"{r['hursat_key']}.nc"))
               for _, r in storm_rows.iterrows()]
    results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(download_file, *t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="  Downloading", unit="file"):
            key = futs[fut]
            if fut.result():
                results[key] = str(tmp_dir / f"{key}.nc")
    return results

# ── STEP 4: Frame extraction ──────────────────────────────────────────

def find_bt_var(ds):
    for name in ["TB", "BT", "Tb", "tb", "brightness_temperature",
                 "IRWIN", "IR", "T_IR", "ch4", "data", "temp"]:
        if name in ds.variables:
            return name
    coords = set(ds.dimensions.keys())
    for name, var in ds.variables.items():
        if name in coords:
            continue
        if var.ndim >= 2 and np.issubdtype(var.dtype, np.number):
            return name
    return None

def extract_frames(nc_path, n_wanted, done_set, out_dir, key):
    saved = 0
    try:
        ds     = nc.Dataset(nc_path, "r")
        bt_var = find_bt_var(ds)

        if bt_var is None:
            log(f"No image variable in {Path(nc_path).name}", warn=True)
            ds.close()
            return 0

        data = np.array(ds.variables[bt_var][:])
        ds.close()

        if data.ndim == 2:
            data = data[np.newaxis]

        n_frames = data.shape[0]
        indices  = np.linspace(0, n_frames - 1,
                               min(n_wanted, n_frames), dtype=int)

        for idx in indices:
            out_path = str(out_dir / f"{key}_f{idx:04d}.png")
            if out_path in done_set:
                saved += 1
                continue

            frame = data[idx]
            if isinstance(frame, np.ma.MaskedArray):
                frame = frame.filled(np.nan)
            if np.isfinite(frame).sum() < frame.size * 0.3:
                continue

            clipped = np.clip(frame, BT_MIN, BT_MAX)
            normed  = 1.0 - (clipped - BT_MIN) / (BT_MAX - BT_MIN)
            img     = Image.fromarray((normed * 255).astype(np.uint8))
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            img.save(out_path)
            done_set.add(out_path)
            saved += 1

    except Exception as e:
        log(f"Extraction error ({Path(nc_path).name}): {e}", warn=True)
    return saved

# ── MAIN ──────────────────────────────────────────────────────────────

def run(trial, target, workers, cleanup_cat1, run_filter, filter_out, filter_mode, serial_extract, extract_workers):
    global DOWNLOAD_SEMAPHORE
    target = TRIAL_TARGET if trial else target

    print()
    print("=" * 60)
    print("  HURSAT-B1 + IBTrACS Cyclone Dataset Fetcher")
    print(f"  Mode:    {'TRIAL' if trial else 'FULL'}")
    print(f"  Target:  {target:,} × {len(CATEGORIES)} categories = {target*len(CATEGORIES):,} images")
    print(f"  Years:   {YEAR_START}–{YEAR_END}  |  Workers: {workers}")
    print("=" * 60)
    print()

    base    = make_dirs()
    done    = load_resume_log(base)
    tmp_dir = base / "_tmp_nc"
    DOWNLOAD_SEMAPHORE = threading.BoundedSemaphore(
        max(1, min(workers, MAX_PARALLEL_DOWNLOADS))
    )

    if cleanup_cat1:
        done = cleanup_category(base, done, "cat1_tropical_depression")
        save_resume_log(base, done)

    counts = {cat: len(list((base / cat).glob("*.png")))
              for cat in CATEGORIES}
    for cat, n in counts.items():
        status = "DONE" if n >= target else f"{n}/{target}"
        log(f"{cat}: {status}")
    print()

    df     = load_ibtracs(base)
    print()
    storms = build_storm_list(df, base, INDEX_WORKERS)
    print()

    for cat in CATEGORIES:
        if counts[cat] >= target:
            log(f"{cat}: already complete, skipping.")
            continue

        cat_storms = storms[storms["category"] == cat].reset_index(drop=True)
        if len(cat_storms) == 0:
            log(f"{cat}: no storms found!", warn=True)
            continue

        still_needed = target - counts[cat]
        print(f"\n  ── {cat}")
        log(f"  {len(cat_storms)} storms available | need {still_needed} images")

        out_dir = base / cat

        storms_needed = max(math.ceil(still_needed / 60) * 2, workers * 2)
        storms_needed = min(storms_needed, len(cat_storms))
        to_submit     = cat_storms.head(storms_needed)

        log(f"  Submitting {len(to_submit)} downloads")

        if serial_extract:
            pbar = tqdm(total=len(to_submit), desc=f"  {cat[:20]}", unit="storm")
            storms_done = 0
            for _, row in to_submit.iterrows():
                key  = row["hursat_key"]
                url  = row["hursat_url"]
                dest = str(tmp_dir / f"{key}.nc")

                nc_paths = download_file(key, url, dest)
                if not nc_paths:
                    storms_done += 1
                    pbar.update(1)
                    continue

                still_needed = target - counts[cat]
                if still_needed <= 0:
                    break

                storms_remaining = len(to_submit) - storms_done
                frames_this_storm = max(1, math.ceil(
                    still_needed / max(storms_remaining, 1)
                ))

                remaining = frames_this_storm
                remaining_files = len(nc_paths)
                for nc_path in nc_paths:
                    if remaining <= 0:
                        break
                    per_file = max(1, math.ceil(remaining / max(remaining_files, 1)))
                    got = extract_frames(nc_path, per_file, done, out_dir, key)
                    counts[cat] += got
                    remaining -= per_file
                    remaining_files -= 1

                storms_done += 1
                pbar.update(1)
                pbar.set_postfix({"saved": counts[cat], "target": target})

                if counts[cat] >= target:
                    break
            pbar.close()
        else:
            # Parallelize downloads and extraction: downloads run in download_pool
            # while extraction runs concurrently in extract_pool to maximize throughput.
              with ThreadPoolExecutor(max_workers=workers) as download_pool, \
                  ThreadPoolExecutor(max_workers=extract_workers) as extract_pool:
                download_futs = {}
                for _, row in to_submit.iterrows():
                    key  = row["hursat_key"]
                    url  = row["hursat_url"]
                    dest = str(tmp_dir / f"{key}.nc")
                    download_futs[download_pool.submit(download_file, key, url, dest)] = (key, row)

                extract_futs = {}
                pbar = tqdm(as_completed(download_futs), total=len(download_futs),
                            desc=f"  {cat[:20]}", unit="storm")
                storms_done = 0

                for fut in pbar:
                    key, row = download_futs[fut]
                    try:
                        nc_paths = fut.result()
                    except Exception:
                        nc_paths = []

                    if not nc_paths:
                        storms_done += 1
                        # check any finished extractions and update counts
                        for ef in list(extract_futs):
                            if ef.done():
                                try:
                                    got = ef.result()
                                except Exception:
                                    got = 0
                                counts[cat] += got
                                del extract_futs[ef]
                        continue

                    still_needed = target - counts[cat]
                    if still_needed <= 0:
                        for f in download_futs:
                            f.cancel()
                        break

                    storms_remaining = len(to_submit) - storms_done
                    frames_this_storm = max(1, math.ceil(
                        still_needed / max(storms_remaining, 1)
                    ))

                    # schedule extraction concurrently across all .nc files
                    remaining = frames_this_storm
                    remaining_files = len(nc_paths)
                    for idx, nc_path in enumerate(nc_paths):
                        if remaining <= 0:
                            break
                        per_file = max(1, math.ceil(remaining / max(remaining_files, 1)))
                        ef = extract_pool.submit(extract_frames, nc_path,
                                                 per_file, done, out_dir, key)
                        extract_futs[ef] = key
                        remaining -= per_file
                        remaining_files -= 1

                    # opportunistically collect finished extractions
                    for ef in list(extract_futs):
                        if ef.done():
                            try:
                                got = ef.result()
                            except Exception:
                                got = 0
                            counts[cat] += got
                            del extract_futs[ef]

                    storms_done += 1
                    pbar.set_postfix({"saved": counts[cat], "target": target})

                    if counts[cat] >= target:
                        for f in download_futs:
                            f.cancel()
                        break

                # wait for any remaining extractions to finish
                for ef in as_completed(list(extract_futs)):
                    try:
                        got = ef.result()
                    except Exception:
                        got = 0
                    counts[cat] += got

        save_resume_log(base, done)
        log(f"  {cat}: {counts[cat]:,} images")

    print()
    print("=" * 60)
    total = sum(counts.values())
    for cat in CATEGORIES:
        n   = counts[cat]
        bar = "█" * int(n / max(target, 1) * 24)
        ok  = "✓" if n >= target else "⚠"
        print(f"  {ok}  {bar:<24}  {n:>5,}  {cat}")
    print(f"\n  Total: {total:,}  |  {base.resolve()}")
    print("=" * 60)
    save_resume_log(base, done)

    if run_filter:
        try:
            from bugbite_morphology_filter import run_filter as run_morph_filter
        except Exception:
            log("Filter import failed: bugbite_morphology_filter.py", warn=True)
        else:
            log("Running morphology filter...")
            run_morph_filter(
                input_specs=[str(base / "*")],
                out=filter_out,
                mode=filter_mode,
            )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trial",   action="store_true")
    p.add_argument("--target",  type=int, default=TARGET)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--output",  type=str, default=None)
    p.add_argument("--years",   nargs=2, type=int, default=None,
                   metavar=("START", "END"))
    p.add_argument("--index-workers", type=int, default=INDEX_WORKERS)
    p.add_argument("--cleanup-cat1", action="store_true",
                   help="Delete cat1 output folder and remove its entries from the resume log")
    p.add_argument("--run-filter", action="store_true",
                   help="Run bugbite morphology filter after fetching")
    p.add_argument("--filter-out", type=str, default="bugbite_filter_output",
                   help="Output folder for morphology filter")
    p.add_argument("--filter-mode", choices=["copy", "move"], default="copy",
                   help="Morphology filter file handling")
    p.add_argument("--serial-extract", action="store_true",
                   help="Download then extract sequentially (single-threaded)")
    p.add_argument("--extract-workers", type=int, default=EXTRACT_WORKERS,
                   help="Number of parallel extract workers")
    args = p.parse_args()

    if args.output:
        OUTPUT_DIR = args.output
    if args.years:
        YEAR_START, YEAR_END = args.years

    if args.index_workers is not None:
        INDEX_WORKERS = args.index_workers

    run(trial=args.trial, target=args.target, workers=args.workers,
        cleanup_cat1=args.cleanup_cat1, run_filter=args.run_filter,
        filter_out=args.filter_out, filter_mode=args.filter_mode,
        serial_extract=args.serial_extract, extract_workers=args.extract_workers)
