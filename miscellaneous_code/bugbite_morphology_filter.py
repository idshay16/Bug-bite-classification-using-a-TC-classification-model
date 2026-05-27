#!/usr/bin/env python3
"""
Bug bite morphology filter: scores circularity, spot patterns, blob clusters,
and cyclone-specific radial structure (symmetry, gradient, coherence, core contrast).
"""

import argparse
import csv
import glob
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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

CYCLONE_SCORE_THRESHOLDS = {
    "strict": 0.60,
    "medium": 0.50,
    "loose": 0.45,
}

# Hard-reject: profile range below this = no radial structure (artifacts/flat images)
PROFILE_RANGE_MIN = 0.12

# Hard-reject: row/col mean OR percentile spike above neighbours = scan-line artifact
STREAK_SPIKE_MAX = 0.12
# Hard-reject: bright-pixel-fraction spike (catches streaks on bright cloud backgrounds)
STREAK_BRIGHT_FRAC_SPIKE      = 0.25   # immediate-neighbor spike
STREAK_WIDE_FRAC_SPIKE        = 0.10   # wide-window (11-row) spike — catches streaks at cloud edges
STREAK_WIDE_WINDOW            = 11

# Hard-reject: scan-line smearing — vertical gradient energy >> horizontal = banded rows
GRAD_ANISOTROPY_MAX = 1.55

# Hard-reject: fraction of image pixels near-white above this = missing-data hole
BRIGHT_FILL_MAX        = 0.12
BRIGHT_FILL_THRESHOLD  = 0.88    # pixels above this count as "near-white"

# Hard-reject: fraction of any edge strip (EDGE_STRIP px wide) that is near-white
EDGE_BRIGHT_MAX = 0.20
EDGE_STRIP_PX   = 12


# ── IMAGE LOADING / UTILITIES ─────────────────────────────────────────────────

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
                                yield f
            elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                if p not in seen:
                    seen.add(p)
                    yield p


# ── BLOB / COMPONENT ANALYSIS ────────────────────────────────────────────────

def find_components(mask):
    """Vectorized connected-component analysis using scipy."""
    struct = ndimage.generate_binary_structure(2, 1)
    labeled, n = ndimage.label(mask, structure=struct)
    if n == 0:
        return []

    label_ids = np.arange(1, n + 1)

    m = mask.astype(np.int32)
    padded = np.pad(m, 1, constant_values=0)
    perim_map = m * (
        (padded[:-2, 1:-1] == 0).astype(np.int32)
        + (padded[2:,  1:-1] == 0).astype(np.int32)
        + (padded[1:-1, :-2] == 0).astype(np.int32)
        + (padded[1:-1, 2:]  == 0).astype(np.int32)
    )

    areas   = ndimage.sum(mask, labeled, label_ids)
    perims  = ndimage.sum(perim_map, labeled, label_ids)
    raw_cen = ndimage.center_of_mass(mask, labeled, label_ids)
    if n == 1:
        raw_cen = [raw_cen]

    return [
        {"area": int(areas[i]), "perim": int(perims[i]), "centroid": raw_cen[i]}
        for i in range(n)
    ]


def detect_cluster(comps, h, w, min_blob_circ=0.35, min_count=3, proximity_ratio=0.35):
    """Find largest connected component of circular blobs within proximity."""
    circular = []
    for c in comps:
        a, p = c["area"], c["perim"]
        circ = (4.0 * math.pi * a) / (p ** 2) if p > 0 else 0.0
        if max(0.0, min(1.0, circ)) >= min_blob_circ:
            circular.append(c["centroid"])

    if len(circular) < min_count:
        return False, 0

    max_dist = proximity_ratio * math.hypot(h, w)
    n = len(circular)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(circular[i][0] - circular[j][0],
                          circular[i][1] - circular[j][1]) <= max_dist:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    best = 0
    for start in range(n):
        if visited[start]:
            continue
        cluster, stack = [], [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            cluster.append(node)
            stack.extend(adj[node])
        if len(cluster) > best:
            best = len(cluster)

    return best >= min_count, best


# ── CYCLONE-SPECIFIC METRICS ──────────────────────────────────────────────────

def compute_cyclone_metrics(gray):
    """
    Four vectorized metrics targeting TC radial structure:

    radial_symmetry   — low intra-ring intensity variance (concentric rings = symmetric)
    radial_gradient   — std of the radial intensity profile (ring-to-ring contrast)
    gradient_coherence — how well Sobel gradients align with the radial direction
    core_contrast     — |center_mean − mid_ring_mean| (eye / cold-core signature)
    """
    h, w = gray.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    Y = (np.arange(h, dtype=np.float32) - cy).reshape(-1, 1)
    X = (np.arange(w, dtype=np.float32) - cx).reshape(1, -1)
    R = np.sqrt(Y ** 2 + X ** 2)

    max_r = min(cy, cx) * 0.95
    if max_r < 4:
        return {"radial_symmetry": 0.0, "radial_gradient": 0.0,
                "gradient_coherence": 0.0, "core_contrast": 0.0,
                "cyclone_score": 0.0}

    n_rings = 24
    ring_edges = np.linspace(0, max_r, n_rings + 1)

    # Vectorized ring statistics via digitize + bincount
    ring_idx = np.clip(np.digitize(R.ravel(), ring_edges) - 1, 0, n_rings - 1)
    flat     = gray.ravel().astype(np.float64)

    ring_sum  = np.bincount(ring_idx, weights=flat,        minlength=n_rings)
    ring_sum2 = np.bincount(ring_idx, weights=flat ** 2,   minlength=n_rings)
    ring_cnt  = np.maximum(np.bincount(ring_idx,           minlength=n_rings), 1).astype(np.float64)

    profile   = ring_sum / ring_cnt
    variance  = np.maximum(ring_sum2 / ring_cnt - profile ** 2, 0.0)
    ring_stds = np.sqrt(variance)

    # Radial gradient: blend absolute range + normalised consistency.
    # Absolute range rewards strong structure; normalised ratio alone inflates flat profiles.
    p_range = profile.max() - profile.min()
    abs_component  = float(np.clip(p_range / 0.45, 0.0, 1.0))          # 0 at flat, 1 at range≥0.45
    norm_component = float(np.clip(profile.std() / (p_range + 1e-6) * 2.0, 0.0, 1.0))
    radial_gradient = float(np.clip(0.65 * abs_component + 0.35 * norm_component, 0.0, 1.0))

    # Hard-reject flag: profile range too small = artifact / featureless image
    profile_range_ok = p_range >= PROFILE_RANGE_MIN

    # Streak-line detector:
    #   1. mean spike (catches obvious full-width streaks)
    #   2. 95th-pct spike (catches partial streaks)
    #   3. bright-pixel-fraction spike (catches streaks on bright cloud backgrounds)
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    row_spike = float((row_means[1:-1] - (row_means[:-2] + row_means[2:]) / 2).max())
    col_spike = float((col_means[1:-1] - (col_means[:-2] + col_means[2:]) / 2).max())
    row_pct   = np.percentile(gray, 95, axis=1)
    col_pct   = np.percentile(gray, 95, axis=0)
    row_pct_spike = float((row_pct[1:-1] - (row_pct[:-2] + row_pct[2:]) / 2).max())
    col_pct_spike = float((col_pct[1:-1] - (col_pct[:-2] + col_pct[2:]) / 2).max())
    # bright-fraction spike: fraction of pixels >0.90 per row/col
    row_bf = (gray > 0.90).mean(axis=1).astype(np.float64)
    col_bf = (gray > 0.90).mean(axis=0).astype(np.float64)
    row_bf_spike = float((row_bf[1:-1] - (row_bf[:-2] + row_bf[2:]) / 2).max())
    col_bf_spike = float((col_bf[1:-1] - (col_bf[:-2] + col_bf[2:]) / 2).max())
    # Wide-window spike: compare each row to 11-row smoothed context, excluding self.
    # Catches streaks at cloud/ocean boundaries where immediate neighbors are bright.
    W = STREAK_WIDE_WINDOW
    from scipy.ndimage import uniform_filter1d
    smooth_r = uniform_filter1d(row_bf, size=W, mode='nearest').astype(np.float64)
    smooth_c = uniform_filter1d(col_bf, size=W, mode='nearest').astype(np.float64)
    ctx_r = (smooth_r * W - row_bf) / max(W - 1, 1)
    ctx_c = (smooth_c * W - col_bf) / max(W - 1, 1)
    row_bf_wide_spike = float((row_bf - ctx_r).max())
    col_bf_wide_spike = float((col_bf - ctx_c).max())
    streak_ok = (
        max(row_spike, col_spike, row_pct_spike, col_pct_spike) <= STREAK_SPIKE_MAX
        and max(row_bf_spike, col_bf_spike) <= STREAK_BRIGHT_FRAC_SPIKE
        and max(row_bf_wide_spike, col_bf_wide_spike) <= STREAK_WIDE_FRAC_SPIKE
    )

    # Scan-line smearing: vertical gradient >> horizontal = banded rows artifact
    gy_mean = float(np.mean(np.abs(np.diff(gray.astype(np.float64), axis=0))))
    gx_mean = float(np.mean(np.abs(np.diff(gray.astype(np.float64), axis=1))))
    grad_anisotropy = gy_mean / (gx_mean + 1e-6)
    smear_ok = grad_anisotropy <= GRAD_ANISOTROPY_MAX

    # Missing-data hole: too many near-white pixels = smeared / masked region
    bright_fill_ok = float(np.mean(gray > BRIGHT_FILL_THRESHOLD)) <= BRIGHT_FILL_MAX

    # Partial frame: any edge strip predominantly bright = data boundary cut-off
    s = EDGE_STRIP_PX
    edge_ok = all(
        float(np.mean(strip > 0.90)) <= EDGE_BRIGHT_MAX
        for strip in (gray[:, -s:], gray[:, :s], gray[:s, :], gray[-s:, :])
    )

    # Radial symmetry: mean (1 - CV) per ring
    cv_per_ring = ring_stds / (np.abs(profile) + 1e-6)
    radial_symmetry = float(np.clip(1.0 - cv_per_ring.mean(), 0.0, 1.0))

    # Gradient coherence: fraction of gradient energy pointing radially
    gy = np.gradient(gray.astype(np.float64), axis=0)
    gx = np.gradient(gray.astype(np.float64), axis=1)
    gm = np.sqrt(gx ** 2 + gy ** 2) + 1e-8

    R_safe = R + 1e-8
    ry = Y / R_safe
    rx = X / R_safe

    cos_a    = np.abs(gx * rx + gy * ry) / gm
    interior = R <= max_r
    gm_int   = gm[interior]
    gradient_coherence = float(np.clip(
        (cos_a[interior] * gm_int).sum() / (gm_int.sum() + 1e-8), 0.0, 1.0
    ))

    # Core contrast: |center_mean − mid_ring_mean| (eye / cold-core detection)
    inner_mask = R <= max_r * 0.20
    mid_mask   = (R > max_r * 0.25) & (R <= max_r * 0.55)
    if inner_mask.any() and mid_mask.any():
        diff = abs(float(gray[inner_mask].mean()) - float(gray[mid_mask].mean()))
        core_contrast = float(np.clip(diff * 4.0, 0.0, 1.0))
    else:
        core_contrast = 0.0

    cyclone_score = (
        0.30 * radial_symmetry
        + 0.30 * radial_gradient
        + 0.20 * gradient_coherence
        + 0.20 * core_contrast
    )

    return {
        "radial_symmetry":    radial_symmetry,
        "radial_gradient":    radial_gradient,
        "gradient_coherence": gradient_coherence,
        "core_contrast":      core_contrast,
        "cyclone_score":      float(np.clip(cyclone_score, 0.0, 1.0)),
        "profile_range_ok":   profile_range_ok,
        "streak_ok":          streak_ok,
        "bright_fill_ok":     bright_fill_ok,
        "edge_ok":            edge_ok,
        "smear_ok":           smear_ok,
        "grad_anisotropy":    grad_anisotropy,
    }


# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────

def analyze_image(
    path,
    allow_single_spot,
    min_area_ratio,
    max_area_ratio,
    circ_level,
    cluster_min_count=3,
    cluster_proximity_ratio=0.35,
    min_blob_circularity=0.35,
    cyclone_score_threshold=None,
):
    if cyclone_score_threshold is None:
        cyclone_score_threshold = CYCLONE_SCORE_THRESHOLDS[circ_level]

    img  = Image.open(path).convert("L")
    gray = np.asarray(img, dtype=np.float32) / 255.0
    if gray.size == 0:
        return None

    thresh = otsu_threshold(gray)
    below  = gray <= thresh
    above  = ~below
    if below.sum() == 0 or above.sum() == 0:
        return None

    mask = below if gray[below].mean() < gray[above].mean() else above
    h, w = mask.shape

    area_min = max(1, int(min_area_ratio * h * w))
    area_max = max(area_min, int(max_area_ratio * h * w))

    comps = find_components(mask)
    if not comps:
        return None

    sized = [c for c in comps if area_min <= c["area"] <= area_max] or comps

    has_cluster, cluster_size = detect_cluster(
        sized, h, w,
        min_blob_circ=min_blob_circularity,
        min_count=cluster_min_count,
        proximity_ratio=cluster_proximity_ratio,
    )

    sized.sort(key=lambda c: c["area"], reverse=True)
    largest  = sized[0]
    area     = largest["area"]
    perim    = largest["perim"]
    circularity = max(0.0, min(1.0,
        (4.0 * math.pi * area) / (perim ** 2) if perim > 0 else 0.0
    ))

    cy, cx     = largest["centroid"]
    center_y   = (h - 1) / 2.0
    center_x   = (w - 1) / 2.0
    dist       = math.hypot(cy - center_y, cx - center_x)
    diag       = math.hypot(center_y, center_x)
    center_dist  = dist / diag if diag > 0 else 0.0
    center_score = max(0.0, 1.0 - center_dist)

    spot_count = len(sized)
    spot_score = (
        min(1.0, spot_count / 2.0) if allow_single_spot
        else (1.0 if spot_count >= 2 else 0.0)
    )

    cyc = compute_cyclone_metrics(gray)

    cluster_bonus = 1.0 if has_cluster else 0.0
    score = float(np.clip(
        0.12 * circularity
        + 0.08 * center_score
        + 0.05 * spot_score
        + 0.10 * cluster_bonus
        + 0.22 * cyc["radial_symmetry"]
        + 0.22 * cyc["radial_gradient"]
        + 0.11 * cyc["gradient_coherence"]
        + 0.10 * cyc["core_contrast"],
        0.0, 1.0,
    ))

    # Hard rejects override all other criteria
    artifact_free = (
        cyc["profile_range_ok"]
        and cyc["streak_ok"]
        and cyc["bright_fill_ok"]
        and cyc["edge_ok"]
        and cyc["smear_ok"]
    )

    blob_keep = (
        circularity >= CIRCULARITY_THRESHOLDS[circ_level]
        and score >= SCORE_THRESHOLDS[circ_level]
        and (allow_single_spot or spot_count >= 2)
    )
    keep = artifact_free and (
        blob_keep
        or has_cluster
        or (cyc["cyclone_score"] >= cyclone_score_threshold)
    )

    return {
        "circularity":        circularity,
        "score":              score,
        "spot_count":         spot_count,
        "center_dist":        center_dist,
        "has_cluster":        has_cluster,
        "cluster_size":       cluster_size,
        "radial_symmetry":    cyc["radial_symmetry"],
        "radial_gradient":    cyc["radial_gradient"],
        "gradient_coherence": cyc["gradient_coherence"],
        "core_contrast":      cyc["core_contrast"],
        "cyclone_score":      cyc["cyclone_score"],
        "profile_range_ok":   cyc["profile_range_ok"],
        "streak_ok":          cyc["streak_ok"],
        "bright_fill_ok":     cyc["bright_fill_ok"],
        "edge_ok":            cyc["edge_ok"],
        "smear_ok":           cyc["smear_ok"],
        "grad_anisotropy":    cyc["grad_anisotropy"],
        "keep":               keep,
    }


# ── WORKER / PIPELINE ─────────────────────────────────────────────────────────

def _analyze_worker(args):
    (
        path_str,
        allow_single,
        min_area_ratio,
        max_area_ratio,
        circ_level,
        cluster_min_count,
        cluster_proximity_ratio,
        min_blob_circularity,
        cyclone_score_threshold,
    ) = args
    try:
        metrics = analyze_image(
            Path(path_str),
            allow_single,
            min_area_ratio,
            max_area_ratio,
            circ_level,
            cluster_min_count,
            cluster_proximity_ratio,
            min_blob_circularity,
            cyclone_score_threshold,
        )
    except Exception:
        metrics = None
    return path_str, metrics


def pick_base(path, bases):
    path = path.resolve()
    candidates = [
        b.resolve() for b in bases
        if b.resolve() in [path] + list(path.parents)
    ]
    return max(candidates, key=lambda p: len(str(p))) if candidates else path.parent


def run_filter(
    input_specs,
    out,
    mode="copy",
    circularity="loose",
    allow_single_spot=True,
    require_multiple_spots=False,
    min_area_ratio=0.001,
    max_area_ratio=0.30,
    cluster_min_count=3,
    cluster_proximity_ratio=0.35,
    min_blob_circularity=0.35,
    cyclone_score_threshold=None,
    write_csv=True,
    dry_run=False,
    max_files=0,
    workers=None,
):
    allow_single = allow_single_spot and not require_multiple_spots
    workers      = workers or cpu_count()
    cyc_thresh   = (cyclone_score_threshold
                    if cyclone_score_threshold is not None
                    else CYCLONE_SCORE_THRESHOLDS[circularity])

    out_base   = Path(out)
    keep_dir   = out_base / "KEEP"
    review_dir = out_base / "REVIEW"
    if not dry_run:
        keep_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

    # Use parent of each spec dir so relative paths preserve the category subfolder.
    # e.g. base = cyclone_dataset/ → rel = cat1_tropical_depression/filename.png
    bases = []
    for s in input_specs:
        b = glob_base(s)
        if not any(ch in s for ch in "*?["):
            b = b.parent
        bases.append(b)
    out_resolved = out_base.resolve()

    all_paths = [
        p for p in iter_images(input_specs)
        if out_resolved not in [p.resolve()] + list(p.resolve().parents)
    ]
    if max_files:
        all_paths = all_paths[:max_files]

    worker_args = [
        (
            str(p),
            allow_single,
            min_area_ratio,
            max_area_ratio,
            circularity,
            cluster_min_count,
            cluster_proximity_ratio,
            min_blob_circularity,
            cyc_thresh,
        )
        for p in all_paths
    ]

    csv_path = out_base / "filter_report.csv"
    csv_file = writer = None
    if write_csv and not dry_run:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=[
            "path", "decision", "score", "circularity", "spot_count",
            "center_dist", "has_cluster", "cluster_size",
            "radial_symmetry", "radial_gradient",
            "gradient_coherence", "core_contrast", "cyclone_score",
            "profile_range_ok", "streak_ok", "bright_fill_ok", "edge_ok",
            "smear_ok", "grad_anisotropy",
        ])
        writer.writeheader()

    processed = kept = reviewed = 0
    progress = tqdm(total=len(all_paths), unit="img") if HAS_TQDM else None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_analyze_worker, a): a[0] for a in worker_args}

        for future in as_completed(futures):
            path_str, metrics = future.result()
            path     = Path(path_str)
            decision = "keep" if (metrics is not None and metrics["keep"]) else "review"

            if not dry_run:
                base     = pick_base(path, bases)
                rel      = path.resolve().relative_to(base)
                dest     = (keep_dir if decision == "keep" else review_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                (shutil.copy2 if mode == "copy" else shutil.move)(str(path), str(dest))

            if writer is not None:
                def _fmt(k, fmt=".4f"):
                    return "" if metrics is None else (
                        f"{metrics[k]:{fmt}}" if isinstance(metrics[k], float)
                        else str(metrics[k])
                    )
                writer.writerow({
                    "path":              path_str,
                    "decision":          decision,
                    "score":             _fmt("score"),
                    "circularity":       _fmt("circularity"),
                    "spot_count":        _fmt("spot_count", "d") if metrics else "",
                    "center_dist":       _fmt("center_dist"),
                    "has_cluster":       "" if metrics is None else int(metrics["has_cluster"]),
                    "cluster_size":      _fmt("cluster_size", "d") if metrics else "",
                    "radial_symmetry":   _fmt("radial_symmetry"),
                    "radial_gradient":   _fmt("radial_gradient"),
                    "gradient_coherence": _fmt("gradient_coherence"),
                    "core_contrast":     _fmt("core_contrast"),
                    "cyclone_score":     _fmt("cyclone_score"),
                    "profile_range_ok":  "" if metrics is None else int(metrics["profile_range_ok"]),
                    "streak_ok":         "" if metrics is None else int(metrics["streak_ok"]),
                    "bright_fill_ok":    "" if metrics is None else int(metrics["bright_fill_ok"]),
                    "edge_ok":           "" if metrics is None else int(metrics["edge_ok"]),
                    "smear_ok":          "" if metrics is None else int(metrics["smear_ok"]),
                    "grad_anisotropy":   _fmt("grad_anisotropy") if metrics else "",
                })

            processed += 1
            kept      += decision == "keep"
            reviewed  += decision == "review"
            if progress:
                progress.update(1)

    if progress:
        progress.close()
    if csv_file:
        csv_file.close()

    print(f"Processed : {processed}")
    print(f"Keep      : {kept}")
    print(f"Review    : {reviewed}")
    print(f"Output    : {out_base.resolve()}")

    return {"processed": processed, "keep": kept,
            "review": reviewed, "output": str(out_base.resolve())}


# ── MANUAL REVIEW UI ──────────────────────────────────────────────────────────

def manual_review_mode(out_dir):
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError:
        print("tkinter or Pillow not available — cannot run manual review.")
        return

    review_dir = Path(out_dir) / "REVIEW"
    keep_dir   = Path(out_dir) / "KEEP"

    images = sorted([
        p for p in review_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])
    if not images:
        print("No images in REVIEW folder.")
        return

    decisions  = {}
    idx        = [0]
    photo_ref  = [None]

    root = tk.Tk()
    root.title("Manual Review")
    root.configure(bg="#1e1e1e")

    status_var = tk.StringVar()
    tk.Label(root, textvariable=status_var, bg="#1e1e1e", fg="white",
             font=("Helvetica", 12)).pack(side=tk.TOP, pady=(8, 0))

    img_label = tk.Label(root, bg="#1e1e1e")
    img_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    tk.Label(root, text="y = Keep   n = Skip   <-> = Navigate   q = Quit & apply",
             bg="#1e1e1e", fg="#888888",
             font=("Helvetica", 10)).pack(side=tk.BOTTOM, pady=(0, 8))

    def load_current():
        p = images[idx[0]]
        img = Image.open(p)
        img.thumbnail((900, 700), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        photo_ref[0] = photo
        img_label.config(image=photo)
        dec = decisions.get(p, "-")
        status_var.set(f"[{idx[0] + 1}/{len(images)}]  {p.name}  ->  {dec}")

    def decide(decision):
        decisions[images[idx[0]]] = decision
        if idx[0] < len(images) - 1:
            idx[0] += 1
            load_current()
        else:
            apply_and_quit()

    def nav(delta):
        new_idx = idx[0] + delta
        if 0 <= new_idx < len(images):
            idx[0] = new_idx
            load_current()

    def apply_and_quit():
        root.destroy()
        kept = 0
        for p, dec in decisions.items():
            if dec == "keep":
                rel  = p.relative_to(review_dir)
                dest = keep_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
                kept += 1
        skipped   = sum(1 for v in decisions.values() if v == "skip")
        undecided = len(images) - len(decisions)
        print(f"Manual review: {kept} moved to KEEP, "
              f"{skipped} skipped, {undecided} undecided.")

    root.bind("y", lambda e: decide("keep"))
    root.bind("n", lambda e: decide("skip"))
    root.bind("<Right>", lambda e: nav(1))
    root.bind("<Left>",  lambda e: nav(-1))
    root.bind("q",       lambda e: apply_and_quit())

    load_current()
    root.mainloop()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out",   type=str,  default="bugbite_filter_output")
    p.add_argument("--mode",  choices=["copy", "move"], default="copy")
    p.add_argument("--circularity", choices=["strict", "medium", "loose"],
                   default="loose")
    p.add_argument("--allow-single-spot",     action="store_true", default=True)
    p.add_argument("--require-multiple-spots", action="store_true")
    p.add_argument("--min-area-ratio",         type=float, default=0.001)
    p.add_argument("--max-area-ratio",         type=float, default=0.30)
    p.add_argument("--cluster-min-count",      type=int,   default=3)
    p.add_argument("--cluster-proximity-ratio", type=float, default=0.35)
    p.add_argument("--min-blob-circularity",   type=float, default=0.35)
    p.add_argument("--cyclone-score-threshold", type=float, default=None,
                   help="Override cyclone score keep threshold (default from --circularity level)")
    p.add_argument("--workers",   type=int, default=None)
    p.add_argument("--write-csv", action="store_true", default=True)
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--manual-review", action="store_true",
                   help="Launch interactive review of REVIEW/ folder after filtering")
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
        cluster_min_count=args.cluster_min_count,
        cluster_proximity_ratio=args.cluster_proximity_ratio,
        min_blob_circularity=args.min_blob_circularity,
        cyclone_score_threshold=args.cyclone_score_threshold,
        write_csv=args.write_csv,
        dry_run=args.dry_run,
        max_files=args.max_files,
        workers=args.workers,
    )

    if args.manual_review:
        manual_review_mode(args.out)


if __name__ == "__main__":
    main()
