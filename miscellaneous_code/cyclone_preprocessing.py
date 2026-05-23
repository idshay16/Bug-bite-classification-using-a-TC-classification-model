import os

import cv2
import numpy as np


# ── module-level helpers ───────────────────────────────────────────────────────

def _highpass(profile, smooth_size=63):
    """Remove broad background variation, keeping only sharp local anomalies.

    Subtracts a box-smoothed version of the profile from itself.  The result
    highlights sharp transitions (tile seams, 1-8px wide) while suppressing
    gradual variation (tile-to-tile brightness difference, cloud bands).

    smooth_size is intentionally larger than max grid-cell width (≈111px in
    a 333px image) so that broad brightness ramps are removed but the seam
    signal (a 1-3px spike/trough) is fully preserved.
    """
    k       = min(smooth_size, len(profile) - 1) | 1  # ensure odd, < length
    kernel  = np.ones(k) / k
    smooth  = np.convolve(profile, kernel, mode='same')
    return profile - smooth


def _flag_ranges(profile, length, thresh_sigma=2.0, max_line_width=8, edge_margin=5):
    """Return (start, end) spans for bright OR dark sharp anomalies.

    Operates on the HIGH-PASS residual of `profile`, so broad tile-brightness
    variation does not trigger detection.  Catches seams regardless of polarity
    (brighter or darker than neighboring tiles).
    """
    hp = _highpass(profile)
    m, s = hp.mean(), hp.std()
    if s == 0:
        return []
    bright   = hp > m + thresh_sigma * s
    dark     = hp < m - thresh_sigma * s
    combined = bright | dark
    idx = np.where(
        np.diff(np.concatenate([[False], combined, [False]]).astype(int))
    )[0]
    return [
        (int(a), int(b))
        for a, b in zip(idx[0::2], idx[1::2])
        if (b - a) <= max_line_width
        and a >= edge_margin
        and b <= length - edge_margin
    ]


def _clean_neighbor(pos, direction, all_spans, limit):
    """Walk outward from pos until clear of all grid spans.

    Prevents using a grid-span pixel as the interpolation anchor when two
    grid spans are adjacent.
    """
    while 0 <= pos < limit and any(s <= pos < e for s, e in all_spans):
        pos += direction
    return max(0, min(limit - 1, pos))


def _interp_rows(out, rows, h):
    """In-place skip-aware linear interpolation across all row grid spans."""
    for s, e in rows:
        above = _clean_neighbor(s - 1, -1, rows, h)
        below = _clean_neighbor(e,     +1, rows, h)
        span  = e - s
        if span == 0:
            continue
        for i in range(span):
            t = (i + 1) / (span + 1)
            out[s + i, :] = (1.0 - t) * out[above, :] + t * out[below, :]


def _interp_cols(out, cols, w):
    """In-place skip-aware linear interpolation across all column grid spans."""
    for s, e in cols:
        left  = _clean_neighbor(s - 1, -1, cols, w)
        right = _clean_neighbor(e,     +1, cols, w)
        span  = e - s
        if span == 0:
            continue
        for i in range(span):
            t = (i + 1) / (span + 1)
            out[:, s + i] = (1.0 - t) * out[:, left] + t * out[:, right]


def _exclude_overlapping(spans, known):
    """Remove spans that overlap with any span in `known`."""
    return [
        (s, e) for s, e in spans
        if not any(s < ke and e > ks for ks, ke in known)
    ]


def _grid_position_seams(profile, length, already_done,
                          thresh_sigma=2.0, max_line_width=6,
                          edge_margin=5, search_half=8):
    """Detect seams only at positions that correspond to regular grid spacings.

    Checks multiples of length//3 and length//4 (the two expected grid
    templates).  TC storm cloud bands — which appear at arbitrary positions —
    are therefore never detected, regardless of their sharpness.

    For each candidate position not already covered by `already_done`,
    a ±search_half window in the high-pass profile is examined; if any
    pixel in that window exceeds thresh_sigma * std, the local peak and its
    contiguous anomaly span are returned as a seam.
    """
    # Build candidate positions from divisors 3 and 4
    candidates = set()
    for divisor in (3, 4):
        spacing = length // divisor
        for i in range(1, divisor):
            pos = i * spacing
            if edge_margin < pos < length - edge_margin:
                candidates.add(pos)

    hp = _highpass(profile)
    m, s = hp.mean(), hp.std()
    if s == 0:
        return []

    results = []
    for pos in sorted(candidates):
        # Skip if this position is already covered by the canonical mask
        if any(sp_s <= pos < sp_e for sp_s, sp_e in already_done):
            continue
        # Examine ±search_half window around candidate
        lo = max(edge_margin, pos - search_half)
        hi = min(length - edge_margin, pos + search_half)
        window = hp[lo:hi]
        if len(window) == 0:
            continue
        peak_abs = np.abs(window).max()
        if peak_abs <= thresh_sigma * s:
            continue
        # Locate the peak and expand to the full contiguous anomaly span
        peak = lo + int(np.argmax(np.abs(window)))
        start, end = peak, peak + 1
        while start > edge_margin and abs(hp[start - 1]) > s:
            start -= 1
        while end < length - edge_margin and abs(hp[end]) > s:
            end += 1
        if (end - start) <= max_line_width:
            results.append((int(start), int(end)))

    return results


def _make_grid_mask(grid_rows, grid_cols, h, w):
    """Return uint8 mask array with 255 at detected seam positions."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for s, e in grid_rows:
        mask[s:e, :] = 255
    for s, e in grid_cols:
        mask[:, s:e] = 255
    return mask


# ── public API ─────────────────────────────────────────────────────────────────

def build_grid_masks(src_dir, thresh_sigma=0.7, max_line_width=8,
                     coast_sigma=2.5, merge_tolerance=15, edge_margin=5):
    """Build per-canonical-size-group removal masks for TC satellite imagery.

    Phase 1 (grid seams only):
        Coastline detection and inpainting are disabled. coast_mask in the
        returned dict is an all-zeros array.

    Key improvements over v1:

    High-pass profile detection:
        Before thresholding, the mean profile is high-pass filtered (box
        smooth subtracted).  This removes broad tile-to-tile brightness
        variation (which spans 80-150px per tile) that was causing false
        detections at low sigma.  Only sharp features (1-8px seam width)
        remain in the residual.  The result: thresh_sigma=2.0 on the
        high-pass residual catches both the dominant (3x3) and minority
        (4x4) seam patterns without false positives from cloud bands or
        tile-brightness gradients.

    Bidirectional detection:
        Catches seams regardless of polarity (brighter OR darker tile seam).

    Skip-aware interpolation:
        _clean_neighbor walks outward from a span boundary to find a truly
        clean reference pixel, preventing adjacent spans from using each
        other as interpolation anchors.

    Args:
        src_dir:         Root directory: src_dir/<class>/<image files>.
        thresh_sigma:    Threshold multiplier on high-pass residual std.
        max_line_width:  Max run length (px) counted as a grid seam.
        coast_sigma:     (Phase 1: unused) Coastline gradient threshold.
        merge_tolerance: Max per-axis px difference to merge size variants.
        edge_margin:     Grid detections within this many px of border discarded.

    Returns:
        masks:   {(h, w): {'grid_rows', 'grid_cols', 'coast_mask', 'visual'}}
        means:   {(h, w): (mean_img float32, n_images int)}
        samples: {(h, w): [path, ...]}
    """
    # ── collect paths by exact size ───────────────────────────────────────────
    by_size = {}
    for cls in os.listdir(src_dir):
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(os.path.join(cls_path, fname))
            if img is None:
                continue
            by_size.setdefault(img.shape[:2], []).append(
                os.path.join(cls_path, fname))

    # ── merge near-identical sizes (largest group = canonical) ───────────────
    canonical = {}
    for (h, w), paths in sorted(by_size.items(), key=lambda x: -len(x[1])):
        placed = False
        for (ch, cw) in canonical:
            if abs(h - ch) <= merge_tolerance and abs(w - cw) <= merge_tolerance:
                canonical[(ch, cw)].extend(paths)
                placed = True
                break
        if not placed:
            canonical[(h, w)] = list(paths)

    counts = {f'{cw}x{ch}': len(ps)
              for (ch, cw), ps in sorted(canonical.items(), key=lambda x: -len(x[1]))}
    print(f'Canonical groups (merge_tolerance={merge_tolerance}px): {counts}')

    # ── build mask per canonical group ────────────────────────────────────────
    masks, means, samples = {}, {}, {}

    for (ch, cw), paths in sorted(canonical.items(), key=lambda x: -len(x[1])):
        accum = None
        valid = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            if img.shape[:2] != (ch, cw):
                img = cv2.resize(img, (cw, ch))
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            accum = gray if accum is None else accum + gray
            valid.append(p)
        if accum is None:
            continue

        n        = len(valid)
        mean_img = (accum / n).astype(np.float32)

        # Grid seam detection on high-pass mean profiles
        grid_rows = _flag_ranges(mean_img.mean(axis=1), ch,
                                 thresh_sigma, max_line_width, edge_margin)
        grid_cols = _flag_ranges(mean_img.mean(axis=0), cw,
                                 thresh_sigma, max_line_width, edge_margin)

        # Pass 2 — coastline detection: DISABLED for Phase 1.
        coast_mask = np.zeros((ch, cw), dtype=np.uint8)

        visual = _make_grid_mask(grid_rows, grid_cols, ch, cw)

        pct  = 100 * (visual > 0).mean()
        note = ' (small group — less reliable mean)' if n < 10 else ''
        print(f'  {cw}x{ch} ({n} imgs): {pct:.1f}% masked  '
              f'grid rows={len(grid_rows)} cols={len(grid_cols)}{note}')

        masks[(ch, cw)]   = {'grid_rows': grid_rows, 'grid_cols': grid_cols,
                              'coast_mask': coast_mask, 'visual': visual}
        means[(ch, cw)]   = (mean_img, n)
        samples[(ch, cw)] = valid

    return masks, means, samples


def apply_grid_mask(img_bgr, masks):
    """Remove grid seams with minimal blurring.

    Steps:
      1. (Phase 1) Coastline inpainting: no-op (coast_mask all zeros).
      2. Skip-aware linear interpolation for canonical grid rows.
      3. Skip-aware linear interpolation for canonical grid cols.
      4. Per-image fallback (high-pass, sigma=2.0): re-detect on the
         partially-cleaned image using the same high-pass approach as the
         canonical detection.  The high-pass filter removes broad TC storm
         features (which span 50-100px in the profile) before thresholding,
         so only narrow grid seams (1-6px) are detected.
         max_line_width=6 additionally rejects any wider false positive.

    Falls back to nearest canonical group for anomalous sizes.
    """
    PER_IMG_SIGMA = 1.2  # position-constrained → safe to go low; 2.0 missed visible seams
    PER_IMG_MW    = 8    # allow slightly wider spans for individual-image seams

    h, w = img_bgr.shape[:2]
    if (h, w) in masks:
        m          = masks[(h, w)]
        grid_rows  = m['grid_rows']
        grid_cols  = m['grid_cols']
        coast_mask = m['coast_mask']
    else:
        nearest    = min(masks.keys(), key=lambda k: abs(k[0] - h) + abs(k[1] - w))
        m          = masks[nearest]
        sh, sw     = h / nearest[0], w / nearest[1]
        grid_rows  = [(int(s * sh), int(e * sh)) for s, e in m['grid_rows']]
        grid_cols  = [(int(s * sw), int(e * sw)) for s, e in m['grid_cols']]
        coast_mask = cv2.resize(m['coast_mask'], (w, h), interpolation=cv2.INTER_NEAREST)

    out = img_bgr.copy().astype(np.float32)

    # Step 1: coastline inpainting (Phase 1: coast_mask all zeros — no-op)
    coast_mask8 = (coast_mask > 0).astype(np.uint8) * 255
    if coast_mask8.any():
        out = cv2.inpaint(
            out.astype(np.uint8), coast_mask8,
            inpaintRadius=7, flags=cv2.INPAINT_TELEA
        ).astype(np.float32)

    # Steps 2-3: canonical grid seam interpolation (smooths the hard tile edges
    # left by step 0, as well as removing thin seam-line pixels)
    _interp_rows(out, grid_rows, h)
    _interp_cols(out, grid_cols, w)

    # Step 4: position-constrained per-image fallback, applied sequentially so
    #   that col cleaning can expose row seams that were masked by col artifacts
    #   and vice versa.  Two passes: cols first, then rows on the updated image.
    gray_clean = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    raw_cols   = _flag_ranges(gray_clean.mean(axis=0), w, PER_IMG_SIGMA, PER_IMG_MW, edge_margin=5)
    extra_cols = _exclude_overlapping(raw_cols, grid_cols)
    if extra_cols:
        _interp_cols(out, extra_cols, w)
        grid_cols = list(grid_cols) + extra_cols

    gray_clean = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    extra_rows = _grid_position_seams(gray_clean.mean(axis=1), h,
                                      grid_rows, PER_IMG_SIGMA, PER_IMG_MW)
    if extra_rows:
        _interp_rows(out, extra_rows, h)

    return np.clip(out, 0, 255).astype(np.uint8)
