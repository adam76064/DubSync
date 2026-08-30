"""
Robust RANSAC line fitting for anchor alignment.

Re-implements (as original numpy) the line-fitting core of `sc0ty/subsync`
(its C++ `gizmo` LineFinder) — documented in `docs/SUBSYNC_ANALYSIS.md` and
`docs/SUBSYNC_INTEGRATION_PLAN.md`. No GPL code is copied; only the algorithm
(incremental RANSAC over a (ref_time, tar_time) point cloud + iterative
outlier removal + Pearson r^2 quality) is reproduced.

The fit maps reference time `x` to target time `y` via `y = a*x + b`:
  - `a` (slope)  is the global speed ratio (tar/ref).
  - `b` (intercept) is the global offset.
Anchors that do not lie on the consensus line are treated as either false
matches (scattered outliers) or real cut boundaries (used elsewhere).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

__all__ = ["LineFit", "pearson_r2", "fit_ransac_line"]


@dataclass
class LineFit:
    """Result of a robust line fit over an (x=ref_time, y=tar_time) point cloud."""

    slope: float          # a in y = a*x + b  (speed ratio)
    intercept: float      # b in y = a*x + b  (offset)
    r_squared: float      # Pearson r^2 of the (weighted) least-squares inlier fit
    inlier_ratio: float   # n_inliers / n_total  (coverage)
    n_inliers: int
    n_total: int
    inlier_mask: np.ndarray   # boolean mask over the input points
    confidence: float         # r^2 * inlier_ratio * coverage_ratio (composite 0..1)
    coverage_ratio: float = 1.0   # min(1, distinct_ref_buckets / min_buckets)
    n_buckets: int = 0            # distinct ref-time buckets spanned by inliers

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept


def pearson_r2(x: Sequence[float], y: Sequence[float],
               w: Optional[Sequence[float]] = None) -> Tuple[float, float, float]:
    """
    Weighted least-squares line through (x, y) plus the squared Pearson
    correlation coefficient (r^2) of that fit. Returns (r2, slope, intercept).

    Degenerate cases (fewer than 2 points, zero variance) return a sentinel
    r2 of 0.0 and the identity line so callers can detect them.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        return 0.0, 1.0, 0.0

    if w is None:
        w = np.ones(n)
    else:
        w = np.asarray(w, dtype=float)
        w = np.maximum(w, 1e-6)

    sw = w.sum()
    if sw <= 0.0:
        w = np.ones(n)
        sw = float(n)

    xw = (w * x).sum() / sw
    yw = (w * y).sum() / sw
    txx = (w * (x - xw) * (x - xw)).sum()
    tyy = (w * (y - yw) * (y - yw)).sum()
    txy = (w * (x - xw) * (y - yw)).sum()

    if txx <= 1e-12 or tyy <= 1e-12:
        return 0.0, 1.0, 0.0

    a = txy / txx
    b = yw - a * xw
    r2 = (txy * txy) / (txx * tyy)
    return float(r2), float(a), float(b)


def fit_ransac_line(
    points: Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
    *,
    inlier_tol: float = 1.0,
    slope_lo: float = 0.85,
    slope_hi: float = 1.15,
    y_bounds: Optional[Tuple[float, float]] = None,
    coverage_bucket_sec: Optional[float] = None,
    min_coverage_buckets: Optional[int] = None,
) -> LineFit:
    """
    Fit the dominant line `y = a*x + b` through a (ref_time, tar_time) point
    cloud using incremental RANSAC + iterative furthest-outlier removal.

    Parameters
    ----------
    points : (N, 2) array-like of [x=ref_time, y=tar_time].
    weights : optional (N,) per-point weight (e.g. acoustic-confirmation
        strength). Used for candidate ranking and the final weighted OLS.
    inlier_tol : perpendicular distance (seconds) within which a point counts
        as an inlier.
    slope_lo / slope_hi : allowed slope range (broadcast-speed band).
    y_bounds : optional (y_lo, y_hi). Candidate lines must map the observed
        x-range into this target-time range (global-consistency check, adapted
        from subsync's endpoint constraint to tolerate real offsets).
    coverage_bucket_sec / min_coverage_buckets : optional diversity constraint.
        When both are set, inliers must span at least `min_coverage_buckets`
        distinct `floor(ref_time / coverage_bucket_sec)` buckets; otherwise the
        fit's confidence is scaled down by `coverage_ratio`. Prevents a single
        dense region (e.g. a false black-frame cluster) from over-determining
        the line.

    Returns a LineFit. If no valid line is found (e.g. < 2 points), returns a
    degenerate fit (identity slope, 0 inliers, confidence 0.0).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array")

    n = pts.shape[0]
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must match points length")
        w = np.maximum(w, 0.0)

    def empty_fit() -> LineFit:
        mask = np.zeros(n, dtype=bool)
        return LineFit(1.0, 0.0, 0.0, 0.0, 0, n, mask, 0.0)

    if n < 2:
        return empty_fit()

    x = pts[:, 0]
    y = pts[:, 1]
    xmin, xmax = float(x.min()), float(x.max())

    a2p1 = inlier_tol * inlier_tol  # dist^2 = (a*x + b - y)^2 / (a^2 + 1)

    def inlier_mask(a: float, b: float) -> Optional[np.ndarray]:
        if not (slope_lo <= a <= slope_hi):
            return None
        if y_bounds is not None:
            y_lo, y_hi = y_bounds
            if a * xmin + b < y_lo or a * xmin + b > y_hi:
                return None
            if a * xmax + b < y_lo or a * xmax + b > y_hi:
                return None
        d2 = (a * x + b - y) ** 2 / (a * a + 1.0)
        return d2 <= a2p1

    # Candidate lines: identity slope (broadcast ~1.0) with a couple of offsets,
    # plus the line through every pair of points.
    best_score = -1.0
    best_mask: Optional[np.ndarray] = None
    best_a, best_b = 1.0, 0.0

    def consider(a: float, b: float) -> None:
        nonlocal best_score, best_mask, best_a, best_b
        m = inlier_mask(a, b)
        if m is None:
            return
        score = float(w[m].sum())
        if score > best_score:
            best_score = score
            best_mask = m
            best_a, best_b = a, b

    consider(1.0, 0.0)
    consider(1.0, float(np.median(y - x)))

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] == x[i]:
                continue
            a = (y[j] - y[i]) / (x[j] - x[i])
            consider(a, y[i] - a * x[i])

    if best_mask is None:
        return empty_fit()

    # Refine: weighted OLS on the current inliers, then iteratively drop the
    # furthest inlier until all remaining are within tolerance.
    idx = np.where(best_mask)[0]
    a, b = best_a, best_b
    while idx.size >= 2:
        r2, a, b = pearson_r2(x[idx], y[idx], w[idx])
        d2 = (a * x[idx] + b - y[idx]) ** 2 / (a * a + 1.0)
        if d2.max() <= a2p1:
            break
        idx = np.delete(idx, int(d2.argmax()))
        if idx.size < 2:
            break

    if idx.size < 2:
        # Degenerate: keep the raw candidate line but mark low confidence.
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return LineFit(a, b, 0.0, mask.sum() / n, int(mask.sum()), n, mask, 0.0)

    r2, a, b = pearson_r2(x[idx], y[idx], w[idx])
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    n_in = int(mask.sum())
    inlier_ratio = n_in / n

    # Diversity / coverage constraint: inliers must span distinct ref-time
    # buckets, otherwise one dense region could over-determine the line.
    coverage_ratio = 1.0
    n_buckets = 0
    if coverage_bucket_sec is not None and min_coverage_buckets and min_coverage_buckets > 1:
        buckets = np.floor(x[idx] / coverage_bucket_sec).astype(int)
        n_buckets = int(np.unique(buckets).size)
        coverage_ratio = min(1.0, n_buckets / float(min_coverage_buckets))

    confidence = float(r2 * inlier_ratio * coverage_ratio)
    return LineFit(float(a), float(b), float(r2), float(inlier_ratio),
                   n_in, n, mask, confidence,
                   coverage_ratio=coverage_ratio, n_buckets=n_buckets)
