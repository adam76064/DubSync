"""Phase-subsync idea 1: robust RANSAC line fitting (reimplemented from subsync)."""
import numpy as np
import pytest

from dub_sync_engine.line_fit import fit_ransac_line, fit_piecewise_lines, pearson_r2, LineFit


def _cloud(n, slope, intercept, rng, noise=0.1, lo=0.0, hi=600.0):
    ref = np.sort(rng.uniform(lo, hi, n))
    tar = slope * ref + intercept + rng.normal(0, noise, n)
    return np.c_[ref, tar]


def test_pearson_r2_exact():
    x = [0, 1, 2, 3, 4]
    y = [2, 4, 6, 8, 10]  # y = 2x + 2
    r2, a, b = pearson_r2(x, y)
    assert abs(a - 2.0) < 1e-9
    assert abs(b - 2.0) < 1e-9
    assert abs(r2 - 1.0) < 1e-9


def test_clean_1x():
    rng = np.random.default_rng(0)
    pts = _cloud(50, 1.0, 0.0, rng)
    fit = fit_ransac_line(pts)
    assert abs(fit.slope - 1.0) < 0.01
    assert abs(fit.intercept) < 0.2
    assert fit.r_squared > 0.99
    assert fit.inlier_ratio == 1.0
    assert fit.confidence > 0.99


def test_pal_speed():
    rng = np.random.default_rng(1)
    pts = _cloud(50, 0.96, 2.0, rng)
    fit = fit_ransac_line(pts)
    assert abs(fit.slope - 0.96) < 0.01, f"slope={fit.slope}"
    assert abs(fit.intercept - 2.0) < 0.3, f"intercept={fit.intercept}"
    assert fit.inlier_ratio == 1.0


def test_false_anchors_discarded():
    """Planted scattered outliers must be discarded; true line recovered."""
    rng = np.random.default_rng(2)
    good = _cloud(50, 0.96, 2.0, rng)
    bad_ref = rng.uniform(0, 600, 8)
    bad_tar = rng.uniform(0, 600, 8)
    pts = np.vstack([good, np.c_[bad_ref, bad_tar]])
    fit = fit_ransac_line(pts)
    assert abs(fit.slope - 0.96) < 0.01
    assert abs(fit.intercept - 2.0) < 0.3
    assert fit.inlier_mask[:50].all(), "true inliers must be kept"
    assert not fit.inlier_mask[50:].any(), "false anchors must be rejected"


def test_two_speed_low_confidence():
    """A two-speed episode should not fit one line well -> low inlier ratio."""
    rng = np.random.default_rng(3)
    ref = np.sort(rng.uniform(0, 600, 60))
    tar = np.where(ref < 300, ref, 0.96 * ref + 12.0) + rng.normal(0, 0.1, 60)
    fit = fit_ransac_line(np.c_[ref, tar])
    assert fit.inlier_ratio < 0.9, f"inlier_ratio={fit.inlier_ratio}"
    # confidence = r2 * inlier_ratio must drop below a near-perfect single speed
    assert fit.confidence < 0.9


def test_weights_influence_selection():
    """A high-weight cluster should beat a low-weight cluster of equal size."""
    rng = np.random.default_rng(4)
    # 30 strong points on the true line (slope 1.0), 30 weak points on a wrong line.
    strong = _cloud(30, 1.0, 0.0, rng, lo=0, hi=300)
    weak = _cloud(30, 1.05, 50.0, rng, lo=300, hi=600)
    pts = np.vstack([strong, weak])
    weights = np.concatenate([np.full(30, 10.0), np.full(30, 0.1)])
    fit = fit_ransac_line(pts, weights=weights)
    assert abs(fit.slope - 1.0) < 0.02, f"slope={fit.slope} (strong cluster should win)"
    assert fit.inlier_mask[:30].all()


def test_y_bounds_constraint():
    """A line whose endpoints map outside the target timeline is rejected."""
    rng = np.random.default_rng(5)
    # True slope 1.0 but with a huge offset (300s) that would map ref->tar out of bounds.
    ref = np.sort(rng.uniform(0, 600, 40))
    tar = ref + 300.0 + rng.normal(0, 0.1, 40)
    pts = np.c_[ref, tar]
    # Bounds: target timeline is only [0, 600]; +300s offset maps x=600 -> 900 (out).
    fit = fit_ransac_line(pts, y_bounds=(0.0, 600.0))
    assert fit.confidence < 0.9, "out-of-bounds line must be rejected"


def test_coverage_constraint_downgrades_dense_cluster():
    """Inliers confined to one ref-time bucket must not earn full confidence."""
    rng = np.random.default_rng(7)
    # 40 points all within a single 60s bucket, on a bogus line (slope 1.05).
    ref = rng.uniform(0, 59, 40)
    tar = 1.05 * ref + 0.0 + rng.normal(0, 0.1, 40)
    fit = fit_ransac_line(np.c_[ref, tar],
                          coverage_bucket_sec=60.0, min_coverage_buckets=3)
    assert fit.n_buckets < 3
    assert fit.coverage_ratio < 1.0
    assert fit.confidence < fit.r_squared * fit.inlier_ratio


def test_coverage_constraint_satisfied_when_spread():
    """Inliers spanning many buckets keep full confidence."""
    rng = np.random.default_rng(8)
    ref = np.sort(rng.uniform(0, 600, 60))
    tar = ref + rng.normal(0, 0.1, 60)
    fit = fit_ransac_line(np.c_[ref, tar],
                          coverage_bucket_sec=60.0, min_coverage_buckets=3)
    assert fit.n_buckets >= 3
    assert fit.coverage_ratio == 1.0


def test_degenerate_single_point():
    fit = fit_ransac_line([[10.0, 12.0]])
    assert isinstance(fit, LineFit)
    assert fit.n_inliers == 0
    assert fit.confidence == 0.0


# --------------------------------------------------------------------------- #
# idea 5: recursive piecewise refinement
# --------------------------------------------------------------------------- #

def test_piecewise_single_speed_is_one_segment():
    rng = np.random.default_rng(10)
    ref = np.sort(rng.uniform(0, 600, 80))
    tar = ref + rng.normal(0, 0.1, 80)
    segs = fit_piecewise_lines(np.c_[ref, tar], min_inlier_ratio=0.8)
    assert len(segs) == 1
    assert abs(segs[0].fit.slope - 1.0) < 0.01


def test_piecewise_two_speeds_splits():
    """A PAL act followed by a 1.0x act must split into two segments."""
    rng = np.random.default_rng(11)
    ref1 = np.sort(rng.uniform(0, 300, 40))
    ref2 = np.sort(rng.uniform(300, 600, 40))
    tar1 = 0.96 * ref1 + 0.0 + rng.normal(0, 0.1, 40)
    tar2 = 1.00 * ref2 - 12.0 + rng.normal(0, 0.1, 40)  # offset continuity ~ 0.96*300=288 vs 300-12=288
    pts = np.c_[np.concatenate([ref1, ref2]), np.concatenate([tar1, tar2])]
    segs = fit_piecewise_lines(pts, min_inlier_ratio=0.8)
    assert len(segs) == 2, f"expected 2 segments, got {len(segs)}"
    segs.sort(key=lambda s: s.ref_start)
    assert abs(segs[0].fit.slope - 0.96) < 0.02
    assert abs(segs[1].fit.slope - 1.00) < 0.02


def test_piecewise_recovers_cut_segments():
    """A real cut (removed span) should yield two segments on the same slope."""
    rng = np.random.default_rng(12)
    ref1 = np.sort(rng.uniform(0, 300, 40))
    ref2 = np.sort(rng.uniform(309, 600, 40))  # 9s removed
    tar1 = ref1 + rng.normal(0, 0.1, 40)
    tar2 = (ref2 - 9.0) + rng.normal(0, 0.1, 40)  # same speed, offset shifted by -9s
    pts = np.c_[np.concatenate([ref1, ref2]), np.concatenate([tar1, tar2])]
    segs = fit_piecewise_lines(pts, min_inlier_ratio=0.8)
    # Both acts share slope 1.0 but differ in offset -> 2 segments.
    assert len(segs) == 2
    assert all(abs(s.fit.slope - 1.0) < 0.02 for s in segs)


def test_degenerate_zero_x_variance():
    pts = np.c_[np.full(5, 10.0), np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    fit = fit_ransac_line(pts)
    assert fit.confidence == 0.0
