"""Dense sync-path estimator: measure the true (ref->tar) path and recover
re-cut structure (mid-episode cut + tail trim + speed) instead of assuming one
continuous speed."""
import numpy as np

from dub_sync_engine.config import DubSyncConfig
from dub_sync_engine.path_estimator import SyncPathEstimator, estimate_speed_ratio
from tests.fixtures import generate as gen


def _recut_audio(tmp_path, sr=16000, seed=7, duration=100.0,
                 cut_at=30.0, cut_len=2.0, tail_at=60.0):
    """ref is distinctive audio; tar = ref with a `cut_len` cut at `cut_at` and
    everything after `tail_at` trimmed."""
    ref = gen.make_distinctive_audio(duration, sr, seed)
    chunks = [ref[: int(cut_at * sr)], ref[int((cut_at + cut_len) * sr): int(tail_at * sr)]]
    tar = np.concatenate(chunks)
    rw = str(tmp_path / "ref.wav")
    tw = str(tmp_path / "tar.wav")
    gen.write_wav(rw, ref, sr)
    gen.write_wav(tw, tar, sr)
    return rw, tw, ref, tar


def test_estimate_speed_recovers_ratio():
    rng = np.random.default_rng(0)
    n = 40000
    ref = np.abs(np.cumsum(rng.standard_normal(n))).astype(np.float32)
    for ratio in (0.96, 1.0, 1.02):
        tar = np.concatenate([ref[: int(n * 0.3)], ])  # placeholder to keep shape
        # resample target to `ratio` length
        from scipy.signal import resample
        tar = resample(ref, int(n * ratio)).astype(np.float32)
        est = estimate_speed_ratio(ref[:, None], tar[:, None], 50.0)
        assert abs(est - ratio) < 0.02, f"ratio={ratio} est={est}"


def test_recut_recovers_cut_and_tail(tmp_path):
    """A 2s cut at 30s + tail trim at 60s must yield 2 segments, correct offsets,
    and NOT cover the trimmed tail (ref >= 60)."""
    rw, tw, ref, tar = _recut_audio(tmp_path)
    est = SyncPathEstimator(DubSyncConfig())
    segs = est.extract_path(rw, tw, 100.0, len(tar) / 16000)

    assert len(segs) == 2, f"expected 2 segments, got {len(segs)}"

    s1, s2 = segs
    # Segment 1: offset 0 (slope 1.0), before the cut.
    assert abs(s1.slope - 1.0) < 0.03, f"s1 slope {s1.slope}"
    assert abs(s1.intercept) < 0.5, f"s1 intercept {s1.intercept}"
    assert s1.ref_start == 0.0
    # Segment 2: offset -2 (tar = ref - 2), after the cut.
    assert abs(s2.slope - 1.0) < 0.03, f"s2 slope {s2.slope}"
    assert abs(s2.intercept - (-2.0)) < 0.6, f"s2 intercept {s2.intercept}"

    # The cut is detected between the two segments (within tolerance of 30s).
    cut_pos = (s1.ref_end + s2.ref_start) / 2.0
    assert 20.0 <= cut_pos <= 38.0, f"cut at {cut_pos}"

    # Tail trim: the last segment ends near 60s (the content end), NOT 100s.
    assert 55.0 <= s2.ref_end <= 62.0, f"tail end {s2.ref_end}"

    # No segment covers the trimmed tail (ref > 60 maps to nothing).
    assert s2.ref_end <= 62.0


def test_build_edl_marks_cut_and_tail_as_fallback(tmp_path):
    """The measured path must become dub segments for synced regions and
    fallback segments for the cut (ref 30-32) and the tail trim (ref 60-100)."""
    rw, tw, ref, tar = _recut_audio(tmp_path)
    est = SyncPathEstimator(DubSyncConfig())
    segs = est.extract_path(rw, tw, 100.0, len(tar) / 16000)
    edl = est.build_edl(segs, 100.0, len(tar) / 16000)

    dubs = [s for s in edl if s.segment_type == "dub"]
    fallbacks = [s for s in edl if s.segment_type == "fallback"]

    assert len(dubs) == 2, f"expected 2 dub segments, got {len(dubs)}"
    assert len(fallbacks) >= 1, f"expected fallback for the cut + tail"

    # The tail trim (ref > 60) must be represented as a fallback, not force-dubbed.
    tail = [f for f in fallbacks if f.ref_start >= 55.0]
    assert tail, "tail trim not marked as fallback"

    # The total reference timeline must be fully covered by dub + fallback.
    assert edl[0].ref_start == 0.0
    assert abs(edl[-1].ref_end - 100.0) < 0.1


def test_clean_episode_is_one_segment(tmp_path):
    """No cuts, no trim -> a single segment covering the whole episode."""
    sr = 16000
    ref = gen.make_distinctive_audio(80.0, sr, seed=3)
    rw = str(tmp_path / "ref.wav")
    tw = str(tmp_path / "tar.wav")
    gen.write_wav(rw, ref, sr)
    gen.write_wav(tw, ref, sr)  # target == reference

    est = SyncPathEstimator(DubSyncConfig())
    segs = est.extract_path(rw, tw, 80.0, 80.0)
    assert len(segs) == 1, f"expected 1 segment, got {len(segs)}"
    s = segs[0]
    assert abs(s.slope - 1.0) < 0.03
    assert abs(s.intercept) < 0.5
    assert s.ref_start == 0.0
    assert s.ref_end == 80.0
