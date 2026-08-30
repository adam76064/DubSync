"""Diagnostics honesty: the consensus engine must expose *why* anchors are sparse
(raw visual matches found vs. gated in/out) so a forensic report can be diagnosed
without the source videos."""
import numpy as np
import scipy.signal

from dub_sync_engine.config import DubSyncConfig
from dub_sync_engine.consensus_engine import MultiModalConsensusEngine
from dub_sync_engine.visual_anchors import AnchorMatch, VisualAnchor


def _anchor(t):
    return VisualAnchor(index=int(t * 24), pts_time=t, image_path="", phash=None,
                        dhash=None, color_hist=np.zeros(1))


def test_diagnostics_initialized_and_reset_on_empty():
    eng = MultiModalConsensusEngine(DubSyncConfig())
    assert eng.last_diagnostics == {}

    # No anchors and no acoustic signal (nonexistent wavs -> steps fail gracefully).
    out = eng.discover_consensus_anchors([], [], "/no/ref.wav", "/no/tar.wav", 100.0, 100.0)
    assert out == []
    assert eng.last_diagnostics["raw_visual_matches_found"] == 0
    assert eng.last_diagnostics["visual_matches_gated_in"] == 0
    assert eng.last_diagnostics["visual_matches_gated_out"] == 0


def test_visual_matches_all_gated_out_when_no_acoustic_spine(monkeypatch):
    """With raw visual matches but zero acoustic candidates, every visual match is
    gated OUT and the diagnostic must say so (never silently claim success)."""
    eng = MultiModalConsensusEngine(DubSyncConfig())

    fake_matches = [
        AnchorMatch(ref_idx=0, tar_idx=0, ref_time=10.0, tar_time=9.6, hash_dist=0.0,
                    confidence=0.9, offset=-0.4, source="visual"),
        AnchorMatch(ref_idx=1, tar_idx=1, ref_time=20.0, tar_time=19.2, hash_dist=0.0,
                    confidence=0.9, offset=-0.8, source="visual"),
    ]
    monkeypatch.setattr(eng.visual_engine, "match_anchors", lambda r, t: fake_matches)

    ref_anchors = [_anchor(10.0), _anchor(20.0)]
    tar_anchors = [_anchor(9.6), _anchor(19.2)]

    out = eng.discover_consensus_anchors(
        ref_anchors, tar_anchors, "/no/ref.wav", "/no/tar.wav", 100.0, 100.0
    )

    d = eng.last_diagnostics
    assert d["raw_visual_matches_found"] == 2
    assert d["visual_matches_gated_in"] == 0
    assert d["visual_matches_gated_out"] == 2
    # No acoustic spine -> nothing admitted -> empty result.
    assert out == []


def test_estimate_global_speed_recovers_ratio():
    """The speed estimator must recover the true target/ref ratio (not assume
    0.96). A VFR 24.17fps source is ~1.01, not 0.96 — a hardcoded assumption
    smears every correlation peak."""
    eng = MultiModalConsensusEngine(DubSyncConfig())
    rng = np.random.default_rng(5)
    n = 40000  # ~800s at 50Hz bin rate
    ref = np.abs(np.cumsum(rng.standard_normal(n))).astype(np.float32)
    for ratio, label in ((0.96, "PAL"), (1.02, "VFR-ish"), (1.0, "native")):
        tar = scipy.signal.resample(ref, int(n * ratio)).astype(np.float32)
        est = eng._estimate_global_speed(ref, tar, 50.0)
        assert abs(est - ratio) < 0.02, f"{label}: est={est} ratio={ratio}"
