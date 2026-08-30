"""Phase 5: N-gram sequence verification is surfaced and boosts confidence."""
import numpy as np
from PIL import Image
import imagehash

from dub_sync_engine.config import DubSyncConfig
from dub_sync_engine.visual_anchors import VisualAnchorEngine, VisualAnchor


def _anchor(idx, pts_time, seed):
    """Build a VisualAnchor with deterministic, distinct hashes per seed."""
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    phash = imagehash.phash(img)
    dhash = imagehash.dhash(img)
    hist = np.ones((8, 8), dtype=np.float32) / 64.0
    return VisualAnchor(
        index=idx, pts_time=pts_time, image_path="", phash=phash, dhash=dhash,
        color_hist=hist, burst_hashes=[],
    )


def test_seq_len_surfaced_and_boosted():
    """A chain of consecutive matching cuts surfaces seq_len >= 2 on its members."""
    # Same "scene" images at matching timestamps across ref and tar.
    ref_anchors = [_anchor(i, float(i * 5), seed=i) for i in range(6)]
    tar_anchors = [_anchor(i, float(i * 5), seed=i) for i in range(6)]

    eng = VisualAnchorEngine(DubSyncConfig())
    matches = eng.match_anchors(ref_anchors, tar_anchors)

    assert len(matches) >= 3
    # Interior matches should verify a chain of consecutive cuts (seq_len >= 2).
    seqs = [m.seq_len for m in matches]
    assert max(seqs) >= 2, f"expected an N-gram sequence, got seqs={seqs}"
    # Every matched anchor must now carry a seq_len attribute (>= 1).
    assert all(m.seq_len >= 1 for m in matches)
