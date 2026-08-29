"""
Unit tests for the block segmentation / EDL math.

These are the parts of the engine that are correct — they are pinned here so a
refactor cannot silently break them. No media required.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dub_sync_engine.block_segmenter import BlockSegmenterEngine  # noqa: E402
from dub_sync_engine.config import DubSyncConfig  # noqa: E402
from dub_sync_engine.visual_anchors import AnchorMatch  # noqa: E402

CFG = DubSyncConfig()
BS = BlockSegmenterEngine(CFG)

PAL = 24.0 / 25.0          # 0.96  - PAL slowdown
INV_PAL = 25.0 / 24.0      # 1.041667


def M(rt, tt, c=0.9, hd=3):
    return AnchorMatch(ref_idx=0, tar_idx=0, ref_time=rt, tar_time=tt,
                       hash_dist=hd, confidence=c, offset=tt - rt)


def edl_for(matches, ref_dur, tar_dur):
    blocks = BS.cluster_into_blocks(ref_dur, tar_dur, matches)
    return BS.build_macro_edl(ref_dur, tar_dur, blocks, matches=matches)


def total_ref_coverage(edl):
    return sum(s.ref_duration for s in edl)


# ----------------------------------------------------------- slope estimation
@pytest.mark.parametrize("matches,expected", [
    ([M(0, 0), M(10, 10), M(20, 20), M(30, 30)], 1.0),
    ([M(0, 0), M(10, 9.6), M(20, 19.2), M(30, 28.8)], PAL),
    ([M(0, 0), M(10, 10.41667), M(20, 20.8333)], INV_PAL),
])
def test_global_slope_detects_broadcast_standards(matches, expected):
    assert BS.calibrate_global_slope(matches) == pytest.approx(expected, abs=1e-4)


def test_global_slope_ignores_short_and_implausible_intervals():
    # 1s gaps (below the 3s minimum) and a 2x speed ratio are both rejected
    assert BS.calibrate_global_slope([M(0, 0), M(1, 1), M(2, 2)]) == 1.0
    assert BS.calibrate_global_slope([M(0, 0), M(10, 20)]) == 1.0


# -------------------------------------------------------------- EDL invariants
def test_edl_covers_the_whole_reference_timeline():
    """The EDL must be contiguous from 0.0 to ref_duration - no holes, no overlap."""
    cases = [
        ([M(0, 0), M(10, 10), M(20, 20), M(30, 30)], 60.0, 60.0),
        ([M(12, 12.5), M(24, 25.0), M(36, 31.25), M(48, 43.75)], 60.0, 56.25),
        ([M(10, 0), M(20, 10), M(30, 20)], 60.0, 50.0),      # missing opening logo
    ]
    for matches, ref_dur, tar_dur in cases:
        edl = edl_for(matches, ref_dur, tar_dur)
        assert total_ref_coverage(edl) == pytest.approx(ref_dur, abs=0.05), edl
        assert edl[0].ref_start == pytest.approx(0.0, abs=1e-6)
        assert edl[-1].ref_end == pytest.approx(ref_dur, abs=0.05)
        for prev, nxt in zip(edl[:-1], edl[1:]):
            assert nxt.ref_start == pytest.approx(prev.ref_end, abs=1e-3)


def test_dub_slice_duration_matches_reference_at_detected_speed():
    """Each dub segment must stretch its source slice to exactly fill the ref gap."""
    matches = [M(0, 0), M(10, 9.6), M(20, 19.2), M(30, 28.8)]
    edl = edl_for(matches, 60.0, 57.6)
    for s in edl:
        if s.segment_type == "dub":
            # tar source length * playback stretch == reference length
            assert s.tar_duration == pytest.approx(s.ref_duration * PAL, abs=0.01)


def test_omission_at_end_of_interval_is_bridged():
    """Dub present for ref[0,25), then a 5s cut -> one fallback of ~5s."""
    matches = [M(0, 0), M(10, 10), M(20, 20), M(30, 25), M(40, 35), M(50, 45)]
    edl = edl_for(matches, 60.0, 55.0)
    fallbacks = [s for s in edl if s.segment_type == "fallback"]
    assert len(fallbacks) == 1
    assert fallbacks[0].ref_start == pytest.approx(25.0, abs=0.1)
    assert fallbacks[0].ref_duration == pytest.approx(5.0, abs=0.1)


def test_missing_opening_logo_bridges_the_head():
    matches = [M(10, 0), M(20, 10), M(30, 20)]
    edl = edl_for(matches, 60.0, 50.0)
    assert edl[0].segment_type == "fallback"
    assert edl[0].ref_start == pytest.approx(0.0, abs=1e-6)
    assert edl[0].ref_duration == pytest.approx(10.0, abs=0.1)


def test_adjacent_contiguous_dub_segments_are_merged():
    matches = [M(0, 0), M(10, 10), M(20, 20), M(30, 30), M(40, 40)]
    edl = edl_for(matches, 60.0, 60.0)
    assert len(edl) == 1, f"expected one merged segment, got {len(edl)}"
    assert edl[0].segment_type == "dub"


def test_no_anchors_degrades_to_a_single_segment():
    edl = BS.build_macro_edl(60.0, 55.0, BS.cluster_into_blocks(60.0, 55.0, []), matches=[])
    assert len(edl) == 1
    assert total_ref_coverage(edl) == pytest.approx(60.0, abs=0.05)


def test_segment_ids_are_contiguous_from_zero():
    matches = [M(12, 12.5), M(24, 25.0), M(36, 31.25), M(48, 43.75)]
    edl = edl_for(matches, 60.0, 56.25)
    assert [s.seg_id for s in edl] == list(range(len(edl)))


def test_speed_factor_stays_within_playable_bounds():
    matches = [M(0, 0), M(10, 4), M(20, 8)]     # absurd 0.4x ratio
    edl = edl_for(matches, 60.0, 24.0)
    for s in edl:
        assert 0.90 <= s.speed_factor <= 1.10
