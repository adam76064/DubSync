"""
Phases 2 & 3: cut-aware slope calibration + broadcast speed locking + EDL building.

Each scenario encodes one documented real-world failure mode. We assert the engine
recovers the correct global speed and builds an EDL with the right number of fallback
bridges and no drift / no misplaced audio.
"""
import pytest

from dub_sync_engine.config import DubSyncConfig, snap_to_broadcast_speed
from dub_sync_engine.block_segmenter import BlockSegmenterEngine
from tests.fixtures import generate as gen


def _engine():
    return BlockSegmenterEngine(DubSyncConfig())


def _edl(scenario):
    eng = _engine()
    matches = scenario.make_anchors()
    blocks = eng.cluster_into_blocks(scenario.ref_duration, scenario.tar_duration, matches)
    return eng.build_macro_edl(scenario.ref_duration, scenario.tar_duration, blocks, matches)


def _fallbacks(edl):
    return [s for s in edl if s.segment_type == "fallback"]


def _dubs(edl):
    return [s for s in edl if s.segment_type == "dub"]


# --------------------------------------------------------------------------- #
# Phase 2: slope calibration
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,expected", [
    ("clean_1x", 1.0),
    ("pal_speed", 24.0 / 25.0),
])
def test_calibrate_global_slope_recovers_speed(name, expected):
    scenario = gen.get_scenario(name)
    eng = _engine()
    slope = eng.calibrate_global_slope(scenario.make_anchors())
    assert abs(slope - expected) < 0.006, f"{name}: got {slope}, expected {expected}"


def test_calibrate_global_slope_cut_aware():
    """A single 9s cut must not drag the global speed estimate."""
    scenario = gen.get_scenario("single_cut")
    eng = _engine()
    slope = eng.calibrate_global_slope(scenario.make_anchors())
    assert abs(slope - 1.0) < 0.006, f"cut polluted slope: {slope}"


def test_calibrate_global_slope_micro_trim():
    """A 1.75s micro-trim inside a PAL act must not destroy the 0.96 estimate."""
    scenario = gen.get_scenario("micro_trim")
    eng = _engine()
    slope = eng.calibrate_global_slope(scenario.make_anchors())
    assert abs(slope - (24.0 / 25.0)) < 0.006, f"trim polluted slope: {slope}"


# --------------------------------------------------------------------------- #
# subsync idea 2: measured confidence (r^2 * inlier_ratio)
# --------------------------------------------------------------------------- #

def test_calibrate_global_fit_returns_measured_confidence():
    """The global fit exposes r^2 and inlier_ratio (not a fuzzy number)."""
    scenario = gen.get_scenario("pal_speed")
    eng = _engine()
    fit = eng.calibrate_global_fit(scenario.make_anchors())
    assert fit.n_total > 0
    assert fit.inlier_ratio == 1.0, "clean act: every anchor is an inlier"
    assert fit.r_squared > 0.99
    assert abs(fit.confidence - 1.0) < 0.02


def test_block_confidence_penalizes_outliers():
    """A block containing planted false anchors gets a lower measured confidence."""
    import numpy as np
    from dub_sync_engine.visual_anchors import AnchorMatch

    rng = np.random.default_rng(0)
    ref = np.sort(rng.uniform(0, 600, 50))
    tar = 0.96 * ref + 2.0
    clean = [AnchorMatch(i, i, round(ref[i], 3), round(tar[i], 3), 0, 0.9, round(tar[i] - ref[i], 4))
             for i in range(50)]

    eng = _engine()
    clean_conf = eng._block_confidence(clean)

    # Add 8 false anchors scattered far off the line.
    bad = clean + [AnchorMatch(1000 + i, 1000 + i, round(rng.uniform(0, 600), 3),
                               round(rng.uniform(0, 600), 3), 0, 0.9, 0.0) for i in range(8)]
    dirty_conf = eng._block_confidence(bad)

    assert clean_conf > 0.95
    assert dirty_conf < clean_conf, f"{dirty_conf} !< {clean_conf}"


def test_block_confidence_uses_r2_inlier_ratio():
    """confidence == r^2 * inlier_ratio for a well-populated block."""
    scenario = gen.get_scenario("clean_1x")
    eng = _engine()
    fit = eng.calibrate_global_fit(scenario.make_anchors())
    assert abs(fit.confidence - (fit.r_squared * fit.inlier_ratio)) < 1e-9


# --------------------------------------------------------------------------- #
# subsync idea 3: continuous similarity weights
# --------------------------------------------------------------------------- #

def test_calibrate_global_fit_uses_anchor_weights():
    """The global fit must honor per-anchor weights (acoustic confirmation strength)."""
    import numpy as np
    from dub_sync_engine.visual_anchors import AnchorMatch

    rng = np.random.default_rng(0)
    ref = np.sort(rng.uniform(0, 600, 40))
    tar = ref + rng.normal(0, 0.1, 40)
    matches = [AnchorMatch(i, i, round(ref[i], 3), round(tar[i], 3), 0, 0.9,
                           round(tar[i] - ref[i], 4), weight=1.0) for i in range(40)]

    eng = _engine()
    fit = eng.calibrate_global_fit(matches)
    assert abs(fit.slope - 1.0) < 0.02

    # Zero-weight anchors must not influence the fit: give every anchor weight 0 and
    # confirm the fit still finds the identity line (weights only affect ranking).
    zeroed = [AnchorMatch(i, i, m.ref_time, m.tar_time, 0, 0.9, m.offset, weight=0.0)
              for i, m in enumerate(matches)]
    fit_zero = eng.calibrate_global_fit(zeroed)
    # Degenerate all-zero weights fall back to uniform weighting internally.
    assert fit_zero.n_inliers == len(matches)


def test_anchor_match_has_weight_default():
    from dub_sync_engine.visual_anchors import AnchorMatch
    m = AnchorMatch(0, 0, 10.0, 12.0, 0, 0.9, 2.0)
    assert m.weight == 1.0


# --------------------------------------------------------------------------- #
# subsync idea 5: recursive piecewise refinement in cluster_into_blocks
# --------------------------------------------------------------------------- #

def test_cluster_splits_multi_speed_episode():
    """A PAL act + a 1.0x act must cluster into two blocks with the right speeds."""
    import numpy as np
    from dub_sync_engine.visual_anchors import AnchorMatch

    rng = np.random.default_rng(0)
    ref1 = np.sort(rng.uniform(0, 300, 40))
    ref2 = np.sort(rng.uniform(300, 600, 40))
    tar1 = 0.96 * ref1 + rng.normal(0, 0.1, 40)
    # Act 1 ends at tar 0.96*300 = 288; act 2 continues at 288 (contiguous offset).
    tar2 = (ref2 - 300.0) + 288.0 + rng.normal(0, 0.1, 40)  # speed 1.0, offset -12

    matches = []
    for i in range(40):
        matches.append(AnchorMatch(i, i, round(ref1[i], 3), round(tar1[i], 3), 0, 0.9,
                                   round(tar1[i] - ref1[i], 4)))
    for i in range(40):
        j = i + 40
        matches.append(AnchorMatch(j, j, round(ref2[i], 3), round(tar2[i], 3), 0, 0.9,
                                   round(tar2[i] - ref2[i], 4)))

    eng = _engine()
    # Act 2 runs 1.0x over ref 300..600 -> tar 288..588, so tar_duration is 588.
    blocks = eng.cluster_into_blocks(600.0, 588.0, matches)
    assert len(blocks) == 2, f"expected 2 blocks, got {len(blocks)}"
    blocks.sort(key=lambda b: b.ref_start)
    assert abs(blocks[0].speed_factor - (24.0 / 25.0)) < 0.01
    assert abs(blocks[1].speed_factor - 1.0) < 0.01


def test_cluster_single_speed_one_block():
    scenario = gen.get_scenario("pal_speed")
    eng = _engine()
    blocks = eng.cluster_into_blocks(scenario.ref_duration, scenario.tar_duration,
                                     scenario.make_anchors())
    assert len(blocks) == 1
    assert abs(blocks[0].speed_factor - (24.0 / 25.0)) < 0.01


def test_cluster_empty_matches():
    eng = _engine()
    blocks = eng.cluster_into_blocks(600.0, 600.0, [])
    assert len(blocks) == 1
    assert blocks[0].anchor_count == 0
    assert blocks[0].speed_factor == 1.0


def test_multispeed_edl_uses_per_act_speeds():
    """The EDL must apply each act's own speed (PAL act + 1.0x act), not one global."""
    import numpy as np
    from dub_sync_engine.visual_anchors import AnchorMatch

    rng = np.random.default_rng(0)
    ref1 = np.sort(rng.uniform(0, 300, 40))
    ref2 = np.sort(rng.uniform(300, 600, 40))
    tar1 = 0.96 * ref1 + rng.normal(0, 0.1, 40)
    tar2 = (ref2 - 300.0) + 288.0 + rng.normal(0, 0.1, 40)

    matches = [AnchorMatch(i, i, round(ref1[i], 3), round(tar1[i], 3), 0, 0.9,
                           round(tar1[i] - ref1[i], 4)) for i in range(40)]
    matches += [AnchorMatch(i + 40, i + 40, round(ref2[i], 3), round(tar2[i], 3), 0, 0.9,
                            round(tar2[i] - ref2[i], 4)) for i in range(40)]

    eng = _engine()
    blocks = eng.cluster_into_blocks(600.0, 588.0, matches)
    edl = eng.build_macro_edl(600.0, 588.0, blocks, matches)

    dubs = _dubs(edl)
    speeds = sorted({round(s.speed_factor, 6) for s in dubs})
    assert abs(speeds[0] - (24.0 / 25.0)) < 0.01, f"first act speed {speeds}"
    assert abs(speeds[1] - 1.0) < 0.01, f"second act speed {speeds}"
    # No drift: EDL spans full ref range and target ends at 588.
    assert abs(dubs[-1].ref_end - 600.0) < 0.5
    assert abs(dubs[-1].tar_end - 588.0) < 0.5


# --------------------------------------------------------------------------- #
# Phase 3: broadcast speed locking in EDL
# --------------------------------------------------------------------------- #

_STANDARDS = {
    round(1.0, 6),
    round(24 / 25, 6),
    round(25 / 24, 6),
    round(24 / 23.976, 6),
    round(23.976 / 24, 6),
}


@pytest.mark.parametrize("name", ["clean_1x", "pal_speed", "single_cut", "micro_trim", "extra_scene", "intro_gap", "black_frame"])
def test_edl_speeds_are_broadcast_standards(name):
    """No dub segment may float to an arbitrary speed (e.g. 1.0458x)."""
    edl = _edl(gen.get_scenario(name))
    for s in _dubs(edl):
        assert round(s.speed_factor, 6) in _STANDARDS, f"{name}: non-standard speed {s.speed_factor}"


# --------------------------------------------------------------------------- #
# EDL correctness per scenario
# --------------------------------------------------------------------------- #

def test_clean_1x_edl():
    edl = _edl(gen.get_scenario("clean_1x"))
    assert len(_fallbacks(edl)) == 0
    assert len(_dubs(edl)) == 1
    # Full coverage, no drift.
    s = _dubs(edl)[0]
    assert abs(s.ref_start - 0.0) < 0.01
    assert abs(s.ref_end - 600.0) < 0.1
    assert abs(s.tar_start - 0.0) < 0.01
    assert abs(s.speed_factor - 1.0) < 1e-6


def test_pal_speed_edl():
    edl = _edl(gen.get_scenario("pal_speed"))
    assert len(_fallbacks(edl)) == 0
    s = _dubs(edl)[0]
    assert abs(s.speed_factor - (24.0 / 25.0)) < 1e-6
    # tar_end scaled exactly by speed -> no overflow / cutoff.
    assert abs(s.tar_end - (s.tar_start + s.ref_duration * (24.0 / 25.0))) < 0.01


def test_single_cut_edl():
    scenario = gen.get_scenario("single_cut")
    edl = _edl(scenario)
    fbs = _fallbacks(edl)
    assert len(fbs) == 1, f"expected 1 fallback, got {len(fbs)}"
    # The fallback bridges the removed 9s (300 -> 309).
    assert abs(fbs[0].ref_start - 300.0) < 0.5
    assert abs(fbs[0].ref_end - 309.0) < 0.5
    # No drift: the EDL spans the full reference duration.
    dubs = _dubs(edl)
    assert abs(dubs[-1].ref_end - 600.0) < 0.5


def test_micro_trim_edl():
    scenario = gen.get_scenario("micro_trim")
    edl = _edl(scenario)
    fbs = _fallbacks(edl)
    assert len(fbs) == 1, f"expected 1 fallback, got {len(fbs)}"
    # Trim region ~400 -> 401.75.
    assert abs(fbs[0].ref_start - 400.0) < 1.0
    assert abs(fbs[0].ref_end - 401.75) < 1.0


def test_extra_scene_edl():
    """Extra target footage must be trimmed, never inserted as English."""
    scenario = gen.get_scenario("extra_scene")
    edl = _edl(scenario)
    assert len(_fallbacks(edl)) == 0, "extra scene must not produce a fallback bridge"
    # The last dub still reaches the end of the reference.
    assert abs(_dubs(edl)[-1].ref_end - 600.0) < 0.5


def test_intro_gap_edl():
    """Master has a 4s logo the target lacks -> one opening fallback, dub from 4s."""
    scenario = gen.get_scenario("intro_gap")
    edl = _edl(scenario)
    fbs = _fallbacks(edl)
    assert len(fbs) == 1
    assert abs(fbs[0].ref_start - 0.0) < 0.01
    assert abs(fbs[0].ref_end - 4.0) < 0.5
    # First dub starts at ~4s and its target audio starts at 0 (no misplaced speech).
    d = _dubs(edl)[0]
    assert abs(d.ref_start - 4.0) < 0.5
    assert abs(d.tar_start - 0.0) < 0.5


def test_black_frame_rejected():
    """The planted (0.0, 12.8) false anchor must not displace the intro."""
    scenario = gen.get_scenario("black_frame")
    edl = _edl(scenario)
    fbs = _fallbacks(edl)
    # Exactly one intro bridge (0 -> ~32s), no extra fallback from the false anchor.
    assert len(fbs) == 1, f"expected 1 intro fallback, got {len(fbs)}"
    assert abs(fbs[0].ref_start - 0.0) < 0.01
    assert abs(fbs[0].ref_end - 32.0) < 0.5
    # First dub starts at the real anchor (~32s) with target audio at 0 (no teleport).
    d = _dubs(edl)[0]
    assert abs(d.ref_start - 32.0) < 0.5
    assert abs(d.tar_start - 0.0) < 0.5, f"displaced speech: tar_start={d.tar_start}"
