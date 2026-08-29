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
