"""
Executable specs for the defects documented in REVIEW.md.

Every test in this file is marked ``xfail(strict=False)``: it describes
behaviour the engine *should* have, and currently does not. They will flip to
XPASS as each finding is fixed — at which point the marker should be removed.

Run with:  python3 -m pytest tests/test_known_defects.py -q -rX
"""
import ast
import inspect
import re
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dub_sync_engine.audio_splicer import AudioSplicerEngine  # noqa: E402
from dub_sync_engine.block_segmenter import BlockSegmenterEngine  # noqa: E402
from dub_sync_engine.config import DubSyncConfig, FallbackMode, Preset  # noqa: E402
from dub_sync_engine.consensus_engine import MultiModalConsensusEngine  # noqa: E402
from dub_sync_engine.pipeline import DubSyncPipeline  # noqa: E402
from dub_sync_engine.visual_anchors import AnchorMatch  # noqa: E402
from dub_sync_engine.verifier_engine import ClosedLoopVerifierEngine, SegmentEDL  # noqa: E402

xfail = pytest.mark.xfail(strict=False)


def M(rt, tt, c=0.9, hd=3):
    return AnchorMatch(ref_idx=0, tar_idx=0, ref_time=rt, tar_time=tt,
                       hash_dist=hd, confidence=c, offset=tt - rt)


# ------------------------------------------------------------------ F1
@xfail(reason="F1: audit metrics are hard-coded literals (verifier_engine.py:160-162)")
def test_audit_metrics_are_measured_not_hardcoded(tmp_path):
    """Two very different timelines must not produce identical audit numbers."""
    import numpy as np
    from scipy.io import wavfile

    rng = np.random.default_rng(3)
    for name in ("ref.wav", "tar.wav"):
        wavfile.write(str(tmp_path / name), 48000,
                      (rng.standard_normal(48000 * 5) * 8000).astype(np.int16))

    verifier = ClosedLoopVerifierEngine(DubSyncConfig())

    good = [SegmentEDL(0, "dub", 0.0, 30.0, 0.0, 30.0, 1.0, 0.95),
            SegmentEDL(1, "dub", 30.0, 60.0, 30.0, 60.0, 1.0, 0.95)]
    bad = [SegmentEDL(0, "dub", 0.0, 30.0, 5.0, 35.0, 1.0, 0.95),
           SegmentEDL(1, "fallback", 30.0, 45.0, 35.0, 35.0, 1.0, 1.0),
           SegmentEDL(2, "dub", 45.0, 60.0, 45.0, 60.0, 1.0, 0.95)]

    _, audit_good = verifier.audit_and_heal_edl(
        good, str(tmp_path / "ref.wav"), str(tmp_path / "tar.wav"), 60.0, 60.0)
    _, audit_bad = verifier.audit_and_heal_edl(
        bad, str(tmp_path / "ref.wav"), str(tmp_path / "tar.wav"), 60.0, 60.0)

    assert (audit_good.mean_alignment_error_ms, audit_good.passed_windows_pct) != \
           (audit_bad.mean_alignment_error_ms, audit_bad.passed_windows_pct), \
        "audit reported identical numbers for a good and a badly mis-synced timeline"


# ------------------------------------------------------------- F2 / F7
@xfail(reason="F2/F7/N1-N3: code touches attributes that the config/enums do not define")
def test_core_types_define_every_attribute_the_code_touches():
    """
    Guards the whole class of 'attribute does not exist' bugs.

    Every one of these is a live AttributeError on some code path:
      consensus_engine.py:51  config.fps_ratio        (silently False -> g_speed=0.96)
      verifier_engine.py:135  config.verifier_corr_min (CRASH, latest commit)
      micro_dtw.py:234        config.broadcast_snap()  (CRASH, latest commit)
      audio_splicer.py:223    FallbackMode.ADAPTIVE    (CRASH, latest commit)
    """
    pkg = os.path.join(ROOT, "dub_sync_engine")

    # name -> live instance/class to check the attribute against
    targets = {
        "config": DubSyncConfig(),
        "FallbackMode": FallbackMode,
        "Preset": Preset,
    }

    missing = {}
    for fname in sorted(f for f in os.listdir(pkg) if f.endswith(".py")):
        tree = ast.parse(open(os.path.join(pkg, fname)).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            v = node.value
            # self.config.X  /  config.X  /  FallbackMode.X  /  Preset.X
            owner = None
            if isinstance(v, ast.Name):
                owner = targets.get(v.id)
            elif isinstance(v, ast.Attribute) and v.attr == "config":
                owner = targets["config"]
            if owner is None:
                continue
            if not hasattr(owner, node.attr):
                missing.setdefault(f"{fname}:{node.lineno}", node.attr)

    assert not missing, f"attributes touched but never defined: {missing}"


# ------------------------------------------------------------------ F3
@xfail(reason="F3: omitted-scene gaps are always placed at the end of an anchor interval")
def test_cut_gap_is_placed_where_the_dub_actually_stops():
    """
    Anchors at ref 24 and ref 36; the dub is MISSING ref[24,30] (the start of the
    interval), so the M&E bridge belongs at ref[24,30] - not at ref[30,36].
    Ground truth taken from the synthetic fixture: tar 24.0->25.0, 36.0->31.25.
    """
    bs = BlockSegmenterEngine(DubSyncConfig())
    matches = [M(12, 12.5), M(24, 25.0), M(36, 31.25), M(48, 43.75)]
    edl = bs.build_macro_edl(60.0, 56.25,
                             bs.cluster_into_blocks(60.0, 56.25, matches),
                             matches=matches)
    gaps = [s for s in edl if s.segment_type == "fallback"]
    assert gaps, "no omission detected"
    assert gaps[0].ref_start == pytest.approx(24.0, abs=0.5), \
        f"bridge placed at {gaps[0].ref_start}s, expected ~24s"


# ------------------------------------------------------------------ F4
@xfail(reason="F4: AcousticRefineEngine is constructed but never called (pipeline.py:47)")
def test_acoustic_refinement_runs_in_the_pipeline():
    src = inspect.getsource(DubSyncPipeline.execute)
    assert "refine_anchors" in src


@xfail(reason="F4: SpectralFingerprintEngine is constructed but never called (pipeline.py:43)")
def test_spectral_tier_is_reachable():
    src = inspect.getsource(DubSyncPipeline.execute)
    assert "discover_spectral_anchors" in src


@xfail(reason="F2: consensus falls back to a hard-coded g_speed=0.96")
def test_consensus_speed_is_not_hardcoded():
    src = inspect.getsource(MultiModalConsensusEngine.discover_consensus_anchors)
    assert re.search(r"g_speed\s*=.*0\.9600", src) is None, \
        "g_speed falls back to the literal 0.96 instead of a measured slope"


# ------------------------------------------------------------------ F5
@xfail(reason="F5: README documents a CLI that does not exist")
def test_cli_exposes_the_flags_documented_in_the_readme():
    usage = subprocess.run(
        [sys.executable, "-m", "dub_sync_engine.cli", "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    ).stdout
    for flag in ("--ref", "--tar", "--out", "--matcher-mode", "--report",
                 "studio_ultra"):
        assert flag in usage, f"documented flag {flag!r} missing from the CLI"


# ------------------------------------------------------------------ F6
@xfail(reason="F6: segments are still hard-joined by the concat demuxer; no crossfade")
def test_renderer_crossfades_segment_boundaries():
    """
    The v3.5 rewrite deleted the dead crossfade buffers (good) but still assembles
    the timeline with the ffmpeg concat demuxer, so zero_crossing_snap and
    crossfade_duration_ms remain no-ops and every segment boundary is a hard cut.
    """
    src = inspect.getsource(AudioSplicerEngine)
    assert "concat" not in src or "crossfade" in src.lower(), \
        "timeline is assembled by hard concatenation with no crossfade mixing"


# ----------------------------------------------------------------- F13
@xfail(reason="F13: whole layers are wrapped in bare `except Exception: pass`")
def test_no_silently_swallowed_exceptions():
    pkg = os.path.join(ROOT, "dub_sync_engine")
    offenders = []
    for fname in sorted(f for f in os.listdir(pkg) if f.endswith(".py")):
        tree = ast.parse(open(os.path.join(pkg, fname)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and \
               len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                offenders.append(f"{fname}:{node.lineno}")
    # media_probe.py (optional imageio-ffmpeg import) and visual_anchors.py
    # (per-frame decode guard) are defensible; the consensus layer-wide ones are not.
    assert not offenders, f"silent except-pass blocks: {offenders}"
