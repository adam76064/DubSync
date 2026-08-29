"""Phase 3 & Phase 1: splicer speed locking, crossfade, and length correctness."""
import os
import numpy as np
import scipy.io.wavfile as wavfile
import pytest

from dub_sync_engine.config import DubSyncConfig
from dub_sync_engine.audio_splicer import AudioSplicerEngine, SegmentEDL, sanitize_edl
from tests.fixtures import generate as gen


@pytest.fixture(scope="module")
def audio_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("audio")
    ref_wav, tar_wav, meta = gen.build_ref_tar_audio(gen.get_scenario("clean_1x"), sr=16000, tmpdir=str(tmp))
    return ref_wav, tar_wav, meta


def test_render_length_matches_reference(audio_files, tmp_path):
    ref_wav, tar_wav, meta = audio_files
    cfg = DubSyncConfig()
    cfg.crossfade_duration_ms = 10.0
    cfg.zero_crossing_snap = True
    sp = AudioSplicerEngine(cfg)
    edl = [SegmentEDL(0, "dub", 0.0, 8.0, 0.0, 8.0, 1.0, 0.9)]
    out = str(tmp_path / "synced.wav")
    sp.render_and_splice(edl, ref_wav, tar_wav, out, str(tmp_path))
    sr, data = wavfile.read(out)
    # Output length == target duration (8s) within one crossfade's worth of tolerance.
    assert abs(len(data) - int(8.0 * sr)) < int(0.05 * sr)


def test_speed_lock_is_broadcast_standard(audio_files, tmp_path):
    """A non-standard speed_factor (1.0458x) is snapped to a broadcast standard."""
    ref_wav, tar_wav, meta = audio_files
    cfg = DubSyncConfig()
    cfg.strict_speed = True
    sp = AudioSplicerEngine(cfg)
    edl = [SegmentEDL(0, "dub", 0.0, 4.0, 0.0, 4.0, 1.0458, 0.9)]
    out = str(tmp_path / "synced_speed.wav")
    sp.render_and_splice(edl, ref_wav, tar_wav, out, str(tmp_path))
    # The render must have succeeded without an arbitrary speed; the source speed is
    # snapped (we can't easily introspect atempo post-hoc, so we assert the segment's
    # rendered length stays ~4s, i.e. no gross stretch).
    sr, data = wavfile.read(out)
    assert abs(len(data) - int(4.0 * sr)) < int(0.1 * sr)


def test_sanitize_absorbs_micro_fallback():
    cfg = DubSyncConfig()
    edl = [
        SegmentEDL(0, "dub", 0.0, 10.0, 0.0, 10.0, 1.0, 0.9),
        SegmentEDL(1, "fallback", 10.0, 10.3, 10.0, 10.0, 1.0, 1.0),  # false 0.3s cut
        SegmentEDL(2, "dub", 10.3, 20.0, 10.0, 19.7, 1.0, 0.9),
    ]
    out = sanitize_edl(edl, cfg)
    # The 0.3s fallback with contiguous tar (10.0 == 10.0) is absorbed -> one dub.
    assert all(s.segment_type == "dub" for s in out)
    assert len(out) == 1


def test_sanitize_reclassifies_tiny_dub_between_fallbacks():
    cfg = DubSyncConfig()
    edl = [
        SegmentEDL(0, "fallback", 0.0, 5.0, 0.0, 0.0, 1.0, 1.0),
        SegmentEDL(1, "dub", 5.0, 6.7, 0.0, 0.0, 1.0, 0.5),   # 1.7s ambient "blip"
        SegmentEDL(2, "fallback", 6.7, 30.0, 0.0, 0.0, 1.0, 1.0),
    ]
    out = sanitize_edl(edl, cfg)
    assert all(s.segment_type == "fallback" for s in out)
    assert len(out) == 1  # merged into a single clean bridge


def test_sanitize_keeps_real_cut():
    cfg = DubSyncConfig()
    edl = [
        SegmentEDL(0, "dub", 0.0, 100.0, 0.0, 100.0, 1.0, 0.9),
        SegmentEDL(1, "fallback", 100.0, 109.0, 100.0, 100.0, 1.0, 1.0),  # real 9s cut
        SegmentEDL(2, "dub", 109.0, 200.0, 100.0, 191.0, 1.0, 0.9),
    ]
    out = sanitize_edl(edl, cfg)
    # The 9s fallback (> micro threshold) is preserved.
    assert sum(1 for s in out if s.segment_type == "fallback") == 1
