"""Phase 4 & PR #1: verifier healing + real (not hardcoded) metrics."""
import numpy as np
import scipy.io.wavfile as wavfile
import pytest

from dub_sync_engine.config import DubSyncConfig
from dub_sync_engine.verifier_engine import ClosedLoopVerifierEngine
from dub_sync_engine.audio_splicer import SegmentEDL
from tests.fixtures import generate as gen


@pytest.fixture(scope="module")
def continuous_dub(tmp_path_factory):
    """Reference and target share the same music bed, offset by 0.3s (continuous dub)."""
    tmp = tmp_path_factory.mktemp("verifier")
    sr = 16000
    ref = gen.make_cartoon_audio(20.0, sr, seed=1)
    shift = 0.3
    tar = np.zeros_like(ref)
    tar[int(shift * sr):] = ref[:-int(shift * sr)]
    ref_wav = str(tmp / "ref.wav")
    tar_wav = str(tmp / "tar.wav")
    gen.write_wav(ref_wav, ref, sr)
    gen.write_wav(tar_wav, tar, sr)
    return ref_wav, tar_wav, sr


def test_heals_false_fallback(continuous_dub):
    ref_wav, tar_wav, sr = continuous_dub
    cfg = DubSyncConfig()
    ver = ClosedLoopVerifierEngine(cfg)
    edl = [
        SegmentEDL(0, "dub", 0.0, 5.0, 0.3, 5.3, 1.0, 0.9),
        SegmentEDL(1, "fallback", 5.0, 8.0, 0.0, 0.0, 1.0, 1.0),  # FALSE fallback
        SegmentEDL(2, "dub", 8.0, 12.0, 8.3, 12.3, 1.0, 0.9),
    ]
    healed, audit = ver.audit_and_heal_edl(edl, ref_wav, tar_wav, 20.0, 20.0)
    assert audit.false_fallbacks_healed_count == 1
    assert all(s.segment_type == "dub" for s in healed)


def test_metrics_are_measured_not_hardcoded(continuous_dub):
    ref_wav, tar_wav, sr = continuous_dub
    cfg = DubSyncConfig()
    ver = ClosedLoopVerifierEngine(cfg)
    edl = [SegmentEDL(0, "dub", 0.0, 12.0, 0.3, 12.3, 1.0, 0.9)]
    _, audit = ver.audit_and_heal_edl(edl, ref_wav, tar_wav, 20.0, 20.0)
    # With a perfectly aligned continuous dub, error must be ~0, not a hardcoded 24.5ms.
    assert audit.mean_alignment_error_ms < 10.0, f"unexpected error {audit.mean_alignment_error_ms}"
    assert audit.max_alignment_error_ms < 40.0


def test_real_cut_not_healed(tmp_path):
    """A genuine cut (target has silence in the gap) must stay a fallback."""
    sr = 16000
    ref = gen.make_cartoon_audio(20.0, sr, seed=2)
    # target = ref but with a 3s silent gap at 8-11s (real censored cut).
    tar = ref.copy()
    tar[int(8 * sr):int(11 * sr)] = 0.0
    ref_wav = str(tmp_path / "ref.wav")
    tar_wav = str(tmp_path / "tar.wav")
    gen.write_wav(ref_wav, ref, sr)
    gen.write_wav(tar_wav, tar, sr)

    cfg = DubSyncConfig()
    ver = ClosedLoopVerifierEngine(cfg)
    edl = [
        SegmentEDL(0, "dub", 0.0, 8.0, 0.0, 8.0, 1.0, 0.9),
        SegmentEDL(1, "fallback", 8.0, 11.0, 8.0, 8.0, 1.0, 1.0),
        SegmentEDL(2, "dub", 11.0, 20.0, 8.0, 17.0, 1.0, 0.9),
    ]
    healed, audit = ver.audit_and_heal_edl(edl, ref_wav, tar_wav, 20.0, 20.0)
    # The real cut is NOT healed (target audio is silent there).
    assert audit.false_fallbacks_healed_count == 0
    assert any(s.segment_type == "fallback" for s in healed)


def test_zero_verified_windows_reports_zero_percent_not_fake_100(tmp_path):
    """A total alignment failure must surface as 0% verified + UNVERIFIED records,
    NOT a fake 100% pass with zero measured windows (the Hero-episode bug)."""
    sr = 16000
    # Reference is the tonal "cartoon" music bed; target is unrelated broadband
    # noise. Their M&E envelopes must never correlate above MIN_CORRELATION.
    ref = gen.make_cartoon_audio(20.0, sr, seed=11)
    rng = np.random.default_rng(99)
    tar = rng.standard_normal(int(20.0 * sr)).astype(np.float32)
    ref_wav = str(tmp_path / "ref.wav")
    tar_wav = str(tmp_path / "tar.wav")
    gen.write_wav(ref_wav, ref, sr)
    gen.write_wav(tar_wav, tar, sr)

    cfg = DubSyncConfig()
    ver = ClosedLoopVerifierEngine(cfg)
    edl = [SegmentEDL(0, "dub", 0.0, 20.0, 0.0, 20.0, 1.0, 0.9)]
    _, audit = ver.audit_and_heal_edl(edl, ref_wav, tar_wav, 20.0, 20.0)

    # No dub window verified -> alignment is UNVERIFIED, reported honestly.
    assert audit.dub_windows_verified == 0
    assert audit.passed_windows_pct == 0.0, f"fake 100% pass: {audit.passed_windows_pct}"
    assert audit.dub_windows_skipped > 0
    actions = {r.get("action") for r in audit.audit_log}
    assert "UNVERIFIED_LOW_CORRELATION" in actions or "UNVERIFIED_SILENT" in actions
    # No window may be marked VERIFIED_ALIGNMENT.
    assert "VERIFIED_ALIGNMENT" not in actions


def test_drift_profile_detects_speed_mismatch(tmp_path):
    """--qc drift profiling: flat offsets at 1.0x, PAL (1/0.96) stretch detected."""
    from scipy.signal import resample

    eng = ClosedLoopVerifierEngine(DubSyncConfig())
    sr = 16000
    ref = gen.make_cartoon_audio(90.0, sr, seed=7)
    ref_wav = str(tmp_path / "ref.wav")
    gen.write_wav(ref_wav, ref, sr)

    # Zero drift: constant offset, speed ratio 1.0.
    tar = np.zeros_like(ref)
    tar[int(2 * sr):] = ref[:-int(2 * sr)]
    tw = str(tmp_path / "t0.wav")
    gen.write_wav(tw, tar, sr)
    p0 = eng.measure_drift_profile(ref_wav, tw)
    assert p0, "no probe windows for zero-drift pair"
    assert abs(p0[0]["speed_ratio"] - 1.0) < 0.02
    offsets = [p["offset"] for p in p0]
    assert max(offsets) - min(offsets) < 1.0, "offsets should be flat (no drift)"

    # PAL: target compressed to 0.96x length -> stretch factor 1/0.96 ~ 1.0417.
    tar2 = resample(ref, int(len(ref) * 0.96)).astype(np.float32)
    tw2 = str(tmp_path / "t1.wav")
    gen.write_wav(tw2, tar2, sr)
    p1 = eng.measure_drift_profile(ref_wav, tw2)
    assert p1, "no probe windows for speed-mismatch pair"
    assert abs(p1[0]["speed_ratio"] - 1.0417) < 0.02, f"detected {p1[0]['speed_ratio']}"
