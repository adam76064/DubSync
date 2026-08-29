"""
Tests for the closed-loop verifier's alignment measurement (REVIEW.md F1).

The audit used to return hard-coded literals (24.5 ms / 38.0 ms / 99.2 %) on
every run. These tests pin the replacement: numbers that come from actually
probing the timeline, verified against timelines whose error is known.

Requires the synthetic fixture:  python3 tests/synthetic_media.py
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dub_sync_engine.audio_splicer import SegmentEDL  # noqa: E402
from dub_sync_engine.config import DubSyncConfig  # noqa: E402
from dub_sync_engine.media_probe import MediaProbe  # noqa: E402
from dub_sync_engine.verifier_engine import ClosedLoopVerifierEngine  # noqa: E402

MEDIA = os.path.join(ROOT, "tests", "_media")
SPEED = 1.0 / 0.96          # tar seconds per ref second on the fixture

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MEDIA, "ref.mp4")),
    reason="synthetic fixture missing - run: python3 tests/synthetic_media.py",
)


@pytest.fixture(scope="module")
def pcm(tmp_path_factory):
    """Extract reference and dub PCM once for the whole module."""
    d = tmp_path_factory.mktemp("pcm")
    ref_wav = str(d / "ref.wav")
    tar_wav = str(d / "tar.wav")
    MediaProbe.extract_pcm_wav(os.path.join(MEDIA, "ref.mp4"), ref_wav, sample_rate=48000, channels=1)
    MediaProbe.extract_pcm_wav(os.path.join(MEDIA, "tar.mp4"), tar_wav, sample_rate=48000, channels=1)
    return ref_wav, tar_wav


def measure(edl, pcm, **cfg):
    c = DubSyncConfig()
    for k, v in cfg.items():
        setattr(c, k, v)
    return ClosedLoopVerifierEngine(c)._measure_alignment_error(edl, pcm[0], pcm[1])


def seg(i, kind, r0, r1, t0, t1, speed=round(SPEED, 6), conf=1.0):
    return SegmentEDL(i, kind, r0, r1, t0, t1, speed, conf)


# Ground truth: dub missing ref[24,30]; tar_time = (ref-6)/0.96 for ref >= 30
PERFECT = [
    seg(0, "dub", 0.0, 24.0, 0.0, 24.0 * SPEED),
    seg(1, "fallback", 24.0, 30.0, 0.0, 0.0, 1.0),
    seg(2, "dub", 30.0, 60.0, 25.0, 25.0 + 30.0 * SPEED),
]

SHIFTED_250MS = [
    seg(0, "dub", 0.0, 24.0, 0.25, 0.25 + 24.0 * SPEED),
    seg(1, "fallback", 24.0, 30.0, 0.0, 0.0, 1.0),
    seg(2, "dub", 30.0, 60.0, 25.25, 25.25 + 30.0 * SPEED),
]

# The F3 defect: the omitted scene bridged at the END of the interval, so 6s of
# dub is shifted onto the wrong ref span.
GAP_AT_END = [
    seg(0, "dub", 0.0, 30.03, 0.0, 30.03 * SPEED),
    seg(1, "fallback", 30.03, 36.0, 0.0, 0.0, 1.0),
    seg(2, "dub", 36.0, 60.0, 31.2, 56.18),
]


def test_audit_numbers_are_measured_not_constant(pcm):
    """The whole point of F1: two different timelines must not score identically."""
    good = ClosedLoopVerifierEngine(DubSyncConfig()).audit_and_heal_edl(
        list(PERFECT), pcm[0], pcm[1], 60.0, 56.25)[1]
    bad = ClosedLoopVerifierEngine(DubSyncConfig()).audit_and_heal_edl(
        list(GAP_AT_END), pcm[0], pcm[1], 60.0, 56.25)[1]
    assert (good.mean_alignment_error_ms, good.passed_windows_pct) != \
           (bad.mean_alignment_error_ms, bad.passed_windows_pct)


def test_perfect_timeline_measures_near_zero(pcm):
    mean, mx, pct, n = measure(list(PERFECT), pcm)
    assert n > 10, "expected a meaningful number of probe windows"
    assert mean < 50.0, f"a perfect timeline measured {mean} ms mean error"
    assert pct > 90.0, f"a perfect timeline only passed {pct}% of windows"


def test_known_offset_is_measured_accurately(pcm):
    """A deliberate 250 ms shift must be reported as ~250 ms, not 24.5 ms."""
    mean, mx, pct, n = measure(list(SHIFTED_250MS), pcm)
    assert 200.0 < mean < 320.0, f"a 250ms shift measured as {mean} ms"


def test_gross_misplacement_is_detected(pcm):
    """
    A 6-second mis-sync must show up as a multi-second error, not be clipped to
    the search band.

    Note that passed_windows_pct stays high here: the broken span is ~10% of
    the runtime, so most windows genuinely are fine. Severity is carried by the
    mean and max, which is why all three statistics are reported.
    """
    base_mean, _, base_pct, _ = measure(list(PERFECT), pcm)
    mean, mx, pct, _ = measure(list(GAP_AT_END), pcm)
    assert mx > 5000.0, f"a 6s mis-placement produced only {mx} ms max error"
    assert mean > 10 * base_mean, f"broken timeline ({mean} ms) not clearly worse than perfect ({base_mean} ms)"
    assert pct < base_pct - 5.0, f"broken timeline passed {pct}% vs perfect {base_pct}%"


def test_unmeasurable_timeline_reports_none_not_a_number(pcm):
    """When nothing correlates, say so - never fall back to a plausible-looking number."""
    nonsense = [seg(0, "dub", 0.0, 60.0, 0.0, 60.0, 1.0)]
    mean, mx, pct, n = measure(nonsense, pcm)
    # Either it measures the mismatch honestly, or it reports nothing at all -
    # but it must not report a passing score for a wrong mapping.
    assert not (pct is not None and pct > 90.0 and (mean or 0) < 50.0), \
        "a wrong 1:1 mapping was reported as a clean sync"


def test_config_has_no_hardcoded_audit_defaults():
    """Regression guard: the literals 24.5 / 38.0 / 99.2 must never come back."""
    src = open(os.path.join(ROOT, "dub_sync_engine", "verifier_engine.py")).read()
    for literal in ("24.5", "38.0", "99.2"):
        assert literal not in src, f"hard-coded audit literal {literal} is back"
