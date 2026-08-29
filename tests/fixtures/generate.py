"""
Deterministic synthetic fixtures for DubSync Pro regression tests.

Each scenario encodes one real-world failure mode observed in the development
history (see docs/CHAT_HISTORY_SUMMARY.md). Ground truth is a piecewise-linear
mapping: a list of segments, where each segment maps reference time to target
time at a constant speed:

    tar(ref) = tar_start + (ref - ref_start) * speed

Cuts are boundaries between segments where the mapping jumps in target time.
These fixtures exercise the *engines* directly (no real media required), so the
whole suite runs in seconds.

Audio fixtures are also generated (via numpy, or via ffmpeg for pitch-preserving
time-stretch) for the splicer / verifier / acoustic-refine engines.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from dub_sync_engine.visual_anchors import AnchorMatch

# Broadcast standards used across the engine (24fps film vs 25fps PAL, etc.)
SPEED_1_0 = 1.0
SPEED_PAL = 24.0 / 25.0          # 0.96   (PAL slowdown)
SPEED_PAL_UP = 25.0 / 24.0       # 1.041667 (PAL speedup)
SPEED_NTSC = 24.0 / 23.976       # 1.001001 (NTSC pulldown)
SPEED_FILM_SLOW = 23.976 / 24.0  # 0.999


@dataclass
class Segment:
    """A continuous piece of the piecewise-linear ground-truth mapping."""
    ref_start: float
    ref_end: float
    tar_start: float
    speed: float

    @property
    def ref_duration(self) -> float:
        return self.ref_end - self.ref_start

    @property
    def tar_duration(self) -> float:
        return self.ref_duration * self.speed

    @property
    def tar_end(self) -> float:
        return self.tar_start + self.tar_duration


@dataclass
class Scenario:
    """A full ground-truth mapping plus optional planted false anchors."""
    name: str
    segments: List[Segment]
    ref_duration: float
    tar_duration: float
    # Optional planted false anchors: (ref_time, tar_time) that the engine should reject.
    false_anchors: List[Tuple[float, float]] = field(default_factory=list)
    # Expected number of fallback (English bridge) segments after a correct EDL build.
    expected_fallbacks: int = 0

    def tar_of(self, ref: float) -> float:
        for seg in self.segments:
            if seg.ref_start - 1e-9 <= ref <= seg.ref_end + 1e-9:
                return seg.tar_start + (ref - seg.ref_start) * seg.speed
        raise ValueError(f"ref {ref} outside all segments")

    def make_anchors(self, step: float = 15.0, confidence: float = 0.9) -> List[AnchorMatch]:
        """Generate anchor matches sampled from the ground-truth mapping."""
        anchors: List[AnchorMatch] = []
        idx = 0
        for seg in self.segments:
            r = seg.ref_start
            while r <= seg.ref_end + 1e-9:
                t = seg.tar_start + (r - seg.ref_start) * seg.speed
                anchors.append(AnchorMatch(
                    ref_idx=idx, tar_idx=idx,
                    ref_time=round(r, 3), tar_time=round(t, 3),
                    hash_dist=0, confidence=confidence,
                    offset=round(t - r, 4),
                ))
                idx += 1
                if r >= seg.ref_end - 1e-9:
                    break  # segment end already emitted
                r = min(r + step, seg.ref_end)
        for ft, (fr, fa) in enumerate(self.false_anchors):
            anchors.append(AnchorMatch(
                ref_idx=idx + ft, tar_idx=idx + ft,
                ref_time=round(fr, 3), tar_time=round(fa, 3),
                hash_dist=0, confidence=0.9,
                offset=round(fa - fr, 4),
            ))
        anchors.sort(key=lambda m: (m.ref_time, m.tar_time))
        return anchors


# --------------------------------------------------------------------------- #
# Scenario definitions (each = one documented failure mode)
# --------------------------------------------------------------------------- #

def clean_1x() -> Scenario:
    return Scenario(
        name="clean_1x",
        segments=[Segment(0.0, 600.0, 0.0, SPEED_1_0)],
        ref_duration=600.0,
        tar_duration=600.0,
        expected_fallbacks=0,
    )


def pal_speed() -> Scenario:
    return Scenario(
        name="pal_speed",
        segments=[Segment(0.0, 600.0, 0.0, SPEED_PAL)],
        ref_duration=600.0,
        tar_duration=600.0 * SPEED_PAL,
        expected_fallbacks=0,
    )


def single_cut() -> Scenario:
    """A 9s scene removed by the broadcaster (one real cut)."""
    cut_start, cut_end = 300.0, 309.0
    return Scenario(
        name="single_cut",
        segments=[
            Segment(0.0, cut_start, 0.0, SPEED_1_0),
            Segment(cut_end, 600.0, cut_start, SPEED_1_0),
        ],
        ref_duration=600.0,
        tar_duration=600.0 - (cut_end - cut_start),
        expected_fallbacks=1,
    )


def micro_trim() -> Scenario:
    """A 1.75s micro-trim inside an otherwise continuous PAL-speed act."""
    trim_start, trim_end = 400.0, 401.75
    tar_mid = 400.0 * SPEED_PAL
    return Scenario(
        name="micro_trim",
        segments=[
            Segment(0.0, trim_start, 0.0, SPEED_PAL),
            Segment(trim_end, 600.0, tar_mid, SPEED_PAL),
        ],
        ref_duration=600.0,
        tar_duration=600.0 * SPEED_PAL - (trim_end - trim_start) * SPEED_PAL,
        expected_fallbacks=1,
    )


def extra_scene() -> Scenario:
    """A 10s TV bumper inserted into the target that has no master equivalent."""
    bumper_start = 200.0
    bumper_len = 10.0
    return Scenario(
        name="extra_scene",
        segments=[
            Segment(0.0, bumper_start, 0.0, SPEED_1_0),
            Segment(bumper_start, 600.0, bumper_start + bumper_len, SPEED_1_0),
        ],
        ref_duration=600.0,
        tar_duration=600.0 + bumper_len,
        expected_fallbacks=0,  # extra target audio is trimmed, not bridged
    )


def intro_gap() -> Scenario:
    """Master has a 4s logo the target lacks (target starts 4s in)."""
    gap = 4.0
    return Scenario(
        name="intro_gap",
        segments=[Segment(gap, 600.0, 0.0, SPEED_1_0)],
        ref_duration=600.0,
        tar_duration=600.0 - gap,
        expected_fallbacks=1,
    )


def black_frame() -> Scenario:
    """
    The historical Episode 2 bug: a black intro frame at 0:00 falsely matched a
    black transition at 12.8s in the target. The master has a 32s logo the target
    lacks, so the real first anchor is ~32s in. The false (0.0, 12.8) anchor must
    be rejected, otherwise dialogue is teleported to 0:00.
    """
    gap = 32.0
    return Scenario(
        name="black_frame",
        segments=[Segment(gap, 600.0, 0.0, SPEED_1_0)],
        ref_duration=600.0,
        tar_duration=600.0 - gap,
        false_anchors=[(0.0, 12.8)],
        expected_fallbacks=1,  # the intro bridge (0 -> 32s)
    )


ALL_SCENARIOS = [
    clean_1x, pal_speed, single_cut, micro_trim, extra_scene, intro_gap, black_frame,
]


def get_scenario(name: str) -> Scenario:
    for fn in ALL_SCENARIOS:
        s = fn()
        if s.name == name:
            return s
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# Audio fixtures (for splicer / verifier / acoustic-refine engines)
# --------------------------------------------------------------------------- #

def make_cartoon_audio(duration: float, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """
    A deterministic, NON-STATIONARY "cartoon" signal: a music bed whose frequency
    content sweeps over time (so no two windows are alike — unlike real periodic
    tones, which would alias in cross-correlation), plus speech-like bursts at
    irregular positions. Returns float32 in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float64)

    # A tonal "cartoon music" bed: a melody built from a random walk over a
    # pentatonic scale, one note every 0.4s. This is highly distinctive in time
    # (unlike band-limited noise) and lives inside the 800-3200 Hz M&E band, so
    # cross-correlation gives sharp, unambiguous peaks that survive resampling.
    scale = [880.0, 990.0, 1174.0, 1318.0, 1480.0, 1760.0]
    note = int(scale[2])
    t_note = np.arange(0.0, duration, 0.4)
    prev_note = 2
    for k, t0 in enumerate(t_note):
        step = int(rng.integers(-2, 3))
        prev_note = int(np.clip(prev_note + step, 0, len(scale) - 1))
        f = scale[prev_note]
        s0 = int(t0 * sr)
        s1 = min(n, int((t0 + 0.4) * sr))
        local = np.arange(s1 - s0) / sr
        env = np.exp(-local * 18.0)  # plucky decay
        # Fundamental + a couple of harmonics for a richer tone.
        tone = (np.sin(2 * np.pi * f * local)
                + 0.5 * np.sin(2 * np.pi * 2 * f * local)
                + 0.25 * np.sin(2 * np.pi * 3 * f * local))
        signal[s0:s1] += tone * env * 0.35

    # Percussive hits (rhythmic transients) for extra time structure.
    for beat in np.arange(0.0, duration, 0.8):
        s0 = int(beat * sr)
        s1 = min(n, int((beat + 0.05) * sr))
        local = np.arange(s1 - s0) / sr
        signal[s0:s1] += np.sin(2 * np.pi * 2000 * local) * np.exp(-local * 120.0) * 0.6

    # Speech-like bursts at irregular (deterministic) positions.
    pos = 3.0
    while pos + 2.5 < duration:
        s0 = int(pos * sr)
        s1 = int((pos + 2.5) * sr)
        local = np.arange(s1 - s0) / sr
        env = np.sin(np.pi * local / 2.5) ** 2
        f0 = 150.0 + 80.0 * np.sin(pos)  # varying voice pitch
        signal[s0:s1] += np.sin(2 * np.pi * f0 * local) * env * 0.5
        pos += 5.0 + 4.0 * rng.random()

    peak = np.max(np.abs(signal)) + 1e-9
    return (signal / peak).astype(np.float32)


def write_wav(path: str, data: np.ndarray, sr: int = 16000) -> str:
    import scipy.io.wavfile as wavfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, sr, (np.clip(data, -1, 1) * 32767).astype(np.int16))
    return path


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def time_stretch(data: np.ndarray, sr: int, speed: float, out_path: str) -> str:
    """Pitch-preserving time-stretch via ffmpeg atempo (writes a temp WAV)."""
    import tempfile
    tmpdir = tempfile.mkdtemp()
    src = os.path.join(tmpdir, "src.wav")
    write_wav(src, data, sr)
    cmd = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", src, "-af", f"atempo={speed:.6f}", "-c:a", "pcm_s16le",
        "-ar", str(sr), "-ac", "1", out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def build_ref_tar_audio(scenario: Scenario, sr: int = 16000, seed: int = 0,
                        tmpdir: str = None) -> Tuple[str, str, dict]:
    """
    Build reference and target WAV files for a scenario (uses the same music bed on
    both sides so M&E correlation succeeds), returning (ref_wav, tar_wav, meta).
    The target is assembled from the reference per the ground-truth mapping.
    """
    import tempfile
    tmpdir = tmpdir or tempfile.mkdtemp()
    ref_data = make_cartoon_audio(scenario.ref_duration, sr, seed)

    # Assemble target by concatenating each ground-truth segment's source span.
    chunks = []
    for seg in scenario.segments:
        s0 = int(seg.ref_start * sr)
        s1 = int(seg.ref_end * sr)
        chunk = ref_data[s0:s1]
        if abs(seg.speed - 1.0) > 1e-3:
            # pitch-preserving stretch to the segment speed
            n_out = int(len(chunk) / seg.speed)
            out = os.path.join(tmpdir, f"seg_{int(seg.ref_start)}.wav")
            time_stretch(chunk, sr, seg.speed, out)
            import scipy.io.wavfile as wavfile
            _, chunk = wavfile.read(out)
            chunk = chunk.astype(np.float32) / 32767.0
        chunks.append(chunk)
    tar_data = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    ref_wav = os.path.join(tmpdir, "ref.wav")
    tar_wav = os.path.join(tmpdir, "tar.wav")
    write_wav(ref_wav, ref_data, sr)
    write_wav(tar_wav, tar_data, sr)

    meta = {
        "sr": sr,
        "ref_duration": scenario.ref_duration,
        "tar_duration": scenario.tar_duration,
        "expected_fallbacks": scenario.expected_fallbacks,
        "tmpdir": tmpdir,
    }
    return ref_wav, tar_wav, meta
