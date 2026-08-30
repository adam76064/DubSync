"""
Dense Sync-Path Estimator (similarity matrix + ridge extraction).

Instead of assuming a single continuous speed (which fails on re-cut dubs), this
engine *measures* the true (ref_time -> tar_time) path directly from a
multi-band spectral (M&E) fingerprint:

  1. Build a vocal-suppressed multi-band energy fingerprint for both tracks
     (~10 Hz per band, several log-spaced bands).
  2. Estimate the global speed ratio (target/ref) over a grid of candidates.
  3. Resample the target to the reference timescale and, for every dense probe
     window, find the best-matching target position via normalized
     cross-correlation summed across bands -> a dense
     (ref_time, tar_time, confidence) point cloud.
  4. Fit a *piecewise-linear* path through the dense cloud (reusing the
     subsync-style RANSAC in `line_fit`). Each contiguous segment is a synced
     region; the jumps/gaps between segments are cuts, and the un-covered tail is
     a trim.

The result is the "ground truth" the downstream EDL builder consumes, so cuts
and tail trims are represented honestly instead of being smoothed into one
wrong continuous block.
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .config import DubSyncConfig


# Candidate speed ratios (target timeline / reference timeline) for the global
# speed estimate. Includes broadcast standards plus intermediate values for VFR
# sources that do not land exactly on a standard.
CANDIDATE_SPEED_RATIOS = (
    0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
    1.0, 1.001, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10,
)

# Log-spaced band edges (Hz) for the M&E spectral fingerprint. The lowest band
# starts at 300 Hz to suppress dialogue fundamentals; bands span the music/SFX
# range. A multi-band fingerprint is far more distinctive than a single energy
# envelope (repetitive/percussive music otherwise false-matches everywhere).
BAND_EDGES = (300.0, 500.0, 700.0, 900.0, 1200.0, 1600.0, 2200.0, 3000.0, 3900.0)


@dataclass
class PathSegment:
    """One contiguous synced region of the measured path (y = slope*x + intercept)."""

    ref_start: float
    ref_end: float
    tar_start: float
    tar_end: float
    slope: float        # speed ratio (tar/ref) within this region
    intercept: float    # tar = slope*ref + intercept
    n_points: int       # dense points supporting this region
    confidence: float   # mean correlation of supporting points (0..1)


def estimate_speed_ratio(env_r: np.ndarray, env_t: np.ndarray, bin_sr: float = 10.0) -> float:
    """
    Estimate the target/ref speed ratio by correlating a central reference
    window against the target resampled at each candidate ratio. Accepts either
    a 1-D envelope or a (T, B) multi-band fingerprint (bands are summed).
    Returns the ratio with the strongest normalized peak (1.0 when degenerate).
    """
    env_r, env_t = np.asarray(env_r, dtype=float), np.asarray(env_t, dtype=float)
    if env_r.ndim == 1:
        env_r = env_r[:, None]
        env_t = env_t[:, None]

    n_r = env_r.shape[0]
    w0 = int(n_r * 0.35)
    w1 = min(n_r, w0 + int(n_r * 0.30))
    ref_win = env_r[w0:w1]
    ref_win = ref_win - np.mean(ref_win, axis=0, keepdims=True)
    r_norm = np.linalg.norm(ref_win)
    if r_norm < 1e-6 or (w1 - w0) < 32:
        return 1.0

    best_ratio, best_peak = 1.0, -1.0
    for ratio in CANDIDATE_SPEED_RATIOS:
        n_t2 = int(env_t.shape[0] / ratio)
        if n_t2 < (w1 - w0):
            continue
        tar_scaled = scipy.signal.resample(env_t, n_t2, axis=0).astype(np.float32)
        # Sum per-band normalized correlation of the central window.
        corr_sum = _band_correlations(tar_scaled, ref_win)
        peak = float(np.max(corr_sum) / ref_win.shape[1])
        if peak > best_peak:
            best_peak, best_ratio = peak, ratio
    return best_ratio


def _band_correlations(target_feat: np.ndarray, ref_win: np.ndarray) -> np.ndarray:
    """Sum of per-band normalized cross-correlations of `ref_win` against `target_feat`."""
    W = ref_win.shape[0]
    B = ref_win.shape[1]
    acc = np.zeros(target_feat.shape[0] - W + 1, dtype=np.float64)
    for b in range(B):
        rw = ref_win[:, b]
        rn = np.linalg.norm(rw)
        if rn < 1e-6:
            continue
        tband = target_feat[:, b]
        corr = scipy.signal.correlate(tband, rw, mode="valid")
        t_sq = tband ** 2
        cum = np.concatenate(([0.0], np.cumsum(t_sq)))
        norms = np.sqrt(np.maximum(cum[W:] - cum[:-W], 0.0))
        acc += corr / (rn * norms + 1e-8)
    return acc


class SyncPathEstimator:
    """Measures the piecewise-linear sync path (speed + cuts + tail trim)."""

    def __init__(self, config: Optional[DubSyncConfig] = None):
        self.config = config or DubSyncConfig()
        # Populated during extract_path() for the forensic report.
        self.last_diagnostics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def spectral_envelope(audio: np.ndarray, sr: int, bin_sr: float = 10.0) -> np.ndarray:
        """
        Multi-band M&E spectral fingerprint: per-band band-pass energy envelopes
        downsampled to ~bin_sr Hz, stacked as a (T, B) array. Each band is
        zero-mean normalized over the whole track.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        step = max(1, sr // 8000)
        a = audio[::step]
        sr2 = float(sr) / step
        nyq = sr2 / 2.0

        bands = []
        for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
            low = lo / nyq
            high = min(0.95, hi / nyq)
            if high <= low:
                high = min(0.95, low + 0.05)
            b, a_coef = scipy.signal.butter(3, [low, high], btype="band")
            filtered = scipy.signal.filtfilt(b, a_coef, a)
            env = np.abs(scipy.signal.hilbert(filtered))

            w = int(sr2 * 0.05)
            if w > 1:
                env = np.convolve(env, np.ones(w) / w, mode="same")

            dec = max(1, int(round(sr2 / bin_sr)))
            env = env[::dec]
            env = env - np.mean(env)
            nrm = np.linalg.norm(env)
            if nrm > 1e-8:
                env = env / nrm
            bands.append(env.astype(np.float32))

        return np.stack(bands, axis=1)  # (T, B)

    def _load_features(self, ref_wav_path, tar_wav_path, bin_sr=10.0):
        sr_r, a_r = wavfile.read(ref_wav_path)
        sr_t, a_t = wavfile.read(tar_wav_path)
        env_r = self.spectral_envelope(a_r, sr_r, bin_sr)
        env_t = self.spectral_envelope(a_t, sr_t, bin_sr)
        return env_r, env_t, bin_sr

    # ------------------------------------------------------------------ #
    # Dense point cloud
    # ------------------------------------------------------------------ #
    def build_point_cloud(
        self,
        env_r: np.ndarray,
        env_t: np.ndarray,
        bin_sr: float,
        speed_ratio: Optional[float] = None,
        window_sec: Optional[float] = None,
        hop_sec: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """
        Dense (ref_time, tar_time, confidence) points via multi-band normalized
        cross-correlation of each reference window against the speed-corrected
        target. Returns a list of point dicts.
        """
        window_sec = self.config.path_window_sec if window_sec is None else window_sec
        hop_sec = self.config.path_hop_sec if hop_sec is None else hop_sec

        if speed_ratio is None:
            speed_ratio = estimate_speed_ratio(env_r, env_t, bin_sr)

        env_t_res = scipy.signal.resample(env_t, int(env_t.shape[0] / speed_ratio), axis=0).astype(np.float32)
        n_bands = env_r.shape[1]

        win = max(8, int(round(window_sec * bin_sr)))
        hop = max(1, int(round(hop_sec * bin_sr)))

        points: List[Dict[str, float]] = []
        for i in range(0, env_r.shape[0] - win, hop):
            ref_win = env_r[i:i + win]
            ref_win = ref_win - np.mean(ref_win, axis=0, keepdims=True)
            rn = np.linalg.norm(ref_win)
            if rn < 1e-4:
                continue

            corr_sum = _band_correlations(env_t_res, ref_win)
            best = int(np.argmax(corr_sum))
            peak = float(corr_sum[best] / n_bands)

            ref_time = i / bin_sr
            tar_time = (best / bin_sr) * speed_ratio
            points.append({
                "ref_time": ref_time,
                "tar_time": tar_time,
                "offset": tar_time - ref_time,
                "confidence": peak,
            })
        return points

    # ------------------------------------------------------------------ #
    # Path extraction
    # ------------------------------------------------------------------ #
    def extract_path(
        self,
        ref_wav_path: str,
        tar_wav_path: str,
        ref_duration: float,
        tar_duration: float,
    ) -> List[PathSegment]:
        """
        Measure the piecewise-linear sync path. Returns the synced segments
        (sorted by ref time); reference time not covered by any segment is a
        cut or tail trim (fallback).

        Cuts are detected as *offset steps* (a jump in tar_time/ref_time at a
        fixed reference position), which is the correct model for censored
        trims and scene omissions — a small cut is invisible to a residual-
        based line splitter (it looks like a shallow slope over a long span).
        """
        env_r, env_t, bin_sr = self._load_features(ref_wav_path, tar_wav_path)
        speed_ratio = estimate_speed_ratio(env_r, env_t, bin_sr)
        env_t_res = scipy.signal.resample(env_t, int(env_t.shape[0] / speed_ratio), axis=0).astype(np.float32)
        points = self.build_point_cloud(env_r, env_t, bin_sr, speed_ratio=speed_ratio)

        self.last_diagnostics = {
            "speed_ratio": round(speed_ratio, 5),
            "dense_points": len(points),
            "min_correlation": self.config.path_min_correlation,
        }

        # Drop low-confidence (silence / garbage) points before fitting.
        points = [p for p in points if p["confidence"] >= self.config.path_min_correlation]
        if len(points) < 6:
            return []

        pts = sorted(points, key=lambda p: p["ref_time"])
        refs = np.array([p["ref_time"] for p in pts], dtype=float)
        # Residual offset in reference-time units (divide out the global speed):
        # constant within a continuous region, steps at a cut.
        res_off = np.array([p["tar_time"] / speed_ratio - p["ref_time"] for p in pts], dtype=float)
        confs = np.array([p["confidence"] for p in pts], dtype=float)

        boundaries = _offset_step_boundaries(
            refs, res_off,
            jump=self.config.path_jump_threshold_sec,
            window=self.config.path_step_window_points,
        )

        result: List[PathSegment] = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            rr, ro, cc = refs[a:b], res_off[a:b], confs[a:b]
            if b - a < 3 or (rr[-1] - rr[0]) < self.config.path_min_segment_sec:
                continue
            off = float(np.median(ro))
            slope = speed_ratio
            intercept = speed_ratio * off
            result.append(PathSegment(
                ref_start=round(rr[0], 3),
                ref_end=round(rr[-1], 3),
                tar_start=round(slope * rr[0] + intercept, 3),
                tar_end=round(slope * rr[-1] + intercept, 3),
                slope=round(slope, 6),
                intercept=round(intercept, 4),
                n_points=int(b - a),
                confidence=round(float(np.mean(cc)), 4),
            ))

        result.sort(key=lambda s: s.ref_start)
        self._refine_cut_boundaries(env_r, env_t_res, speed_ratio, bin_sr, result)
        self._complete_seams(result, ref_duration, tar_duration)
        self.last_diagnostics["segments"] = len(result)
        return result

    def _refine_cut_boundaries(self, env_r, env_t_res, speed_ratio, bin_sr, segments) -> None:
        """
        Coarse-to-fine: the long coarse window blurs a cut boundary by up to
        ~window_sec. Re-probe each gap between adjacent segments with short
        windows (3s) and move the boundary to the precise offset transition.
        """
        if len(segments) < 2:
            return
        n_bands = env_r.shape[1]
        win = max(6, int(3.0 * bin_sr))
        hop = max(1, int(1.0 * bin_sr))

        def probe(i: int):
            ref_win = env_r[i:i + win] - np.mean(env_r[i:i + win], axis=0, keepdims=True)
            if np.linalg.norm(ref_win) < 1e-4:
                return None
            corr_sum = _band_correlations(env_t_res, ref_win)
            best = int(np.argmax(corr_sum))
            peak = float(corr_sum[best] / n_bands)
            ref_time = i / bin_sr
            tar_time = best / bin_sr * speed_ratio
            return ref_time, tar_time / speed_ratio - ref_time, peak

        for k in range(len(segments) - 1):
            left, right = segments[k], segments[k + 1]
            o_left = left.intercept / speed_ratio
            o_right = right.intercept / speed_ratio
            r_lo = left.ref_end
            r_hi = right.ref_start
            if r_hi - r_lo > 30.0 or r_hi <= r_lo:
                continue

            transition = None
            last_label = None
            for i in range(int(r_lo * bin_sr), int(r_hi * bin_sr) - win, hop):
                res = probe(i)
                if res is None:
                    continue
                ref_time, off, peak = res
                if peak < self.config.path_min_correlation:
                    continue
                label = "L" if abs(off - o_left) <= abs(off - o_right) else "R"
                if last_label == "L" and label == "R":
                    transition = ref_time
                    break
                last_label = label

            if transition is not None:
                left.ref_end = round(transition, 3)
                left.tar_end = round(left.slope * left.ref_end + left.intercept, 3)
                right.ref_start = round(transition, 3)
                right.tar_start = round(right.slope * right.ref_start + right.intercept, 3)

    def build_edl(
        self,
        segments: List[PathSegment],
        ref_duration: float,
        tar_duration: float,
    ):
        """
        Convert measured path segments into an EDL: each segment becomes a dub
        span, and the reference time *not* covered by a segment (cuts between
        segments, the intro gap, and the tail trim) becomes a fallback span.
        Missing content is represented honestly — never force-dubbed.
        """
        from .audio_splicer import SegmentEDL, sanitize_edl

        edl = []
        seg_id = 0

        # Opening gap (intro the dub lacks).
        if segments and segments[0].ref_start > 0.05:
            edl.append(SegmentEDL(
                seg_id, "fallback",
                0.0, round(segments[0].ref_start, 3),
                0.0, 0.0, 1.0, 1.0,
            ))
            seg_id += 1

        for i, s in enumerate(segments):
            edl.append(SegmentEDL(
                seg_id, "dub",
                s.ref_start, s.ref_end,
                max(0.0, s.tar_start), s.tar_end,
                s.slope, s.confidence,
            ))
            seg_id += 1

            # Gap to the next segment = a cut (censored/omitted scene).
            if i + 1 < len(segments):
                nxt = segments[i + 1]
                if nxt.ref_start > s.ref_end + 0.05:
                    edl.append(SegmentEDL(
                        seg_id, "fallback",
                        round(s.ref_end, 3), round(nxt.ref_start, 3),
                        round(s.tar_end, 3), round(nxt.tar_start, 3),
                        1.0, 1.0,
                    ))
                    seg_id += 1

        # Tail trim (the dub ends before the master).
        if segments and segments[-1].ref_end < ref_duration - 0.05:
            last = segments[-1]
            edl.append(SegmentEDL(
                seg_id, "fallback",
                round(last.ref_end, 3), round(ref_duration, 3),
                round(last.tar_end, 3), round(tar_duration, 3),
                1.0, 1.0,
            ))
            seg_id += 1

        return sanitize_edl(edl, self.config)

    @staticmethod
    def _complete_seams(segments: List[PathSegment], ref_duration: float, tar_duration: float) -> None:
        """
        Extend the first/last segments to the true media boundaries using the
        known durations, so intro gaps and tail trims are located exactly
        (windowed correlation blurs a ~window_sec boundary, and can miss the
        last stretch of valid content before a tail trim).
        """
        if not segments:
            return
        first = segments[0]
        r0 = (0.0 - first.intercept) / first.slope
        if r0 > 0.01:
            # The dub lacks the opening of the master (intro gap): content
            # actually begins at r0, everything before is fallback.
            first.ref_start = round(r0, 3)
            first.tar_start = 0.0
        else:
            first.ref_start = 0.0
            first.tar_start = round(max(0.0, first.intercept), 3)

        last = segments[-1]
        r_end_target = (tar_duration - last.intercept) / last.slope
        if r_end_target < ref_duration:
            # Target runs out before the master ends (tail trim).
            last.ref_end = round(r_end_target, 3)
            last.tar_end = round(tar_duration, 3)
        else:
            last.ref_end = round(ref_duration, 3)
            last.tar_end = round(last.slope * ref_duration + last.intercept, 3)


def _offset_step_boundaries(refs: np.ndarray, res_off: np.ndarray, jump: float, window: int = 5) -> List[int]:
    """
    Detect offset-step (cut) boundaries via before/after median comparison
    (robust change-point detection). Returns boundary indices into `refs`
    (always starting with 0 and ending with len(refs)).
    """
    n = len(refs)
    boundaries = [0]
    for i in range(window, n - window):
        before = np.median(res_off[i - window:i])
        after = np.median(res_off[i:i + window])
        if abs(after - before) > jump:
            boundaries.append(i)
    boundaries.append(n)
    return boundaries
