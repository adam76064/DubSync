"""
Autonomous Closed-Loop Self-Verification & Healing Engine for DubSync Pro.
Audits the candidate timeline, probes foreign audio continuity in fallback gaps,
and self-heals any flagged scenes before final multiplexing.

The key insight: the reference and the foreign dub share the same background
Music & Effects (M&E) bed but differ in *language*. Correlating raw waveforms
(dominated by dialogue) is therefore meaningless across language versions. This
engine instead correlates a vocal-suppressed band-pass energy envelope so that
same-scene backgrounds produce high correlation regardless of the spoken language.
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from .config import DubSyncConfig
from .audio_splicer import SegmentEDL, sanitize_edl


@dataclass
class VerificationAudit:
    total_probed_windows: int
    mean_alignment_error_ms: float
    max_alignment_error_ms: float
    passed_windows_pct: float
    false_fallbacks_healed_count: int
    healed_duration_sec: float
    audit_log: List[Dict[str, Any]]


class ClosedLoopVerifierEngine:
    """
    Autonomously audits the Edit Decision List (EDL) against reference & foreign signals,
    probes candidate audio to eliminate false English fallbacks, and verifies sub-frame precision.
    """

    # A window is considered "passed" if its residual alignment error is below one
    # 24fps video frame (~41.7ms). We use a slightly tighter margin.
    PASS_THRESHOLD_MS: float = 40.0
    # Minimum normalized M&E envelope correlation to accept a window as aligned.
    MIN_CORRELATION: float = 0.10
    # Minimum normalized M&E correlation to consider a fallback gap "false" and heal it.
    HEAL_CORRELATION: float = 0.18
    # Envelope downsampling rate for fast full-episode correlation.
    ENVELOPE_SR: int = 8000
    # Alignment search window (ms) around each projected anchor when measuring residual error.
    SEARCH_WINDOW_MS: float = 200.0

    def __init__(self, config: Optional[DubSyncConfig] = None):
        self.config = config or DubSyncConfig()

    # ------------------------------------------------------------------ #
    # Envelope helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_mono_float(audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio.astype(np.float32)

    def _me_envelope(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Downsample to ~8kHz, band-pass to the M&E bed (800Hz-3200Hz), and return the
        zero-mean Hilbert energy envelope plus the effective envelope sample rate.
        """
        step = max(1, sr // self.ENVELOPE_SR)
        a = audio[::step].astype(np.float32)
        sr2 = float(sr) / step
        nyq = sr2 / 2.0

        low = 800.0 / nyq
        high = min(0.95, 3200.0 / nyq)
        if high <= low:
            high = min(0.95, low + 0.05)

        b, a_coef = scipy.signal.butter(3, [low, high], btype="band")
        filtered = scipy.signal.filtfilt(b, a_coef, a)
        envelope = np.abs(scipy.signal.hilbert(filtered))
        envelope = envelope - np.mean(envelope)
        return envelope, int(round(sr2))

    def _normalized_correlation(
        self,
        ref_env: np.ndarray,
        tar_env: np.ndarray,
        sr2: int,
        r0: float,
        r1: float,
        t0: float,
        t1: float,
        search_ms: Optional[float] = None,
    ):
        """
        Cross-correlates the reference M&E envelope window [r0, r1] against the target
        window [t0, t1] (extended by +/-search_ms) and returns:
            (lag_samples, peak_correlation) where lag_samples == 0 means the target
            window is already perfectly aligned.
        """
        search_ms = self.SEARCH_WINDOW_MS if search_ms is None else search_ms

        wi, wj = int(r0 * sr2), int(r1 * sr2)
        if wj <= wi:
            return None

        pad = int(search_ms / 1000.0 * sr2)
        t0_sample = int(t0 * sr2)
        ti = max(0, t0_sample - pad)
        tj = min(len(tar_env), int(t1 * sr2) + pad)
        if tj <= ti:
            return None

        ref_win = ref_env[wi:wj]
        tar_win = tar_env[ti:tj]
        L = len(ref_win)
        if len(tar_win) < L:
            return None

        r_norm = np.linalg.norm(ref_win)
        if r_norm < 1e-4:
            return None

        corr = scipy.signal.correlate(tar_win, ref_win, mode="valid")

        # Efficient sliding-window normalization via cumulative sums of squares.
        tar_sq = tar_win ** 2
        cum = np.concatenate(([0.0], np.cumsum(tar_sq)))
        window_norms = np.sqrt(np.maximum(cum[L:] - cum[:-L], 0.0))

        denom = r_norm * window_norms
        corr_norm = corr / np.maximum(denom, 1e-8)

        peak = int(np.argmax(corr_norm))
        # lag (samples) relative to the *expected* target position t0, robust to the
        # search window being clamped at the file boundaries.
        lag = peak - (t0_sample - ti)
        return lag, float(corr_norm[peak])

    # ------------------------------------------------------------------ #
    # Main audit
    # ------------------------------------------------------------------ #
    def audit_and_heal_edl(
        self,
        edl: List[SegmentEDL],
        ref_wav_path: str,
        tar_wav_path: str,
        ref_duration: float,
        tar_duration: float
    ) -> Tuple[List[SegmentEDL], VerificationAudit]:
        """
        (1) Heals fallback gaps that actually contain continuous dub (same M&E background),
        and (2) measures the real residual alignment error across dub segments.
        """
        sr_r, audio_r = wavfile.read(ref_wav_path)
        sr_t, audio_t = wavfile.read(tar_wav_path)

        audio_r = self._to_mono_float(audio_r)
        audio_t = self._to_mono_float(audio_t)

        ref_env, sr_r2 = self._me_envelope(audio_r, sr_r)
        tar_env, sr_t2 = self._me_envelope(audio_t, sr_t)
        # Resample envelopes to a common rate for direct comparison.
        if sr_r2 != sr_t2:
            common = max(sr_r2, sr_t2)
            if sr_r2 != common:
                ref_env = scipy.signal.resample(ref_env, int(len(ref_env) * common / sr_r2))
            if sr_t2 != common:
                tar_env = scipy.signal.resample(tar_env, int(len(tar_env) * common / sr_t2))
            sr2 = common
        else:
            sr2 = sr_r2

        healed_edl: List[SegmentEDL] = []
        false_fallbacks_healed = 0
        healed_duration = 0.0
        audit_records: List[Dict[str, Any]] = []

        # --- Pass 1: heal false fallbacks ---
        for seg in edl:
            if seg.segment_type == "fallback":
                r_start, r_end, dur = seg.ref_start, seg.ref_end, seg.ref_duration

                prev_dub = next((s for s in reversed(healed_edl) if s.segment_type == "dub"), None)
                speed = prev_dub.speed_factor if prev_dub else 1.0

                healed = False
                if prev_dub is not None and dur <= 90.0:
                    proj_tar_start = prev_dub.tar_end
                    proj_tar_end = proj_tar_start + (dur * speed)

                    result = self._normalized_correlation(
                        ref_env, tar_env, sr2, r_start, r_end, proj_tar_start, proj_tar_end
                    )
                    if result is not None:
                        lag, peak = result
                        if peak >= self.HEAL_CORRELATION:
                            healed_seg = SegmentEDL(
                                seg_id=seg.seg_id,
                                segment_type="dub",
                                ref_start=r_start,
                                ref_end=r_end,
                                tar_start=round(proj_tar_start, 3),
                                tar_end=round(proj_tar_end, 3),
                                speed_factor=speed,
                                confidence=0.92
                            )
                            healed_edl.append(healed_seg)
                            false_fallbacks_healed += 1
                            healed_duration += dur
                            audit_records.append({
                                "ref_start": r_start,
                                "ref_end": r_end,
                                "action": "HEALED_FALSE_FALLBACK",
                                "healed_duration": dur,
                                "correlation": round(peak, 4),
                                "lag_samples": int(lag),
                            })
                            healed = True

                if not healed:
                    healed_edl.append(seg)
                    audit_records.append({
                        "ref_start": r_start,
                        "ref_end": r_end,
                        "action": "CONFIRMED_GENUINE_CUT",
                        "duration": dur,
                    })
            else:
                healed_edl.append(seg)

        # --- Pass 2: measure real residual alignment error on dub segments ---
        errors_ms: List[float] = []
        for seg in healed_edl:
            if seg.segment_type != "dub":
                continue
            r0, r1 = seg.ref_start, seg.ref_end
            t0, t1 = seg.tar_start, seg.tar_end
            if (r1 - r0) < 2.0:
                continue

            # Probe up to 3 evenly-spaced windows inside the segment.
            n_win = min(3, int((r1 - r0) // 5.0))
            for k in range(n_win):
                frac0 = k / max(1, n_win)
                frac1 = (k + 1) / max(1, n_win)
                wr0 = r0 + frac0 * (r1 - r0)
                wr1 = r0 + frac1 * (r1 - r0)
                wt0 = t0 + frac0 * (t1 - t0)
                wt1 = t0 + frac1 * (t1 - t0)

                result = self._normalized_correlation(
                    ref_env, tar_env, sr2, wr0, wr1, wt0, wt1
                )
                if result is None:
                    continue
                lag, peak = result
                if peak < self.MIN_CORRELATION:
                    continue  # silent / uninformative window — don't count it

                err_ms = (lag / sr2) * 1000.0
                errors_ms.append(err_ms)
                audit_records.append({
                    "ref_start": round(wr0, 3),
                    "ref_end": round(wr1, 3),
                    "action": "VERIFIED_ALIGNMENT",
                    "error_ms": round(err_ms, 2),
                    "correlation": round(peak, 4),
                })

        # --- Pass 3: sanitize (micro-fallback absorption, min-dub-act, contiguity merge) ---
        compact_edl = sanitize_edl(healed_edl, self.config)

        if errors_ms:
            abs_errors = np.abs(np.array(errors_ms, dtype=np.float32))
            mean_err = float(np.mean(abs_errors))
            max_err = float(np.max(abs_errors))
            passed_pct = float(np.mean(abs_errors <= self.PASS_THRESHOLD_MS) * 100.0)
        else:
            mean_err, max_err, passed_pct = 0.0, 0.0, 100.0

        audit = VerificationAudit(
            total_probed_windows=len(audit_records),
            mean_alignment_error_ms=round(mean_err, 2),
            max_alignment_error_ms=round(max_err, 2),
            passed_windows_pct=round(passed_pct, 1),
            false_fallbacks_healed_count=false_fallbacks_healed,
            healed_duration_sec=round(healed_duration, 2),
            audit_log=audit_records
        )

        return compact_edl, audit

    # ------------------------------------------------------------------ #
    # Standalone drift profiling (for `--qc` — no re-render)
    # ------------------------------------------------------------------ #
    # Candidate broadcast speed ratios (target timeline length / reference
    # timeline length). PAL 25fps->24fps is ~0.96, NTSC pull-up ~1.001, etc.
    CANDIDATE_SPEED_RATIOS: Tuple[float, ...] = (
        0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        1.0, 1.001, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10,
    )

    def measure_drift_profile(
        self,
        ref_wav_path: str,
        tar_wav_path: str,
        window_sec: float = 30.0,
        hop_sec: float = 20.0,
        search_sec: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """
        Measure the alignment offset (and its drift over time) between the reference
        and target M&E envelopes, without building an EDL or rendering anything.

        Broadcast-speed mismatches are handled by first estimating the global speed
        ratio (target/ref) over a coarse grid of known standards, resampling the
        target to the reference timescale, and only then measuring per-window
        residual offsets (whose slope is the residual drift, ~0 when speed-locked).

        Returns a list of probe records:
            {ref_time, tar_time, offset, correlation, speed_ratio}.
        """
        sr_r, audio_r = wavfile.read(ref_wav_path)
        sr_t, audio_t = wavfile.read(tar_wav_path)
        audio_r = self._to_mono_float(audio_r)
        audio_t = self._to_mono_float(audio_t)

        ref_env, sr_r2 = self._me_envelope(audio_r, sr_r)
        tar_env, sr_t2 = self._me_envelope(audio_t, sr_t)
        if sr_r2 != sr_t2:
            common = max(sr_r2, sr_t2)
            if sr_r2 != common:
                ref_env = scipy.signal.resample(ref_env, int(len(ref_env) * common / sr_r2))
            if sr_t2 != common:
                tar_env = scipy.signal.resample(tar_env, int(len(tar_env) * common / sr_t2))
            sr2 = common
        else:
            sr2 = sr_r2

        ref_dur = len(ref_env) / float(sr2)
        tar_dur = len(tar_env) / float(sr2)
        if ref_dur < window_sec or tar_dur < window_sec:
            return []

        # --- Step 1: estimate the global speed ratio by correlating a central ---
        # reference window against the target resampled at each candidate ratio.
        c0 = ref_dur * 0.30
        c1 = min(c0 + window_sec, ref_dur - 1.0)
        if c1 <= c0:
            c0, c1 = 0.0, min(window_sec, ref_dur)
        wi, wj = int(c0 * sr2), int(c1 * sr2)
        ref_win = ref_env[wi:wj]
        r_norm = np.linalg.norm(ref_win)
        if r_norm < 1e-4:
            return []

        best_ratio, best_corr = 1.0, -1.0
        for ratio in self.CANDIDATE_SPEED_RATIOS:
            # Resample the target to the reference timescale for this ratio.
            tar_n = int(len(tar_env) * ratio)
            if tar_n < len(ref_win):
                continue
            tar_scaled = scipy.signal.resample(tar_env, tar_n).astype(np.float32)
            corr = scipy.signal.correlate(tar_scaled, ref_win, mode="valid")
            # Normalize by the sliding target-window energy.
            tar_sq = tar_scaled ** 2
            cum = np.concatenate(([0.0], np.cumsum(tar_sq)))
            L = len(ref_win)
            norms = np.sqrt(np.maximum(cum[L:] - cum[:-L], 0.0))
            corr_n = corr / np.maximum(r_norm * norms, 1e-8)
            peak = float(np.max(corr_n))
            if peak > best_corr:
                best_corr, best_ratio = peak, ratio

        # --- Step 2: resample target to the reference timescale at best ratio ---
        tar_aligned = scipy.signal.resample(
            tar_env, int(len(tar_env) * best_ratio)
        ).astype(np.float32)

        # --- Step 3: per-window residual offset (slope = residual drift) ---
        results: List[Dict[str, Any]] = []
        for t_ref in np.arange(0.0, ref_dur - window_sec, hop_sec):
            t1 = t_ref + window_sec
            res = self._normalized_correlation(
                ref_env, tar_aligned, sr2, t_ref, t1, t_ref, t1,
                search_ms=search_sec * 1000.0,
            )
            if res is None:
                continue
            lag, peak = res
            tar_time = t_ref + (lag / sr2)
            results.append({
                "ref_time": round(t_ref, 2),
                "tar_time": round(tar_time, 3),
                "offset": round(tar_time - t_ref, 4),
                "correlation": round(peak, 4),
                "speed_ratio": round(best_ratio, 5),
            })

        return results
