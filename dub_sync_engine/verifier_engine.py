"""
Autonomous Closed-Loop Self-Verification & Healing Engine for DubSync Pro.

v2.3 Enhancements
-----------------
* FIXED: Absolute RMS threshold (> 200.0) replaced with gain-normalised SNR check.
  The old code broke silently on quiet dub sources (e.g. dialogue at -24 dBFS),
  causing every valid dub segment to be incorrectly replaced with English fallback.

* FIXED: Correlation threshold (0.48) is now configurable via config.verifier_corr_min
  and defaults to 0.28, which is more appropriate for spectrally-filtered audio
  (speech band attenuated before correlation).

* NEW: Noise-floor estimation using the 5th-percentile RMS of 0.5-s frames,
  giving a per-file adaptive baseline independent of absolute gain level.

* NEW: Fallback gap merging — adjacent fallback segments separated by < 0.5 s
  of dub are collapsed into one gap, avoiding micro-blip English insertions.

* NEW: Audit log now records per-gap SNR and correlation values for forensic reports.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from .config import DubSyncConfig
from .audio_splicer import SegmentEDL


@dataclass
class VerificationAudit:
    total_probed_windows:       int
    mean_alignment_error_ms:    Optional[float]   # None when it could not be measured
    max_alignment_error_ms:     Optional[float]
    passed_windows_pct:         Optional[float]
    false_fallbacks_healed_count: int
    healed_duration_sec:        float
    audit_log:                  List[Dict[str, Any]] = field(default_factory=list)
    windows_measured:           int = 0           # windows that yielded a usable measurement


class ClosedLoopVerifierEngine:
    """
    Autonomously audits the Edit Decision List against reference and foreign signals,
    probes candidate audio to eliminate false English fallbacks, and verifies
    sub-frame precision alignment.
    """

    def __init__(self, config: Optional[DubSyncConfig] = None):
        self.config = config or DubSyncConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def audit_and_heal_edl(
        self,
        edl:          List[SegmentEDL],
        ref_wav_path: str,
        tar_wav_path: str,
        ref_duration: float,
        tar_duration: float,
    ) -> Tuple[List[SegmentEDL], VerificationAudit]:
        """
        Runs continuity verification on fallback intervals and performs closed-loop healing.
        """
        sr_r, audio_r = wavfile.read(ref_wav_path)
        sr_t, audio_t = wavfile.read(tar_wav_path)

        audio_r = self._to_mono_float(audio_r)
        audio_t = self._to_mono_float(audio_t)

        # ── Compute adaptive noise floors (gain-normalised baseline) ──────────
        noise_floor_r = self._estimate_noise_floor(audio_r, sr_r)
        noise_floor_t = self._estimate_noise_floor(audio_t, sr_t)

        healed_edl:             List[SegmentEDL]     = []
        false_fallbacks_healed: int                  = 0
        healed_duration:        float                = 0.0
        audit_records:          List[Dict[str, Any]] = []

        for seg in edl:
            if seg.segment_type != "fallback":
                healed_edl.append(seg)
                continue

            r_start = seg.ref_start
            r_end   = seg.ref_end
            dur     = seg.ref_duration

            # Don't attempt healing on very long gaps (likely genuine cut)
            if dur > self.config.verifier_max_gap_sec:
                healed_edl.append(seg)
                audit_records.append({
                    "ref_start": r_start, "ref_end": r_end,
                    "action": "SKIP_TOO_LONG", "duration": dur,
                })
                continue

            prev_dub = next(
                (s for s in reversed(healed_edl) if s.segment_type == "dub"),
                None,
            )
            speed = prev_dub.speed_factor if prev_dub else 1.0

            if prev_dub is None:
                healed_edl.append(seg)
                continue

            proj_tar_start = prev_dub.tar_end
            proj_tar_end   = proj_tar_start + (dur * speed)

            idx_t1 = int(proj_tar_start * sr_t)
            idx_t2 = int(proj_tar_end   * sr_t)
            idx_r1 = int(r_start        * sr_r)
            idx_r2 = int(r_end          * sr_r)

            if idx_t2 > len(audio_t) or idx_r2 > len(audio_r) or idx_t2 <= idx_t1:
                healed_edl.append(seg)
                continue

            tar_slice = audio_t[idx_t1:idx_t2]
            ref_slice = audio_r[idx_r1:idx_r2]

            # ── Gain-normalised SNR check (v2.3 fix) ──────────────────────────
            tar_rms = float(np.sqrt(np.mean(tar_slice ** 2)) + 1e-9)
            snr_db  = 20.0 * np.log10(tar_rms / (noise_floor_t + 1e-9))

            # ── Normalised cross-correlation check ────────────────────────────
            corr_val = self._fast_norm_corr(ref_slice, tar_slice, sr_r)

            snr_ok  = snr_db  >= self.config.verifier_rms_snr_db
            corr_ok = corr_val >= self.config.verifier_corr_min

            if snr_ok and corr_ok:
                # Convert false fallback back to continuous dub
                healed_seg = SegmentEDL(
                    seg_id        = seg.seg_id,
                    segment_type  = "dub",
                    ref_start     = r_start,
                    ref_end       = r_end,
                    tar_start     = round(proj_tar_start, 3),
                    tar_end       = round(proj_tar_end, 3),
                    speed_factor  = speed,
                    confidence    = 0.92,
                )
                healed_edl.append(healed_seg)
                false_fallbacks_healed += 1
                healed_duration        += dur
                audit_records.append({
                    "ref_start": r_start, "ref_end": r_end,
                    "action": "HEALED_FALSE_FALLBACK",
                    "healed_duration": dur,
                    "snr_db": round(snr_db, 2),
                    "correlation": round(corr_val, 4),
                })
            else:
                # Confirmed genuine cut omission
                healed_edl.append(seg)
                audit_records.append({
                    "ref_start": r_start, "ref_end": r_end,
                    "action": "CONFIRMED_GENUINE_CUT",
                    "duration": dur,
                    "snr_db": round(snr_db, 2),
                    "correlation": round(corr_val, 4),
                })

        # ── Merge adjacent dub segments with matching speed ───────────────────
        compact_edl = self._merge_adjacent_dub(healed_edl)

        # ── Merge micro-blip fallbacks (< 0.5 s) sandwiched between dub ──────
        compact_edl = self._merge_micro_fallbacks(compact_edl, min_gap_sec=0.5)

        for idx, s in enumerate(compact_edl):
            s.seg_id = idx

        # ── Measure the residual sync error on the FINAL timeline ─────────────
        # This is a real measurement, not a constant: every dub segment is
        # probed with normalised cross-correlation against the reference.
        mean_ms, max_ms, passed_pct, n_measured = self._measure_alignment_error(
            compact_edl, ref_wav_path, tar_wav_path
        )

        audit = VerificationAudit(
            total_probed_windows        = len(audit_records),
            mean_alignment_error_ms     = mean_ms,
            max_alignment_error_ms      = max_ms,
            passed_windows_pct          = passed_pct,
            false_fallbacks_healed_count= false_fallbacks_healed,
            healed_duration_sec         = round(healed_duration, 2),
            audit_log                   = audit_records,
            windows_measured            = n_measured,
        )
        return compact_edl, audit

    # ── Alignment measurement ─────────────────────────────────────────────────

    @staticmethod
    def _me_envelope(audio: np.ndarray, sr: int, band: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, float]:
        """
        RMS envelope at ~1 ms resolution.

        Defaults to the FULL band, because what makes a measurement reliable is
        aperiodic structure - the pattern of speech bursts and pauses, which a
        dub preserves even though the words change. Narrow-banded M&E energy
        tends to be quasi-periodic (a steady beat), which produces spurious
        period-shifted correlation peaks over a wide search window.
        """
        sig = audio.astype(np.float64)
        if band is not None:
            nyq = sr / 2.0
            lo, hi = band[0] / nyq, min(band[1] / nyq, 0.95)
            if hi > lo:
                b, a = scipy.signal.butter(3, [lo, hi], btype="band")
                sig = scipy.signal.filtfilt(b, a, sig)

        win = max(1, int(0.001 * sr))          # 1 ms bins
        csum = np.cumsum(np.insert(sig * sig, 0, 0.0))
        env = np.sqrt(np.maximum(csum[win:] - csum[:-win], 0.0) / win)[::win]
        env = env - env.mean()                  # remove DC so silence cannot bias the peak
        return env, win / sr

    def _measure_alignment_error(
        self,
        edl: List[SegmentEDL],
        ref_wav_path: str,
        tar_wav_path: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
        """
        Probes dub segments across the finished timeline and measures how far
        the dub content sitting at reference time T is from the reference
        content at T.

        Method: each segment's dub envelope is resampled onto the reference
        timeline using that segment's EDL mapping (tar = tar_start +
        (ref - ref_start) * speed) and window-normalised cross-correlated
        against the reference envelope. The peak lag is the residual sync error.

        Returns (mean_ms, max_ms, passed_pct, windows_measured). Any statistic
        is None when nothing could be measured - the caller must render that as
        "not measured", never as a fabricated number.
        """
        try:
            sr_r, audio_r = wavfile.read(ref_wav_path)
            sr_t, audio_t = wavfile.read(tar_wav_path)
            audio_r = self._to_mono_float(audio_r)
            audio_t = self._to_mono_float(audio_t)

            env_r, bin_r = self._me_envelope(audio_r, sr_r)
            env_t, bin_t = self._me_envelope(audio_t, sr_t)
            if np.linalg.norm(env_r) < 1e-9 or np.linalg.norm(env_t) < 1e-9:
                return None, None, None, 0

            ref_total = len(env_r) * bin_r
            tar_grid = np.arange(len(env_t)) * bin_t

            dub_segs = [s for s in edl if s.segment_type == "dub" and s.ref_duration > 0.5]
            if not dub_segs:
                return None, None, None, 0

            n_windows  = max(1, int(self.config.verifier_probe_windows))
            win_sec    = float(self.config.verifier_probe_window_sec)
            # Wide enough to catch gross mis-sync (a whole scene bridged or
            # shifted), which a sub-second search window would clip.
            max_lag_sec = 10.0
            pass_ms     = float(self.config.verifier_pass_threshold_ms)

            errors_ms      = []
            n_probed       = 0
            n_unmeasurable = 0

            total_dub = sum(s.ref_duration for s in dub_segs)
            for seg in dub_segs:
                speed = seg.speed_factor if seg.speed_factor > 0 else 1.0
                share = max(1, int(round(n_windows * seg.ref_duration / total_dub)))
                step  = max(0.5, (seg.ref_duration - win_sec) / max(1, share))
                nb    = max(1, int(round(win_sec / bin_r)))
                lag_b = int(max_lag_sec / bin_r)

                # Reference-time grid covering the segment plus search margin
                g0 = max(0.0, seg.ref_start - max_lag_sec)
                g1 = min(ref_total, seg.ref_end + max_lag_sec + win_sec)
                if g1 - g0 < win_sec:
                    continue
                grid = np.arange(g0, g1, bin_r)

                # The dub envelope resampled onto the reference timeline via the
                # EDL mapping: ref time r  ->  tar time tar_start + (r-ref_start)*speed
                tar_times = seg.tar_start + (grid - seg.ref_start) * speed
                dub_on_ref = np.interp(tar_times, tar_grid, env_t, left=0.0, right=0.0)
                ref_on_ref = np.interp(grid, np.arange(len(env_r)) * bin_r, env_r,
                                       left=0.0, right=0.0)

                for i in range(share):
                    off = min(i * step, max(0.0, seg.ref_duration - win_sec))
                    i0 = int(round((seg.ref_start + off - g0) / bin_r))
                    if i0 < 0 or i0 + nb > len(ref_on_ref):
                        continue
                    n_probed += 1

                    ref_win = ref_on_ref[i0 : i0 + nb]
                    if np.linalg.norm(ref_win) < 1e-9:
                        n_unmeasurable += 1      # silent reference: nothing to align to
                        continue

                    lo = max(0, i0 - lag_b)
                    hi = min(len(dub_on_ref), i0 + lag_b + nb)
                    region = dub_on_ref[lo:hi]
                    if len(region) < nb + 1:
                        n_unmeasurable += 1
                        continue

                    corr = scipy.signal.correlate(region, ref_win, mode="valid", method="fft")
                    csum = np.cumsum(np.insert(region.astype(np.float64) ** 2, 0, 0.0))
                    energy = csum[nb:] - csum[:-nb]
                    ncc = corr[: len(energy)] / (
                        np.sqrt(np.maximum(energy, 0)) * np.linalg.norm(ref_win) + 1e-9
                    )
                    if len(ncc) == 0:
                        n_unmeasurable += 1
                        continue

                    k = int(np.argmax(ncc))
                    peak = float(ncc[k])
                    if peak < 0.15:
                        # The dub does not resemble the reference here at all.
                        # That IS a sync failure - count it, never skip it.
                        n_unmeasurable += 1
                        continue

                    # Parabolic interpolation for sub-bin precision
                    if 0 < k < len(ncc) - 1:
                        alpha, beta, gamma = float(ncc[k - 1]), peak, float(ncc[k + 1])
                        denom = 2.0 * (2.0 * beta - alpha - gamma)
                        if abs(denom) > 1e-9:
                            k = k + (alpha - gamma) / denom

                    lag_bins = (lo + k) - i0
                    errors_ms.append(abs(lag_bins * bin_r) * 1000.0)

            if not errors_ms:
                # Either nothing at all was probed, or every window failed to
                # correlate - in both cases no accuracy figure may be claimed.
                return (None, None, 0.0, 0) if n_unmeasurable else (None, None, None, 0)

            n_pass = sum(1 for e in errors_ms if e <= pass_ms)
            return (
                round(float(np.mean(errors_ms)), 2),
                round(float(np.max(errors_ms)), 2),
                round(100.0 * n_pass / max(1, n_probed), 1),
                len(errors_ms),
            )
        except Exception:
            # Measurement failed - report nothing rather than invent a number.
            return None, None, None, 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_mono_float(audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio.astype(np.float32)

    @staticmethod
    def _estimate_noise_floor(audio: np.ndarray, sr: int, frame_sec: float = 0.5) -> float:
        """
        Estimates signal noise floor as the 5th-percentile RMS of 0.5-s frames.
        This is gain-invariant and robust to quiet dub tracks.
        """
        frame_len = int(frame_sec * sr)
        if frame_len <= 0 or len(audio) < frame_len:
            return float(np.sqrt(np.mean(audio ** 2)) + 1e-9)

        n_frames = len(audio) // frame_len
        frames   = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms_vals = np.sqrt(np.mean(frames ** 2, axis=1))
        return float(np.percentile(rms_vals, 5) + 1e-9)

    @staticmethod
    def _fast_norm_corr(ref: np.ndarray, tar: np.ndarray, sr: int, max_win: int = 4) -> float:
        """
        Computes normalised cross-correlation on a central window of up to max_win seconds.
        Returns scalar in [0, 1].
        """
        win = min(len(ref), len(tar), int(max_win * sr))
        if win < sr // 4:
            return 0.0
        r_s = ref[: win].astype(np.float64)
        t_s = tar[: win].astype(np.float64)
        r_n = r_s / (np.linalg.norm(r_s) + 1e-9)
        t_n = t_s / (np.linalg.norm(t_s) + 1e-9)
        corr = np.correlate(t_n, r_n, mode="valid")
        return float(np.max(np.abs(corr)) / win) if len(corr) > 0 else 0.0

    @staticmethod
    def _merge_adjacent_dub(edl: List[SegmentEDL]) -> List[SegmentEDL]:
        merged: List[SegmentEDL] = []
        for s in edl:
            if merged and merged[-1].segment_type == s.segment_type == "dub":
                prev = merged[-1]
                if (
                    abs(prev.speed_factor - s.speed_factor) < 0.003
                    and abs(prev.tar_end - s.tar_start) < 0.2
                ):
                    prev.ref_end = s.ref_end
                    prev.tar_end = s.tar_end
                    continue
            merged.append(s)
        return merged

    @staticmethod
    def _merge_micro_fallbacks(edl: List[SegmentEDL], min_gap_sec: float = 0.5) -> List[SegmentEDL]:
        """
        Absorbs tiny fallback gaps (< min_gap_sec) sandwiched between dub segments
        into the preceding dub, avoiding micro-blip English audio insertions.
        """
        if len(edl) < 3:
            return edl

        result: List[SegmentEDL] = [edl[0]]
        i = 1
        while i < len(edl):
            seg = edl[i]
            if (
                seg.segment_type == "fallback"
                and seg.ref_duration < min_gap_sec
                and i + 1 < len(edl)
                and edl[i + 1].segment_type == "dub"
                and result
                and result[-1].segment_type == "dub"
            ):
                # Absorb the micro fallback into the preceding dub tail
                result[-1].ref_end = seg.ref_end
                result[-1].tar_end = result[-1].tar_end + seg.ref_duration * result[-1].speed_factor
            else:
                result.append(seg)
            i += 1
        return result
