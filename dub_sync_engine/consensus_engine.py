"""
Multi-Modal Consensus Matcher Engine for DubSync Pro.
Fuses Visual Keyframes, Spectral Music Transients, and Neural Silero VAD into a Unified Confidence Lattice.
"""

import numpy as np
import scipy.signal
from typing import List, Dict, Any, Optional, Tuple

from .config import DubSyncConfig
from .visual_anchors import VisualAnchorEngine, AnchorMatch, VisualAnchor
from .spectral_fingerprint import SpectralFingerprintEngine
from .vad_engine import SileroVADEngine


class MultiModalConsensusEngine:
    """
    Fuses visual scene changes, background music transient beats, and
    neural vocal cord probabilities into a joint multi-sensor alignment lattice.
    """

    # Candidate speed ratios searched when estimating the global target/ref
    # speed (target timeline length / reference timeline length). Includes the
    # broadcast standards plus intermediate values — VFR sources (e.g. 24.17fps)
    # rarely land exactly on a standard, so a hardcoded 0.96 assumption smears
    # every correlation peak and starves the anchor set.
    CANDIDATE_SPEED_RATIOS = (
        0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        1.0, 1.001, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10,
    )

    def __init__(self, config: Optional[DubSyncConfig] = None):
        self.config = config or DubSyncConfig()
        self.visual_engine = VisualAnchorEngine(self.config)
        self.spectral_engine = SpectralFingerprintEngine(self.config)
        self.vad_engine = SileroVADEngine(self.config)
        # Populated during discover_consensus_anchors() so the pipeline/report can
        # see *why* the anchor count is what it is (e.g. visual matches found but
        # gated out vs. never found).
        self.last_diagnostics: Dict[str, Any] = {}

    def _estimate_global_speed(self, env_r: np.ndarray, env_t: np.ndarray, bin_sr: float) -> float:
        """
        Estimate the target/ref speed ratio by correlating a central reference
        window against the target resampled at each candidate ratio and keeping
        the ratio that yields the strongest normalized peak.

        This replaces the former hardcoded PAL assumption (24.0/25.0). A wrong
        speed smears the cross-correlation peaks used to discover acoustic
        anchors, which is exactly how the Hero episode collapsed to 7 weak
        anchors.
        """
        ds = max(1, int(round(bin_sr / 10.0)))
        env_r = env_r[::ds]
        env_t = env_t[::ds]

        n_r = len(env_r)
        w0 = int(n_r * 0.35)
        w1 = min(n_r, w0 + int(n_r * 0.30))
        ref_win = env_r[w0:w1]
        ref_win = ref_win - np.mean(ref_win)
        r_norm = np.linalg.norm(ref_win)
        if r_norm < 1e-6 or len(ref_win) < 32:
            return 1.0

        best_ratio, best_peak = 1.0, -1.0
        for ratio in self.CANDIDATE_SPEED_RATIOS:
            n_t2 = int(len(env_t) / ratio)
            if n_t2 < len(ref_win):
                continue
            tar_scaled = scipy.signal.resample(env_t, n_t2).astype(np.float32)
            corr = scipy.signal.correlate(tar_scaled, ref_win, mode="valid")
            tar_sq = tar_scaled ** 2
            cum = np.concatenate(([0.0], np.cumsum(tar_sq)))
            L = len(ref_win)
            norms = np.sqrt(np.maximum(cum[L:] - cum[:-L], 0.0))
            corr_n = corr / np.maximum(r_norm * norms, 1e-8)
            peak = float(np.max(corr_n))
            if peak > best_peak:
                best_peak, best_ratio = peak, ratio
        return best_ratio

    def discover_consensus_anchors(
        self,
        ref_anchors: List[VisualAnchor],
        tar_anchors: List[VisualAnchor],
        ref_wav_path: str,
        tar_wav_path: str,
        ref_duration: float,
        tar_duration: float
    ) -> List[AnchorMatch]:
        """
        Acoustic-First Gated Multi-Frame Architecture:
        1. Uses Background Music (800Hz-3500Hz) & Neural Speech VAD as the primary immutable spine.
        2. Gates visual keyframe matching within tight +/-2.0s windows around acoustic anchors.
        3. Enforces multi-frame sequence consistency to eliminate false cartoon matches.
        """
        candidates: List[Dict[str, Any]] = []
        self.last_diagnostics = {
            "raw_visual_matches_found": 0,
            "visual_matches_gated_in": 0,
            "visual_matches_gated_out": 0,
            "acoustic_candidates": 0,
            "vad_candidates": 0,
            "estimated_speed_ratio": None,
        }

        # --- STEP 1: Acoustic Background Music & Speech Envelopes (Primary Master Spine) ---
        try:
            import scipy.io.wavfile as wavfile
            sr_r, a_r = wavfile.read(ref_wav_path)
            sr_t, a_t = wavfile.read(tar_wav_path)

            # Mono downmix
            if a_r.ndim > 1: a_r = np.mean(a_r, axis=1)
            if a_t.ndim > 1: a_t = np.mean(a_t, axis=1)

            # Downsample to 8kHz for fast acoustic processing
            step_r = max(1, sr_r // 8000)
            step_t = max(1, sr_t // 8000)
            a_r = a_r[::step_r].astype(np.float32)
            a_t = a_t[::step_t].astype(np.float32)
            sr = 8000

            # Bandpass music & ambient sound effects (800Hz to 3200Hz)
            b, a = scipy.signal.butter(3, [800.0 / (sr/2), 3200.0 / (sr/2)], btype='band')
            env_r = np.abs(scipy.signal.filtfilt(b, a, a_r))
            env_t = np.abs(scipy.signal.filtfilt(b, a, a_t))

            # 20ms envelope binning
            w_smooth = int(sr * 0.020)
            env_r_ds = np.convolve(env_r, np.ones(w_smooth)/w_smooth, mode='same')[::w_smooth]
            env_t_ds = np.convolve(env_t, np.ones(w_smooth)/w_smooth, mode='same')[::w_smooth]
            dt = 0.020  # 20ms bins

            # Estimate the true speed ratio instead of assuming PAL 0.960000x.
            # A hardcoded assumption smears the correlation peaks for VFR/non-PAL
            # sources (the Hero case is 640x480 @ 24.17fps VFR), starving anchors.
            g_speed = self._estimate_global_speed(env_r_ds, env_t_ds, 1.0 / dt)
            self.last_diagnostics["estimated_speed_ratio"] = round(g_speed, 5)
            t_resampled = scipy.signal.resample(env_t_ds, int(len(env_t_ds) / g_speed))

            # Scan master in 15-second sliding windows with 10-second hop
            win_len = int(12.0 / dt)
            for sec in range(10, int(len(env_r_ds)*dt) - 15, 10):
                f1 = int(sec / dt)
                f2 = f1 + win_len
                if f2 > len(env_r_ds): break

                r_slice = env_r_ds[f1:f2] - np.mean(env_r_ds[f1:f2])
                r_norm = np.linalg.norm(r_slice)
                if r_norm < 10.0: continue

                corr = scipy.signal.correlate(t_resampled, r_slice, mode='valid')
                best_lag = int(np.argmax(corr))
                tar_t_norm = best_lag * dt
                tar_actual_t = tar_t_norm * g_speed
                offset = tar_actual_t - sec

                t_slice = t_resampled[best_lag : best_lag + win_len] - np.mean(t_resampled[best_lag : best_lag + win_len])
                t_norm = np.linalg.norm(t_slice)
                peak = corr[best_lag] / (r_norm * t_norm + 1e-8)

                if peak >= self.config.min_acoustic_peak and tar_actual_t > 0:
                    candidates.append({
                        "ref_time": float(sec),
                        "tar_time": float(tar_actual_t),
                        "offset": float(offset),
                        "confidence": float(peak),
                        "weight": float(peak),  # continuous confirmation strength
                        "source": "acoustic_music",
                        "score": float(peak) * 14.0
                    })
        except Exception:
            pass

        # --- STEP 2: Neural Silero VAD Dialogue Bursts (Secondary Spine) ---
        try:
            p_ref, dt_r = self.vad_engine.compute_speech_probabilities(ref_wav_path)
            p_tar, dt_t = self.vad_engine.compute_speech_probabilities(tar_wav_path)

            ker = np.ones(5) / 5.0
            p_r_s = np.convolve(p_ref, ker, mode='same')
            p_t_s = np.convolve(p_tar, ker, mode='same')

            dt_vad = dt_r
            win_frames = int(12.0 / dt_vad)
            hop_frames = int(10.0 / dt_vad)

            for f_ref in range(0, len(p_r_s) - win_frames, hop_frames):
                t_ref = f_ref * dt_vad
                r_slice = p_r_s[f_ref : f_ref + win_frames] - np.mean(p_r_s[f_ref : f_ref + win_frames])
                r_norm = np.linalg.norm(r_slice)
                if r_norm < 0.20: continue

                expected_tar_f = int(f_ref * (len(p_t_s) / max(1, len(p_r_s))))
                search_band = int(45.0 / dt_vad)
                s_start = max(0, expected_tar_f - search_band)
                s_end = min(len(p_t_s), expected_tar_f + win_frames + search_band)

                t_win = p_t_s[s_start : s_end] - np.mean(p_t_s[s_start : s_end])
                if len(t_win) < win_frames: continue

                corr = scipy.signal.correlate(t_win, r_slice, mode='valid')
                best_lag = int(np.argmax(corr))
                t_norm = np.linalg.norm(t_win[best_lag : best_lag + win_frames])
                norm_peak = corr[best_lag] / (r_norm * t_norm + 1e-8)

                if norm_peak >= self.config.min_vad_peak:
                    tar_f = s_start + best_lag
                    t_tar = tar_f * dt_vad
                    offset = t_tar - t_ref
                    candidates.append({
                        "ref_time": float(t_ref),
                        "tar_time": float(t_tar),
                        "offset": float(offset),
                        "confidence": min(1.0, float(norm_peak)),
                        "weight": min(1.0, float(norm_peak)),  # continuous confirmation strength
                        "source": "vad_speech",
                        "score": float(norm_peak) * 11.0
                    })
        except Exception:
            pass

        # --- STEP 3: Audio-Gated Multi-Frame Visual Verification ---
        if ref_anchors and tar_anchors:
            raw_visual_matches = self.visual_engine.match_anchors(ref_anchors, tar_anchors)
            self.last_diagnostics["raw_visual_matches_found"] = len(raw_visual_matches)

            # Compute acoustic median offset baseline
            acoustic_offsets = [c["offset"] for c in candidates if c["source"] in ["acoustic_music", "vad_speech"]]
            med_offset = float(np.median(acoustic_offsets)) if acoustic_offsets else None

            for m in raw_visual_matches:
                # A visual match is admitted only when confirmed by an acoustic anchor
                # within a tight window with a consistent offset (no confidence bypass:
                # an isolated frame match with no acoustic support is always rejected).
                # The *strength* of that confirmation becomes a continuous weight for
                # the downstream RANSAC fit, instead of a hard accept/reject.
                win = self.config.acoustic_gate_window_sec
                off = self.config.acoustic_gate_offset_sec
                confirming = [
                    c for c in candidates
                    if c["source"] in ["acoustic_music", "vad_speech"]
                    and abs(m.ref_time - c["ref_time"]) <= win
                    and abs(m.offset - c["offset"]) <= off
                ]

                if confirming:
                    strength = max(float(c["confidence"]) for c in confirming)
                    self.last_diagnostics["visual_matches_gated_in"] += 1
                    candidates.append({
                        "ref_time": m.ref_time,
                        "tar_time": m.tar_time,
                        "offset": m.offset,
                        "confidence": m.confidence,
                        "weight": round(strength, 4),
                        "source": "visual_gated",
                        "score": m.confidence * 14.0
                    })
                else:
                    self.last_diagnostics["visual_matches_gated_out"] += 1

        self.last_diagnostics["acoustic_candidates"] = sum(
            1 for c in candidates if c["source"] == "acoustic_music")
        self.last_diagnostics["vad_candidates"] = sum(
            1 for c in candidates if c["source"] == "vad_speech")

        if not candidates:
            return []

        # --- STEP 4: Monotonic Physical Speed Dynamic Programming Lattice ---
        candidates.sort(key=lambda x: (x["ref_time"], x["tar_time"]))
        N = len(candidates)
        dp = [c["score"] for c in candidates]
        parent = [-1] * N

        for i in range(N):
            ci = candidates[i]
            for j in range(i):
                cj = candidates[j]
                dt_r = ci["ref_time"] - cj["ref_time"]
                dt_t = ci["tar_time"] - cj["tar_time"]

                # Physical Causality Constraint:
                # Both reference and target time MUST progress forward monotonically
                if dt_r > 0.2 and dt_t > 0.1:
                    speed = dt_t / dt_r

                    # Broadcast Continuous Act: 0.93x <= speed <= 1.05x
                    if 0.93 <= speed <= 1.05:
                        speed_penalty = abs(speed - 0.960) * 4.0
                        gain = ci["score"] - speed_penalty
                        if dp[j] + gain > dp[i]:
                            dp[i] = dp[j] + gain
                            parent[i] = j

                    # Genuine TV Cut / Censored Scene (Speed < 0.93x across genuine gaps)
                    elif speed < 0.93 and dt_r >= 3.5:
                        cut_penalty = 1.0
                        gain = ci["score"] - cut_penalty
                        if dp[j] + gain > dp[i]:
                            dp[i] = dp[j] + gain
                            parent[i] = j

        best_idx = int(np.argmax(dp))
        chain = []
        curr = best_idx
        idx = 0

        while curr != -1:
            c = candidates[curr]
            chain.append(AnchorMatch(
                ref_idx=idx,
                tar_idx=idx,
                ref_time=round(c["ref_time"], 3),
                tar_time=round(c["tar_time"], 3),
                hash_dist=0,
                confidence=round(c["confidence"], 3),
                offset=round(c["offset"], 4),
                weight=round(float(c.get("weight", 1.0)), 4),
                source=c.get("source", "unknown")
            ))
            curr = parent[curr]
            idx += 1

        chain.reverse()
        return chain
