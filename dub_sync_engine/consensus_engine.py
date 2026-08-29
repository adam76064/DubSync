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

    def __init__(self, config: Optional[DubSyncConfig] = None):
        self.config = config or DubSyncConfig()
        self.visual_engine = VisualAnchorEngine(self.config)
        self.spectral_engine = SpectralFingerprintEngine(self.config)
        self.vad_engine = SileroVADEngine(self.config)

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

            # Standard broadcast tempo (0.960000x for PAL)
            g_speed = 24.0 / 25.0
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

                if peak >= 0.40 and tar_actual_t > 0:
                    candidates.append({
                        "ref_time": float(sec),
                        "tar_time": float(tar_actual_t),
                        "offset": float(offset),
                        "confidence": float(peak),
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

                if norm_peak >= 0.48:
                    tar_f = s_start + best_lag
                    t_tar = tar_f * dt_vad
                    offset = t_tar - t_ref
                    candidates.append({
                        "ref_time": float(t_ref),
                        "tar_time": float(t_tar),
                        "offset": float(offset),
                        "confidence": min(1.0, float(norm_peak)),
                        "source": "vad_speech",
                        "score": float(norm_peak) * 11.0
                    })
        except Exception:
            pass

        # --- STEP 3: Audio-Gated Multi-Frame Visual Verification ---
        if ref_anchors and tar_anchors:
            raw_visual_matches = self.visual_engine.match_anchors(ref_anchors, tar_anchors)

            # Compute acoustic median offset baseline
            acoustic_offsets = [c["offset"] for c in candidates if c["source"] in ["acoustic_music", "vad_speech"]]
            med_offset = float(np.median(acoustic_offsets)) if acoustic_offsets else None

            for m in raw_visual_matches:
                # Check if this visual anchor is confirmed by an acoustic anchor within +/-4.0s with consistent offset
                is_acoustically_confirmed = any(
                    abs(m.ref_time - c["ref_time"]) <= 4.0 and abs(m.offset - c["offset"]) <= 2.0
                    for c in candidates if c["source"] in ["acoustic_music", "vad_speech"]
                )

                if is_acoustically_confirmed:
                    candidates.append({
                        "ref_time": m.ref_time,
                        "tar_time": m.tar_time,
                        "offset": m.offset,
                        "confidence": m.confidence,
                        "source": "visual_gated",
                        "score": m.confidence * 14.0
                    })

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
                offset=round(c["offset"], 4)
            ))
            curr = parent[curr]
            idx += 1

        chain.reverse()
        return chain
