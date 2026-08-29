"""
Tier 3 Fallback: Spectral Music & Sound Effect Landmark Fingerprint Matcher.
Matches audio tracks by isolating invariant background music chords and percussive transients,
completely bypassing speech language differences and video distortion.
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
from typing import List, Dict, Tuple, Optional

from .visual_anchors import AnchorMatch
from .config import DubSyncConfig


class SpectralFingerprintEngine:
    """Discovers audio sync offsets via vocal-attenuated spectral energy and transient correlation."""

    def __init__(self, config: DubSyncConfig):
        self.config = config

    def extract_me_envelope(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Applies vocal band attenuation (speech suppression) and computes the energy envelope.
        """
        # Bandpass filter around 1000Hz - 4000Hz or notch filter dialogue (300 - 3000Hz)
        nyq = sr / 2.0
        # Emphasize high frequencies & percussion transients
        b, a = scipy.signal.butter(2, [1200 / nyq, min(0.95, 3800 / nyq)], btype='bandpass')
        filtered = scipy.signal.filtfilt(b, a, audio_data.astype(np.float32))

        # Compute smoothed Hilbert envelope
        analytic = scipy.signal.hilbert(filtered)
        envelope = np.abs(analytic)
        
        # Smooth envelope
        smooth_win = int(0.05 * sr)
        if smooth_win > 1:
            kernel = np.ones(smooth_win) / smooth_win
            envelope = np.convolve(envelope, kernel, mode='same')

        return envelope

    def discover_spectral_anchors(
        self,
        ref_wav_path: str,
        tar_wav_path: str,
        ref_duration: float,
        tar_duration: float,
        step_seconds: float = 30.0,
        probe_duration: float = 15.0
    ) -> List[AnchorMatch]:
        """
        Probes the reference track at regular intervals and locates matching music/SFX
        transients in the target track.
        """
        sr_r, a_r = wavfile.read(ref_wav_path)
        sr_t, a_t = wavfile.read(tar_wav_path)

        if a_r.ndim > 1:
            a_r = np.mean(a_r, axis=1)
        if a_t.ndim > 1:
            a_t = np.mean(a_t, axis=1)

        # Downsample to 8000Hz for fast macro correlation
        target_sr = 8000
        factor_r = max(1, sr_r // target_sr)
        factor_t = max(1, sr_t // target_sr)

        a_r_ds = a_r[::factor_r].astype(np.float32)
        a_t_ds = a_t[::factor_t].astype(np.float32)

        env_r = self.extract_me_envelope(a_r_ds, target_sr)
        env_t = self.extract_me_envelope(a_t_ds, target_sr)

        env_r -= np.mean(env_r)
        env_t -= np.mean(env_t)

        win_samples = int(probe_duration * target_sr)
        candidates = []

        # Probe every step_seconds
        for t_ref in np.arange(15.0, ref_duration - probe_duration - 10.0, step_seconds):
            idx_r = int(t_ref * target_sr)
            if idx_r + win_samples > len(env_r):
                break

            ref_slice = env_r[idx_r : idx_r + win_samples]
            r_norm = np.linalg.norm(ref_slice)
            if r_norm < 1e-3:
                continue

            # Correlate across full target audio
            corr = scipy.signal.correlate(env_t, ref_slice, mode='valid')
            best_lag = int(np.argmax(corr))
            tar_slice = env_t[best_lag : best_lag + win_samples]
            t_norm = np.linalg.norm(tar_slice)

            norm_peak = float(corr[best_lag] / (r_norm * t_norm + 1e-8))

            if norm_peak >= 0.12:
                tar_time = best_lag / float(target_sr)
                offset = tar_time - t_ref
                candidates.append({
                    "ref_time": float(t_ref),
                    "tar_time": float(tar_time),
                    "offset": float(offset),
                    "confidence": min(1.0, norm_peak * 2.0),
                    "score": norm_peak * 10.0
                })

        if not candidates:
            return []

        # DP monotonic path filtering
        candidates.sort(key=lambda x: (x["ref_time"], x["tar_time"]))
        N = len(candidates)
        dp = [c["score"] for c in candidates]
        parent = [-1] * N

        for i in range(N):
            ci = candidates[i]
            for j in range(i):
                cj = candidates[j]
                if cj["ref_time"] < ci["ref_time"] and cj["tar_time"] < ci["tar_time"]:
                    dt_r = ci["ref_time"] - cj["ref_time"]
                    dt_t = ci["tar_time"] - cj["tar_time"]
                    if dt_r <= 0:
                        continue
                    speed = dt_t / dt_r
                    if 0.90 <= speed <= 1.10:
                        gain = ci["score"] - abs(speed - 1.0) * 8.0
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
                ref_time=c["ref_time"],
                tar_time=c["tar_time"],
                hash_dist=0,
                confidence=round(c["confidence"], 3),
                offset=round(c["offset"], 4)
            ))
            curr = parent[curr]
            idx += 1

        chain.reverse()
        return chain
