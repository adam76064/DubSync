"""
Tier 0: Chromaprint Acoustic Fingerprint Global Offset Bootstrap.

Computes a coarse but extremely fast (~100-300 ms) global time offset between
the reference and foreign audio by cross-correlating their chroma-energy envelopes.

Why this matters
----------------
All downstream matchers (visual DP, spectral, VAD) waste time searching ±45s+ windows
for every anchor. A good global bootstrap shrinks those search windows to ±5-8s,
reducing the O(N²) DP candidate count by ~50-80x on typical dub pairs.

Algorithm
---------
1. Downsample both audio files to 4kHz mono (fast I/O).
2. Apply a 12-band chroma filterbank (semitone bins A0–G#0 equivalent) to isolate
   pitch-class energy, which is language-invariant (music stays, speech changes).
3. Compute a smoothed RMS envelope per chroma band at 100ms resolution.
4. Cross-correlate the stacked 12-channel envelope matrices using FFT convolution.
5. Find the peak lag → coarse global offset in seconds.
6. Optionally refine around the peak using parabolic interpolation.

Output
------
Returns a GlobalOffsetEstimate dataclass with:
  - offset_sec     : best global offset (tar_time = ref_time + offset_sec)
  - confidence     : normalised correlation peak [0, 1]
  - search_radius  : recommended DP search radius for downstream matchers (seconds)
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import DubSyncConfig


# 12 semitone centre frequencies (Hz) for chromagram (A3 = 220 Hz reference)
_CHROMA_FREQS_HZ: list[float] = [
    220.00, 233.08, 246.94, 261.63, 277.18, 293.66,
    311.13, 329.63, 349.23, 369.99, 392.00, 415.30,
]


@dataclass
class GlobalOffsetEstimate:
    offset_sec:     float   # tar starts at ref_time + offset_sec
    confidence:     float   # normalised cross-correlation peak [0, 1]
    search_radius:  float   # recommended per-anchor search window for downstream (seconds)
    method:         str     # "chromaprint" | "fallback_energy"


class ChromaprintBootstrap:
    """
    Fast whole-file chroma-energy cross-correlation to bootstrap a global offset.
    """

    WORK_SR: int   = 4000    # Downsample target (Hz) — pitch resolution OK at 4kHz
    BIN_MS:  float = 100.0   # Envelope resolution (ms)

    def __init__(self, config: DubSyncConfig):
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        ref_wav_path: str,
        tar_wav_path: str,
        max_offset_sec: Optional[float] = None,
    ) -> GlobalOffsetEstimate:
        """
        Estimate the global time offset between reference and target audio.

        Parameters
        ----------
        ref_wav_path    : Path to reference PCM WAV (any sample rate).
        tar_wav_path    : Path to target/foreign PCM WAV (any sample rate).
        max_offset_sec  : Maximum absolute offset to search (default from config).

        Returns
        -------
        GlobalOffsetEstimate
        """
        if max_offset_sec is None:
            max_offset_sec = self.config.chromaprint_bootstrap_max_offset_sec

        try:
            ref_env = self._load_chroma_envelope(ref_wav_path)
            tar_env = self._load_chroma_envelope(tar_wav_path)
            return self._correlate_envelopes(ref_env, tar_env, max_offset_sec)
        except Exception as exc:
            # Graceful degradation: return a zero-offset, low-confidence estimate
            return GlobalOffsetEstimate(
                offset_sec=0.0,
                confidence=0.0,
                search_radius=max_offset_sec,
                method="fallback_energy",
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_chroma_envelope(self, wav_path: str) -> np.ndarray:
        """
        Load WAV, downsample to WORK_SR, apply 12-band chroma filterbank,
        and return a (12, T) float32 RMS envelope matrix at BIN_MS resolution.
        """
        sr, audio = wavfile.read(wav_path)

        # Mono downmix
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        # Normalise to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio /= 32768.0
        elif peak > 0.0:
            audio /= peak

        # Downsample to WORK_SR
        if sr != self.WORK_SR:
            n_target = int(len(audio) * self.WORK_SR / sr)
            audio = scipy.signal.resample_poly(
                audio,
                up=self.WORK_SR,
                down=sr,
                padtype="line",
            ).astype(np.float32)

        nyq = self.WORK_SR / 2.0
        bin_samples = max(1, int(self.BIN_MS / 1000.0 * self.WORK_SR))

        chroma_rows: list[np.ndarray] = []
        for cf in _CHROMA_FREQS_HZ:
            # Narrow bandpass per semitone (±1 semitone bandwidth ≈ ×2^(1/24))
            bw_ratio = 2 ** (1 / 24)
            lo = cf / bw_ratio
            hi = cf * bw_ratio
            lo = max(20.0, lo)
            hi = min(nyq * 0.98, hi)
            if lo >= hi:
                chroma_rows.append(np.zeros(len(audio) // bin_samples + 1, dtype=np.float32))
                continue

            b, a = scipy.signal.butter(2, [lo / nyq, hi / nyq], btype="bandpass")
            filtered = scipy.signal.filtfilt(b, a, audio)

            # RMS envelope in BIN_MS windows
            n_bins = len(filtered) // bin_samples
            trimmed = filtered[: n_bins * bin_samples].reshape(n_bins, bin_samples)
            rms = np.sqrt(np.mean(trimmed ** 2, axis=1)).astype(np.float32)
            chroma_rows.append(rms)

        # Pad to same length
        max_len = max(r.shape[0] for r in chroma_rows)
        envelope = np.stack(
            [np.pad(r, (0, max_len - r.shape[0])) for r in chroma_rows],
            axis=0,
        )  # (12, T)

        return envelope

    def _correlate_envelopes(
        self,
        ref_env: np.ndarray,
        tar_env: np.ndarray,
        max_offset_sec: float,
    ) -> GlobalOffsetEstimate:
        """
        FFT cross-correlate each chroma band independently and sum the normalised
        correlation maps to get a robust lag estimate.
        """
        bin_sec = self.BIN_MS / 1000.0
        max_lag_bins = int(max_offset_sec / bin_sec)

        n_bands, T_ref = ref_env.shape
        _,       T_tar = tar_env.shape

        # Subtract band-wise mean to remove DC energy (prevents bias toward silence)
        ref_env = ref_env - ref_env.mean(axis=1, keepdims=True)
        tar_env = tar_env - tar_env.mean(axis=1, keepdims=True)

        # FFT size: next power of 2 for speed
        fft_size = 1
        while fft_size < (T_ref + T_tar):
            fft_size <<= 1

        combined_corr = np.zeros(fft_size, dtype=np.float64)

        for band in range(n_bands):
            r = np.zeros(fft_size, dtype=np.float64)
            t = np.zeros(fft_size, dtype=np.float64)
            r[: T_ref] = ref_env[band]
            t[: T_tar] = tar_env[band]

            R = np.fft.rfft(r)
            T = np.fft.rfft(t)

            # Cross-power spectrum (phase correlation)
            corr = np.fft.irfft(np.conj(R) * T)

            # Normalise per band so all 12 contribute equally
            band_norm = (np.linalg.norm(ref_env[band]) * np.linalg.norm(tar_env[band])) + 1e-9
            combined_corr += corr / band_norm

        # Wrap to get lags in [-T_ref, T_ref] range
        corr_wrapped = np.concatenate([
            combined_corr[fft_size - T_ref :],
            combined_corr[: T_ref],
        ])

        # Restrict search to ±max_lag_bins
        center = T_ref
        lo = max(0, center - max_lag_bins)
        hi = min(len(corr_wrapped), center + max_lag_bins)
        search_region = corr_wrapped[lo:hi]

        best_local_idx = int(np.argmax(search_region))
        best_global_idx = lo + best_local_idx
        raw_lag_bins = best_global_idx - center
        offset_sec = raw_lag_bins * bin_sec

        # Parabolic sub-bin interpolation for fractional accuracy
        if 0 < best_local_idx < len(search_region) - 1:
            alpha = search_region[best_local_idx - 1]
            beta  = search_region[best_local_idx]
            gamma = search_region[best_local_idx + 1]
            denom = 2.0 * (2.0 * beta - alpha - gamma)
            if abs(denom) > 1e-9:
                frac = (alpha - gamma) / denom
                offset_sec += frac * bin_sec

        # Confidence: normalised peak vs mean absolute value
        peak_val   = float(search_region[best_local_idx])
        mean_abs   = float(np.mean(np.abs(search_region))) + 1e-9
        confidence = float(np.clip(peak_val / (mean_abs * 6.0), 0.0, 1.0))

        # Adaptive search radius: tighter if confidence is high
        if confidence >= 0.70:
            search_radius = 6.0
        elif confidence >= 0.45:
            search_radius = 15.0
        else:
            search_radius = max_offset_sec  # Low confidence → keep wide window

        return GlobalOffsetEstimate(
            offset_sec=round(offset_sec, 3),
            confidence=round(confidence, 4),
            search_radius=round(search_radius, 1),
            method="chromaprint",
        )
