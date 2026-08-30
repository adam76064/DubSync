"""
Localized sub-millisecond acoustic cross-correlation refinement engine.
Snaps visual scene cut points to sample-accurate audio transients and beats.
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .visual_anchors import AnchorMatch
from .config import DubSyncConfig


@dataclass
class RefinedAnchor:
    ref_time: float
    tar_time: float
    acoustic_offset_ms: float
    visual_offset_s: float
    acoustic_confidence: float
    combined_confidence: float
    # Preserved from the source AnchorMatch (so the forensic report keeps the
    # modality, N-gram rhythm and confirmation weight after refinement).
    seq_len: int = 1
    weight: float = 1.0
    source: str = "unknown"
    hash_dist: int = 0


class AcousticRefineEngine:
    """Refines visual anchor cut points using localized 48kHz audio cross-correlation."""

    def __init__(self, config: DubSyncConfig):
        self.config = config

    def refine_anchors(
        self,
        ref_wav_path: str,
        tar_wav_path: str,
        matches: List[AnchorMatch],
        progress_callback: Optional[callable] = None
    ) -> List[RefinedAnchor]:
        """
        Loads uncompressed 48kHz PCM audio and performs localized normalized cross-correlation
        around each visual anchor match.
        """
        if not matches or not self.config.enable_acoustic_refine:
            # Fallback: pass-through visual matches without acoustic shift
            return [
                RefinedAnchor(
                    ref_time=m.ref_time,
                    tar_time=m.tar_time,
                    acoustic_offset_ms=0.0,
                    visual_offset_s=m.offset,
                    acoustic_confidence=1.0,
                    combined_confidence=m.confidence,
                    seq_len=m.seq_len,
                    weight=getattr(m, "weight", 1.0),
                    source=getattr(m, "source", "unknown"),
                    hash_dist=m.hash_dist,
                )
                for m in matches
            ]

        # Read audio files into numpy arrays
        sr_ref, ref_audio = wavfile.read(ref_wav_path)
        sr_tar, tar_audio = wavfile.read(tar_wav_path)

        # Convert to mono float32
        if ref_audio.ndim > 1:
            ref_mono = np.mean(ref_audio, axis=1).astype(np.float32)
        else:
            ref_mono = ref_audio.astype(np.float32)

        if tar_audio.ndim > 1:
            tar_mono = np.mean(tar_audio, axis=1).astype(np.float32)
        else:
            tar_mono = tar_audio.astype(np.float32)

        # Optional speech band attenuation (notch / high-pass) to focus on M&E
        if self.config.speech_band_attenuation:
            # High-pass filter above 1000Hz or band-pass to highlight percussion and sound effects
            b, a = scipy.signal.butter(2, [800 / (sr_ref / 2), 12000 / (sr_ref / 2)], btype='bandpass')
            ref_mono = scipy.signal.filtfilt(b, a, ref_mono)
            tar_mono = scipy.signal.filtfilt(b, a, tar_mono)

        refined = []
        win_samples = int((self.config.acoustic_window_ms / 1000.0) * sr_ref)
        search_radius = int((self.config.acoustic_window_ms / 1000.0) * sr_ref)

        for idx, m in enumerate(matches):
            r_center = int(m.ref_time * sr_ref)
            t_center = int(m.tar_time * sr_tar)

            def _passthrough(acoustic_confidence: float) -> RefinedAnchor:
                """Keep visual timing, tag the acoustic refinement result, preserve metadata."""
                return RefinedAnchor(
                    ref_time=m.ref_time,
                    tar_time=m.tar_time,
                    acoustic_offset_ms=0.0,
                    visual_offset_s=m.offset,
                    acoustic_confidence=acoustic_confidence,
                    combined_confidence=m.confidence,
                    seq_len=m.seq_len,
                    weight=getattr(m, "weight", 1.0),
                    source=getattr(m, "source", "unknown"),
                    hash_dist=m.hash_dist,
                )

            # Check bounds
            if r_center - win_samples < 0 or r_center + win_samples >= len(ref_mono):
                refined.append(_passthrough(0.5))
                continue

            if t_center - search_radius - win_samples < 0 or t_center + search_radius + win_samples >= len(tar_mono):
                refined.append(_passthrough(0.5))
                continue

            ref_slice = ref_mono[r_center - win_samples : r_center + win_samples]
            tar_slice = tar_mono[t_center - search_radius - win_samples : t_center + search_radius + win_samples]

            ref_norm = np.linalg.norm(ref_slice)
            if ref_norm < 1e-4:
                # Silent region: retain visual timing
                refined.append(_passthrough(0.5))
                continue

            # Compute cross-correlation
            corr = scipy.signal.correlate(tar_slice, ref_slice, mode='valid')
            corr_norm = corr / (ref_norm * np.linalg.norm(tar_slice) + 1e-8)

            peak_idx = int(np.argmax(corr_norm))
            peak_val = float(corr_norm[peak_idx])

            # Zero lag is at search_radius
            sample_shift = peak_idx - search_radius

            # Parabolic interpolation for sub-sample accuracy
            if 0 < peak_idx < len(corr_norm) - 1:
                alpha = corr_norm[peak_idx - 1]
                beta = corr_norm[peak_idx]
                gamma = corr_norm[peak_idx + 1]
                denom = 2 * (2 * beta - alpha - gamma)
                if denom > 1e-6:
                    sub_sample_shift = (alpha - gamma) / denom
                    sample_shift += sub_sample_shift

            time_shift_s = sample_shift / sr_ref
            shift_ms = time_shift_s * 1000.0

            # If acoustic correlation peak is confident (> 0.25) and within reasonable bounds (+/- 150ms)
            if peak_val >= 0.20 and abs(shift_ms) <= 200.0:
                refined_tar_time = m.tar_time + time_shift_s
                ac_conf = min(1.0, peak_val * 1.5)
            else:
                # Keep visual anchor
                refined_tar_time = m.tar_time
                shift_ms = 0.0
                ac_conf = 0.5

            comb_conf = round(0.4 * m.confidence + 0.6 * ac_conf, 3)

            refined.append(RefinedAnchor(
                ref_time=m.ref_time,
                tar_time=refined_tar_time,
                acoustic_offset_ms=round(shift_ms, 2),
                visual_offset_s=m.offset,
                acoustic_confidence=round(ac_conf, 3),
                combined_confidence=comb_conf,
                seq_len=m.seq_len,
                weight=getattr(m, "weight", 1.0),
                source=getattr(m, "source", "unknown"),
                hash_dist=m.hash_dist,
            ))

            if progress_callback:
                progress_callback(idx + 1, len(matches))

        return refined
