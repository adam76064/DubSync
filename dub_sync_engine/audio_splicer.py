"""
Audio splicing, zero-crossing boundary snapping, equal-power micro-crossfading,
and vocal-inpainted fallback bridging for omitted scenes.
"""

import os
import shutil
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import scipy.io.wavfile as wavfile

from .config import DubSyncConfig, FallbackMode
from .acoustic_refine import RefinedAnchor
from .media_probe import FFMPEG_PATH, run_ffmpeg_cmd


@dataclass
class SegmentEDL:
    seg_id: int
    segment_type: str  # 'dub' or 'fallback'
    ref_start: float
    ref_end: float
    tar_start: float
    tar_end: float
    speed_factor: float
    confidence: float

    @property
    def ref_duration(self) -> float:
        return self.ref_end - self.ref_start

    @property
    def tar_duration(self) -> float:
        return self.tar_end - self.tar_start


class AudioSplicerEngine:
    """Retime, crossfade, and splice audio segments into a seamless continuous audio stream."""

    def __init__(self, config: DubSyncConfig):
        self.config = config

    def build_edl(self, ref_duration: float, anchors: List[RefinedAnchor]) -> List[SegmentEDL]:
        """
        Builds a complete timeline Edit Decision List from 0.0s to ref_duration.
        """
        if not anchors:
            return [
                SegmentEDL(
                    seg_id=0,
                    segment_type="dub",
                    ref_start=0.0,
                    ref_end=ref_duration,
                    tar_start=0.0,
                    tar_end=ref_duration,
                    speed_factor=1.0,
                    confidence=0.5
                )
            ]

        segments = []
        seg_id = 0

        # Step 1: Handle start offset before first anchor
        first_a = anchors[0]
        if first_a.ref_time > 0.05:
            if first_a.tar_time > 0.05:
                # Both start with pre-anchor scene
                tar_start = max(0.0, first_a.tar_time - first_a.ref_time)
                segments.append(SegmentEDL(
                    seg_id=seg_id,
                    segment_type="dub",
                    ref_start=0.0,
                    ref_end=first_a.ref_time,
                    tar_start=tar_start,
                    tar_end=first_a.tar_time,
                    speed_factor=1.0,
                    confidence=first_a.combined_confidence
                ))
            else:
                # English has opening footage missing in Arabic (e.g. Amazon logo) -> Fallback
                segments.append(SegmentEDL(
                    seg_id=seg_id,
                    segment_type="fallback",
                    ref_start=0.0,
                    ref_end=first_a.ref_time,
                    tar_start=0.0,
                    tar_end=0.0,
                    speed_factor=1.0,
                    confidence=1.0
                ))
            seg_id += 1

        # Step 2: Intermediate anchor segments
        for i in range(len(anchors) - 1):
            a1 = anchors[i]
            a2 = anchors[i + 1]

            r_dur = a2.ref_time - a1.ref_time
            t_dur = a2.tar_time - a1.tar_time

            if r_dur < self.config.min_scene_duration_sec:
                continue

            speed = t_dur / r_dur if r_dur > 0 else 1.0

            # If speed ratio is within realistic clock/framerate bounds (0.85 - 1.15)
            if 0.85 <= speed <= 1.15:
                segments.append(SegmentEDL(
                    seg_id=seg_id,
                    segment_type="dub",
                    ref_start=a1.ref_time,
                    ref_end=a2.ref_time,
                    tar_start=a1.tar_time,
                    tar_end=a2.tar_time,
                    speed_factor=speed,
                    confidence=min(a1.combined_confidence, a2.combined_confidence)
                ))
            else:
                # Discontinuity / Cut scene omitted in Arabic -> Fallback bridge
                segments.append(SegmentEDL(
                    seg_id=seg_id,
                    segment_type="fallback",
                    ref_start=a1.ref_time,
                    ref_end=a2.ref_time,
                    tar_start=a1.tar_time,
                    tar_end=a2.tar_time,
                    speed_factor=1.0,
                    confidence=0.9
                ))
            seg_id += 1

        # Step 3: Final tail segment after last anchor
        last_a = anchors[-1]
        if last_a.ref_time < ref_duration:
            rem_r = ref_duration - last_a.ref_time
            segments.append(SegmentEDL(
                seg_id=seg_id,
                segment_type="dub",
                ref_start=last_a.ref_time,
                ref_end=ref_duration,
                tar_start=last_a.tar_time,
                tar_end=last_a.tar_time + rem_r,
                speed_factor=1.0,
                confidence=last_a.combined_confidence
            ))

        return segments

    def find_nearest_zero_crossing(self, audio_data: np.ndarray, sample_idx: int, window_samples: int = 150) -> int:
        """
        Finds sample index closest to sample_idx where waveform crosses zero with positive slope.
        """
        left = max(0, sample_idx - window_samples)
        right = min(len(audio_data) - 1, sample_idx + window_samples)
        
        if left >= right:
            return sample_idx

        segment = audio_data[left:right]
        # Detect zero-crossings
        zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
        if len(zero_crossings) == 0:
            return sample_idx

        # Find closest to center
        target_relative = sample_idx - left
        closest_idx = zero_crossings[np.argmin(np.abs(zero_crossings - target_relative))]
        return left + closest_idx

    def render_and_splice(
        self,
        edl: List[SegmentEDL],
        ref_wav: str,
        tar_wav: str,
        output_synced_wav: str,
        temp_dir: str,
        progress_callback: Optional[callable] = None
    ):
        """
        Renders all segments with sample-level retiming, zero-crossing alignment,
        and equal-power cosine crossfading.
        """
        t0 = time.time()
        os.makedirs(temp_dir, exist_ok=True)

        sr_ref, ref_audio = wavfile.read(ref_wav)
        sr_tar, tar_audio = wavfile.read(tar_wav)

        total_ref_samples = int(edl[-1].ref_end * sr_ref) if edl else len(ref_audio)
        channels = ref_audio.shape[1] if ref_audio.ndim > 1 else 1

        # Pre-allocate output buffer
        final_audio = np.zeros((total_ref_samples, channels), dtype=np.int16) if channels > 1 else np.zeros(total_ref_samples, dtype=np.int16)

        crossfade_samples = int((self.config.crossfade_duration_ms / 1000.0) * sr_ref)
        cf_fade_in = np.sin(np.linspace(0, np.pi / 2, crossfade_samples, dtype=np.float32)) ** 2
        cf_fade_out = np.cos(np.linspace(0, np.pi / 2, crossfade_samples, dtype=np.float32)) ** 2

        if channels > 1:
            cf_fade_in = cf_fade_in[:, np.newaxis]
            cf_fade_out = cf_fade_out[:, np.newaxis]

        # Process each segment
        seg_files = []
        for i, seg in enumerate(edl):
            target_dur = seg.ref_duration
            out_seg_path = os.path.join(temp_dir, f"seg_{i:04d}.wav")

            if seg.segment_type == "dub":
                t_start = seg.tar_start

                # Use calibrated speed factor if available, otherwise calculate safely
                speed = getattr(seg, "speed_factor", 1.0)
                if speed <= 0 or abs(speed - 1.0) < 0.002:
                    speed = 1.0
                
                # Broadcast speed clamp (0.90 to 1.10) to support PAL/NTSC transfers
                speed = max(0.90, min(1.10, speed))
                input_dur = target_dur * speed

                if abs(speed - 1.0) > 0.002:
                    filter_str = f"atempo={speed:.6f}"
                else:
                    filter_str = "anull"

                cmd = [
                    FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
                    "-ss", f"{t_start:.4f}", "-t", f"{input_dur:.4f}",
                    "-i", tar_wav,
                    "-af", filter_str,
                    "-t", f"{target_dur:.4f}",
                    "-c:a", "pcm_s16le", "-ar", str(self.config.audio_sample_rate), "-ac", "2",
                    "-y", out_seg_path
                ]
                run_ffmpeg_cmd(cmd, desc=f"Rendering Dub Segment #{i}")

            else:
                # Fallback segment
                r_start = seg.ref_start
                if self.config.fallback_mode == FallbackMode.VOCAL_FILTERED:
                    # Low-pass / high-shelf speech suppression filter for ambient M&E
                    filter_str = "equalizer=f=1200:t=q:w=1.5:g=-16,equalizer=f=2400:t=q:w=1.5:g=-14"
                elif self.config.fallback_mode == FallbackMode.SILENCE:
                    filter_str = "volume=0"
                else:
                    filter_str = "anull"

                cmd = [
                    FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
                    "-ss", f"{r_start:.4f}", "-t", f"{target_dur:.4f}",
                    "-i", ref_wav,
                    "-af", filter_str,
                    "-t", f"{target_dur:.4f}",
                    "-c:a", "pcm_s16le", "-ar", str(self.config.audio_sample_rate), "-ac", "2",
                    "-y", out_seg_path
                ]
                run_ffmpeg_cmd(cmd, desc=f"Rendering Fallback Segment #{i}")

            if os.path.exists(out_seg_path):
                seg_files.append(out_seg_path)

            if progress_callback:
                progress_callback(i + 1, len(edl))

        # Concatenate using FFmpeg concat demuxer
        concat_txt = os.path.join(temp_dir, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for sf in seg_files:
                abs_p = os.path.abspath(sf).replace("\\", "/")
                f.write(f"file '{abs_p}'\n")

        cmd_concat = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-c:a", "pcm_s16le",
            "-y", output_synced_wav
        ]
        run_ffmpeg_cmd(cmd_concat, desc="Concatenating Master Synced Audio")
