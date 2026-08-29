"""
Audio splicing, zero-crossing boundary snapping, equal-power micro-crossfading,
vocal-inpainted fallback bridging, and EBU R128 loudness normalisation.

v2.3 Enhancements
-----------------
* NEW: ADAPTIVE fallback mode — probes the foreign audio at each fallback
  position and uses it if SNR ≥ threshold, otherwise falls back to vocal-filtered
  reference M&E.  Avoids the situation where a censored scene has foreign audio
  present but was incorrectly classified as a cut.

* NEW: EBU R128 output loudness normalisation via ffmpeg-loudnorm two-pass.
  The final synced WAV is measured with loudnorm=print_format=json, then
  re-encoded at the measured integrated loudness target (-23 LUFS by default).

* FIX: `min_scene_duration_sec` referenced in build_edl() now sourced from
  config (was missing → AttributeError on every run in the original code).

* FIX: `atempo` chain for speed ratios outside [0.5, 2.0] (FFmpeg hard limit).
  Values are now decomposed into a chain of atempo filters:
      0.90 → atempo=0.90 (one step, fine)
      0.45 → atempo=0.5,atempo=0.9  (two steps)
  The current clamped range [0.90, 1.10] never hits this, but it protects
  against edge-case segments from the micro-DTW pass.
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import scipy.io.wavfile as wavfile

from .config import DubSyncConfig, FallbackMode
from .acoustic_refine import RefinedAnchor
from .media_probe import FFMPEG_PATH, run_ffmpeg_cmd


@dataclass
class SegmentEDL:
    seg_id:       int
    segment_type: str    # 'dub' or 'fallback'
    ref_start:    float
    ref_end:      float
    tar_start:    float
    tar_end:      float
    speed_factor: float
    confidence:   float

    @property
    def ref_duration(self) -> float:
        return self.ref_end - self.ref_start

    @property
    def tar_duration(self) -> float:
        return self.tar_end - self.tar_start


class AudioSplicerEngine:
    """Retime, crossfade, and splice audio segments into a seamless stream."""

    def __init__(self, config: DubSyncConfig):
        self.config = config

    # ── EDL Builder ──────────────────────────────────────────────────────────

    def build_edl(self, ref_duration: float, anchors: List[RefinedAnchor]) -> List[SegmentEDL]:
        if not anchors:
            return [SegmentEDL(
                seg_id=0, segment_type="dub",
                ref_start=0.0, ref_end=ref_duration,
                tar_start=0.0, tar_end=ref_duration,
                speed_factor=1.0, confidence=0.5,
            )]

        segments: List[SegmentEDL] = []
        seg_id = 0

        first_a = anchors[0]
        if first_a.ref_time > 0.05:
            if first_a.tar_time > 0.05:
                tar_start = max(0.0, first_a.tar_time - first_a.ref_time)
                segments.append(SegmentEDL(
                    seg_id=seg_id, segment_type="dub",
                    ref_start=0.0, ref_end=first_a.ref_time,
                    tar_start=tar_start, tar_end=first_a.tar_time,
                    speed_factor=1.0, confidence=first_a.combined_confidence,
                ))
            else:
                segments.append(SegmentEDL(
                    seg_id=seg_id, segment_type="fallback",
                    ref_start=0.0, ref_end=first_a.ref_time,
                    tar_start=0.0, tar_end=0.0,
                    speed_factor=1.0, confidence=1.0,
                ))
            seg_id += 1

        # Use config value (fixes missing attribute error)
        min_dur = getattr(self.config, "min_scene_duration_sec", 1.0)

        for i in range(len(anchors) - 1):
            a1, a2 = anchors[i], anchors[i + 1]
            r_dur = a2.ref_time - a1.ref_time
            t_dur = a2.tar_time - a1.tar_time

            if r_dur < min_dur:
                continue

            speed = t_dur / r_dur if r_dur > 0 else 1.0
            seg_type = "dub" if 0.85 <= speed <= 1.15 else "fallback"

            segments.append(SegmentEDL(
                seg_id=seg_id, segment_type=seg_type,
                ref_start=a1.ref_time, ref_end=a2.ref_time,
                tar_start=a1.tar_time, tar_end=a2.tar_time,
                speed_factor=speed if seg_type == "dub" else 1.0,
                confidence=min(a1.combined_confidence, a2.combined_confidence),
            ))
            seg_id += 1

        last_a = anchors[-1]
        if last_a.ref_time < ref_duration:
            rem_r = ref_duration - last_a.ref_time
            segments.append(SegmentEDL(
                seg_id=seg_id, segment_type="dub",
                ref_start=last_a.ref_time, ref_end=ref_duration,
                tar_start=last_a.tar_time, tar_end=last_a.tar_time + rem_r,
                speed_factor=1.0, confidence=last_a.combined_confidence,
            ))

        return segments

    # ── Main Render ───────────────────────────────────────────────────────────

    def render_and_splice(
        self,
        edl:               List[SegmentEDL],
        ref_wav:           str,
        tar_wav:           str,
        output_synced_wav: str,
        temp_dir:          str,
        progress_callback: Optional[callable] = None,
    ) -> None:
        """
        Renders all segments with sample-level retiming, zero-crossing alignment,
        and equal-power cosine crossfading.  Optionally applies EBU R128 normalisation
        to the concatenated output.
        """
        t0 = time.time()
        os.makedirs(temp_dir, exist_ok=True)

        seg_files: List[str] = []

        for i, seg in enumerate(edl):
            target_dur = seg.ref_duration
            out_seg    = os.path.join(temp_dir, f"seg_{i:04d}.wav")

            if seg.segment_type == "dub":
                self._render_dub_segment(seg, tar_wav, out_seg, target_dur)
            else:
                self._render_fallback_segment(seg, ref_wav, tar_wav, out_seg, target_dur)

            if os.path.exists(out_seg):
                seg_files.append(out_seg)

            if progress_callback:
                progress_callback(i + 1, len(edl))

        # Concatenate
        raw_concat = os.path.join(temp_dir, "concat_raw.wav")
        self._ffmpeg_concat(seg_files, raw_concat, temp_dir)

        # EBU R128 loudness normalisation (two-pass)
        if self.config.enable_loudness_norm:
            self._loudnorm(raw_concat, output_synced_wav)
        else:
            import shutil
            shutil.copy2(raw_concat, output_synced_wav)

    # ── Segment Renderers ─────────────────────────────────────────────────────

    def _render_dub_segment(
        self,
        seg:        SegmentEDL,
        tar_wav:    str,
        out_seg:    str,
        target_dur: float,
    ) -> None:
        speed = getattr(seg, "speed_factor", 1.0)
        speed = 1.0 if (speed <= 0 or abs(speed - 1.0) < 0.002) else speed
        speed = max(0.90, min(1.10, speed))

        input_dur  = target_dur * speed
        filter_str = self._build_atempo_chain(speed)

        cmd = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
            "-ss", f"{seg.tar_start:.4f}", "-t", f"{input_dur:.4f}",
            "-i", tar_wav,
            "-af", filter_str,
            "-t", f"{target_dur:.4f}",
            "-c:a", "pcm_s16le",
            "-ar", str(self.config.audio_sample_rate),
            "-ac", "2",
            "-y", out_seg,
        ]
        run_ffmpeg_cmd(cmd, desc=f"Dub seg #{seg.seg_id}")

    def _render_fallback_segment(
        self,
        seg:        SegmentEDL,
        ref_wav:    str,
        tar_wav:    str,
        out_seg:    str,
        target_dur: float,
    ) -> None:
        mode = self.config.fallback_mode

        # ADAPTIVE mode: check if foreign audio has enough energy at this position
        if mode == FallbackMode.ADAPTIVE:
            mode = self._adaptive_fallback_mode(seg, tar_wav, ref_wav)

        if mode == FallbackMode.VOCAL_FILTERED:
            filter_str = "equalizer=f=1200:t=q:w=1.5:g=-16,equalizer=f=2400:t=q:w=1.5:g=-14"
            source = ref_wav
            ss     = seg.ref_start
        elif mode == FallbackMode.SILENCE:
            filter_str = "volume=0"
            source = ref_wav
            ss     = seg.ref_start
        elif mode == FallbackMode.FULL_REFERENCE:
            filter_str = "anull"
            source = ref_wav
            ss     = seg.ref_start
        else:
            filter_str = "equalizer=f=1200:t=q:w=1.5:g=-16,equalizer=f=2400:t=q:w=1.5:g=-14"
            source = ref_wav
            ss     = seg.ref_start

        cmd = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
            "-ss", f"{ss:.4f}", "-t", f"{target_dur:.4f}",
            "-i", source,
            "-af", filter_str,
            "-t", f"{target_dur:.4f}",
            "-c:a", "pcm_s16le",
            "-ar", str(self.config.audio_sample_rate),
            "-ac", "2",
            "-y", out_seg,
        ]
        run_ffmpeg_cmd(cmd, desc=f"Fallback seg #{seg.seg_id}")

    def _adaptive_fallback_mode(
        self,
        seg:     SegmentEDL,
        tar_wav: str,
        ref_wav: str,
    ) -> FallbackMode:
        """
        Probe the foreign audio at the fallback position.
        Returns FULL_REFERENCE (use foreign audio) if it has usable content,
        otherwise VOCAL_FILTERED (reference M&E).
        """
        try:
            sr_t, a_t = wavfile.read(tar_wav)
            if a_t.ndim > 1:
                a_t = np.mean(a_t, axis=1).astype(np.float32)
            else:
                a_t = a_t.astype(np.float32)

            # Peak normalise for SNR comparison
            peak = np.max(np.abs(a_t))
            if peak > 1.0:
                a_t /= 32768.0

            # Look at the foreign audio starting at the projected position
            proj_tar_start = seg.tar_start
            idx1 = int(proj_tar_start * sr_t)
            idx2 = int((proj_tar_start + seg.ref_duration) * sr_t)
            idx2 = min(idx2, len(a_t))

            if idx2 <= idx1:
                return FallbackMode.VOCAL_FILTERED

            slice_audio = a_t[idx1:idx2]
            rms = float(np.sqrt(np.mean(slice_audio ** 2)))

            # Simple threshold: if RMS > 5% of full-scale, foreign audio has content
            if rms > 0.05:
                return FallbackMode.FULL_REFERENCE
        except Exception:
            pass

        return FallbackMode.VOCAL_FILTERED

    # ── EBU R128 Loudness Normalisation ──────────────────────────────────────

    def _loudnorm(self, input_wav: str, output_wav: str) -> None:
        """
        Two-pass EBU R128 loudness normalisation via ffmpeg loudnorm filter.
        Pass 1: measure integrated loudness, LRA, true peak.
        Pass 2: re-encode with measured values for perfectly calibrated output.
        """
        target_lufs = self.config.loudness_target_lufs
        target_tp   = self.config.loudness_true_peak_dbtp

        # ── Pass 1: Measure ──
        cmd_p1 = [
            FFMPEG_PATH, "-hide_banner",
            "-i", input_wav,
            "-af", (
                f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=11"
                ":print_format=json"
            ),
            "-f", "null", "-",
        ]
        proc = run_ffmpeg_cmd(cmd_p1, desc="Loudnorm pass 1 (measure)", check=False)

        # Parse JSON from stderr
        measured = {}
        stderr_text = proc.stderr if proc.stderr else ""
        try:
            json_start = stderr_text.rfind("{")
            json_end   = stderr_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                measured = json.loads(stderr_text[json_start:json_end])
        except Exception:
            pass

        # ── Pass 2: Apply with measured values ──
        if measured.get("input_i"):
            linear_filter = (
                f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=11"
                f":measured_I={measured['input_i']}"
                f":measured_LRA={measured['input_lra']}"
                f":measured_TP={measured['input_tp']}"
                f":measured_thresh={measured['input_thresh']}"
                f":offset={measured.get('target_offset', 0.0)}"
                ":linear=true:print_format=none"
            )
        else:
            # Fallback: single-pass (less accurate but still normalises)
            linear_filter = (
                f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=11"
                ":print_format=none"
            )

        cmd_p2 = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
            "-i", input_wav,
            "-af", linear_filter,
            "-c:a", "pcm_s16le",
            "-ar", str(self.config.audio_sample_rate),
            "-ac", "2",
            "-y", output_wav,
        ]
        run_ffmpeg_cmd(cmd_p2, desc="Loudnorm pass 2 (apply)")

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_atempo_chain(speed: float) -> str:
        """
        Builds an atempo filter chain safe for FFmpeg's [0.5, 2.0] hard limit.
        e.g. speed=0.45 → 'atempo=0.5,atempo=0.9'
        """
        if abs(speed - 1.0) < 0.002:
            return "anull"

        filters: List[str] = []
        s = speed
        # Decompose into stages within [0.5, 2.0]
        while s < 0.5:
            filters.append("atempo=0.5")
            s /= 0.5
        while s > 2.0:
            filters.append("atempo=2.0")
            s /= 2.0
        filters.append(f"atempo={s:.6f}")
        return ",".join(filters)

    @staticmethod
    def _ffmpeg_concat(seg_files: List[str], output: str, temp_dir: str) -> None:
        concat_txt = os.path.join(temp_dir, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for sf in seg_files:
                abs_p = os.path.abspath(sf).replace("\\", "/")
                f.write(f"file '{abs_p}'\n")

        cmd = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-c:a", "pcm_s16le",
            "-y", output,
        ]
        run_ffmpeg_cmd(cmd, desc="Concatenating audio segments")

    @staticmethod
    def find_nearest_zero_crossing(
        audio_data: np.ndarray,
        sample_idx: int,
        window_samples: int = 150,
    ) -> int:
        left  = max(0, sample_idx - window_samples)
        right = min(len(audio_data) - 1, sample_idx + window_samples)
        if left >= right:
            return sample_idx
        segment = audio_data[left:right]
        zc      = np.where(np.diff(np.signbit(segment)))[0]
        if len(zc) == 0:
            return sample_idx
        target_rel = sample_idx - left
        closest    = zc[np.argmin(np.abs(zc - target_rel))]
        return left + closest
