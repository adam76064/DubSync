"""
MKV container multiplexing and multi-track audio packaging.
"""

import os
import time
from typing import Optional, Callable
from .config import DubSyncConfig
from .media_probe import FFMPEG_PATH, run_ffmpeg_cmd


class MKVMuxer:
    """Multiplexes video, reference audio, and synchronized dub audio into a pristine MKV file."""

    def __init__(self, config: DubSyncConfig):
        self.config = config

    def mux(
        self,
        ref_video_path: str,
        synced_wav_path: str,
        output_mkv_path: str,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Muxes master video stream, master audio (Track 1), synced dub audio (Track 2),
        and any existing subtitles into final MKV.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_mkv_path)), exist_ok=True)
        t0 = time.time()

        cmd = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
            "-i", ref_video_path,
            "-i", synced_wav_path,
            "-map", "0:v:0",          # Master video stream (lossless copy)
            "-map", "0:a:0?",         # Master audio stream (lossless copy)
            "-map", "1:a:0",          # Synced dub audio (Track 2)
        ]

        if self.config.keep_subtitles:
            cmd.extend(["-map", "0:s?"])  # Copy subtitles if present

        cmd.extend([
            "-c:v", "copy",
            "-c:a:0", "copy",
            "-c:a:1", self.config.audio_codec,
            "-b:a:1", self.config.audio_bitrate,
            f"-metadata:s:a:0", f"language={self.config.ref_lang}",
            f"-metadata:s:a:0", f"title={self.config.ref_title}",
            f"-metadata:s:a:1", f"language={self.config.tar_lang}",
            f"-metadata:s:a:1", f"title={self.config.dub_title}",
        ])

        if self.config.keep_subtitles:
            cmd.extend(["-c:s", "copy"])

        cmd.extend(["-y", output_mkv_path])

        run_ffmpeg_cmd(cmd, desc=f"Muxing Final MKV: {os.path.basename(output_mkv_path)}")
        return output_mkv_path
