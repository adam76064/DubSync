"""
Media probing, stream discovery, and high-precision audio extraction.
"""

import os
import shutil
import subprocess
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


def get_ffmpeg_path() -> str:
    """Finds FFmpeg executable from system PATH or imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    candidates = [
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise RuntimeError("FFmpeg executable not found. Please install FFmpeg or run: pip install imageio-ffmpeg")


def get_ffprobe_path() -> Optional[str]:
    """Finds FFprobe executable if available."""
    path = shutil.which("ffprobe")
    if path:
        return path
    candidates = [
        r"C:\ProgramData\chocolatey\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


FFMPEG_PATH = get_ffmpeg_path()
FFPROBE_PATH = get_ffprobe_path()


def run_ffmpeg_cmd(cmd: List[str], desc: str = "", check: bool = True) -> subprocess.CompletedProcess:
    """Executes an FFmpeg command with sanitized logging and error handling."""
    if cmd and cmd[0] != FFMPEG_PATH:
        cmd[0] = FFMPEG_PATH
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=check
        )
        return proc
    except subprocess.CalledProcessError as e:
        err_msg = f"[ERROR] FFmpeg failed: {desc}\nCommand: {' '.join(cmd)}\n"
        if e.stderr:
            err_msg += "\n".join(e.stderr.strip().splitlines()[-15:])
        raise RuntimeError(err_msg) from e


@dataclass
class StreamInfo:
    index: int
    stream_type: str  # 'video', 'audio', 'subtitle'
    codec: str
    language: str = "und"
    title: str = ""
    # Video specific
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    # Audio specific
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    bitrate: Optional[str] = None


@dataclass
class MediaInfo:
    filepath: str
    filename: str
    duration: float
    filesize_mb: float
    video_streams: List[StreamInfo] = field(default_factory=list)
    audio_streams: List[StreamInfo] = field(default_factory=list)
    subtitle_streams: List[StreamInfo] = field(default_factory=list)

    @property
    def primary_video(self) -> Optional[StreamInfo]:
        return self.video_streams[0] if self.video_streams else None

    @property
    def primary_audio(self) -> Optional[StreamInfo]:
        return self.audio_streams[0] if self.audio_streams else None


class MediaProbe:
    """Probes media files and extracts streams with high fidelity."""

    @staticmethod
    def probe(filepath: str) -> MediaInfo:
        """Inspects media file and returns structured MediaInfo."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Media file not found: {filepath}")

        cmd = [FFMPEG_PATH, "-hide_banner", "-i", filepath]
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, errors="ignore")

        filesize = os.path.getsize(filepath) / (1024 * 1024)
        duration = 0.0

        # Parse duration
        m_dur = re.search(r"Duration:\s*(\d+):(\d+):([0-9.]+)", proc.stderr)
        if m_dur:
            hours, mins, secs = int(m_dur.group(1)), int(m_dur.group(2)), float(m_dur.group(3))
            duration = hours * 3600 + mins * 60 + secs

        media_info = MediaInfo(
            filepath=os.path.abspath(filepath),
            filename=os.path.basename(filepath),
            duration=duration,
            filesize_mb=round(filesize, 2)
        )

        for line in proc.stderr.splitlines():
            if "Stream #" in line:
                m_stream = re.search(r"Stream #0:(\d+)(?:\[.*?\])?(?:\((.*?)\))?:\s*(Video|Audio|Subtitle):\s*(.*)", line)
                if m_stream:
                    idx = int(m_stream.group(1))
                    lang = m_stream.group(2) or "und"
                    stype = m_stream.group(3).lower()
                    details = m_stream.group(4).strip()

                    codec = details.split(",")[0].split()[0]

                    stream = StreamInfo(
                        index=idx,
                        stream_type=stype,
                        codec=codec,
                        language=lang
                    )

                    if stype == "video":
                        m_res = re.search(r"(\d{3,5})x(\d{3,5})", details)
                        if m_res:
                            stream.width, stream.height = int(m_res.group(1)), int(m_res.group(2))
                        m_fps = re.search(r"([0-9.]+)\s*fps", details)
                        if m_fps:
                            stream.fps = float(m_fps.group(1))
                        media_info.video_streams.append(stream)

                    elif stype == "audio":
                        m_hz = re.search(r"(\d+)\s*Hz", details)
                        if m_hz:
                            stream.sample_rate = int(m_hz.group(1))
                        if "stereo" in details:
                            stream.channels = 2
                        elif "5.1" in details:
                            stream.channels = 6
                        elif "mono" in details:
                            stream.channels = 1
                        media_info.audio_streams.append(stream)

                    elif stype == "subtitle":
                        media_info.subtitle_streams.append(stream)

        return media_info

    @staticmethod
    def select_audio_stream(info: "MediaInfo", language: str = "") -> Optional[int]:
        """
        Picks the audio stream index best matching the requested ISO 639-2 language code.
        Falls back to the primary (first) audio stream, then to None when no audio exists.
        """
        if not info.audio_streams:
            return None

        lang = (language or "").strip().lower()
        if lang:
            for s in info.audio_streams:
                if (s.language or "").lower() == lang:
                    return s.index
            for s in info.audio_streams:
                if (s.language or "").lower().startswith(lang[:2]):
                    return s.index

        return info.primary_audio.index

    @staticmethod
    def extract_pcm_wav(
        input_media: str,
        output_wav: str,
        sample_rate: int = 48000,
        channels: int = 2,
        stream_index: Optional[int] = None
    ) -> str:
        """
        Extracts pristine uncompressed 16-bit PCM WAV audio.

        When ``stream_index`` is provided (an absolute container stream index, as
        returned by :meth:`select_audio_stream`), that stream is explicitly mapped
        (``-map 0:<index>``) so multi-audio containers select the intended language
        track instead of relying on FFmpeg's best-stream heuristic.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)

        cmd = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "warning",
            "-i", input_media,
            "-vn",
        ]
        if stream_index is not None:
            cmd.extend(["-map", f"0:{stream_index}"])
        cmd.extend([
            "-c:a", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-y", output_wav
        ])
        run_ffmpeg_cmd(cmd, desc=f"Extracting PCM Audio: {os.path.basename(input_media)}")
        return output_wav
