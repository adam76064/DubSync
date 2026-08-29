"""Shared FFmpeg discovery for the test harness."""
import shutil


def find_ffmpeg() -> str:
    """Return a usable ffmpeg executable.

    Prefers a system ffmpeg on PATH, then imageio-ffmpeg's bundled binary.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "No ffmpeg found. Install FFmpeg on PATH or run: "
            "pip install -r requirements-dev.txt"
        ) from exc
