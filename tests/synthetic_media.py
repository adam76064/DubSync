#!/usr/bin/env python3
"""
Builds a synthetic reference/dub pair whose ground-truth timing is known exactly.

Reference : 60.0s @ 24fps, 320x180, hard scene cuts at t = 12, 24, 36, 48.
            Audio = deterministic "M&E" bed (800-3200Hz) + speech-like bursts
            (300-3000Hz, syllable-rate AM, real pauses).
Dub       : reference with the scene ref[24, 30] removed (a censorship cut),
            then slowed to 0.96x and letterboxed to a different picture scale.

Ground truth:  tar_time = (ref_time - 6.0 if ref_time >= 30.0 else ref_time) / 0.96

Usage:  python3 tests/synthetic_media.py [output_dir]
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffmpeg_util import find_ffmpeg  # noqa: E402

REF_DUR, FPS, W, H = 60.0, 24, 320, 180
CUTS = [12.0, 24.0, 36.0, 48.0]
SR = 44100
REMOVED_START, REMOVED_END = 24.0, 30.0
SPEED = 0.96          # tar plays at 0.96x -> tar_time = ref_time / SPEED


def make_pattern(i, path):
    """A distinctive static keyframe pattern for segment i."""
    rng = np.random.default_rng(1000 + i)
    img = Image.new("RGB", (W, H), (20 + 40 * i, 30, 90 - 15 * i))
    d = ImageDraw.Draw(img)
    for _ in range(14):
        x0, x1 = sorted((int(rng.integers(0, W)), int(rng.integers(0, W))))
        y0, y1 = sorted((int(rng.integers(0, H)), int(rng.integers(0, H))))
        d.rectangle([x0, y0, x1, y1], fill=tuple(int(c) for c in rng.integers(40, 255, 3)))
    for _ in range(6):
        x0, y0 = int(rng.integers(0, W - 40)), int(rng.integers(0, H - 40))
        r = int(rng.integers(8, 40))
        d.ellipse([x0, y0, x0 + r, y0 + r], fill=tuple(int(c) for c in rng.integers(40, 255, 3)))
    img.save(path)
    return path


def band_noise(rng, n, lo, hi, sr=SR):
    """Deterministic noise band-limited to [lo, hi] via FFT masking."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    X[(f < lo) | (f > hi)] = 0.0
    y = np.fft.irfft(X, n)
    return y / (np.max(np.abs(y)) + 1e-9)


def build_audio(rng):
    n = int(REF_DUR * SR)
    t = np.arange(n) / SR

    bed = band_noise(rng, n, 800, 3200)
    bed *= 0.35 + 0.65 * (np.abs(np.sin(2 * np.pi * t / 0.5)) ** 3)   # 0.5s beat

    speech = np.zeros(n)
    bounds = [0.0] + CUTS + [REF_DUR]
    events = []
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        cur = s0 + 0.5
        while cur < s1 - 1.0:
            dur = float(rng.uniform(0.6, 2.2))
            if cur + dur > s1 - 0.3:
                break
            i0, i1 = int(cur * SR), int((cur + dur) * SR)
            tt = np.arange(i1 - i0) / SR
            chunk = band_noise(rng, i1 - i0, 300, 3000)
            chunk *= 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 3.5 * tt + 0.4))
            ramp = int(0.03 * SR)
            chunk[:ramp] *= np.linspace(0, 1, ramp)
            chunk[-ramp:] *= np.linspace(1, 0, ramp)
            speech[i0:i1] = chunk
            events.append((round(cur, 3), round(cur + dur, 3)))
            cur += dur + float(rng.uniform(0.25, 1.1))

    mix = 0.45 * bed + 0.55 * speech
    mix *= 0.85 / (np.max(np.abs(mix)) + 1e-9)
    return mix, events


def main():
    FF = find_ffmpeg()
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_media")
    os.makedirs(out, exist_ok=True)

    def run(cmd):
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise SystemExit(f"FFmpeg failed:\n{' '.join(cmd)}\n{p.stderr[-3000:]}")

    # ---- reference video: one static pattern per scene
    bounds = [0.0] + CUTS + [REF_DUR]
    segs = []
    for i, (s0, s1) in enumerate(zip(bounds[:-1], bounds[1:])):
        png = make_pattern(i, f"{out}/pat_{i}.png")
        seg = f"{out}/seg_{i}.mp4"
        run([FF, "-hide_banner", "-loglevel", "error", "-loop", "1", "-framerate", str(FPS),
             "-t", f"{s1 - s0:.3f}", "-i", png,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", "48", "-y", seg])
        segs.append(seg)
    with open(f"{out}/vconcat.txt", "w") as f:
        for p in segs:
            f.write(f"file '{p}'\n")

    rng = np.random.default_rng(7)
    ref_audio, events = build_audio(rng)
    wavfile.write(f"{out}/ref_audio.wav", SR, (ref_audio * 32767).astype(np.int16))

    run([FF, "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", f"{out}/vconcat.txt",
         "-i", f"{out}/ref_audio.wav",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-y", f"{out}/ref.mp4"])

    # ---- dub: cut the censored scene ...
    run([FF, "-hide_banner", "-loglevel", "error", "-i", f"{out}/ref.mp4",
         "-filter_complex",
         f"[0:v]trim=0:{REMOVED_START},setpts=PTS-STARTPTS[v0];"
         f"[0:a]atrim=0:{REMOVED_START},asetpts=PTS-STARTPTS[a0];"
         f"[0:v]trim={REMOVED_END}:{REF_DUR},setpts=PTS-STARTPTS[v1];"
         f"[0:a]atrim={REMOVED_END}:{REF_DUR},asetpts=PTS-STARTPTS[a1];"
         f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]",
         "-map", "[outv]", "-map", "[outa]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-y", f"{out}/cut.mp4"])

    # ... then slow to 0.96x and letterbox to a different picture geometry
    run([FF, "-hide_banner", "-loglevel", "error", "-i", f"{out}/cut.mp4",
         "-vf", f"setpts=PTS/{SPEED},scale=288:162,pad=320:180:(ow-iw)/2:(oh-ih)/2",
         "-af", f"atempo={SPEED}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-y", f"{out}/tar.mp4"])

    truth = {
        "ref_duration": REF_DUR,
        "tar_duration": round((REF_DUR - (REMOVED_END - REMOVED_START)) / SPEED, 3),
        "speed": SPEED,
        "removed_ref_span": [REMOVED_START, REMOVED_END],
        "scene_cuts_ref": CUTS,
        "map": "tar_time = (ref_time - 6.0 if ref_time >= 30.0 else ref_time) / 0.96",
        "speech_events_ref": events,
    }
    with open(f"{out}/ground_truth.json", "w") as f:
        json.dump(truth, f, indent=2)

    print(json.dumps({k: v for k, v in truth.items() if k != "speech_events_ref"}, indent=2))
    print(f"speech events (ref time): {len(events)}")
    print("media written to", out)


if __name__ == "__main__":
    main()
