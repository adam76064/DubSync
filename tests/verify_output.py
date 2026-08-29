#!/usr/bin/env python3
"""
Independently verify a DubSync output against known ground truth.

This deliberately does NOT trust the tool's own audit (see REVIEW.md finding F1).
It re-measures everything from the rendered audio:

  1. output duration vs reference duration
  2. per-window sync error, via envelope cross-correlation of the synced dub
     track against the ORIGINAL reference audio (lag 0 ms == perfectly in sync)
  3. which regions are English M&E fallback vs real dub, detected through the
     spectral notch the vocal_filtered fallback leaves at 1200Hz / 2400Hz

Usage:  python3 tests/verify_output.py <output.mkv> [media_dir] [label]
"""
import json
import os
import subprocess
import sys

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffmpeg_util import find_ffmpeg  # noqa: E402

FF = find_ffmpeg()
HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "_media")
OUT_MKV = sys.argv[1]
LABEL = sys.argv[3] if len(sys.argv) > 3 else OUT_MKV

gt = json.load(open(os.path.join(MEDIA, "ground_truth.json")))
REF_DUR = gt["ref_duration"]
R_START, R_END = gt["removed_ref_span"]
WIN, HOP, SEARCH_S = 4.0, 2.0, 8.0
DT = 0.010


def to_wav(path, stream="0:a:0", tmp="/tmp/_dubsync_probe.wav"):
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-i", path,
                    "-map", stream, "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1", "-y", tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sr, x = wavfile.read(tmp)
    return sr, x.astype(np.float32) / 32768.0


def envelope(x, sr, bin_s=DT):
    w = max(1, int(bin_s * sr))
    return np.sqrt(np.convolve(x * x, np.ones(w) / w, mode="same") + 1e-12)[::w]


def notch_score(x, sr, t0, t1):
    """Energy at 1200/2400Hz relative to neighbouring bands.
    The vocal-filtered fallback notches these hard -> low score."""
    seg = x[int(t0 * sr):int(t1 * sr)]
    if len(seg) < sr // 4:
        return float("nan")
    f, P = sps.welch(seg, sr, nperseg=min(4096, len(seg)))
    band = lambda lo, hi: float(np.sum(P[(f >= lo) & (f <= hi)]))  # noqa: E731
    notch = band(1100, 1300) + band(2300, 2500)
    neigh = band(700, 1000) + band(1500, 1900) + band(2800, 3300)
    return notch / (neigh + 1e-12)


def main():
    sr_r, ref = to_wav(os.path.join(MEDIA, "ref_audio.wav"))
    sr_o, out = to_wav(OUT_MKV, "0:a:1")            # track 2 = synced dub
    out_dur = len(out) / sr_o

    print(f"\n================ VERIFICATION: {LABEL} ================")
    print("[1] DURATION")
    print(f"    reference  : {REF_DUR:.3f}s")
    print(f"    output dub : {out_dur:.3f}s   (delta {out_dur - REF_DUR:+.3f}s)")

    env_r, env_o = envelope(ref, sr_r), envelope(out, sr_o)
    print(f"\n[2] PER-WINDOW SYNC ERROR (envelope xcorr, {WIN}s windows, +/-{SEARCH_S}s search)")
    print(f"    {'out t':>8} {'matched ref t':>14} {'error':>10} {'peak':>6}  verdict")

    rows = []
    t = 0.0
    while t + WIN <= min(out_dur, REF_DUR):
        o = env_o[int(t / DT):int((t + WIN) / DT)]
        o = o - o.mean()
        if np.linalg.norm(o) < 1e-6:
            t += HOP
            continue
        ir0 = int(max(0.0, t - SEARCH_S) / DT)
        ir1 = int(min(REF_DUR, t + WIN + SEARCH_S) / DT)
        r = env_r[ir0:ir1]
        k = int(np.argmax(sps.correlate(r, o, mode="valid")))
        err = (((ir0 + k) * DT) - t) * 1000.0        # 0 ms == perfectly in sync
        rows.append(abs(err))
        print(f"    {t:7.1f}s {t + err / 1000:13.2f}s {err:+9.0f}ms "
              f"{'OK ' if abs(err) < 120 else 'BAD'}")
        t += HOP

    if rows:
        print(f"\n    mean |error| (all windows) : {np.mean(rows):8.1f} ms")
        print(f"    max  |error| (all windows) : {np.max(rows):8.1f} ms")
        print(f"    windows within +/-120ms    : {100 * sum(e < 120 for e in rows) / len(rows):5.1f}%")

    print("\n[3] REGION CLASSIFICATION (low notch = English M&E fallback)")
    print(f"    ground truth: ref [{R_START:.0f}s, {R_END:.0f}s] is MISSING from the dub")
    print(f"    {'region':>16} {'notch':>8}  {'classified':>12}  {'truth':>12}")
    baseline = notch_score(ref, sr_r, 0.5, 55)
    mismatches = 0
    for a, b in [(0, 12), (12, 24), (24, 30), (30, 36), (36, 48), (48, 60)]:
        nz = notch_score(out, sr_o, a + 0.3, b - 0.3)
        cls = "FALLBACK" if nz < baseline * 0.35 else "DUB"
        truth = "FALLBACK" if (a >= R_START and b <= R_END) else "DUB"
        ok = cls == truth
        mismatches += (not ok)
        print(f"    [{a:5.1f},{b:5.1f}) {nz:8.4f}  {cls:>12}  {truth:>12}  {'ok' if ok else 'MISMATCH'}")
    print(f"    (untouched reference baseline notch = {baseline:.4f})")
    print(f"    region mismatches: {mismatches}")

    out_dir = os.path.dirname(os.path.abspath(OUT_MKV))
    out_base = os.path.splitext(os.path.basename(OUT_MKV))[0]
    rp = os.path.join(out_dir, f"{out_base}_forensic_report.json")
    if os.path.exists(rp):
        d = json.load(open(rp))
        a = d.get("verifier_audit", {})
        print("\n[4] WHAT THE TOOL'S OWN AUDIT CLAIMED")
        print(f"    mean_alignment_error_ms : {a.get('mean_alignment_error_ms')}")
        print(f"    max_alignment_error_ms  : {a.get('max_alignment_error_ms')}")
        print(f"    passed_windows_pct      : {a.get('passed_windows_pct')}")
        print(f"    healed false fallbacks  : {a.get('false_fallbacks_healed_count')}")
        print(f"    reported omitted gaps   : {d.get('omitted_censored_gaps')}")


if __name__ == "__main__":
    main()
