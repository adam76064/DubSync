#!/usr/bin/env python3
"""
DubSync: Automated Intelligent Dub Synchronization Engine
Aligns foreign/dubbed audio tracks with high-quality reference video releases,
handling scene omissions, cuts, commercials, and speed/framerate drift.
"""

import os
import sys
import shutil
import subprocess
import argparse
import time
import json
import re
import tempfile
from pathlib import Path
from PIL import Image
import imagehash
import numpy as np

# --- FFmpeg Discovery ---
def get_ffmpeg_path():
    """Finds FFmpeg executable from system PATH or imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    # Check common locations
    candidates = [
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise RuntimeError("FFmpeg executable not found. Please install FFmpeg or run: pip install imageio-ffmpeg")

FFMPEG_EXE = get_ffmpeg_path()


def run_cmd(cmd, desc="", check=True, capture_output=True):
    """Executes a subprocess command with error handling."""
    if isinstance(cmd, list):
        cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    else:
        cmd_str = cmd
    
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=check
        )
        return proc
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {desc}")
        print(f"Command: {cmd_str}")
        if e.stderr:
            lines = e.stderr.strip().splitlines()
            print("Stderr:\n" + "\n".join(lines[-15:]))
        raise


def get_media_info(filepath):
    """Extracts media duration, streams, and properties using FFmpeg."""
    cmd = [FFMPEG_EXE, "-hide_banner", "-i", filepath]
    proc = run_cmd(cmd, desc=f"Probing {os.path.basename(filepath)}", check=False)
    
    info = {
        "filepath": filepath,
        "duration": 0.0,
        "video_streams": [],
        "audio_streams": []
    }
    
    # Parse duration
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):([0-9.]+)", proc.stderr)
    if m_dur:
        hours = int(m_dur.group(1))
        mins = int(m_dur.group(2))
        secs = float(m_dur.group(3))
        info["duration"] = hours * 3600 + mins * 60 + secs
        
    # Parse streams
    for line in proc.stderr.splitlines():
        if "Stream #" in line:
            m_stream = re.search(r"Stream #0:(\d+)(?:\((.*?)\))?:\s*(Video|Audio):(.*)", line)
            if m_stream:
                idx = int(m_stream.group(1))
                lang = m_stream.group(2) or "und"
                stype = m_stream.group(3).lower()
                details = m_stream.group(4)
                
                stream_dict = {
                    "index": idx,
                    "language": lang,
                    "details": details.strip()
                }
                
                if stype == "video":
                    # Parse FPS
                    m_fps = re.search(r"([0-9.]+)\s*fps", details)
                    stream_dict["fps"] = float(m_fps.group(1)) if m_fps else 24.0
                    info["video_streams"].append(stream_dict)
                elif stype == "audio":
                    m_hz = re.search(r"(\d+)\s*Hz", details)
                    stream_dict["sample_rate"] = int(m_hz.group(1)) if m_hz else 48000
                    info["audio_streams"].append(stream_dict)
                    
    return info


def extract_scene_keyframes(video_path, output_dir, prefix, scene_threshold=0.25, max_duration=None):
    """
    Extracts keyframes at visual scene transitions with exact pts_time timestamps.
    """
    print(f"--- Extracting scene keyframes: {os.path.basename(video_path)} (threshold={scene_threshold}) ---")
    t0 = time.time()
    
    cmd = [
        FFMPEG_EXE, "-hide_banner",
        "-threads", "0",
    ]
    if max_duration:
        cmd.extend(["-t", str(max_duration)])
        
    cmd.extend([
        "-i", video_path,
        "-vf", f"scale=320:180,select='gt(scene,{scene_threshold})',showinfo",
        "-fps_mode", "vfr",
        "-q:v", "3",
        os.path.join(output_dir, f"{prefix}_%05d.jpg"),
        "-y"
    ])
    
    proc = run_cmd(cmd, desc="Extracting keyframes", check=True)
    
    # Parse timestamps from showinfo output
    keyframes = []
    for line in proc.stderr.splitlines():
        if "pts_time:" in line:
            m = re.search(r"n:\s*(\d+)\s+pts:\s*(\d+)\s+pts_time:([0-9.]+)", line)
            if m:
                n = int(m.group(1))
                pts_time = float(m.group(3))
                img_file = os.path.join(output_dir, f"{prefix}_{n+1:05d}.jpg")
                if os.path.exists(img_file):
                    try:
                        pil_img = Image.open(img_file)
                        h = imagehash.phash(pil_img)
                        keyframes.append({
                            "n": n,
                            "pts_time": pts_time,
                            "file": img_file,
                            "hash": h
                        })
                    except Exception as e:
                        pass
                        
    elapsed = time.time() - t0
    print(f"  [OK] Extracted {len(keyframes)} scene anchors in {elapsed:.2f}s")
    return keyframes


def match_anchors_dp(ref_keyframes, tar_keyframes, max_hash_diff=7):
    """
    Matches reference keyframes to target keyframes using Dynamic Programming / Monotonic chaining.
    Guarantees temporal ordering: ref_time[i] < ref_time[j] => tar_time[i] < tar_time[j].
    """
    print("--- Matching Visual Anchors & Calculating Alignments ---")
    
    # Step 1: Candidate pairs with pHash distance <= max_hash_diff
    candidates = []
    for r_idx, r_k in enumerate(ref_keyframes):
        for t_idx, t_k in enumerate(tar_keyframes):
            h_diff = r_k["hash"] - t_k["hash"]
            if h_diff <= max_hash_diff:
                candidates.append({
                    "r_idx": r_idx,
                    "t_idx": t_idx,
                    "r_time": r_k["pts_time"],
                    "t_time": t_k["pts_time"],
                    "h_diff": h_diff,
                    "score": max_hash_diff - h_diff + 1.0
                })
                
    if not candidates:
        return []
        
    # Step 2: Sort by reference time
    candidates.sort(key=lambda x: (x["r_time"], x["t_time"]))
    
    # Step 3: Longest strictly increasing chain (Monotonic Sequence)
    N = len(candidates)
    dp = [c["score"] for c in candidates]
    parent = [-1] * N
    
    for i in range(N):
        ci = candidates[i]
        for j in range(i):
            cj = candidates[j]
            # Must be strictly forward in both reference and target timelines
            if cj["r_time"] < ci["r_time"] and cj["t_time"] < ci["t_time"]:
                dt_r = ci["r_time"] - cj["r_time"]
                dt_t = ci["t_time"] - cj["t_time"]
                
                ratio = dt_t / dt_r if dt_r > 0 else 0
                if 0.5 <= ratio <= 2.0 or dt_r > 10.0:
                    if dp[j] + ci["score"] > dp[i]:
                        dp[i] = dp[j] + ci["score"]
                        parent[i] = j
                        
    # Backtrack best chain
    best_end = int(np.argmax(dp))
    chain = []
    curr = best_end
    while curr != -1:
        chain.append(candidates[curr])
        curr = parent[curr]
    chain.reverse()
    
    print(f"  [OK] Found {len(chain)} consistent visual anchor matches.")
    return chain


def build_piecewise_edl(ref_duration, anchor_chain, min_scene_dur=1.0):
    """
    Constructs an Edit Decision List (EDL) mapping the master timeline [0, ref_duration]
    to corresponding foreign segments or fallback fills.
    """
    print("--- Building Piecewise Edit Decision List (EDL) ---")
    
    if not anchor_chain:
        return [{
            "type": "dub",
            "ref_start": 0.0,
            "ref_end": ref_duration,
            "tar_start": 0.0,
            "tar_end": ref_duration,
            "speed": 1.0
        }]
        
    segments = []
    
    # Initial Segment: from 0.0 to first matched anchor
    first_anchor = anchor_chain[0]
    if first_anchor["r_time"] > 0.1:
        if first_anchor["t_time"] > 0.1:
            segments.append({
                "type": "dub",
                "ref_start": 0.0,
                "ref_end": first_anchor["r_time"],
                "tar_start": max(0.0, first_anchor["t_time"] - first_anchor["r_time"]),
                "tar_end": first_anchor["t_time"],
                "speed": 1.0
            })
        else:
            # English has intro/footage missing in Arabic -> Fill with Reference English Audio
            segments.append({
                "type": "fallback",
                "ref_start": 0.0,
                "ref_end": first_anchor["r_time"],
                "tar_start": 0.0,
                "tar_end": 0.0,
                "speed": 1.0
            })
            
    # Intermediate Segments: between consecutive matched anchors
    for i in range(len(anchor_chain) - 1):
        a1 = anchor_chain[i]
        a2 = anchor_chain[i + 1]
        
        r_start, r_end = a1["r_time"], a2["r_time"]
        t_start, t_end = a1["t_time"], a2["t_time"]
        
        r_dur = r_end - r_start
        t_dur = t_end - t_start
        
        if r_dur < min_scene_dur:
            continue
            
        speed = t_dur / r_dur if r_dur > 0 else 1.0
        
        if 0.85 <= speed <= 1.15:
            # Valid continuous scene
            segments.append({
                "type": "dub",
                "ref_start": r_start,
                "ref_end": r_end,
                "tar_start": t_start,
                "tar_end": t_end,
                "speed": speed
            })
        else:
            # Cut/Omission detected inside this window
            segments.append({
                "type": "fallback",
                "ref_start": r_start,
                "ref_end": r_end,
                "tar_start": t_start,
                "tar_end": t_end,
                "speed": 1.0
            })
            
    # Final Segment: from last matched anchor to ref_duration
    last_anchor = anchor_chain[-1]
    if last_anchor["r_time"] < ref_duration:
        remaining_r = ref_duration - last_anchor["r_time"]
        segments.append({
            "type": "dub",
            "ref_start": last_anchor["r_time"],
            "ref_end": ref_duration,
            "tar_start": last_anchor["t_time"],
            "tar_end": last_anchor["t_time"] + remaining_r,
            "speed": 1.0
        })
        
    print(f"  [OK] Generated {len(segments)} timeline segments:")
    dub_count = sum(1 for s in segments if s["type"] == "dub")
    fallback_count = sum(1 for s in segments if s["type"] == "fallback")
    print(f"       -> {dub_count} Synced Dub Segments")
    print(f"       -> {fallback_count} Fallback/Bridge Segments (Cuts/Omissions)")
    
    return segments


def render_synced_audio(edl, ref_audio_wav, tar_audio_wav, output_wav, temp_dir):
    """
    Slices, retimes, and concatenates audio segments according to the EDL.
    """
    print("--- Retiming and Rendering Synced Audio Track ---")
    t0 = time.time()
    
    segment_files = []
    
    for i, seg in enumerate(edl):
        target_dur = seg["ref_end"] - seg["ref_start"]
        out_seg = os.path.join(temp_dir, f"seg_{i:04d}.wav")
        
        if seg["type"] == "dub":
            t_start = seg["tar_start"]
            t_dur = seg["tar_end"] - seg["tar_start"]
            
            speed_ratio = t_dur / target_dur if target_dur > 0 else 1.0
            speed_ratio = max(0.5, min(2.0, speed_ratio))
            
            if abs(speed_ratio - 1.0) > 0.002:
                filter_str = f"atempo={speed_ratio}"
            else:
                filter_str = "anull"
                
            cmd = [
                FFMPEG_EXE, "-hide_banner", "-loglevel", "warning",
                "-ss", f"{t_start:.4f}", "-t", f"{t_dur:.4f}",
                "-i", tar_audio_wav,
                "-af", filter_str,
                "-t", f"{target_dur:.4f}",
                "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
                "-y", out_seg
            ]
            run_cmd(cmd, desc=f"Processing dub seg #{i}")
            
        else:
            r_start = seg["ref_start"]
            cmd = [
                FFMPEG_EXE, "-hide_banner", "-loglevel", "warning",
                "-ss", f"{r_start:.4f}", "-t", f"{target_dur:.4f}",
                "-i", ref_audio_wav,
                "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
                "-y", out_seg
            ]
            run_cmd(cmd, desc=f"Processing fallback seg #{i}")
            
        if os.path.exists(out_seg):
            segment_files.append(out_seg)
            
    # Concatenate segments
    concat_list = os.path.join(temp_dir, "concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for sf in segment_files:
            abs_p = os.path.abspath(sf).replace("\\", "/")
            f.write(f"file '{abs_p}'\n")
            
    cmd_concat = [
        FFMPEG_EXE, "-hide_banner", "-loglevel", "warning",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c:a", "pcm_s16le",
        "-y", output_wav
    ]
    run_cmd(cmd_concat, desc="Concatenating audio segments")
    print(f"  [OK] Synced audio rendered successfully in {time.time()-t0:.2f}s")


def mux_final_mkv(ref_video, synced_audio_wav, output_video, tar_lang="ara", ref_lang="eng"):
    """
    Combines Reference Video + Reference Audio (Track 1) + Synced Dub Audio (Track 2) into final MKV.
    """
    print(f"--- Muxing Final Output Video: {os.path.basename(output_video)} ---")
    t0 = time.time()
    
    cmd = [
        FFMPEG_EXE, "-hide_banner", "-loglevel", "warning",
        "-i", ref_video,
        "-i", synced_audio_wav,
        "-map", "0:v:0",          # Video from reference
        "-map", "0:a:0?",         # Audio Track 1 from reference
        "-map", "1:a:0",          # Audio Track 2: Synced Dub
        "-c:v", "copy",
        "-c:a:0", "copy",
        "-c:a:1", "aac", "-b:a:1", "192k",
        f"-metadata:s:a:0", f"language={ref_lang}",
        f"-metadata:s:a:0", f"title=Original ({ref_lang.upper()})",
        f"-metadata:s:a:1", f"language={tar_lang}",
        f"-metadata:s:a:1", f"title=Synced Dub ({tar_lang.upper()})",
        "-y", output_video
    ]
    
    run_cmd(cmd, desc="Muxing final MKV")
    print(f"  [OK] Video created successfully in {time.time()-t0:.2f}s -> {output_video}")


def dub_sync(ref_path, tar_path, output_path, scene_threshold=0.25, tar_lang="ara", ref_lang="eng"):
    """
    Main synchronization pipeline.
    """
    start_total = time.time()
    print("=" * 70)
    print(" DubSync: Automated Intelligent Dub Synchronization")
    print("=" * 70)
    print(f"  Master (HQ Reference): {ref_path}")
    print(f"  Target (Foreign Dub):  {tar_path}")
    print(f"  Output Destination:    {output_path}\n")
    
    temp_dir = tempfile.mkdtemp(prefix="dub_sync_")
    try:
        # Step 1: Probe metadata
        ref_info = get_media_info(ref_path)
        tar_info = get_media_info(tar_path)
        print(f"Ref Duration: {ref_info['duration']:.2f}s ({ref_info['duration']/60:.2f}m)")
        print(f"Tar Duration: {tar_info['duration']:.2f}s ({tar_info['duration']/60:.2f}m)")
        
        # Step 2: Extract audio stems to WAV
        ref_wav = os.path.join(temp_dir, "ref_audio.wav")
        tar_wav = os.path.join(temp_dir, "tar_audio.wav")
        
        print("\n--- Extracting Audio Tracks ---")
        run_cmd([FFMPEG_EXE, "-hide_banner", "-loglevel", "warning", "-i", ref_path, "-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-y", ref_wav], desc="Extracting reference audio")
        run_cmd([FFMPEG_EXE, "-hide_banner", "-loglevel", "warning", "-i", tar_path, "-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-y", tar_wav], desc="Extracting foreign audio")
        
        # Step 3: Keyframe Scene Extraction
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        ref_keys = extract_scene_keyframes(ref_path, frames_dir, "ref", scene_threshold)
        tar_keys = extract_scene_keyframes(tar_path, frames_dir, "tar", scene_threshold)
        
        # Step 4: Anchor Matching & DP Monotonic Sequence
        anchors = match_anchors_dp(ref_keys, tar_keys)
        if not anchors:
            print("[WARN] No visual anchors matched. Falling back to whole-file offset.")
            
        # Step 5: Construct EDL
        edl = build_piecewise_edl(ref_info["duration"], anchors)
        
        # Step 6: Render Synced Audio
        synced_wav = os.path.join(temp_dir, "synced_dub.wav")
        render_synced_audio(edl, ref_wav, tar_wav, synced_wav, temp_dir)
        
        # Step 7: Final Mux
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mux_final_mkv(ref_path, synced_wav, output_path, tar_lang=tar_lang, ref_lang=ref_lang)
        
        total_time = time.time() - start_total
        print("\n" + "=" * 70)
        print(f" SUCCESS! Completed in {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f" Output File: {output_path}")
        print("=" * 70)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="DubSync: Automated Dub Audio Synchronization Tool")
    parser.add_argument("ref_video", help="High-quality reference video (e.g. English WEB-DL/BluRay)")
    parser.add_argument("foreign_video", help="Dubbed video (e.g. Arabic TV/web rip)")
    parser.add_argument("output_video", help="Output file path (e.g. output.mkv)")
    parser.add_argument("--scene_threshold", type=float, default=0.25, help="Scene-change detection sensitivity (default: 0.25)")
    parser.add_argument("--tar_lang", default="ara", help="ISO 639-2 code for dub language (default: ara)")
    parser.add_argument("--ref_lang", default="eng", help="ISO 639-2 code for reference language (default: eng)")
    
    args = parser.parse_args()
    dub_sync(
        ref_path=args.ref_video,
        tar_path=args.foreign_video,
        output_path=args.output_video,
        scene_threshold=args.scene_threshold,
        tar_lang=args.tar_lang,
        ref_lang=args.ref_lang
    )

if __name__ == "__main__":
    main()
