import os
import subprocess
import numpy as np
import auditok
import tempfile
import shutil
import sys

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True)

def extract_segment(input_file, output_wav, start_time, duration=240):
    """Extracts a 4-minute chunk for analysis."""
    # Ensure start_time isn't negative
    start_time = max(0, start_time)
    cmd = f'ffmpeg -y -ss {start_time} -t {duration} -i "{input_file}" -vn -ac 1 -ar 16000 "{output_wav}"'
    run_cmd(cmd)

def get_voice_map(wav_path):
    """Creates a binary speech map using explicit time slicing."""
    audio_regions = auditok.load(wav_path)
    analysis_window = 0.1 
    tokens = []
    total_dur = audio_regions.duration
    
    # Using .sec[start:end] ensures we are slicing by time, not by sample index
    for i in np.arange(0, total_dur, analysis_window):
        try:
            # Explicitly slice by seconds
            region = audio_regions.sec[i : i + analysis_window]
            tokens.append(1 if region.is_event() else 0)
        except Exception:
            # Fallback if the very last slice is too small
            tokens.append(0)
    return np.array(tokens)

def find_offset(master_map, target_map):
    # Cross-correlation to find the highest point of overlap
    correlation = np.correlate(master_map, target_map, mode='full')
    best_idx = np.argmax(correlation)
    # Convert index back to seconds (0.1s per index)
    return (best_idx - (len(target_map) - 1)) * 0.1

def main():
    en_path = input("English Video: ").strip('"')
    ar_path = input("Arabic Video: ").strip('"')
    out_name = input("Output Filename: ").strip('"')

    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Get Duration
        dur_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{en_path}"'
        total_duration = float(run_cmd(dur_cmd).stdout)

        # 2. Sample Start and End
        # Shift checkpoints in to ensure we don't hit the file boundary
        checkpoints = [60, total_duration - 300] 
        found_offsets = []

        print(f"\n--- Analyzing Drift (Total Duration: {total_duration/60:.2f}m) ---")
        
        for i, cp in enumerate(checkpoints):
            en_seg = os.path.join(temp_dir, f"en_{i}.wav")
            ar_seg = os.path.join(temp_dir, f"ar_{i}.wav")
            
            sys.stdout.write(f"Processing segment {i+1}/2 at {cp/60:.1f}m... ")
            sys.stdout.flush()
            
            extract_segment(en_path, en_seg, cp)
            extract_segment(ar_path, ar_seg, cp)
            
            off = find_offset(get_voice_map(en_seg), get_voice_map(ar_seg))
            found_offsets.append(off)
            print(f"Done. Offset: {off:.3f}s")

        # 3. Calculate Stretch
        start_off = found_offsets[0]
        end_off = found_offsets[1]
        drift_amount = end_off - start_off
        
        dist = checkpoints[1] - checkpoints[0]
        # Stretch ratio: (English Duration) / (Arabic Equivalent Duration)
        stretch_ratio = dist / (dist + drift_amount)

        print(f"\n--- Analysis Results ---")
        print(f"Initial Sync Delay: {-start_off:.3f}s")
        print(f"Drift Detected: {drift_amount:.3f}s")
        print(f"Required Stretch Ratio: {stretch_ratio:.6f}")

        # 4. Final Processing
        print("\n--- Applying Fix and Muxing (This takes a moment) ---")
        
        # Calculate delay/trim based on the START offset
        delay_ms = int(-start_off * 1000)
        
        if delay_ms > 0:
            audio_filter = f"adelay={delay_ms}|{delay_ms},atempo={stretch_ratio}"
        else:
            # Trim the start, reset timestamps, then stretch
            audio_filter = f"atrim=start={abs(start_off)},asetpts=PTS-STARTPTS,atempo={stretch_ratio}"

        # Combine English Video with the Filtered Arabic Audio
        mux_cmd = (
            f'ffmpeg -y -i "{en_path}" -i "{ar_path}" '
            f'-filter_complex "[1:a]{audio_filter}[outa]" '
            f'-map 0:v -map 0:a -map [outa] '
            f'-c:v copy -c:a:0 copy -c:a:1 ac3 -b:a:1 192k '
            f'-metadata:s:a:1 language=ara -metadata:s:a:1 title="Arabic (Auto-Stretched)" '
            f'"{out_name}"'
        )
        
        run_cmd(mux_cmd)
        print(f"\n✅ SUCCESS! File saved as: {out_name}")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
