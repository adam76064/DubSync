import os
import subprocess
import audalign
import tempfile
import shutil

def run(cmd):
    """Runs command and prints errors if they occur."""
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr}")
        raise

def extract_audio(input_file, output_wav):
    print(f"--- Extracting: {os.path.basename(input_file)}")
    # 44100Hz Mono is the 'sweet spot' for alignment algorithms
    cmd = f'ffmpeg -y -i "{input_file}" -vn -ac 1 -ar 44100 "{output_wav}"'
    run(cmd)

def adjust_sync(input_wav, delay, output_wav):
    # If delay is positive, the target starts LATER, so we pad it.
    # If delay is negative, the target starts EARLIER, so we trim it.
    if delay > 0:
        delay_ms = int(delay * 1000)
        print(f"--- Applying DELAY of {delay_ms}ms...")
        cmd = f'ffmpeg -y -i "{input_wav}" -af "adelay={delay_ms}|{delay_ms}" "{output_wav}"'
    else:
        trim_sec = abs(delay)
        print(f"--- Applying TRIM of {trim_sec:.3f}s...")
        cmd = f'ffmpeg -y -ss {trim_sec} -i "{input_wav}" "{output_wav}"'
    run(cmd)

def main():
    en_path = input("Master Video (English): ").strip('"')
    ar_path = input("Target Video (Arabic): ").strip('"')
    out_name = input("Output Filename (movie_fixed.mkv): ").strip('"')

    temp_dir = tempfile.mkdtemp()
    
    try:
        en_wav = os.path.join(temp_dir, "en.wav")
        ar_wav = os.path.join(temp_dir, "ar.wav")
        fixed_ar_wav = os.path.join(temp_dir, "ar_fixed.wav")

        extract_audio(en_path, en_wav)
        extract_audio(ar_path, ar_wav)

        print("--- Analyzing Alignment (This may take a minute)...")
        
        # We use 'align_files' which defaults to fingerprinting.
        # If fingerprinting fails or gives low confidence, consider using 
        # result = audalign.align_files(en_wav, ar_wav, technique="correlation") 
        # but it is very slow.
        result = audalign.align_files(en_wav, ar_wav)

        # Navigation based on your specific JSON log structure
        try:
            # The keys in the result dict are the filenames exactly as passed
            matches = result['match_info']['en.wav']['match_info']['ar.wav']
            
            # Pull the top candidate
            best_offset = matches['offset_seconds'][0]
            confidence = matches['confidence'][0]
            
            print(f"\nMatch Found!")
            print(f"Offset: {best_offset:.3f} seconds")
            print(f"Confidence: {confidence}")

            # IMPORTANT: Based on your log, ar.wav has an offset of 11.47 in the global dict,
            # but the match info says 0.66. We will use the match_info offset.
            # We use negative offset to align Arabic BACK to English.
            actual_shift = -best_offset

            adjust_sync(ar_wav, actual_shift, fixed_ar_wav)

            print("--- Muxing final MKV with synced audio...")
            # We re-encode to AC3 to ensure compatibility and prevent 'stutter'
            cmd = (
                f'ffmpeg -y -i "{en_path}" -i "{fixed_ar_wav}" '
                f'-map 0:v -map 0:a -map 1:a '
                f'-c:v copy -c:a:0 copy -c:a:1 ac3 -b:a:1 192k '
                f'-metadata:s:a:1 language=ara -metadata:s:a:1 title="Arabic (Synced)" '
                f'"{out_name}"'
            )
            run(cmd)

            print(f"\n✅ SUCCESS! File saved as: {out_name}")

        except (KeyError, IndexError) as e:
            print(f"❌ Could not find a match. Audalign output was unexpected: {e}")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
