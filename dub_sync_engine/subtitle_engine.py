"""
Native ASS/SSA and SRT Styled Subtitle Retiming Engine for DubSync Pro.
Retimes foreign subtitle dialogue events against the master EDL while preserving
100% of custom anime fonts, vector drawings, styles, colors, positioning, and karaoke tags.
"""

import os
import re
from typing import List, Tuple, Optional
from .audio_splicer import SegmentEDL


def parse_ass_time(time_str: str) -> float:
    """Converts ASS timestamp H:MM:SS.cs (e.g. '0:01:23.45') to seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return h * 3600.0 + m * 60.0 + s
    return 0.0


def format_ass_time(seconds: float) -> str:
    """Converts seconds to ASS timestamp format H:MM:SS.cs."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    rem = seconds % 3600
    m = int(rem // 60)
    s = rem % 60
    # ASS uses centiseconds (2 decimal places)
    return f"{h}:{m:02d}:{s:05.2f}"


def parse_srt_time(time_str: str) -> float:
    """Converts SRT timestamp HH:MM:SS,mmm to seconds."""
    clean = time_str.strip().replace(",", ".")
    parts = clean.split(":")
    if len(parts) == 3:
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return h * 3600.0 + m * 60.0 + s
    return 0.0


def format_srt_time(seconds: float) -> str:
    """Converts seconds to SRT timestamp format HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    rem = seconds % 3600
    m = int(rem // 60)
    s = int(rem % 60)
    ms = int(round((rem % 1) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class SubtitleEngine:
    """Retimes styled ASS/SSA and SRT subtitles according to the master sync EDL."""

    def __init__(self, edl: List[SegmentEDL]):
        self.edl = edl

    def map_target_to_reference(self, t_tar: float) -> Optional[float]:
        """
        Maps a target timestamp (in original foreign video time) to reference master time.
        Returns None if target timestamp falls in a permanently omitted/deleted scene.
        """
        for seg in self.edl:
            if seg.segment_type == "dub":
                if seg.tar_start <= t_tar <= seg.tar_end:
                    offset_within_tar = t_tar - seg.tar_start
                    speed = seg.speed_factor if seg.speed_factor > 0 else 1.0
                    ref_offset = offset_within_tar / speed
                    return seg.ref_start + ref_offset

        # If slightly before first segment or after last, extrapolate using closest segment
        if self.edl:
            first_dub = next((s for s in self.edl if s.segment_type == "dub"), None)
            if first_dub and t_tar < first_dub.tar_start:
                diff = first_dub.tar_start - t_tar
                speed = first_dub.speed_factor if first_dub.speed_factor > 0 else 1.0
                return max(0.0, first_dub.ref_start - (diff / speed))

            last_dub = next((s for s in reversed(self.edl) if s.segment_type == "dub"), None)
            if last_dub and t_tar > last_dub.tar_end:
                diff = t_tar - last_dub.tar_end
                speed = last_dub.speed_factor if last_dub.speed_factor > 0 else 1.0
                return last_dub.ref_end + (diff / speed)

        return None

    def retime_ass(self, input_path: str, output_path: str) -> bool:
        """
        Retimes an Advanced SubStation Alpha (.ass / .ssa) subtitle file.
        Preserves 100% of custom styles, fonts, vector drawings, and tags.
        """
        if not os.path.exists(input_path):
            return False

        with open(input_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            lines = f.readlines()

        in_events = False
        format_cols = []
        start_idx = -1
        end_idx = -1

        out_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if "[events]" in stripped.lower():
                    in_events = True
                else:
                    in_events = False
                out_lines.append(line)
                continue

            if in_events and stripped.lower().startswith("format:"):
                format_cols = [c.strip().lower() for c in stripped[7:].split(",")]
                if "start" in format_cols:
                    start_idx = format_cols.index("start")
                if "end" in format_cols:
                    end_idx = format_cols.index("end")
                out_lines.append(line)
                continue

            if in_events and (stripped.lower().startswith("dialogue:") or stripped.lower().startswith("comment:")):
                prefix_len = line.find(":") + 1
                prefix = line[:prefix_len]
                content = line[prefix_len:].strip()

                num_commas = len(format_cols) - 1 if format_cols else 9
                parts = content.split(",", num_commas)

                if start_idx != -1 and end_idx != -1 and len(parts) > max(start_idx, end_idx):
                    t_start_orig = parse_ass_time(parts[start_idx])
                    t_end_orig = parse_ass_time(parts[end_idx])

                    t_start_new = self.map_target_to_reference(t_start_orig)
                    t_end_new = self.map_target_to_reference(t_end_orig)

                    if t_start_new is not None and t_end_new is not None:
                        if t_end_new > t_start_new:
                            parts[start_idx] = format_ass_time(t_start_new)
                            parts[end_idx] = format_ass_time(t_end_new)
                            new_line = prefix + " " + ",".join(parts) + "\n"
                            out_lines.append(new_line)
                            continue

                # Fallback if unparseable
                out_lines.append(line)
            else:
                out_lines.append(line)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(out_lines)

        return True

    def retime_srt(self, input_path: str, output_path: str) -> bool:
        """
        Retimes a SubRip (.srt) subtitle file according to the master EDL.
        """
        if not os.path.exists(input_path):
            return False

        with open(input_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            content = f.read()

        blocks = re.split(r"\n\s*\n", content.strip())
        out_blocks = []

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 2:
                continue

            time_line_idx = 1 if lines[0].strip().isdigit() else 0
            time_line = lines[time_line_idx]

            if "-->" in time_line:
                time_parts = time_line.split("-->")
                t_start_orig = parse_srt_time(time_parts[0])
                t_end_orig = parse_srt_time(time_parts[1])

                t_start_new = self.map_target_to_reference(t_start_orig)
                t_end_new = self.map_target_to_reference(t_end_orig)

                if t_start_new is not None and t_end_new is not None and t_end_new > t_start_new:
                    new_time_line = f"{format_srt_time(t_start_new)} --> {format_srt_time(t_end_new)}"
                    lines[time_line_idx] = new_time_line
                    out_blocks.append("\n".join(lines))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n\n".join(out_blocks) + "\n")

        return True

    def auto_retime(self, input_sub_path: str, output_sub_path: str) -> bool:
        """Automatically detects format and retimes subtitle file."""
        ext = os.path.splitext(input_sub_path)[1].lower()
        if ext in [".ass", ".ssa"]:
            return self.retime_ass(input_sub_path, output_sub_path)
        elif ext in [".srt"]:
            return self.retime_srt(input_sub_path, output_sub_path)
        return False
