"""
Quality Control (QC) Report Generator and Confidence Analyzer for DubSync.
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from .visual_anchors import AnchorMatch
from .acoustic_refine import RefinedAnchor
from .audio_splicer import SegmentEDL
from .media_probe import MediaInfo


@dataclass
class QCReport:
    timestamp: str
    ref_filename: str
    tar_filename: str
    output_filename: str
    ref_duration_sec: float
    tar_duration_sec: float
    duration_delta_sec: float
    total_anchors_extracted_ref: int
    total_anchors_extracted_tar: int
    matched_anchors_count: int
    refined_anchors_count: int
    edl_total_segments: int
    edl_dub_segments: int
    edl_fallback_segments: int
    average_confidence: float
    omitted_scenes: List[Dict[str, float]]
    processing_time_sec: float


class QCReportGenerator:
    """Calculates synchronization quality metrics and generates reports."""

    @staticmethod
    def generate(
        ref_info: MediaInfo,
        tar_info: MediaInfo,
        output_path: str,
        ref_anchors_count: int,
        tar_anchors_count: int,
        refined_anchors: List[RefinedAnchor],
        edl: List[SegmentEDL],
        total_time: float
    ) -> QCReport:
        omitted = []
        for seg in edl:
            if seg.segment_type == "fallback":
                omitted.append({
                    "start_time": round(seg.ref_start, 2),
                    "end_time": round(seg.ref_end, 2),
                    "duration": round(seg.ref_duration, 2)
                })

        if refined_anchors:
            avg_conf = sum(a.combined_confidence for a in refined_anchors) / len(refined_anchors)
            matched_count = len(refined_anchors)
        elif edl:
            avg_conf = sum(s.confidence for s in edl) / len(edl)
            matched_count = sum(1 for s in edl if s.segment_type == "dub")
        else:
            avg_conf = 1.0
            matched_count = 0

        return QCReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            ref_filename=ref_info.filename,
            tar_filename=tar_info.filename,
            output_filename=os.path.basename(output_path),
            ref_duration_sec=round(ref_info.duration, 2),
            tar_duration_sec=round(tar_info.duration, 2),
            duration_delta_sec=round(ref_info.duration - tar_info.duration, 2),
            total_anchors_extracted_ref=ref_anchors_count,
            total_anchors_extracted_tar=tar_anchors_count,
            matched_anchors_count=matched_count,
            refined_anchors_count=matched_count,
            edl_total_segments=len(edl),
            edl_dub_segments=sum(1 for s in edl if s.segment_type == "dub"),
            edl_fallback_segments=len(omitted),
            average_confidence=round(avg_conf, 3),
            omitted_scenes=omitted,
            processing_time_sec=round(total_time, 2)
        )

    @staticmethod
    def save_json(report: QCReport, output_json_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)


@dataclass
class ForensicReport:
    timestamp: str
    dub_sync_version: str
    execution_time_sec: float
    pipeline_configuration: Dict[str, Any]
    media_specs: Dict[str, Any]
    matching_engine_telemetry: Dict[str, Any]
    continuous_blocks: List[Dict[str, Any]]
    timeline_edl: List[Dict[str, Any]]
    omitted_censored_gaps: List[Dict[str, Any]]
    quality_summary: Dict[str, Any]


class ForensicReportGenerator:
    """Generates detailed JSON & Markdown diagnostic reports for deep forensic inspection."""

    @staticmethod
    def generate_and_save(
        report_data: Dict[str, Any],
        base_output_path: str
    ) -> Tuple[str, str]:
        """Saves both .json and .md forensic reports alongside the output video."""
        base_dir = os.path.dirname(os.path.abspath(base_output_path))
        base_name = os.path.splitext(os.path.basename(base_output_path))[0]

        json_path = os.path.join(base_dir, f"{base_name}_forensic_report.json")
        md_path = os.path.join(base_dir, f"{base_name}_forensic_report.md")

        # 1. Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 2. Save Markdown
        md_content = ForensicReportGenerator._build_markdown(report_data)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path

    @staticmethod
    def _build_markdown(data: Dict[str, Any]) -> str:
        cfg = data.get("pipeline_configuration", {})
        media = data.get("media_specs", {})
        matching = data.get("matching_engine_telemetry", {})
        blocks = data.get("continuous_blocks", [])
        edl = data.get("timeline_edl", [])
        gaps = data.get("omitted_censored_gaps", [])
        summary = data.get("quality_summary", {})
        audit = data.get("verifier_audit", {})

        md = []
        md.append(f"# DubSync Pro — Comprehensive Forensic Diagnostic Report\n")
        md.append(f"**Generated:** {data.get('timestamp', 'N/A')} | **Processing Time:** {data.get('execution_time_sec', 0):.2f}s | **Engine Version:** {data.get('dub_sync_version', 'v2.2.0')}\n")
        md.append("---\n")

        # Media Specs
        md.append("## 1. Media Ingestion Specifications\n")
        ref_v = media.get("ref_video", {})
        tar_v = media.get("tar_video", {})
        ref_a = media.get("ref_audio", {})
        tar_a = media.get("tar_audio", {})

        md.append("| Stream | Reference (HQ Master) | Target (Foreign Dub) | Delta / Notes |")
        md.append("| :--- | :--- | :--- | :--- |")
        md.append(f"| **File** | `{media.get('ref_filename')}` | `{media.get('tar_filename')}` | - |")
        md.append(f"| **Duration** | {media.get('ref_duration_sec'):.3f}s ({int(media.get('ref_duration_sec', 0)//60)}m {int(media.get('ref_duration_sec', 0)%60):02d}s) | {media.get('tar_duration_sec'):.3f}s ({int(media.get('tar_duration_sec', 0)//60)}m {int(media.get('tar_duration_sec', 0)%60):02d}s) | {media.get('duration_delta_sec'):+.3f}s |")
        md.append(f"| **Video** | {ref_v.get('resolution')} @ {ref_v.get('fps'):.3f}fps ({ref_v.get('codec')}) | {tar_v.get('resolution')} @ {tar_v.get('fps'):.3f}fps ({tar_v.get('codec')}) | FPS Ratio: {ref_v.get('fps', 24.0)/max(0.01, tar_v.get('fps', 25.0)):.4f} |")
        md.append(f"| **Audio** | {ref_a.get('codec')} {ref_a.get('sample_rate')}Hz ({ref_a.get('channels')}ch, {ref_a.get('lang')}) | {tar_a.get('codec')} {tar_a.get('sample_rate')}Hz ({tar_a.get('channels')}ch, {tar_a.get('lang')}) | Internal Resample: 48,000Hz PCM |")
        md.append("\n---\n")

        # Pipeline Configuration
        md.append("## 2. Active Pipeline Configuration & Variables\n")
        md.append("| Variable / Parameter | Configured Value | Description |")
        md.append("| :--- | :--- | :--- |")
        for k, v in cfg.items():
            md.append(f"| `{k}` | `{v}` | - |")
        md.append("\n---\n")

        # Matching Telemetry
        md.append("## 3. Multi-Modal Alignment & Complete Anchor Registry\n")
        md.append(f"* **Active Strategy:** `{cfg.get('strategy', 'hybrid')}` | **Matcher Mode:** `{matching.get('active_mode')}`")
        md.append(f"* **Cascade Selection:** `{matching.get('cascade_tier_selected')}`")
        md.append(f"* **Total Keyframes Extracted:** Reference: {matching.get('ref_anchors_extracted', 0)} | Foreign Target: {matching.get('tar_anchors_extracted', 0)}")
        md.append(f"* **Total Matched Keypoints:** **{matching.get('matched_anchors_count', 0)} keypoints**\n")

        raw_anchors = matching.get("anchors", [])
        if raw_anchors:
            # Calculate global calibrated speed from blocks if available
            g_speed = blocks[0].get("speed_factor", 1.0) if blocks else 1.0
            first_offset = raw_anchors[0].get("offset", 0.0)

            md.append("### Complete Chronological Anchor Registry Table\n")
            md.append("| # | Ref Time (mm:ss / Frame) | Tar Time (mm:ss / Frame) | Measured Offset | Expected Offset (at speed) | Local Drift | Confidence / Hash |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            
            r_fps = ref_v.get('fps', 24.0)
            t_fps = tar_v.get('fps', 25.0)

            for idx, a in enumerate(raw_anchors):
                r_t = a['ref_time']
                t_t = a['tar_time']
                r_f = int(r_t * r_fps)
                t_f = int(t_t * t_fps)
                r_str = f"{int(r_t//60)}:{int(r_t%60):02d}.{int((r_t%1)*1000):03d} (`#{r_f}`)"
                t_str = f"{int(t_t//60)}:{int(t_t%60):02d}.{int((t_t%1)*1000):03d} (`#{t_f}`)"
                
                exp_offset = first_offset - (r_t * (1.0 - g_speed))
                drift = a['offset'] - exp_offset
                drift_str = f"**{drift:+6.3f}s**" if abs(drift) > 0.40 else f"{drift:+6.3f}s"
                
                md.append(f"| {idx+1} | {r_t:7.3f}s ({r_str}) | {t_t:7.3f}s ({t_str}) | {a['offset']:+7.4f}s | {exp_offset:+7.4f}s | {drift_str} | {a.get('confidence', 1.0)*100:.1f}% ({a.get('metrics', '')}) |")
            md.append("\n---\n")

        # Continuous Macro-Blocks
        md.append("## 4. Continuous Macro-Blocks & Independent Speed Slopes\n")
        if blocks:
            md.append("| Block # | Ref Start ➔ End | Ref Duration | Tar Start ➔ End | Tar Duration | Calibrated Speed | Net Offset | Keypoints |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for b in blocks:
                r_d = b['ref_end'] - b['ref_start']
                t_d = b['tar_end'] - b['tar_start']
                r_span_str = f"{int(b['ref_start']//60)}:{int(b['ref_start']%60):02d} ➔ {int(b['ref_end']//60)}:{int(b['ref_end']%60):02d}"
                t_span_str = f"{int(b['tar_start']//60)}:{int(b['tar_start']%60):02d} ➔ {int(b['tar_end']//60)}:{int(b['tar_end']%60):02d}"
                md.append(f"| **Block #{b['block_id']}** | {b['ref_start']:.2f}s ➔ {b['ref_end']:.2f}s ({r_span_str}) | {r_d:.2f}s | {b['tar_start']:.2f}s ➔ {b['tar_end']:.2f}s ({t_span_str}) | {t_d:.2f}s | **{b['speed_factor']:.6f}x** | {b['offset']:+.4f}s | {b['anchor_count']} anchors |")
            md.append("\n---\n")

        # Detected Omissions / Censored Scenes
        md.append("## 5. Detected Omitted / Censored Scenes (Bridged Gaps)\n")
        if gaps:
            md.append("| Cut # | Master Ref Interval | Missing Duration | Bridge Treatment |")
            md.append("| :--- | :--- | :--- | :--- |")
            for i, g in enumerate(gaps):
                cut_span_str = f"{int(g['start_time']//60)}:{int(g['start_time']%60):02d} ➔ {int(g['end_time']//60)}:{int(g['end_time']%60):02d}"
                md.append(f"| **Cut #{i+1}** | {g['start_time']:.2f}s ➔ {g['end_time']:.2f}s ({cut_span_str}) | **{g['duration']:.2f}s** | Vocal-Filtered Ambient M&E Bridge |")
        else:
            md.append("*No omitted or censored scene gaps detected — audio remained continuous throughout the timeline.*")
        md.append("\n---\n")

        # Final Timeline EDL
        md.append("## 6. Complete Final Render Edit Decision List (EDL)\n")
        md.append("| Seg # | Type | Master Ref Interval (Duration) | Foreign Dub Slice (Input Dur) | Speed Ratio | Audio Filter Executed |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for s in edl:
            s_type = f"**{s['type'].upper()}**"
            r_span = f"{s['ref_start']:.3f}s ➔ {s['ref_end']:.3f}s ({s['ref_duration']:.3f}s)"
            if s['type'] == "dub":
                t_dur = s['tar_end'] - s['tar_start']
                t_span = f"{s['tar_start']:.3f}s ➔ {s['tar_end']:.3f}s ({t_dur:.3f}s)"
                filt = f"`atempo={s['speed_factor']:.6f}`" if abs(s['speed_factor'] - 1.0) > 0.002 else "`anull (1.0x)`"
            else:
                t_span = "*- (Master Ref PCM)*"
                filt = "`equalizer vocal-suppressed M&E`"
            md.append(f"| {s['seg_id']} | {s_type} | {r_span} | {t_span} | {s['speed_factor']:.6f}x | {filt} |")
        md.append("\n---\n")

        # Closed Loop Verification Audit
        if audit:
            md.append("## 7. Closed-Loop Auto-Verification Audit Scorecard\n")
            md.append(f"* **Probed Audit Windows:** {audit.get('total_probed_windows', 0)}")
            _mean = audit.get("mean_alignment_error_ms")
            _max = audit.get("max_alignment_error_ms")
            _pct = audit.get("passed_windows_pct")
            if _mean is None:
                md.append(f"* **Alignment Measurement:** `NOT MEASURED` — no probe window produced a usable "
                          f"correlation peak, so no accuracy figure is claimed.")
            else:
                md.append(f"* **Timeline Verification Coverage:** {_pct:.1f}%")
                md.append(f"* **Mean Alignment Error:** `{_mean:.1f} ms`")
                md.append(f"* **Max Peak Error:** `{_max:.1f} ms`")
            md.append(f"* **Probe Windows Measured:** {audit.get('windows_measured', 0)}")
            md.append(f"* **Healed False Fallbacks:** **{audit.get('false_fallbacks_healed_count', 0)} gaps** ({audit.get('healed_duration_sec', 0):.1f}s of continuous dub preserved)\n")
            md.append("\n---\n")

        # Quality Summary
        md.append("## 8. Quality Control & Execution Summary\n")
        md.append(f"* **Output Master MKV:** `{data.get('output_filename')}`")
        md.append(f"* **Dub Segments Rendered:** {summary.get('dub_segments_count', 0)}")
        md.append(f"* **Bridged Fallback Cuts:** {summary.get('fallback_segments_count', 0)}")
        md.append(f"* **Mean Confidence Score:** {summary.get('average_confidence', 1.0)*100:.1f}%\n")

        return "\n".join(md)
