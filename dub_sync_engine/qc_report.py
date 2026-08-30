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
    def _fmt_time(sec: float, fps: Optional[float] = None) -> str:
        """Format seconds as `m:ss.mmm` (optionally with a frame number)."""
        if sec is None:
            return "-"
        sec = float(sec)
        s = f"{int(sec // 60)}:{int(sec % 60):02d}.{int(round((sec % 1) * 1000)):03d}"
        if fps:
            s += f" (#{int(round(sec * fps))})"
        return s

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
        diag = data.get("diagnostics", {})
        anomalies = diag.get("anomalies", [])
        global_fit = diag.get("global_fit")
        source_breakdown = matching.get("source_breakdown", diag.get("source_breakdown", {}))

        md: List[str] = []
        md.append(f"# DubSync Pro — Comprehensive Forensic Diagnostic Report\n")
        md.append(f"**Generated:** {data.get('timestamp', 'N/A')} | "
                  f"**Schema:** v{data.get('schema_version', '1.0')} | "
                  f"**Engine:** {data.get('dub_sync_version', 'v2.2.0')} | "
                  f"**Processing:** {data.get('execution_time_sec', 0):.2f}s\n")
        md.append("---\n")

        # ────────────────────────────────────────────────────────────────
        # 0. Diagnostics — anomaly / flaw analysis (front and center)
        # ────────────────────────────────────────────────────────────────
        md.append("## 0. Diagnostics — Anomaly & Flaw Analysis\n")
        if anomalies:
            md.append(f"**{len(anomalies)} potential flaw(s) auto-flagged.** Review these first.\n")
            md.append("| Severity | Type | Ref Time | Signal | Detail |")
            md.append("| :--- | :--- | :--- | :--- | :--- |")
            for a in anomalies:
                t = a.get("ref_time", a.get("ref_start", a.get("block_id", "-")))
                if isinstance(t, float):
                    t = ForensicReportGenerator._fmt_time(t)
                sig = ""
                for key in ("offset_jump_sec", "local_speed", "r_squared", "confidence",
                            "error_ms", "correlation", "acoustic_confidence"):
                    if key in a:
                        sig += f"{key}={a[key]} "
                md.append(f"| {a.get('severity', 'unknown')} | `{a.get('type')}` | {t} | `{sig.strip()}` | {a.get('detail', '')} |")
        else:
            md.append("*No anomalies detected — the sync ran clean.*")
        md.append("")

        if global_fit:
            md.append("### Global RANSAC Fit Summary\n")
            md.append("| Metric | Value |")
            md.append("| :--- | :--- |")
            md.append(f"| Raw slope (speed ratio) | {global_fit.get('raw_slope'):.6f} |")
            md.append(f"| Snapped slope (broadcast) | {global_fit.get('snapped_slope'):.6f} |")
            md.append(f"| Intercept (global offset) | {global_fit.get('intercept'):+.4f}s |")
            md.append(f"| Pearson r² | {global_fit.get('r_squared'):.4f} |")
            md.append(f"| Inlier ratio | {global_fit.get('inlier_ratio'):.4f} ({global_fit.get('n_inliers')}/{global_fit.get('n_total')}) |")
            md.append(f"| Coverage ratio | {global_fit.get('coverage_ratio'):.4f} ({global_fit.get('n_buckets')} buckets) |")
            md.append(f"| Composite confidence | {global_fit.get('confidence'):.4f} |")
            md.append("")

        if source_breakdown:
            md.append("### Anchor Source Breakdown\n")
            md.append("| Source | Count |")
            md.append("| :--- | :--- |")
            for src, cnt in sorted(source_breakdown.items(), key=lambda kv: -kv[1]):
                md.append(f"| `{src}` | {cnt} |")
            md.append("")

        md.append("---\n")

        # ────────────────────────────────────────────────────────────────
        # 1. Media specs
        # ────────────────────────────────────────────────────────────────
        md.append("## 1. Media Ingestion Specifications\n")
        ref_v = media.get("ref_video", {})
        tar_v = media.get("tar_video", {})
        ref_a = media.get("ref_audio", {})
        tar_a = media.get("tar_audio", {})

        md.append("| Stream | Reference (HQ Master) | Target (Foreign Dub) | Delta / Notes |")
        md.append("| :--- | :--- | :--- | :--- |")
        ref_dur = media.get("ref_duration_sec", 0.0) or 0.0
        tar_dur = media.get("tar_duration_sec", 0.0) or 0.0
        delta = media.get("duration_delta_sec", 0.0) or 0.0
        ref_fps = (ref_v.get("fps") or 24.0)
        tar_fps = (tar_v.get("fps") or 25.0)
        md.append(f"| **File** | `{media.get('ref_filename', 'N/A')}` | `{media.get('tar_filename', 'N/A')}` | - |")
        md.append(f"| **Duration** | {ref_dur:.3f}s ({int(ref_dur//60)}m {int(ref_dur%60):02d}s) | {tar_dur:.3f}s ({int(tar_dur//60)}m {int(tar_dur%60):02d}s) | {delta:+.3f}s |")
        md.append(f"| **Video** | {ref_v.get('resolution', 'N/A')} @ {ref_fps:.3f}fps ({ref_v.get('codec', 'N/A')}) | {tar_v.get('resolution', 'N/A')} @ {tar_fps:.3f}fps ({tar_v.get('codec', 'N/A')}) | FPS Ratio: {ref_fps/max(0.01, tar_fps):.4f} |")
        md.append(f"| **Audio** | {ref_a.get('codec', 'N/A')} {ref_a.get('sample_rate', '?')}Hz ({ref_a.get('channels', '?')}ch, {ref_a.get('lang', '?')}) | {tar_a.get('codec', 'N/A')} {tar_a.get('sample_rate', '?')}Hz ({tar_a.get('channels', '?')}ch, {tar_a.get('lang', '?')}) | Internal Resample: 48,000Hz PCM |")
        md.append("\n---\n")

        # ────────────────────────────────────────────────────────────────
        # 2. Pipeline configuration
        # ────────────────────────────────────────────────────────────────
        md.append("## 2. Active Pipeline Configuration & Variables\n")
        md.append("| Variable / Parameter | Configured Value |")
        md.append("| :--- | :--- |")
        for k in sorted(cfg.keys()):
            md.append(f"| `{k}` | `{cfg[k]}` |")
        md.append("\n---\n")

        # ────────────────────────────────────────────────────────────────
        # 3. Anchor registry
        # ────────────────────────────────────────────────────────────────
        md.append("## 3. Multi-Modal Alignment & Complete Anchor Registry\n")
        md.append(f"* **Active Strategy:** `{cfg.get('strategy', 'hybrid')}` | **Matcher Mode:** `{matching.get('active_mode')}`")
        md.append(f"* **Cascade Selection:** `{matching.get('cascade_tier_selected')}`")
        md.append(f"* **Total Keyframes Extracted:** Reference: {matching.get('ref_anchors_extracted', 0)} | Foreign Target: {matching.get('tar_anchors_extracted', 0)}")
        md.append(f"* **Total Matched Keypoints:** **{matching.get('matched_anchors_count', 0)} keypoints**\n")

        raw_anchors = matching.get("anchors", [])
        if raw_anchors:
            r_fps = ref_v.get('fps', 24.0)
            md.append("### Complete Chronological Anchor Registry Table\n")
            md.append("| # | Ref Time | Tar Time | Offset | Local Speed →next | Offset Jump →next | Source | Weight | Seq | Ac.Shift | Ac.Conf | Conf |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for a in raw_anchors:
                ls = a.get("local_speed_to_next")
                ls_s = f"{ls:.4f}" if ls is not None else "-"
                oj = a.get("offset_jump_to_next")
                oj_s = f"**{oj:+.3f}**" if oj is not None and abs(oj) > 2.0 else (f"{oj:+.3f}" if oj is not None else "-")
                md.append(
                    f"| {a.get('index', '?')} | {ForensicReportGenerator._fmt_time(a.get('ref_time'), r_fps)} | "
                    f"{ForensicReportGenerator._fmt_time(a.get('tar_time'))} | {a.get('offset', 0):+.3f}s | "
                    f"{ls_s} | {oj_s} | `{a.get('source', 'unknown')}` | {a.get('weight', 1.0):.2f} | "
                    f"{a.get('seq_len', 1)} | {a.get('acoustic_shift_ms', 0):+.1f}ms | {a.get('acoustic_confidence', 1.0):.2f} | "
                    f"{a.get('confidence', 1.0) * 100:.0f}% |"
                )
            md.append("\n---\n")

        # ────────────────────────────────────────────────────────────────
        # 4. Continuous macro-blocks
        # ────────────────────────────────────────────────────────────────
        md.append("## 4. Continuous Macro-Blocks & Independent Speed Slopes\n")
        if blocks:
            md.append("| Block | Ref Span | Tar Span | Speed | Raw Slope | Offset | r² | Inliers | Coverage | Anchors | Conf |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for b in blocks:
                r_span = f"{ForensicReportGenerator._fmt_time(b.get('ref_start'))} ➔ {ForensicReportGenerator._fmt_time(b.get('ref_end'))}"
                t_span = f"{ForensicReportGenerator._fmt_time(b.get('tar_start'))} ➔ {ForensicReportGenerator._fmt_time(b.get('tar_end'))}"
                inl = f"{b.get('inlier_ratio', 0):.2f}"
                md.append(
                    f"| **#{b.get('block_id')}** | {r_span} | {t_span} | **{b.get('speed_factor', 1.0):.6f}x** | "
                    f"{b.get('raw_slope', 0):.6f} | {b.get('offset', 0):+.4f}s | {b.get('r_squared', 0):.3f} | "
                    f"{inl} | {b.get('coverage_ratio', 1.0):.2f} ({b.get('n_buckets', 0)}b) | {b.get('anchor_count', 0)} | {b.get('confidence', 0) * 100:.0f}% |"
                )
            md.append("\n---\n")

        # ────────────────────────────────────────────────────────────────
        # 5. Omissions / censored scenes
        # ────────────────────────────────────────────────────────────────
        md.append("## 5. Detected Omitted / Censored Scenes (Bridged Gaps)\n")
        if gaps:
            md.append("| Cut # | Master Ref Interval | Missing Duration | Target Position | Bridge Treatment |")
            md.append("| :--- | :--- | :--- | :--- | :--- |")
            for i, g in enumerate(gaps):
                cut_span = f"{ForensicReportGenerator._fmt_time(g.get('start_time'))} ➔ {ForensicReportGenerator._fmt_time(g.get('end_time'))}"
                md.append(f"| **Cut #{i+1}** | {cut_span} | **{g.get('duration', 0):.2f}s** | {ForensicReportGenerator._fmt_time(g.get('tar_time'))} | Vocal-Filtered Ambient M&E Bridge |")
        else:
            md.append("*No omitted or censored scene gaps detected — audio remained continuous throughout the timeline.*")
        md.append("\n---\n")

        # ────────────────────────────────────────────────────────────────
        # 6. Timeline EDL
        # ────────────────────────────────────────────────────────────────
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

        # ────────────────────────────────────────────────────────────────
        # 7. Closed-loop verification audit
        # ────────────────────────────────────────────────────────────────
        md.append("## 7. Closed-Loop Auto-Verification Audit Scorecard\n")
        if audit:
            md.append(f"* **Probed Audit Windows:** {audit.get('total_probed_windows', 0)}")
            md.append(f"* **Timeline Verification Coverage:** {audit.get('passed_windows_pct', 100.0):.1f}%")
            md.append(f"* **Mean Alignment Error:** `{audit.get('mean_alignment_error_ms', 0):.1f} ms`")
            md.append(f"* **Max Peak Error:** `{audit.get('max_alignment_error_ms', 0):.1f} ms`")
            md.append(f"* **Healed False Fallbacks:** **{audit.get('false_fallbacks_healed_count', 0)} gaps** ({audit.get('healed_duration_sec', 0):.1f}s of continuous dub preserved)\n")

            audit_log = audit.get("audit_log", [])
            if audit_log:
                md.append("### Per-Window Audit Log\n")
                md.append("| Ref Start | Ref End | Action | Error (ms) | Correlation | Detail |")
                md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
                for rec in audit_log:
                    detail = ""
                    if rec.get("action") == "HEALED_FALSE_FALLBACK":
                        detail = f"healed {rec.get('healed_duration', 0):.1f}s, lag={rec.get('lag_samples', 0)}"
                    elif rec.get("action") == "CONFIRMED_GENUINE_CUT":
                        detail = f"duration={rec.get('duration', 0):.1f}s"
                    err = rec.get("error_ms")
                    err_s = f"{err:.1f}" if err is not None else "-"
                    corr = rec.get("correlation")
                    corr_s = f"{corr:.3f}" if corr is not None else "-"
                    md.append(f"| {ForensicReportGenerator._fmt_time(rec.get('ref_start'))} | {ForensicReportGenerator._fmt_time(rec.get('ref_end'))} | `{rec.get('action')}` | {err_s} | {corr_s} | {detail} |")
        else:
            md.append("*Auto-verification was disabled for this run — no audit data collected.*")
        md.append("\n---\n")

        # ────────────────────────────────────────────────────────────────
        # 8. Quality summary
        # ────────────────────────────────────────────────────────────────
        md.append("## 8. Quality Control & Execution Summary\n")
        md.append(f"* **Output Master MKV:** `{data.get('output_filename')}`")
        md.append(f"* **Dub Segments Rendered:** {summary.get('dub_segments_count', 0)}")
        md.append(f"* **Bridged Fallback Cuts:** {summary.get('fallback_segments_count', 0)}")
        md.append(f"* **Mean Confidence Score:** {summary.get('average_confidence', 1.0)*100:.1f}%\n")

        return "\n".join(md)
