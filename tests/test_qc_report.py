"""Forensic report rendering: the report must be self-contained and diagnosable
offline (schema_version, diagnostics/anomalies, global fit, source breakdown,
per-window verifier log, per-anchor/per-block telemetry)."""
import json
import os
import pytest

from dub_sync_engine.qc_report import ForensicReportGenerator


def _sample_payload():
    """A representative forensic payload shaped like what the pipeline emits."""
    return {
        "schema_version": "2.1",
        "timestamp": "2026-08-30 12:00:00",
        "dub_sync_version": "v2.0.0",
        "execution_time_sec": 12.34,
        "output_filename": "episode_Synced.mkv",
        "pipeline_configuration": {
            "strategy": "hybrid", "matcher_mode": "auto",
            "scene_threshold": 0.22, "ransac_inlier_tolerance_sec": 1.0,
            "strict_speed": True, "min_acoustic_peak": 0.5,
        },
        "media_specs": {
            "ref_filename": "ref.mkv", "tar_filename": "tar.mp4",
            "ref_duration_sec": 600.0, "tar_duration_sec": 576.0,
            "duration_delta_sec": 24.0,
            "ref_video": {"resolution": "1920x1080", "fps": 24.0, "codec": "h264"},
            "tar_video": {"resolution": "1280x720", "fps": 25.0, "codec": "h264"},
            "ref_audio": {"codec": "aac", "sample_rate": 48000, "channels": 2, "lang": "eng"},
            "tar_audio": {"codec": "aac", "sample_rate": 44100, "channels": 2, "lang": "ara"},
        },
        "matching_engine_telemetry": {
            "active_mode": "auto",
            "cascade_tier_selected": "Tier 1 (Visual Hash)",
            "ref_anchors_extracted": 120, "tar_anchors_extracted": 130,
            "matched_anchors_count": 43,
            "source_breakdown": {"acoustic_music": 20, "vad_speech": 8, "visual_gated": 15},
            "anchors": [
                {"index": 0, "ref_time": 0.0, "tar_time": 12.8, "offset": 12.8,
                 "confidence": 0.4, "weight": 0.1, "source": "visual_gated", "seq_len": 1,
                 "hash_dist": 8, "acoustic_shift_ms": 0.0, "acoustic_confidence": 0.2,
                 "local_speed_to_next": 0.53, "offset_jump_to_next": -44.8},
                {"index": 1, "ref_time": 32.0, "tar_time": 0.0, "offset": -32.0,
                 "confidence": 0.95, "weight": 0.9, "source": "acoustic_music", "seq_len": 3,
                 "hash_dist": 0, "acoustic_shift_ms": 1.2, "acoustic_confidence": 0.98,
                 "local_speed_to_next": 1.0, "offset_jump_to_next": 0.0},
            ],
        },
        "continuous_blocks": [
            {"block_id": 0, "ref_start": 32.0, "ref_end": 600.0,
             "tar_start": 0.0, "tar_end": 568.0, "speed_factor": 0.96, "raw_slope": 0.9612,
             "offset": -32.0, "anchor_count": 42, "confidence": 0.98,
             "r_squared": 0.99, "inlier_ratio": 0.98, "coverage_ratio": 1.0, "n_buckets": 9},
        ],
        "timeline_edl": [
            {"seg_id": 0, "type": "fallback", "ref_start": 0.0, "ref_end": 32.0,
             "ref_duration": 32.0, "tar_start": 0.0, "tar_end": 0.0, "tar_duration": 0.0,
             "speed_factor": 1.0, "confidence": 1.0},
            {"seg_id": 1, "type": "dub", "ref_start": 32.0, "ref_end": 600.0,
             "ref_duration": 568.0, "tar_start": 0.0, "tar_end": 568.0, "tar_duration": 568.0,
             "speed_factor": 0.96, "confidence": 0.95},
        ],
        "omitted_censored_gaps": [
            {"start_time": 0.0, "end_time": 32.0, "duration": 32.0, "tar_time": 0.0},
        ],
        "verifier_audit": {
            "total_probed_windows": 12,
            "mean_alignment_error_ms": 3.2,
            "max_alignment_error_ms": 41.0,
            "passed_windows_pct": 91.7,
            "false_fallbacks_healed_count": 0,
            "healed_duration_sec": 0.0,
            "audit_log": [
                {"ref_start": 40.0, "ref_end": 55.0, "action": "VERIFIED_ALIGNMENT",
                 "error_ms": 2.1, "correlation": 0.98},
                {"ref_start": 55.0, "ref_end": 70.0, "action": "VERIFIED_ALIGNMENT",
                 "error_ms": 41.0, "correlation": 0.45},
                {"ref_start": 300.0, "ref_end": 309.0, "action": "CONFIRMED_GENUINE_CUT",
                 "duration": 9.0},
            ],
        },
        "diagnostics": {
            "global_fit": {
                "raw_slope": 0.9612, "snapped_slope": 0.96, "intercept": -32.0,
                "r_squared": 0.99, "inlier_ratio": 0.98, "coverage_ratio": 1.0,
                "n_buckets": 9, "n_inliers": 42, "n_total": 43, "confidence": 0.97,
            },
            "source_breakdown": {"acoustic_music": 20, "vad_speech": 8, "visual_gated": 15},
            "anchor_count": 43, "block_count": 1, "anomaly_count": 2,
            "anomalies": [
                {"type": "offset_jump", "severity": "high", "ref_time": 0.0,
                 "offset_jump_sec": -44.8,
                 "detail": "Large offset change between consecutive anchors."},
                {"type": "misaligned_window", "severity": "high", "ref_start": 55.0,
                 "error_ms": 41.0, "correlation": 0.45,
                 "detail": "Residual alignment error > 40ms."},
            ],
        },
        "quality_summary": {
            "dub_segments_count": 1, "fallback_segments_count": 1, "average_confidence": 0.975,
        },
    }


def test_markdown_contains_all_sections():
    md = ForensicReportGenerator._build_markdown(_sample_payload())
    for section in [
        "## 0. Diagnostics — Anomaly & Flaw Analysis",
        "## 1. Media Ingestion Specifications",
        "## 2. Active Pipeline Configuration",
        "## 3. Multi-Modal Alignment",
        "## 4. Continuous Macro-Blocks",
        "## 5. Detected Omitted",
        "## 6. Complete Final Render Edit Decision List",
        "## 7. Closed-Loop Auto-Verification Audit",
        "## 8. Quality Control",
        "Global RANSAC Fit Summary",
        "Anchor Source Breakdown",
        "Per-Window Audit Log",
    ]:
        assert section in md, f"missing section: {section}"


def test_markdown_flags_anomalies():
    md = ForensicReportGenerator._build_markdown(_sample_payload())
    assert "2 potential flaw(s) auto-flagged" in md
    assert "offset_jump" in md
    assert "misaligned_window" in md


def test_markdown_renders_verifier_log():
    md = ForensicReportGenerator._build_markdown(_sample_payload())
    assert "VERIFIED_ALIGNMENT" in md
    assert "CONFIRMED_GENUINE_CUT" in md


def test_generate_and_save_writes_json_and_md(tmp_path):
    out = str(tmp_path / "episode_Synced.mkv")
    json_path, md_path = ForensicReportGenerator.generate_and_save(_sample_payload(), out)
    assert os.path.exists(json_path)
    assert os.path.exists(md_path)

    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["schema_version"] == "2.1"
    assert "diagnostics" in payload
    assert "verifier_audit" in payload
    assert payload["verifier_audit"]["audit_log"], "audit_log must round-trip through JSON"


def test_markdown_empty_payload_no_crash():
    """A minimal/empty payload must render without raising."""
    md = ForensicReportGenerator._build_markdown({})
    assert md  # non-empty string, no exception
