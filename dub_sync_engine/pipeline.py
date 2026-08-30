"""
Main execution pipeline orchestrating probing, keyframing, matching,
3-Tier Hierarchical Fallbacks, RANSAC block aggregation, and MKV multiplexing.
"""

import os
import time
import tempfile
import shutil
from dataclasses import asdict
from typing import Optional, Callable, Dict, List

from rich.console import Console

from .config import DubSyncConfig, snap_to_broadcast_speed
from .media_probe import MediaProbe, MediaInfo
from .visual_anchors import VisualAnchorEngine, AnchorMatch
from .block_segmenter import BlockSegmenterEngine, ContinuousBlock
from .orb_matcher import ORBMatcherEngine
from .spectral_fingerprint import SpectralFingerprintEngine
from .vad_engine import SileroVADEngine
from .acoustic_refine import AcousticRefineEngine
from .audio_splicer import AudioSplicerEngine
from .consensus_engine import MultiModalConsensusEngine
from .path_estimator import SyncPathEstimator
from .verifier_engine import ClosedLoopVerifierEngine
from .mkv_muxer import MKVMuxer
from .qc_report import QCReportGenerator, QCReport
from .tui import DubSyncTUI

console = Console(highlight=False)


class DubSyncPipeline:
    """End-to-end synchronization orchestrator with multi-tier fallback architecture."""

    def __init__(self, config: Optional[DubSyncConfig] = None):
        self.config = config or DubSyncConfig()
        self.tui = DubSyncTUI(self.config)
        self.probe = MediaProbe()
        self.visual_engine = VisualAnchorEngine(self.config)
        self.block_segmenter = BlockSegmenterEngine(self.config)
        self.orb_matcher = ORBMatcherEngine(self.config)
        self.spectral_engine = SpectralFingerprintEngine(self.config)
        self.vad_engine = SileroVADEngine(self.config)
        self.consensus_engine = MultiModalConsensusEngine(self.config)
        self.path_estimator = SyncPathEstimator(self.config)
        self.verifier_engine = ClosedLoopVerifierEngine(self.config)
        self.acoustic_engine = AcousticRefineEngine(self.config)
        self.splicer = AudioSplicerEngine(self.config)
        self.muxer = MKVMuxer(self.config)

    def execute(self, ref_path: str, tar_path: str, output_path: str) -> QCReport:
        """Executes the complete studio accuracy pipeline with intelligent fallbacks."""
        start_time = time.time()
        self.tui.print_banner()

        base_dir = os.path.dirname(os.path.abspath(output_path))
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        temp_dir = os.path.join(base_dir, f"{base_name}_temp_debug")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # --- STAGE 1: Media Probing ---
            with console.status("[bold cyan]Stage 1/7: Probing media streams & properties...[/bold cyan]", spinner="dots"):
                ref_info = self.probe.probe(ref_path)
                tar_info = self.probe.probe(tar_path)
                time.sleep(0.3)

            self.tui.display_media_summary(ref_info, tar_info)
            console.print()

            # --- STAGE 2: Extract PCM WAVs ---
            ref_wav = os.path.join(temp_dir, "ref_pcm.wav")
            tar_wav = os.path.join(temp_dir, "tar_pcm.wav")

            ref_audio_idx = self.probe.select_audio_stream(ref_info, self.config.ref_lang)
            tar_audio_idx = self.probe.select_audio_stream(tar_info, self.config.tar_lang)

            with console.status("[bold cyan]Stage 2/7: Extracting uncompressed 48kHz PCM audio...[/bold cyan]", spinner="dots"):
                self.probe.extract_pcm_wav(ref_path, ref_wav, sample_rate=self.config.audio_sample_rate, stream_index=ref_audio_idx)
                self.probe.extract_pcm_wav(tar_path, tar_wav, sample_rate=self.config.audio_sample_rate, stream_index=tar_audio_idx)
                console.print("  [bold green][OK][/bold green] PCM 48kHz audio extracted successfully.")

            # --- STAGE 3: Safe-Zone Crop & Keyframe Extraction ---
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            ref_anchors, tar_anchors = [], []

            if self.config.matcher_mode not in ["spectral", "vad"]:
                with console.status(f"[bold cyan]Stage 3/7: Safe-zone center-crop ({int(self.config.center_crop_ratio*100)}%) keyframe extraction...[/bold cyan]", spinner="dots"):
                    ref_anchors = self.visual_engine.extract_keyframes(ref_path, frames_dir, "ref")
                    tar_anchors = self.visual_engine.extract_keyframes(tar_path, frames_dir, "tar")
                    console.print(f"  [bold green][OK][/bold green] Extracted {len(ref_anchors)} reference anchors & {len(tar_anchors)} foreign anchors.")
            else:
                console.print(f"  [bold cyan][*][/bold cyan] Matcher mode set to [bold yellow]'{self.config.matcher_mode}'[/bold yellow]: Skipping video keyframing and running direct audio alignment.")

            # --- STAGE 4: Multi-Modal Consensus Anchor Alignment ---
            visual_matches = []
            strategy = self.config.sync_strategy
            mode = self.config.matcher_mode

            if strategy == "hybrid" or mode == "auto":
                with console.status("[bold cyan]Stage 4/7 (Multi-Modal Consensus): Fusing Visual Keyframes, Background Music Transients & Silero Neural VAD...[/bold cyan]", spinner="dots"):
                    visual_matches = self.consensus_engine.discover_consensus_anchors(
                        ref_anchors, tar_anchors, ref_wav, tar_wav, ref_info.duration, tar_info.duration
                    )
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Multi-Modal Consensus Lattice[/bold cyan] constructed [bold green]{len(visual_matches)} strong cross-modal anchors[/bold green].")

            elif mode == "vad":
                with console.status("[bold cyan]Stage 4/7 (Direct ML VAD): Running Neural Silero Voice Activity Speech Burst Discovery...[/bold cyan]", spinner="dots"):
                    visual_matches = self.vad_engine.discover_speech_anchors(ref_wav, tar_wav, ref_info.duration, tar_info.duration)
                console.print(f"  [bold green][OK][/bold green] [bold cyan]ML VAD Engine[/bold cyan] discovered [bold green]{len(visual_matches)} speech dialogue anchors[/bold green].")

            elif mode == "spectral":
                with console.status("[bold cyan]Stage 4/7 (Tier 3 Direct): Running Vocal-Suppressed Spectral Audio Fingerprint Discovery...[/bold cyan]", spinner="dots"):
                    visual_matches = self.spectral_engine.discover_spectral_anchors(ref_wav, tar_wav, ref_info.duration, tar_info.duration)
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Tier 3 (Spectral Fingerprint)[/bold cyan] discovered [bold green]{len(visual_matches)} acoustic background anchors[/bold green].")

            elif mode == "orb":
                with console.status("[bold cyan]Stage 4/7 (Tier 2 Direct): Running Scale- & Aspect-Invariant ORB Line-Art Matcher...[/bold cyan]", spinner="dots"):
                    visual_matches = self.orb_matcher.match_anchors_orb(ref_anchors, tar_anchors)
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Tier 2 (ORB Line-Art)[/bold cyan] recovered [bold green]{len(visual_matches)} scale-invariant anchors[/bold green].")

            else:
                with console.status("[bold cyan]Stage 4/7 (Tier 1 Direct): Running Primary Multi-Descriptor Visual Hash Matching...[/bold cyan]", spinner="dots"):
                    visual_matches = self.visual_engine.match_anchors(ref_anchors, tar_anchors)
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Tier 1 (Visual Hash)[/bold cyan] formed [bold green]{len(visual_matches)} visual anchors[/bold green].")

            # --- STAGE 4b: Sub-Millisecond Acoustic Refinement ---
            # Snap visual/consensus anchor cut points to sample-accurate audio transients.
            # Only meaningful for scene-cut-based anchors (visual, ORB, consensus) — not for
            # pure speech (VAD) or music-envelope (spectral) anchor sets.
            if (
                self.config.enable_acoustic_refine
                and visual_matches
                and mode in ("auto", "visual", "orb")
                and strategy != "dtw"
            ):
                with console.status("[bold cyan]Stage 4b/7 (Acoustic Refine): Sub-millisecond 48kHz cross-correlation snapping...[/bold cyan]", spinner="dots"):
                    refined = self.acoustic_engine.refine_anchors(ref_wav, tar_wav, visual_matches)
                    visual_matches = [
                        AnchorMatch(
                            ref_idx=i,
                            tar_idx=i,
                            ref_time=r.ref_time,
                            tar_time=r.tar_time,
                            hash_dist=r.hash_dist,
                            confidence=round(r.combined_confidence, 3),
                            offset=round(r.tar_time - r.ref_time, 4),
                            seq_len=r.seq_len,
                            weight=r.weight,
                            source=r.source,
                            acoustic_shift_ms=r.acoustic_offset_ms,
                            acoustic_confidence=r.acoustic_confidence,
                        )
                        for i, r in enumerate(refined)
                    ]
                avg_shift_ms = sum(r.acoustic_offset_ms for r in refined) / max(1, len(refined))
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Acoustic Refine[/bold cyan] snapped {len(visual_matches)} anchors (mean acoustic shift: [bold yellow]{avg_shift_ms:+.2f}ms[/bold yellow]).")

            # --- STAGE 5: Adaptive Block Clustering, Dense Path, or Neural DTW ---
            blocks = []
            path_segments = []
            if strategy == "dtw":
                with console.status("[bold cyan]Stage 5/7 (Neural DTW): Computing dense speech probability Dynamic Time Warping path...[/bold cyan]", spinner="dots"):
                    edl = self.vad_engine.compute_neural_dtw_edl(ref_wav, tar_wav, ref_info.duration, tar_info.duration)
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Neural DTW[/bold cyan] constructed [bold green]{len(edl)} continuous dialogue nodes[/bold green].")
            elif strategy == "path":
                with console.status("[bold cyan]Stage 5/7 (Dense Sync-Path): Measuring the true ref->tar path from the M&E fingerprint (speed + cuts + tail)...[/bold cyan]", spinner="dots"):
                    path_segments = self.path_estimator.extract_path(ref_wav, tar_wav, ref_info.duration, tar_info.duration)
                    edl = self.path_estimator.build_edl(path_segments, ref_info.duration, tar_info.duration)
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Dense Sync-Path[/bold cyan] measured [bold green]{len(path_segments)} synced region(s)[/bold green] -> {len(edl)} EDL segment(s).")
            else:
                with console.status("[bold cyan]Stage 5/7: Adaptive Macro-Block Clustering & Independent Speed Calibration...[/bold cyan]", spinner="dots"):
                    blocks = self.block_segmenter.cluster_into_blocks(
                        ref_info.duration,
                        tar_info.duration,
                        visual_matches,
                        discontinuity_threshold_sec=self.config.discontinuity_threshold_sec
                    )
                    # Anchor-fit quality gate: the block segmentation is only
                    # trustworthy when the anchors actually lie on a consistent
                    # line. With repetitive music the acoustic matcher floods
                    # the fit with false matches (low inlier ratio), so a low
                    # composite confidence means "measure the path instead of
                    # trusting these anchors" — regardless of anchor count.
                    gfit = self.block_segmenter.calibrate_global_fit(visual_matches) if visual_matches else None
                    anchor_quality = gfit.confidence if gfit is not None else 0.0

                    if len(blocks) == 1 and blocks[0].anchor_count == 0 and strategy != "blocks":
                        console.print("  [bold yellow][!][/bold yellow] Zero macro anchors found. Activating [bold cyan]Neural DTW Speech Warping[/bold cyan]...")
                        edl = self.vad_engine.compute_neural_dtw_edl(ref_wav, tar_wav, ref_info.duration, tar_info.duration)
                        console.print(f"  [bold green][OK][/bold green] [bold cyan]Neural DTW[/bold cyan] constructed [bold green]{len(edl)} continuous dialogue nodes[/bold green].")
                    elif strategy != "blocks" and (len(visual_matches) < 10 or anchor_quality < 0.5):
                        # Sparse OR low-quality anchors: block segmentation can't
                        # see cuts/tail trims reliably. Measure the true path
                        # directly from the M&E fingerprint.
                        console.print(f"  [bold yellow][!][/bold yellow] Anchor fit quality {anchor_quality:.3f} ({len(visual_matches)} anchors) — too weak for block segmentation. Measuring the dense sync path...")
                        path_segments = self.path_estimator.extract_path(ref_wav, tar_wav, ref_info.duration, tar_info.duration)
                        if path_segments:
                            edl = self.path_estimator.build_edl(path_segments, ref_info.duration, tar_info.duration)
                            console.print(f"  [bold green][OK][/bold green] [bold cyan]Dense Sync-Path[/bold cyan] measured [bold green]{len(path_segments)} synced region(s)[/bold green] -> {len(edl)} EDL segment(s).")
                        else:
                            console.print("  [bold yellow][!][/bold yellow] Dense path estimation also failed; falling back to block segmentation.")
                            edl = self.block_segmenter.build_macro_edl(ref_info.duration, tar_info.duration, blocks, matches=visual_matches)
                    else:
                        console.print(f"  [bold green][OK][/bold green] Clustered into [bold cyan]{len(blocks)} continuous macro-blocks[/bold cyan] with independent speed calibration.")
                        for b in blocks:
                            console.print(f"       -> Block #{b.block_id}: Ref [{b.ref_start:.2f}s -> {b.ref_end:.2f}s] @ [bold yellow]{b.speed_factor:.6f}x[/bold yellow] speed (Offset: {b.offset:+.4f}s, {b.anchor_count} anchors)")
                        edl = self.block_segmenter.build_macro_edl(ref_info.duration, tar_info.duration, blocks, matches=visual_matches)

            # --- STAGE 6: Autonomous Closed-Loop Self-Verification & Healing ---
            if self.config.enable_auto_verification:
                with console.status("[bold cyan]Stage 6/7: Autonomous Closed-Loop Verification & Audio Continuity Probing...[/bold cyan]", spinner="dots"):
                    edl, audit = self.verifier_engine.audit_and_heal_edl(
                        edl, ref_wav, tar_wav, ref_info.duration, tar_info.duration
                    )
                if audit.false_fallbacks_healed_count > 0:
                    console.print(f"  [bold green][OK][/bold green] [bold green]Self-Healing Activated:[/bold green] Healed [bold yellow]{audit.false_fallbacks_healed_count} false fallback gap(s)[/bold yellow] ({audit.healed_duration_sec:.1f}s of continuous dub recovered).")
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Closed-Loop Audit Passed:[/bold cyan] {audit.passed_windows_pct:.1f}% frames verified (Mean Error: [bold green]{audit.mean_alignment_error_ms:.1f}ms[/bold green]).")

            # --- STAGE 7: Continuous Audio Splicing & MKV Muxing ---
            synced_wav = os.path.join(temp_dir, "synced_dub.wav")
            with console.status("[bold cyan]Stage 7/7: Continuous audio rendering, crossfading & MKV muxing...[/bold cyan]", spinner="dots"):
                self.splicer.render_and_splice(edl, ref_wav, tar_wav, synced_wav, temp_dir)
                
                # Mux to final MKV
                self.muxer.mux(ref_path, synced_wav, output_path)
                console.print(f"  [bold green][OK][/bold green] Output MKV muxed with multi-track audio: {output_path}")

            total_elapsed = time.time() - start_time

            # --- STAGE 7: Comprehensive Forensic Telemetry Export ---
            # Build rich, structured telemetry so a diagnostic report can be fully
            # understood offline (no access to the source videos required).

            blocks_data = [
                {
                    "block_id": b.block_id,
                    "ref_start": round(b.ref_start, 3),
                    "ref_end": round(b.ref_end, 3),
                    "ref_duration": round(b.ref_duration, 3),
                    "tar_start": round(b.tar_start, 3),
                    "tar_end": round(b.tar_end, 3),
                    "tar_duration": round(b.tar_duration, 3),
                    "speed_factor": round(b.speed_factor, 6),
                    "raw_slope": round(b.raw_slope, 6),
                    "offset": round(b.offset, 4),
                    "anchor_count": b.anchor_count,
                    "confidence": round(b.confidence, 3),
                    "r_squared": round(b.r_squared, 4),
                    "inlier_ratio": round(b.inlier_ratio, 4),
                    "coverage_ratio": round(b.coverage_ratio, 4),
                    "n_buckets": b.n_buckets,
                }
                for b in (blocks if 'blocks' in locals() else [])
            ]

            path_segments_data = [
                {
                    "ref_start": s.ref_start,
                    "ref_end": s.ref_end,
                    "tar_start": s.tar_start,
                    "tar_end": s.tar_end,
                    "slope": s.slope,
                    "intercept": s.intercept,
                    "n_points": s.n_points,
                    "confidence": s.confidence,
                }
                for s in path_segments
            ]

            anchors_data = []
            source_breakdown: Dict[str, int] = {}
            for i, m in enumerate(visual_matches):
                src = getattr(m, "source", "unknown")
                source_breakdown[src] = source_breakdown.get(src, 0) + 1

                nxt = visual_matches[i + 1] if i + 1 < len(visual_matches) else None
                if nxt is not None:
                    dr = nxt.ref_time - m.ref_time
                    dt = nxt.tar_time - m.tar_time
                    local_speed = round(dt / dr, 6) if dr > 1e-6 else None
                    offset_jump = round(nxt.offset - m.offset, 4)
                else:
                    local_speed = None
                    offset_jump = None

                anchors_data.append({
                    "index": i,
                    "ref_time": round(m.ref_time, 3),
                    "tar_time": round(m.tar_time, 3),
                    "offset": round(m.offset, 4),
                    "confidence": round(m.confidence, 3),
                    "weight": round(getattr(m, "weight", 1.0), 4),
                    "source": src,
                    "seq_len": getattr(m, "seq_len", 1),
                    "hash_dist": m.hash_dist,
                    "acoustic_shift_ms": round(getattr(m, "acoustic_shift_ms", 0.0), 2),
                    "acoustic_confidence": round(getattr(m, "acoustic_confidence", 1.0), 3),
                    "local_speed_to_next": local_speed,
                    "offset_jump_to_next": offset_jump,
                })

            edl_data = [
                {
                    "seg_id": s.seg_id,
                    "type": s.segment_type,
                    "ref_start": round(s.ref_start, 3),
                    "ref_end": round(s.ref_end, 3),
                    "ref_duration": round(s.ref_duration, 3),
                    "tar_start": round(s.tar_start, 3),
                    "tar_end": round(s.tar_end, 3),
                    "tar_duration": round(s.tar_duration, 3),
                    "speed_factor": round(s.speed_factor, 6),
                    "confidence": round(s.confidence, 3)
                }
                for s in edl
            ]

            omitted_gaps = [
                {
                    "start_time": round(s.ref_start, 2),
                    "end_time": round(s.ref_end, 2),
                    "duration": round(s.ref_duration, 2),
                    "tar_time": round(s.tar_start, 2)
                }
                for s in edl if s.segment_type == "fallback"
            ]

            # --- Diagnostics: measured signals + automatically-flagged anomalies ---
            gfit = self.block_segmenter.calibrate_global_fit(visual_matches) if visual_matches else None
            global_fit_summary = None
            if gfit is not None:
                global_fit_summary = {
                    "raw_slope": round(gfit.slope, 6),
                    "snapped_slope": round(snap_to_broadcast_speed(gfit.slope), 6),
                    "intercept": round(gfit.intercept, 4),
                    "r_squared": round(gfit.r_squared, 4),
                    "inlier_ratio": round(gfit.inlier_ratio, 4),
                    "coverage_ratio": round(gfit.coverage_ratio, 4),
                    "n_buckets": int(gfit.n_buckets),
                    "n_inliers": int(gfit.n_inliers),
                    "n_total": int(gfit.n_total),
                    "confidence": round(gfit.confidence, 4),
                }

            audit_obj = locals().get("audit")
            anomalies: List[dict] = []
            consensus_diag = getattr(self.consensus_engine, "last_diagnostics", {})

            # 0. Insufficient anchors: too few matched anchors to build a reliable EDL.
            if len(anchors_data) < 10:
                anomalies.append({
                    "type": "insufficient_anchors",
                    "severity": "critical",
                    "detail": (f"Only {len(anchors_data)} matched anchors "
                               f"({len(ref_anchors)} ref / {len(tar_anchors)} tar keyframes extracted). "
                               "The EDL is built on too sparse a skeleton — likely a matcher failure."),
                })

            # 1. Visual matcher produced nothing despite substantial keyframes.
            if (len(ref_anchors) >= 50
                    and len(tar_anchors) >= 20
                    and not any(src in ("visual", "visual_gated", "orb") for src in source_breakdown)):
                found = consensus_diag.get("raw_visual_matches_found", 0)
                anomalies.append({
                    "type": "no_visual_matches",
                    "severity": "critical",
                    "detail": (f"Visual matcher found {found} raw match(es) but none survived. "
                               "Possible resolution/framerate mismatch (1080p vs 480p, VFR) breaking hashing, "
                               "or the dub/master have structurally different scene cuts."),
                })

            # 2. Visual matches found but all gated out by the acoustic gate.
            if consensus_diag.get("raw_visual_matches_found", 0) > 0 and consensus_diag.get("visual_matches_gated_in", 0) == 0:
                anomalies.append({
                    "type": "visual_matches_all_gated_out",
                    "severity": "high",
                    "detail": (f"{consensus_diag.get('raw_visual_matches_found')} visual matches were found "
                               f"but ALL were rejected by the acoustic gate. The acoustic spine is too weak "
                               f"({consensus_diag.get('acoustic_candidates', 0)} acoustic anchors) to confirm visuals."),
                })

            # 3. Verifier verified zero dub windows -> alignment unverified.
            if audit_obj is not None and getattr(audit_obj, "dub_windows_verified", 0) == 0 and any(s["type"] == "dub" for s in edl_data):
                anomalies.append({
                    "type": "no_dub_verification",
                    "severity": "critical",
                    "detail": (f"Closed-loop verification could not verify ANY dub alignment "
                               f"({getattr(audit_obj, 'dub_windows_skipped', 0)} windows skipped: "
                               "M&E envelopes do not correlate). The dub audio does not match the reference "
                               "in the 'aligned' regions — the alignment is almost certainly wrong."),
                })

            # 4. Anchor offset jumps -> candidate cuts / false-anchor teleports.
            for a in anchors_data:
                if a["offset_jump_to_next"] is not None and abs(a["offset_jump_to_next"]) > 2.0:
                    anomalies.append({
                        "type": "offset_jump",
                        "severity": "high",
                        "ref_time": a["ref_time"],
                        "offset_jump_sec": a["offset_jump_to_next"],
                        "detail": "Large |tar-ref| offset change between consecutive anchors: candidate editorial cut or a false/teleported anchor.",
                    })
            # 5. Local speeds off the broadcast band -> cut or drift.
            for a in anchors_data:
                if a["local_speed_to_next"] is not None and not (0.90 <= a["local_speed_to_next"] <= 1.10):
                    anomalies.append({
                        "type": "local_speed_out_of_range",
                        "severity": "medium",
                        "ref_time": a["ref_time"],
                        "local_speed": a["local_speed_to_next"],
                        "detail": "Per-interval speed outside the broadcast band [0.90, 1.10]: likely a real cut or a misplaced anchor.",
                    })
            # 6. Weak acoustic support on visual anchors.
            for a in anchors_data:
                if a["source"] in ("visual", "visual_gated") and a["acoustic_confidence"] < 0.5:
                    anomalies.append({
                        "type": "weak_acoustic_support",
                        "severity": "low",
                        "ref_time": a["ref_time"],
                        "acoustic_confidence": a["acoustic_confidence"],
                        "detail": "Visual anchor with poor acoustic confirmation at refinement: could be a soft-cut or a false visual match.",
                    })
            # 7. Low-confidence / weak-fit blocks.
            for b in blocks_data:
                if b["confidence"] < 0.5:
                    anomalies.append({
                        "type": "low_confidence_block",
                        "severity": "high",
                        "block_id": b["block_id"],
                        "confidence": b["confidence"],
                        "detail": "Block with low measured fit confidence (r^2 * inlier_ratio * coverage): weak anchor support.",
                    })
                if b["r_squared"] < 0.7 and b["anchor_count"] >= 3:
                    anomalies.append({
                        "type": "weak_block_fit",
                        "severity": "medium",
                        "block_id": b["block_id"],
                        "r_squared": b["r_squared"],
                        "detail": "Block line fit has low Pearson r^2: anchors deviate from a single speed line (drift or mixed speeds).",
                    })
            # 8. Non-standard target framerate (likely VFR) — breaks keyframe timing.
            _STANDARD_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0)
            tar_fps = (tar_info.primary_video.fps if tar_info.primary_video else None)
            if tar_fps is not None and all(abs(tar_fps - s) > 0.05 for s in _STANDARD_FPS):
                anomalies.append({
                    "type": "nonstandard_framerate",
                    "severity": "high",
                    "fps": round(tar_fps, 3),
                    "detail": (f"Target framerate {tar_fps:.3f} is not a broadcast standard — "
                               "likely a variable-frame-rate file. This breaks keyframe/scene-cut alignment."),
                })
            # 9. Undetermined target audio language -> wrong stream may have been selected.
            tar_lang = tar_info.primary_audio.language if tar_info.primary_audio else None
            if tar_lang in (None, "und", "unknown", ""):
                anomalies.append({
                    "type": "undetermined_audio_language",
                    "severity": "high",
                    "lang": tar_lang,
                    "detail": "Target audio stream language is undetermined ('und') — the wrong audio stream may have been selected for syncing.",
                })
            # 10. Misaligned verifier windows.
            if audit_obj is not None and hasattr(audit_obj, "audit_log"):
                for rec in audit_obj.audit_log:
                    if rec.get("action") == "VERIFIED_ALIGNMENT" and abs(rec.get("error_ms", 0)) > 40.0:
                        anomalies.append({
                            "type": "misaligned_window",
                            "severity": "high",
                            "ref_start": rec.get("ref_start"),
                            "error_ms": rec.get("error_ms"),
                            "correlation": rec.get("correlation"),
                            "detail": "Closed-loop verification window with residual alignment error > 40ms (one 24fps frame).",
                        })

            diagnostics = {
                "global_fit": global_fit_summary,
                "source_breakdown": source_breakdown,
                "consensus_diagnostics": consensus_diag,
                "anchor_count": len(anchors_data),
                "block_count": len(blocks_data),
                "anomaly_count": len(anomalies),
                "anomalies": anomalies,
            }

            # Full config, with Enums serialized to their value.
            config_data = {
                k: (v.value if hasattr(v, "value") else v)
                for k, v in asdict(self.config).items()
            }

            forensic_payload = {
                "schema_version": "2.1",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dub_sync_version": "v2.0.0",
                "execution_time_sec": round(total_elapsed, 2),
                "output_filename": os.path.basename(output_path),
                "pipeline_configuration": config_data,
                "media_specs": {
                    "ref_filename": ref_info.filename,
                    "tar_filename": tar_info.filename,
                    "ref_duration_sec": round(ref_info.duration, 3),
                    "tar_duration_sec": round(tar_info.duration, 3),
                    "duration_delta_sec": round(ref_info.duration - tar_info.duration, 3),
                    "ref_video": {
                        "resolution": f"{ref_info.primary_video.width}x{ref_info.primary_video.height}" if ref_info.primary_video else "N/A",
                        "fps": round(ref_info.primary_video.fps, 3) if ref_info.primary_video else 24.0,
                        "codec": ref_info.primary_video.codec if ref_info.primary_video else "N/A"
                    },
                    "tar_video": {
                        "resolution": f"{tar_info.primary_video.width}x{tar_info.primary_video.height}" if tar_info.primary_video else "N/A",
                        "fps": round(tar_info.primary_video.fps, 3) if tar_info.primary_video else 25.0,
                        "codec": tar_info.primary_video.codec if tar_info.primary_video else "N/A"
                    },
                    "ref_audio": {
                        "codec": ref_info.primary_audio.codec if ref_info.primary_audio else "N/A",
                        "sample_rate": ref_info.primary_audio.sample_rate if ref_info.primary_audio else 48000,
                        "channels": ref_info.primary_audio.channels if ref_info.primary_audio else 2,
                        "lang": ref_info.primary_audio.language if ref_info.primary_audio else "eng"
                    },
                    "tar_audio": {
                        "codec": tar_info.primary_audio.codec if tar_info.primary_audio else "N/A",
                        "sample_rate": tar_info.primary_audio.sample_rate if tar_info.primary_audio else 44100,
                        "channels": tar_info.primary_audio.channels if tar_info.primary_audio else 2,
                        "lang": tar_info.primary_audio.language if tar_info.primary_audio else "ara"
                    }
                },
                "matching_engine_telemetry": {
                    "active_mode": self.config.matcher_mode,
                    "cascade_tier_selected": "Tier 4 (Neural VAD)" if self.config.matcher_mode == "vad" else "Tier 1 (Visual Hash)",
                    "ref_anchors_extracted": len(ref_anchors),
                    "tar_anchors_extracted": len(tar_anchors),
                    "matched_anchors_count": len(visual_matches),
                    "source_breakdown": source_breakdown,
                    "consensus_diagnostics": consensus_diag,
                    "anchors": anchors_data
                },
                "continuous_blocks": blocks_data,
                "sync_path_segments": path_segments_data,
                "path_diagnostics": getattr(self.path_estimator, "last_diagnostics", {}),
                "timeline_edl": edl_data,
                "omitted_censored_gaps": omitted_gaps,
                "verifier_audit": asdict(audit_obj) if audit_obj is not None and hasattr(audit_obj, '__dataclass_fields__') else (audit_obj if isinstance(audit_obj, dict) else {}),
                "diagnostics": diagnostics,
                "quality_summary": {
                    "dub_segments_count": sum(1 for s in edl if s.segment_type == "dub"),
                    "fallback_segments_count": len(omitted_gaps),
                    "average_confidence": round(sum(s.confidence for s in edl) / max(1, len(edl)), 3)
                }
            }

            json_report, md_report = None, None
            if self.config.generate_report:
                from .qc_report import ForensicReportGenerator
                json_report, md_report = ForensicReportGenerator.generate_and_save(forensic_payload, output_path)
                console.print(f"  [bold green][OK][/bold green] Comprehensive Forensic Diagnostic Report saved:")
                console.print(f"       -> JSON: [cyan]{json_report}[/cyan]")
                console.print(f"       -> Markdown: [cyan]{md_report}[/cyan]")

            console.print()
            # Summary display
            from .qc_report import QCReport
            qc_summary = QCReport(
                timestamp=forensic_payload["timestamp"],
                ref_filename=ref_info.filename,
                tar_filename=tar_info.filename,
                output_filename=os.path.basename(output_path),
                ref_duration_sec=forensic_payload["media_specs"]["ref_duration_sec"],
                tar_duration_sec=forensic_payload["media_specs"]["tar_duration_sec"],
                duration_delta_sec=forensic_payload["media_specs"]["duration_delta_sec"],
                total_anchors_extracted_ref=len(ref_anchors),
                total_anchors_extracted_tar=len(tar_anchors),
                matched_anchors_count=len(visual_matches),
                refined_anchors_count=len(visual_matches),
                edl_total_segments=len(edl),
                edl_dub_segments=forensic_payload["quality_summary"]["dub_segments_count"],
                edl_fallback_segments=len(omitted_gaps),
                average_confidence=forensic_payload["quality_summary"]["average_confidence"],
                omitted_scenes=omitted_gaps,
                processing_time_sec=forensic_payload["execution_time_sec"]
            )
            self.tui.display_qc_summary(qc_summary)
            return qc_summary

        finally:
            console.print(f"\n[bold yellow][DEBUG ARTIFACTS][/bold yellow] Intermediate WAVs & frames stored at:")
            console.print(f"       -> [cyan]{temp_dir}[/cyan]")
            try:
                from rich.prompt import Confirm
                delete_temp = Confirm.ask("Do you want to DELETE this temporary debug folder?", default=False)
                if delete_temp:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    console.print("  [dim]Temporary debug folder deleted.[/dim]")
                else:
                    console.print(f"  [bold green][OK][/bold green] Debug folder preserved for inspection: [cyan]{temp_dir}[/cyan]")
            except Exception:
                console.print(f"  [bold green][OK][/bold green] Debug folder preserved: [cyan]{temp_dir}[/cyan]")
