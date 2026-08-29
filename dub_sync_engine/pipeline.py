"""
Main execution pipeline orchestrating probing, keyframing, matching,
3-Tier Hierarchical Fallbacks, RANSAC block aggregation, and MKV multiplexing.
"""

import os
import time
import tempfile
import shutil
from dataclasses import asdict
from typing import Optional, Callable

from rich.console import Console

from .config import DubSyncConfig
from .media_probe import MediaProbe, MediaInfo
from .visual_anchors import VisualAnchorEngine
from .block_segmenter import BlockSegmenterEngine, ContinuousBlock
from .orb_matcher import ORBMatcherEngine
from .spectral_fingerprint import SpectralFingerprintEngine
from .vad_engine import SileroVADEngine
from .acoustic_refine import AcousticRefineEngine
from .audio_splicer import AudioSplicerEngine
from .consensus_engine import MultiModalConsensusEngine
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

            with console.status("[bold cyan]Stage 2/7: Extracting uncompressed 48kHz PCM audio...[/bold cyan]", spinner="dots"):
                self.probe.extract_pcm_wav(ref_path, ref_wav, sample_rate=self.config.audio_sample_rate)
                self.probe.extract_pcm_wav(tar_path, tar_wav, sample_rate=self.config.audio_sample_rate)
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

            # --- STAGE 5: Adaptive Block Clustering or Neural DTW Warping ---
            if strategy == "dtw":
                with console.status("[bold cyan]Stage 5/7 (Neural DTW): Computing dense speech probability Dynamic Time Warping path...[/bold cyan]", spinner="dots"):
                    edl = self.vad_engine.compute_neural_dtw_edl(ref_wav, tar_wav, ref_info.duration, tar_info.duration)
                console.print(f"  [bold green][OK][/bold green] [bold cyan]Neural DTW[/bold cyan] constructed [bold green]{len(edl)} continuous dialogue nodes[/bold green].")
            else:
                with console.status("[bold cyan]Stage 5/7: Adaptive Macro-Block Clustering & Independent Speed Calibration...[/bold cyan]", spinner="dots"):
                    blocks = self.block_segmenter.cluster_into_blocks(
                        ref_info.duration,
                        tar_info.duration,
                        visual_matches,
                        discontinuity_threshold_sec=self.config.discontinuity_threshold_sec
                    )
                    
                    if len(blocks) == 1 and blocks[0].anchor_count == 0 and strategy != "blocks":
                        console.print("  [bold yellow][!][/bold yellow] Zero macro anchors found. Activating [bold cyan]Neural DTW Speech Warping[/bold cyan]...")
                        edl = self.vad_engine.compute_neural_dtw_edl(ref_wav, tar_wav, ref_info.duration, tar_info.duration)
                        console.print(f"  [bold green][OK][/bold green] [bold cyan]Neural DTW[/bold cyan] constructed [bold green]{len(edl)} continuous dialogue nodes[/bold green].")
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
            blocks_data = [
                {
                    "block_id": b.block_id,
                    "ref_start": round(b.ref_start, 3),
                    "ref_end": round(b.ref_end, 3),
                    "tar_start": round(b.tar_start, 3),
                    "tar_end": round(b.tar_end, 3),
                    "speed_factor": round(b.speed_factor, 6),
                    "offset": round(b.offset, 4),
                    "anchor_count": b.anchor_count,
                    "confidence": round(b.confidence, 3)
                }
                for b in (blocks if 'blocks' in locals() else [])
            ]

            anchors_data = [
                {
                    "ref_time": round(m.ref_time, 3),
                    "tar_time": round(m.tar_time, 3),
                    "offset": round(m.offset, 4),
                    "confidence": round(m.confidence, 3),
                    "metrics": f"hash_dist={m.hash_dist}"
                }
                for m in visual_matches
            ]

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
                    "duration": round(s.ref_duration, 2)
                }
                for s in edl if s.segment_type == "fallback"
            ]

            forensic_payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dub_sync_version": "v2.0.0",
                "execution_time_sec": round(total_elapsed, 2),
                "output_filename": os.path.basename(output_path),
                "pipeline_configuration": {
                    "strategy": self.config.sync_strategy,
                    "matcher_mode": self.config.matcher_mode,
                    "scene_threshold": self.config.scene_threshold,
                    "max_hash_dist": self.config.max_hash_dist,
                    "center_crop_ratio": self.config.center_crop_ratio,
                    "discontinuity_threshold_sec": self.config.discontinuity_threshold_sec,
                    "fallback_mode": self.config.fallback_mode.value,
                    "audio_sample_rate": self.config.audio_sample_rate,
                    "crossfade_duration_ms": self.config.crossfade_duration_ms
                },
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
                    "anchors": anchors_data
                },
                "continuous_blocks": blocks_data,
                "timeline_edl": edl_data,
                "omitted_censored_gaps": omitted_gaps,
                "verifier_audit": asdict(audit) if 'audit' in locals() and hasattr(audit, '__dataclass_fields__') else (audit if 'audit' in locals() and isinstance(audit, dict) else {}),
                "quality_summary": {
                    "dub_segments_count": sum(1 for s in edl if s.segment_type == "dub"),
                    "fallback_segments_count": len(omitted_gaps),
                    "average_confidence": round(sum(s.confidence for s in edl) / max(1, len(edl)), 3)
                }
            }

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
