"""
Command-Line Interface and interactive entrypoint for DubSync Pro.
"""

import sys
import argparse
import os

from .config import DubSyncConfig, Preset, FallbackMode
from .pipeline import DubSyncPipeline
from .tui import DubSyncTUI


def _run_qc(ref_path: str, tar_path: str, config: DubSyncConfig) -> None:
    """Measure the alignment drift profile between two media files (no re-render)."""
    import json
    import tempfile

    from .media_probe import MediaProbe
    from .verifier_engine import ClosedLoopVerifierEngine
    from rich.console import Console
    from rich.table import Table

    console = Console(highlight=False)
    probe = MediaProbe()

    with console.status("QC: extracting 48kHz PCM audio..."):
        tmp = tempfile.mkdtemp()
        ref_wav = os.path.join(tmp, "ref.wav")
        tar_wav = os.path.join(tmp, "tar.wav")
        ref_info = probe.probe(ref_path)
        tar_info = probe.probe(tar_path)
        probe.extract_pcm_wav(ref_path, ref_wav, sample_rate=config.audio_sample_rate,
                              stream_index=probe.select_audio_stream(ref_info, config.ref_lang))
        probe.extract_pcm_wav(tar_path, tar_wav, sample_rate=config.audio_sample_rate,
                              stream_index=probe.select_audio_stream(tar_info, config.tar_lang))

    with console.status("QC: measuring drift profile..."):
        verifier = ClosedLoopVerifierEngine(config)
        profile = verifier.measure_drift_profile(ref_wav, tar_wav)

    if not profile:
        console.print("[red]QC: no alignable windows found.[/red]")
        return

    offsets = [p["offset"] for p in profile]
    corrs = [p["correlation"] for p in profile]
    ref_times = [p["ref_time"] for p in profile]

    # Linear drift estimate: slope of offset vs ref_time (sec/sec).
    if len(profile) >= 2:
        import numpy as np
        slope, intercept = np.polyfit(ref_times, offsets, 1)
    else:
        slope, intercept = 0.0, offsets[0]

    table = Table(title="DubSync QC — Alignment Drift Profile", show_header=True)
    table.add_column("Ref Time", justify="right")
    table.add_column("Tar Time", justify="right")
    table.add_column("Offset (s)", justify="right")
    table.add_column("Corr", justify="right")
    for p in profile:
        table.add_row(f"{p['ref_time']:.1f}s", f"{p['tar_time']:.2f}s",
                      f"{p['offset']:+.3f}", f"{p['correlation']:.2f}")
    console.print(table)

    console.print()
    console.print(f"[bold]Global offset:[/bold] {intercept:+.3f}s   "
                  f"[bold]Drift slope:[/bold] {slope:+.5f}s/s "
                  f"({slope * 3600:+.1f}s/hour)   "
                  f"[bold]Mean corr:[/bold] {sum(corrs)/max(1, len(corrs)):.3f}")
    console.print(f"[dim]A near-zero drift slope means the pair is speed-locked; "
                  f"a large |slope| indicates a broadcast-speed (PAL/NTSC) mismatch.[/dim]")

    # Save JSON QC report alongside the reference.
    out_json = os.path.splitext(ref_path)[0] + "_qc_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "ref_file": os.path.basename(ref_path),
            "tar_file": os.path.basename(tar_path),
            "ref_duration_sec": ref_info.duration,
            "tar_duration_sec": tar_info.duration,
            "global_offset_sec": round(intercept, 4),
            "drift_slope_sec_per_sec": round(float(slope), 6),
            "mean_correlation": round(sum(corrs) / max(1, len(corrs)), 4),
            "probe_windows": profile,
        }, f, indent=2)
    console.print(f"\n[green]QC report saved:[/green] {out_json}")


def main():
    parser = argparse.ArgumentParser(
        description="DubSync Pro: Studio-Grade Cartoon & Anime Dub Audio Synchronizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("ref_video", nargs="?", help="Reference Master video (e.g. English WEB-DL / BluRay MKV)")
    parser.add_argument("foreign_video", nargs="?", help="Foreign Dub video (e.g. Arabic TV / Web MP4)")
    parser.add_argument("output_video", nargs="?", help="Output MKV file path")

    parser.add_argument("--preset", choices=["studio", "studio_ultra", "balanced", "fast"], default="studio",
                        help="Accuracy preset (studio=maximum precision, balanced=standard, fast=quick)")
    parser.add_argument("--matcher", "--matcher-mode", dest="matcher",
                        choices=["auto", "visual", "orb", "spectral", "vad"], default="auto",
                        help="Matching engine mode: auto (Tier 1->2->3), visual (Tier 1), orb (Tier 2), spectral (Tier 3), vad (Neural ML VAD)")
    parser.add_argument("--strategy", choices=["hybrid", "blocks", "dtw", "auto"], default="hybrid",
                        help="Audio synchronization strategy: hybrid (Multi-Modal Consensus + Closed-Loop Auto-Verification), blocks (Adaptive Macro-Blocks), dtw (Neural DTW)")
    parser.add_argument("--scene_threshold", type=float, default=0.22,
                        help="Visual scene-change detection sensitivity")
    parser.add_argument("--tar_lang", default="ara", help="ISO 639-2 code for foreign dub (default: ara)")
    parser.add_argument("--ref_lang", default="eng", help="ISO 639-2 code for reference audio (default: eng)")
    parser.add_argument("--fallback", "--fallback-mode", dest="fallback",
                        choices=["vocal_filtered", "full_reference", "silence"], default="vocal_filtered",
                        help="Audio fallback mode for scenes omitted in the foreign version")
    parser.add_argument("--report", dest="report", action="store_true", default=True,
                        help="Generate forensic diagnostic reports (JSON + Markdown)")
    parser.add_argument("--no-report", dest="report", action="store_false",
                        help="Disable forensic diagnostic report generation")
    parser.add_argument("--interactive", "-i", action="store_true", help="Force interactive TUI setup mode")
    parser.add_argument("--strict-speed", dest="strict_speed", action="store_true", default=True,
                        help="Lock continuous-act speed to broadcast standards (default on)")
    parser.add_argument("--no-strict-speed", dest="strict_speed", action="store_false",
                        help="Allow continuous-act speed to float between 0.90x and 1.10x")
    parser.add_argument("--qc", action="store_true",
                        help="Measure the alignment drift profile between ref and tar without re-rendering")

    args = parser.parse_args()

    config = DubSyncConfig(
        ref_lang=args.ref_lang,
        tar_lang=args.tar_lang,
        scene_threshold=args.scene_threshold,
        matcher_mode=args.matcher,
        generate_report=args.report
    )

    if args.preset in ("studio", "studio_ultra"):
        config.apply_preset(Preset.STUDIO_ULTRA)
    elif args.preset == "balanced":
        config.apply_preset(Preset.BALANCED)
    else:
        config.apply_preset(Preset.FAST)

    if args.fallback == "vocal_filtered":
        config.fallback_mode = FallbackMode.VOCAL_FILTERED
    elif args.fallback == "full_reference":
        config.fallback_mode = FallbackMode.FULL_REFERENCE
    else:
        config.fallback_mode = FallbackMode.SILENCE

    config.sync_strategy = args.strategy
    config.strict_speed = args.strict_speed

    tui = DubSyncTUI(config)

    # If arguments are missing or interactive flag is passed -> Launch TUI prompt
    if args.interactive or not args.ref_video or not args.foreign_video:
        ref_path, tar_path, out_path = tui.prompt_user_inputs()
    else:
        ref_path = args.ref_video
        tar_path = args.foreign_video
        out_path = args.output_video or (os.path.splitext(ref_path)[0] + "_Synced.mkv")

    # QC mode: measure drift only, no rendering / muxing.
    if args.qc:
        _run_qc(ref_path, tar_path, config)
        return

    # Execute pipeline
    pipeline = DubSyncPipeline(config)
    pipeline.execute(ref_path, tar_path, out_path)


if __name__ == "__main__":
    main()
