"""
Command-Line Interface and interactive entrypoint for DubSync Pro.
"""

import sys
import argparse
import os

from .config import DubSyncConfig, Preset, FallbackMode
from .pipeline import DubSyncPipeline
from .tui import DubSyncTUI


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

    tui = DubSyncTUI(config)

    # If arguments are missing or interactive flag is passed -> Launch TUI prompt
    if args.interactive or not args.ref_video or not args.foreign_video:
        ref_path, tar_path, out_path = tui.prompt_user_inputs()
    else:
        ref_path = args.ref_video
        tar_path = args.foreign_video
        out_path = args.output_video or (os.path.splitext(ref_path)[0] + "_Synced.mkv")

    # Execute pipeline
    pipeline = DubSyncPipeline(config)
    pipeline.execute(ref_path, tar_path, out_path)


if __name__ == "__main__":
    main()
