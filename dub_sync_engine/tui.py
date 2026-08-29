"""
Terminal User Interface (TUI) for DubSync Pro using Rich.
Provides styled dashboards, live multi-stage progress tracking, and interactive controls.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt

from .config import DubSyncConfig, Preset, FallbackMode
from .media_probe import MediaInfo
from .qc_report import QCReport

# Safe console initialization
console = Console(highlight=False)

BANNER = r"""[bold cyan]
  =============================================================
   DUBSYNC PRO : STUDIO-GRADE CARTOON & ANIME DUB SYNCHRONIZER
  =============================================================[/bold cyan]
  [bold yellow] * High-Precision Sub-Millisecond Multi-Stage Audio Sync *[/bold yellow] [dim]v2.0.0[/dim]
"""


class DubSyncTUI:
    """Rich-powered interactive user interface for DubSync."""

    def __init__(self, config: Optional[DubSyncConfig] = None):
        self.config = config or DubSyncConfig()
        self.console = console

    def print_banner(self):
        self.console.print(BANNER)

    def display_media_summary(self, ref_info: MediaInfo, tar_info: MediaInfo):
        """Displays a side-by-side comparison table of the reference and foreign media."""
        table = Table(title="[bold green]Media Inspection Summary[/bold green]", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="dim", width=22)
        table.add_column("Reference (Master English)", style="cyan", width=34)
        table.add_column("Target (Foreign Dub)", style="yellow", width=34)

        table.add_row("Filename", ref_info.filename, tar_info.filename)
        table.add_row("Duration", f"{ref_info.duration:.2f}s ({ref_info.duration/60:.2f}m)", f"{tar_info.duration:.2f}s ({tar_info.duration/60:.2f}m)")
        
        # Duration Delta
        delta = tar_info.duration - ref_info.duration
        delta_str = f"[bold red]{delta:+.2f}s (Missing Scenes)[/bold red]" if delta < -1.0 else f"{delta:+.2f}s"
        table.add_row("Duration Delta", "-", delta_str)

        # Video Specs
        ref_v = ref_info.primary_video
        tar_v = tar_info.primary_video
        v_ref_str = f"{ref_v.width}x{ref_v.height} @ {ref_v.fps:.2f}fps ({ref_v.codec})" if ref_v else "None"
        v_tar_str = f"{tar_v.width}x{tar_v.height} @ {tar_v.fps:.2f}fps ({tar_v.codec})" if tar_v else "None"
        table.add_row("Video Stream", v_ref_str, v_tar_str)

        # Audio Specs
        ref_a = ref_info.primary_audio
        tar_a = tar_info.primary_audio
        a_ref_str = f"{ref_a.codec} {ref_a.sample_rate}Hz ({ref_a.language})" if ref_a else "None"
        a_tar_str = f"{tar_a.codec} {tar_a.sample_rate}Hz ({tar_a.language})" if tar_a else "None"
        table.add_row("Audio Stream", a_ref_str, a_tar_str)

        self.console.print(table)

    def display_qc_summary(self, report: QCReport):
        """Displays a summary panel of the final synchronization results."""
        summary_text = Text()
        summary_text.append("[OK] Synchronization Completed Successfully!\n\n", style="bold green")
        summary_text.append(f"  * Output MKV:       {report.output_filename}\n", style="bold white")
        summary_text.append(f"  * Processing Time:  {report.processing_time_sec:.2f}s ({report.processing_time_sec/60:.2f}m)\n", style="cyan")
        summary_text.append(f"  * Matched Anchors:  {report.matched_anchors_count} scene keyframes\n", style="yellow")
        summary_text.append(f"  * Timeline EDL:     {report.edl_dub_segments} Dub Scenes + {report.edl_fallback_segments} Bridged Cuts\n", style="magenta")
        summary_text.append(f"  * Avg Confidence:   {report.average_confidence * 100:.1f}%\n", style="bold green")

        if report.omitted_scenes:
            summary_text.append("\n[bold red]Detected Omitted/Censored Gaps (Bridged with Ambient M&E):[/bold red]\n", style="bold red")
            for gap in report.omitted_scenes:
                summary_text.append(f"   -> [{gap['start_time']}s -> {gap['end_time']}s] Duration: {gap['duration']}s\n", style="dim")

        panel = Panel(summary_text, title="[bold cyan]DubSync Quality Control Dashboard[/bold cyan]", border_style="green", padding=(1, 2))
        self.console.print(panel)

    def prompt_user_inputs(self) -> Tuple[str, str, str]:
        """Interactive prompt when run without CLI arguments."""
        self.print_banner()
        self.console.print("[bold yellow]Interactive DubSync Setup[/bold yellow]\n")

        ref = Prompt.ask("Enter [cyan]Reference Video[/cyan] (HQ Master English)").strip('"')
        tar = Prompt.ask("Enter [yellow]Foreign Dub Video[/yellow] (Arabic / Other)").strip('"')
        
        default_out = os.path.splitext(ref)[0] + "_Synced.mkv"
        out = Prompt.ask("Enter [green]Output Filename[/green]", default=default_out).strip('"')

        # Preset selection
        preset_choice = Prompt.ask(
            "Select Synchronization Preset",
            choices=["studio", "balanced", "fast"],
            default="studio"
        )
        if preset_choice == "studio":
            self.config.apply_preset(Preset.STUDIO_ULTRA)
        elif preset_choice == "balanced":
            self.config.apply_preset(Preset.BALANCED)
        else:
            self.config.apply_preset(Preset.FAST)

        return ref, tar, out
