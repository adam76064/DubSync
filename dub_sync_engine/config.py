"""
Configuration dataclasses, presets, and constants for DubSync Pro.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, ClassVar


class FallbackMode(Enum):
    VOCAL_FILTERED = "vocal_filtered"  # English audio with speech attenuated (keeps music/SFX)
    FULL_REFERENCE = "full_reference"  # Raw English audio fallback
    SILENCE = "silence"                # Digital silence during cut duration
    ADAPTIVE = "adaptive"              # Probe the foreign audio; only bridge if it is missing


class Preset(Enum):
    STUDIO_ULTRA = "studio_ultra"      # Maximum accuracy: 3-frame burst, multi-hash, sub-ms acoustic correlation
    BALANCED = "balanced"              # Standard high-accuracy: safe-zone crop, visual + acoustic refine
    FAST = "fast"                      # Quick alignment: single-frame pHash, frame-level boundaries


@dataclass
class DubSyncConfig:
    # --- Language & Metadata ---
    ref_lang: str = "eng"
    tar_lang: str = "ara"
    dub_title: str = "Arabic (Studio Synced)"
    ref_title: str = "English (Original)"
    
    # --- Preset ---
    preset: Preset = Preset.STUDIO_ULTRA
    
    # --- Visual Anchor Detection Settings ---
    scene_threshold: float = 0.25      # Sensitivity for scene cut detection (0.15 - 0.40)
    max_hash_dist: int = 14            # Maximum perceptual hash distance threshold (default: 14)
    center_crop_ratio: float = 0.80    # Safe-zone center crop (0.80 = center 80%, ignoring logos/bars)
    use_temporal_burst: bool = True    # 3-frame burst extraction around cuts (t-dt, t, t+dt)
    burst_delta_frames: int = 1        # Delta frame distance for burst
    # --- Broadcast Speed ---
    fps_ratio: float = 0.0           # dt_tar/dt_ref estimated from the probed frame rates.
                                     # 0.0 = unknown; the consensus engine then prefers a
                                     # slope measured from the visual anchors.
    
    # --- Matching Mode & Fallbacks ---
    matcher_mode: str = "hybrid"       # 'hybrid' (Dual-Layer Cross-Validated: Visual Cuts + Music Transients + Neural VAD), 'audio' (Pure Audio), 'visual' (Legacy Visual), 'orb' (Tier 2 ORB)
    sync_strategy: str = "hybrid"      # 'hybrid' (Dual-Layer Consensus + Closed-Loop Verification), 'blocks' (Macro-Blocks), 'dtw' (Neural DTW)
    enable_auto_verification: bool = True  # Closed-loop self-auditing & false fallback healing
    use_ransac_block_clustering: bool = True
    discontinuity_threshold_sec: float = 0.40  # Threshold to identify real cuts/omissions vs frame jitter
    dtw_band_sec: float = 35.0         # Sakoe-Chiba constraint band for DTW
    dtw_node_interval_sec: float = 20.0# Interval between DTW retiming nodes
    
    # --- Acoustic Sub-Millisecond Refinement ---
    enable_acoustic_refine: bool = True# Sub-millisecond acoustic cross-correlation on 48kHz audio
    acoustic_window_ms: float = 350.0  # Search window around visual anchor in milliseconds (+/- 350ms)
    audio_sample_rate: int = 48000     # Internal high-fidelity processing sample rate
    speech_band_attenuation: bool = True# Attenuate 300Hz-3.4kHz dialogue to correlate pure M&E
    
    # --- Tier 0: Chromaprint Acoustic Bootstrap ---
    enable_chromaprint_bootstrap: bool = True
    chromaprint_bootstrap_max_offset_sec: float = 120.0
    
    # --- Sub-Segment Micro-DTW Word-Boundary Tightening ---
    enable_micro_dtw: bool = True      # Intra-segment VAD probability DTW for <10ms word sync
    micro_dtw_band_ms: float = 800.0   # Sakoe-Chiba constraint band (ms)
    
    # --- Audio Splicing & Retiming ---
    zero_crossing_snap: bool = True    # Snap segment cut boundaries to nearest zero amplitude
    zero_crossing_window_ms: float = 3.0# Search window for zero-crossing in ms
    crossfade_duration_ms: float = 10.0# Equal-power cosine crossfade duration in ms
    max_speed_deformation: float = 0.05# Maximum allowable speed stretch per continuous scene (5%)
    min_scene_duration_sec: float = 1.0# Minimum duration for independent scene segments
    fallback_mode: FallbackMode = FallbackMode.VOCAL_FILTERED
    
    # --- EBU R128 Loudness Normalization ---
    enable_loudness_norm: bool = True  # Two-pass broadcast standard loudness normalization
    loudness_target_lufs: float = -23.0# Target integrated loudness (EBU R128 = -23 LUFS)
    loudness_true_peak_dbtp: float = -1.0# Maximum true peak level in dBTP
    
    # --- Closed-Loop Verifier & Healing ---
    verifier_max_gap_sec: float = 90.0
    verifier_rms_snr_db: float = 6.0   # Gain-normalized SNR threshold above noise floor
    verifier_corr_min: float = 0.28    # Gain-normalized correlation threshold (see verifier docstring)
    # Windowed alignment probing used to report a *measured* sync error.
    # Tuned on the synthetic fixture against known-answer timelines: 3s/40 is
    # the largest window that still resolves a 6s mis-placed region, and the
    # smallest that stays clear of spurious matches (2s already degrades,
    # 1.5s collapses: a perfect timeline measures 124ms instead of ~14ms).
    verifier_probe_windows: int = 40       # Probe windows spread across the timeline
    verifier_probe_window_sec: float = 3.0
    verifier_pass_threshold_ms: float = 50.0  # A window "passes" under this residual error
    
    # --- Output Codec & Container ---
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    keep_subtitles: bool = True
    qc_report_html: bool = True
    
    # --- Reporting ---
    enable_reports: bool = True        # Write JSON + Markdown forensic reports
    
    # --- Debug & Performance ---
    num_threads: int = 0               # 0 = auto-detect all available CPU cores
    verbose: bool = False
    
    # Broadcast tempo ratios that a real dub transfer can take.
    # 1.000000 (1:1) | 24/25 (PAL slowdown) | 25/24 (PAL speedup)
    # 24/23.976 (NTSC pull-down) | 23.976/24 (film slowdown)
    BROADCAST_STANDARDS: ClassVar[tuple] = (
        1.000000, 24.0 / 25.0, 25.0 / 24.0, 24.0 / 23.976, 23.976 / 24.0,
    )

    def broadcast_snap(self, speed: float, tol: float = 0.006) -> float:
        """
        Snaps a measured speed ratio to the nearest broadcast standard, so that
        frame-rate conversions land on exact ratios instead of noisy estimates.

        Returns the input unchanged (rounded) when no standard is within `tol`.
        Callers are responsible for clamping to a playable range.
        """
        for std in self.BROADCAST_STANDARDS:
            if abs(speed - std) < tol:
                return round(std, 6)
        return round(speed, 6)

    def apply_preset(self, preset: Preset):
        self.preset = preset
        if preset == Preset.STUDIO_ULTRA:
            self.scene_threshold = 0.22
            self.center_crop_ratio = 0.80
            self.use_temporal_burst = True
            self.enable_acoustic_refine = True
            self.zero_crossing_snap = True
            self.crossfade_duration_ms = 12.0
            self.speech_band_attenuation = True
        elif preset == Preset.BALANCED:
            self.scene_threshold = 0.25
            self.center_crop_ratio = 0.85
            self.use_temporal_burst = False
            self.enable_acoustic_refine = True
            self.zero_crossing_snap = True
            self.crossfade_duration_ms = 8.0
            self.speech_band_attenuation = True
        elif preset == Preset.FAST:
            self.scene_threshold = 0.30
            self.center_crop_ratio = 1.0
            self.use_temporal_burst = False
            self.enable_acoustic_refine = False
            self.zero_crossing_snap = False
            self.crossfade_duration_ms = 5.0
            self.speech_band_attenuation = False
