"""
Configuration dataclasses, presets, and constants for DubSync Pro.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


# Standard broadcast playback speed ratios. A continuous act of a TV episode runs at
# one of these hardware clock speeds — never an arbitrary value (which causes drift).
BROADCAST_STANDARDS = (
    1.0,            # Film / native speed
    24.0 / 25.0,    # 0.960000  (PAL slowdown: 25fps -> 24fps)
    25.0 / 24.0,    # 1.041667  (PAL speedup: 24fps -> 25fps)
    24.0 / 23.976,  # 1.001001  (NTSC film pulldown)
    23.976 / 24.0,  # 0.999000  (film slowdown)
)


def snap_to_broadcast_speed(raw: float, tol: float = 0.004) -> float:
    """
    Snap a measured speed ratio to the nearest broadcast standard within ``tol``.
    Falls back to a clamped raw value when no standard matches (defensive — should
    be rare for well-formed continuous acts).

    The tolerance is tight enough to distinguish the "big" standards (1.0 film,
    0.96 PAL, 1.041667 PAL-speedup, which are ~4% apart) while folding the
    near-unity NTSC pulldown (1.001001) and film slowdown (0.999) into 1.0 —
    matching the historical engine behavior.
    """
    for std in BROADCAST_STANDARDS:
        if abs(raw - std) <= tol:
            return round(std, 6)
    return max(0.90, min(1.10, raw))


class FallbackMode(Enum):
    VOCAL_FILTERED = "vocal_filtered"  # English audio with speech attenuated (keeps music/SFX)
    FULL_REFERENCE = "full_reference"  # Raw English audio fallback
    SILENCE = "silence"                # Digital silence during cut duration


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
    # --- Matching Mode & Fallbacks ---
    matcher_mode: str = "auto"         # 'auto' (Tier 1->2->3 cascade), 'visual' (Tier 1), 'orb' (Tier 2), 'spectral' (Tier 3), 'vad' (ML VAD)
    sync_strategy: str = "hybrid"      # 'hybrid' (Multi-Modal Consensus + Closed-Loop Verification), 'blocks' (Macro-Blocks), 'dtw' (Neural DTW)
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
    
    # --- Audio Splicing & Retiming ---
    zero_crossing_snap: bool = True    # Snap segment cut boundaries to nearest zero amplitude
    zero_crossing_window_ms: float = 3.0# Search window for zero-crossing in ms
    crossfade_duration_ms: float = 10.0# Equal-power cosine crossfade duration in ms
    max_speed_deformation: float = 0.05# Maximum allowable speed stretch per continuous scene (5%)
    min_scene_duration_sec: float = 0.20 # Minimum scene span (s) to retain as a distinct EDL segment
    strict_speed: bool = True         # Lock continuous-act speed to broadcast standards (no floating)
    fallback_mode: FallbackMode = FallbackMode.VOCAL_FILTERED

    # --- Acoustic Gate & Micro-Chopping Guards ---
    min_acoustic_peak: float = 0.50   # Min normalized M&E correlation peak to accept an acoustic anchor
    min_vad_peak: float = 0.55        # Min normalized speech-probability correlation to accept a VAD anchor
    # Dense acoustic anchor discovery (subsync-style point cloud): short windows,
    # small hop, and a *low* acceptance threshold so weak-but-real candidates are
    # kept with soft scores (their correlation peak becomes the RANSAC weight).
    # Geometry (RANSAC + monotonic DP) — not a hard threshold — rejects the false ones.
    acoustic_anchor_window_sec: float = 5.0   # Ref window length for acoustic candidates
    acoustic_anchor_hop_sec: float = 1.0      # Hop between adjacent acoustic probe windows
    acoustic_anchor_min_peak: float = 0.20    # Low soft threshold (was hard min_acoustic_peak=0.50)
    vad_anchor_min_peak: float = 0.20         # Low soft threshold for VAD candidates
    min_dub_act_sec: float = 5.0      # A dub fragment shorter than this (between fallbacks) is ambient noise
    micro_fallback_merge_sec: float = 0.5  # Merge fallback gaps shorter than this into the neighboring dub
    acoustic_gate_window_sec: float = 4.0  # +/- window (s) around an acoustic anchor for visual confirmation
    acoustic_gate_offset_sec: float = 2.0  # Max allowed |delta offset| between visual & acoustic anchors

    # --- RANSAC Global Line Fit (subsync-style) ---
    ransac_inlier_tolerance_sec: float = 1.0  # Perpendicular distance (s) within which an anchor is an inlier
    ransac_slope_lo: float = 0.85   # Min allowed speed ratio for the global line
    ransac_slope_hi: float = 1.15   # Max allowed speed ratio for the global line
    ransac_min_inlier_ratio: float = 0.80  # Coverage target for a single global line (below -> recursive split)
    ransac_coverage_bucket_sec: float = 60.0  # Ref-time bucket size for the coverage constraint
    ransac_min_coverage_buckets: int = 3     # Min distinct ref-time buckets the inliers must span

    # --- Dense Sync-Path Estimator (similarity matrix + ridge extraction) ---
    # Measures the true (ref_time -> tar_time) path directly from the M&E
    # envelope instead of assuming a single continuous speed. Reveals cuts, tail
    # trims, and the real speed ratio — the "ground truth" for re-cut dubs.
    path_window_sec: float = 15.0        # Ref window length for each dense probe
    path_hop_sec: float = 2.0            # Hop between adjacent dense probes
    path_min_correlation: float = 0.4    # Drop dense points below this peak (silence/garbage)
    path_jump_threshold_sec: float = 0.8 # Offset jump that marks a cut boundary (offset step detection)
    path_min_segment_sec: float = 4.0    # Drop recovered segments shorter than this
    path_step_window_points: int = 2     # Before/after median window (points) for step detection
    
    # --- Output Codec & Container ---
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    keep_subtitles: bool = True
    qc_report_html: bool = True
    
    # --- Debug & Performance ---
    num_threads: int = 0               # 0 = auto-detect all available CPU cores
    verbose: bool = False
    generate_report: bool = True       # Write forensic JSON/Markdown diagnostic reports
    
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
