"""
Configuration dataclasses, presets, and constants for DubSync Pro.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


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
    fallback_mode: FallbackMode = FallbackMode.VOCAL_FILTERED
    
    # --- Output Codec & Container ---
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    keep_subtitles: bool = True
    qc_report_html: bool = True
    
    # --- Debug & Performance ---
    num_threads: int = 0               # 0 = auto-detect all available CPU cores
    verbose: bool = False
    
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
