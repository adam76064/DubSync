"""
DubSync Pro: Studio-Grade Dub Audio Synchronization Engine
===========================================================
High-precision, accuracy-first tool for aligning foreign cartoon and anime dubs
with pristine reference video releases, handling cuts, omissions, watermarks,
aspect ratio differences, and frame rate quantization jitter.
"""

__version__ = "2.0.0"
__author__ = "Antigravity & DubSync Contributors"

from .config import DubSyncConfig, Preset
from .media_probe import MediaProbe, MediaInfo
from .visual_anchors import VisualAnchorEngine, VisualAnchor
from .block_segmenter import BlockSegmenterEngine, ContinuousBlock
from .orb_matcher import ORBMatcherEngine
from .spectral_fingerprint import SpectralFingerprintEngine
from .vad_engine import SileroVADEngine
from .consensus_engine import MultiModalConsensusEngine
from .verifier_engine import ClosedLoopVerifierEngine, VerificationAudit
from .chromaprint_bootstrap import ChromaprintBootstrap, GlobalOffsetEstimate
from .micro_dtw import MicroDTWEngine
from .acoustic_refine import AcousticRefineEngine
from .audio_splicer import AudioSplicerEngine, SegmentEDL
from .mkv_muxer import MKVMuxer
from .qc_report import QCReportGenerator, QCReport, ForensicReportGenerator, ForensicReport
from .tui import DubSyncTUI

__all__ = [
    "DubSyncConfig",
    "Preset",
    "MediaProbe",
    "MediaInfo",
    "VisualAnchorEngine",
    "VisualAnchor",
    "BlockSegmenterEngine",
    "ContinuousBlock",
    "ORBMatcherEngine",
    "SpectralFingerprintEngine",
    "SileroVADEngine",
    "MultiModalConsensusEngine",
    "ClosedLoopVerifierEngine",
    "VerificationAudit",
    "ChromaprintBootstrap",
    "GlobalOffsetEstimate",
    "MicroDTWEngine",
    "AcousticRefineEngine",
    "AudioSplicerEngine",
    "SegmentEDL",
    "MKVMuxer",
    "QCReportGenerator",
    "QCReport",
    "ForensicReportGenerator",
    "ForensicReport",
    "DubSyncTUI",
]
