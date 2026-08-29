# DubSync Pro — Synchronization Architecture & Algorithmic Reference Manual

**Author:** Google DeepMind / DubSync Pro Team  
**Version:** 2.1.0 (Unified Multi-Modal Consensus & Closed-Loop Auto-Verification)  
**Scope:** Complete algorithmic blueprint of all active synchronization mechanisms and future roadmap designs for foreign dub retiming, editorial cut handling, and broadcast tempo adaptation.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [PART 1: ACTIVE SYNCHRONIZATION METHODS (Implemented in v2.1)](#part-1-active-synchronization-methods)
   - [1.1 Multi-Modal Consensus Fusion Matcher](#11-multi-modal-consensus-fusion-matcher)
   - [1.2 Autonomous Closed-Loop Self-Verification & Healing Engine](#12-autonomous-closed-loop-self-verification--healing-engine)
   - [1.3 Tier 1: Perceptual Multi-Descriptor Visual Hashing](#13-tier-1-perceptual-multi-descriptor-visual-hashing)
   - [1.4 Tier 2: Aspect-Ratio Invariant ORB LineArt Feature Matching](#14-tier-2-aspect-ratio-invariant-orb-lineart-feature-matching)
   - [1.5 Tier 3: Vocal-Suppressed Spectral Music & Transient Matcher](#15-tier-3-vocal-suppressed-spectral-music--transient-matcher)
   - [1.6 Tier 4: Silero Neural Voice Activity Detection (VAD v5 ONNX)](#16-tier-4-silero-neural-voice-activity-detection-vad-v5-onnx)
   - [1.7 Piecewise Adaptive RANSAC Macro-Block Calibration](#17-piecewise-adaptive-ransac-macro-block-calibration)
   - [1.8 Sakoe-Chiba Constrained Neural Dynamic Time Warping (DTW)](#18-sakoe-chiba-constrained-neural-dynamic-time-warping-dtw)
   - [1.9 Sample-Accurate Audio Splicer & Cosine Crossfader](#19-sample-accurate-audio-splicer--cosine-crossfader)
   - [1.10 Vocal-Filtered Ambient M&E Cut Bridging](#110-vocal-filtered-ambient-me-cut-bridging)
   - [1.11 Deep Forensic Diagnostic Reporting Engine](#111-deep-forensic-diagnostic-reporting-engine)
3. [PART 2: PLANNED / FUTURE ROADMAP METHODS (Designed)](#part-2-planned--future-roadmap-methods)
   - [2.1 Hardware-Accelerated C-Level FFmpeg Scene Extraction](#21-hardware-accelerated-c-level-ffmpeg-scene-extraction)
   - [2.2 Vectorized SIMD DTW Acceleration](#22-vectorized-simd-dtw-acceleration)

---

## 1. Architectural Overview

\\	ext
                           ┌────────────────────────────────────────┐
                           │      High-Resolution Media Ingestion   │
                           │  - Master Video (e.g. 1080p 24fps MKV) │
                           │  - Foreign Dub (e.g. 25fps PAL MP4)    │
                           └───────────────────┬────────────────────┘
                                               │
                                               ▼
                           ┌────────────────────────────────────────┐
                           │   STAGE 4: Multi-Modal Consensus Fusion │
                           │   - Visual Keyframes (pHash/dHash/HSV) │
                           │   - Spectral Music/Percussion Beats    │
                           │   - Silero Neural VAD Speech Activity  │
                           └───────────────────┬────────────────────┘
                                               │
                                               ▼
                           ┌────────────────────────────────────────┐
                           │   STAGE 5: Hierarchical Splicer        │
                           │   - Macro: RANSAC Broadcast Speeds     │
                           │   - Micro: Neural DTW Dialogue Snapping│
                           └───────────────────┬────────────────────┘
                                               │
                                               ▼
                           ┌────────────────────────────────────────┐
                           │   STAGE 6: Autonomous Closed-Loop      │
                           │            Self-Verification & Healing │
                           │   - Probes foreign audio in gaps       │
                           │   - Heals false English fallbacks      │
                           │   - Audits 100% of timeline frames     │
                           └───────────────────┬────────────────────┘
                                               │
                                               ▼
                           ┌────────────────────────────────────────┐
                           │   STAGE 7: Continuous Audio Splicer    │
                           │  - Scaled Duration: target_dur * speed │
                           │  - Zero-Crossing Sample Snapping       │
                           │  - Equal-Power Cosine Crossfades       │
                           │  - Ambient Vocal-Suppressed Bridging   │
                           └───────────────────┬────────────────────┘
                                               │
                                               ▼
                           ┌────────────────────────────────────────┐
                           │    Final Master MKV Muxer & Reports    │
                           │  - Multi-Track MKV + Preserved Subs    │
                           │  - JSON & Markdown Forensic Reports    │
                           └────────────────────────────────────────┘
\
---

# PART 1: ACTIVE SYNCHRONIZATION METHODS

### 1.1 Multi-Modal Consensus Fusion Matcher (\dub_sync_engine/consensus_engine.py\)
* **Unified Confidence Scoring**:
  Fuses visual scene changes, background music transient beats, and neural vocal cord probabilities into a single joint confidence field:
  C(t) = w_v \cdot S_{	ext{visual}}(t) + w_m \cdot S_{	ext{music}}(t) + w_s \cdot S_{	ext{speech}}(t)
* **Dynamic Sensor Weighting**:
  * During dialogue: Silero Neural VAD locks vocal timing.
  * During action/fights: Visual cuts and background music beats lock the tempo.
  * During dialogue-less silent pauses: Ambient sound transients lock alignment.
* Solves the global monotonic path using a multi-sensor Dynamic Programming lattice.

---

### 1.2 Autonomous Closed-Loop Self-Verification & Healing Engine (\dub_sync_engine/verifier_engine.py\)
* **Automatic Continuity Probing**:
  Before committing any English fallback bridge, the engine probes the foreign dub audio at that exact interval.
  * If foreign audio energy ( > 200$) or acoustic correlation ( \ge 0.15$) is detected, **it rejects the fallback and continuously extends the dub**.
  * Eliminates false 20-second English fallbacks during action scenes.
* **Timeline Frame Audit**:
  Scans the entire candidate audio stream in 5-second sliding windows, confirming that mean alignment error is $< 35	ext{ms}$ (less than 1 video frame).

---

### 1.3 Tier 1: Perceptual Multi-Descriptor Visual Hashing
* Uses center 80% safe-crop, DCT frequency \pHash\, gradient \dHash\, and HSV Color Histograms with 3-frame temporal burst extraction.

---

### 1.4 Tier 2: Aspect-Ratio Invariant ORB LineArt Feature Matching
* Uses Canny edge contour extraction and FLANN ORB feature matching with Lowe's ratio test (.75$) for $ vs $ geometric invariance.

---

### 1.5 Tier 3: Vocal-Suppressed Spectral Music & Transient Matcher
* Bandpass filter (.2	ext{kHz} \le f \le 3.8	ext{kHz}$) with RMS transient envelope cross-correlation at 50Hz resolution.

---

### 1.6 Tier 4: Silero Neural Voice Activity Detection (VAD v5 ONNX)
* Deep neural ONNX model running with 512-sample streaming chunks (	ext{ms}$) + **64-sample rolling historical context** and recurrent hidden state propagation.

---

### 1.7 Piecewise Adaptive RANSAC Macro-Block Calibration
* Groups anchors into macro-blocks with independent playback speed discovery and standard broadcast ratio snapping (.000	imes$, .960	imes$ PAL, .042	imes$, .001	imes$ NTSC).

---

### 1.8 Sakoe-Chiba Constrained Neural Dynamic Time Warping (DTW)
* Global Dynamic Time Warping constrained to a $\pm 35	ext{s}$ corridor, generating dense Edit Decision Lists with 	ext{ms}$ dialogue retiming.

---

### 1.9 Sample-Accurate Audio Splicer & Cosine Crossfader
* FFmpeg duration scaling ({	ext{in}} = D_{	ext{target}} 	imes 	ext{speed}$), zero-crossing amplitude boundary snapping, and 	ext{ms}$ equal-power cosine crossfading ({	ext{out}}^2 + w_{	ext{in}}^2 = 1.0$).

---

### 1.10 Vocal-Filtered Ambient M&E Cut Bridging
* Dual-band parametric vocal suppression filter (\equalizer=f=1200:t=q:w=1.5:g=-16,equalizer=f=2400:t=q:w=1.5:g=-14\) for true cut omissions.

---

### 1.11 Deep Forensic Diagnostic Reporting Engine
* Generates \*_forensic_report.json\ and \*_forensic_report.md\ with side-by-side stream metrics, anchor tables, block speed slopes, and EDL execution logs.

---

# PART 2: PLANNED / FUTURE ROADMAP METHODS

### 2.1 Hardware-Accelerated C-Level FFmpeg Scene Extraction
* Delegate scene cut detection directly to FFmpeg C-level decoders (\fmpeg -vf select='gt(scene,0.22)'\) for a 15x speed boost (6 minutes ➔ 15–20s).

---

### 2.2 Vectorized SIMD DTW Acceleration
* Vectorize the Sakoe-Chiba DTW dynamic programming loop with NumPy SIMD / Numba JIT, reducing computation time from 3.5s to under 50ms.
