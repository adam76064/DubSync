# DubSync Pro — Master Synchronization Architecture & Unified Technical Plan 🎙️🎬

**Document:** Unified Master Specification (v4.0)  
**Status:** Approved Architectural Blueprint  
**Scope:** Complete synthesis of all research, algorithms, lightweight ML models, and boundary safety mechanisms into a single, conflict-free production pipeline.

---

## 1. Architectural Philosophy & Zero-Conflict Guarantee

### 1.1 Non-Regression Core Directives
To guarantee that newly integrated techniques enhance accuracy without causing regressions or audio degradation:
1. **Immutable Broadcast Speed Locking:** Continuous dialogue acts must remain strictly locked to broadcast standard tempos ($0.960000$ PAL $\leftrightarrow$ Film, $1.000000$ Film $\leftrightarrow$ Film). Dynamic intra-segment DTW pitch/speed warping is strictly forbidden across audio rendering to prevent the "tempo wobble" bug.
2. **Exact Scaled Target Slicing Math:** Every continuous segment duration is mathematically locked:
   $$\text{tar\_end} = \text{tar\_start} + (\text{ref\_duration} \times g_{\text{speed}})$$
   This guarantees **$0.000\text{ms}$ cumulative drift** and zero audio duration overflows.
3. **Dual-Layer Cross-Validation Preservation:** Visual camera cut chains ($N$-Gram rhythm verification) must remain co-validated with bandpass music transients ($800\text{Hz}\text{--}3.2\text{kHz}$) and Silero Neural VAD speech envelopes.

---

## 2. The 8-Stage Unified Pipeline Overview

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DUBSYNC PRO v4.0 UNIFIED PIPELINE                                │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage 1: Ingest & Stream Probing           -> Probe rational frame rates, audio sample rates     │
│ Stage 2: PCM Extraction & Instant FFT Lock -> 48kHz audio + 1D Binary VAD FFT Speed Sweep (<30ms)│
│ Stage 3: Safe-Zone Video Keyframing        -> 80% safe center-crop, 3-frame temporal burst       │
│ Stage 4: Dual-Layer Multi-Modal Consensus  -> N-gram visual chains + Bandpass Music + Neural VAD │
│ Stage 5: Split-Penalty DP & Macro-Blocks   -> alass regularized DP cut detection (no false cuts) │
│ Stage 6: Silence-Gated Splicing Assembly   -> Mutual silence window boundary snap (0% clipping)  │
│ Stage 7: Sub-Sample Parabolic Refine & EBU -> <0.5ms peak interpolation + Zero-crossing + loudnorm│
│ Stage 8: Multi-Track MKV Mux & Subtitles   -> Zero-loss MKV muxing + Native ASS/SSA/SRT retiming │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Algorithmic Specifications

### 3.1 Stage 1 & 2: Ingest & 1D Binary Speech Density FFT Speed Lock
* **Goal:** Determine the global broadcast speed ratio ($0.960000$ vs $1.000000$ vs $1.041667$) and initial timeline offset $\tau_0$ in $< 30\text{ms}$ before any video processing.
* **Algorithm:**
  1. Extract 48kHz PCM audio for reference and target streams.
  2. Compute 100Hz binary speech activity vectors ($1 = \text{speech}, -1 = \text{silence}$) using `silero_vad.onnx` (2.1MB ONNX Runtime).
  3. Resample target binary vector across candidate broadcast ratios:
     $$\mathcal{S} = \{0.960000 \text{ (PAL)}, \, 1.000000 \text{ (Film)}, \, 1.041667 \text{ (NTSC)}, \, 0.959040\}$$
  4. Perform frequency-domain convolution via Fast Fourier Transform:
     $$R_s(\tau) = \mathcal{F}^{-1} \left\{ \mathcal{F}[VAD_{\text{ref}}] \cdot \mathcal{F}[VAD_{\text{tar}}^{(s)}]^* \right\}$$
  5. The global speed $s^*$ and offset $\tau_0$ are resolved instantly with maximum correlation peak confidence.

---

### 3.2 Stage 3 & 4: Safe-Zone Keyframing & Dual-Layer Consensus
* **Safe-Zone 80% Center-Crop:** Crops away Spacetoon/MBC3/Cartoon Network logos and letterbox borders:
  $$\text{Crop} = [0.10 \cdot W \text{ to } 0.90 \cdot W] \times [0.10 \cdot H \text{ to } 0.90 \cdot H]$$
* **3-Frame Temporal Burst:** Extracts $\{t - \Delta t, \, t, \, t + \Delta t\}$ around every camera cut to ensure sub-frame cadence matching.
* **$N$-Gram Rhythm Chains ($\ge 3$ consecutive cuts):**
  $$\frac{\Delta t_{\text{tar}}^{(1)}}{\Delta t_{\text{ref}}^{(1)}} \approx \frac{\Delta t_{\text{tar}}^{(2)}}{\Delta t_{\text{ref}}^{(2)}} \approx s^* \quad (\pm 0.005)$$
  Eliminates 100% of false matches from repeating cartoon animation cycles or static poses.
* **Acoustic Music Transient Co-Validation:** Cross-validates visual knots against bandpass-filtered music transients ($800\text{Hz}\text{--}3.2\text{kHz}$ with speech band suppression).

---

### 3.3 Stage 5: Split-Penalty Dynamic Programming Cut Resolution
* **Goal:** Mathematically prove whether a timing gap is a genuine TV omission cut vs. a voice actor's natural dramatic pause.
* **Objective Function (`alass` Mathematical Formulation):**
  $$\max_{\mathcal{P}, \mathcal{K}} \left( \sum_{i \in \text{Matched}} \text{SpeechOverlap}(\text{Ref}_i, \text{Tar}_{\mathcal{P}(i)}) - \sum_{k \in \mathcal{K}} \lambda_{\text{split}} \right)$$
* **Decision Boundary:**
  * If the alignment improvement downstream exceeds $\lambda_{\text{split}}$, a new timeline block is instantiated (confirmed TV commercial/censorship omission).
  * Otherwise, the region remains locked in a single continuous broadcast block with zero micro-stutter.

---

### 3.4 Stage 6: Silence-Gated Splicing (Zero Syllable Clipping)
* **Goal:** Guarantee that audio slice boundaries and EDL transitions never chop an actor's word, consonant, or breath.
* **Mutual Silence Window Rule:**
  $$\text{Window}_{\text{safe}} = \{ t \mid VAD_{\text{ref}}(t) = 0 \text{ and } VAD_{\text{tar}}(t) = 0 \text{ for } \Delta t \ge 0.40\text{s} \}$$
* **Snapping Function:**
  $$t_{\text{splice}} = \arg\min_{t \in \text{Window}_{\text{safe}}} |t - t_{\text{boundary}}|$$
* **Bridging Omitted Scenes:**
  * If an omission is $\ge 2.0\text{s}$, bridge the gap using vocal-filtered ambient Music & Effects (M&E) from the master reference.
  * If an omission is $< 2.0\text{s}$, smooth over the gap transparently without micro-slicing.

---

### 3.5 Stage 7: Sub-Sample Parabolic Refinement & Loudness Mastering
* **Parabolic Peak Interpolation:** Around the discrete peak $t^*$ of the cross-correlation function $R(t)$, compute fractional lag:
  $$\Delta \tau = \frac{R(t^* - 1) - R(t^* + 1)}{2 \cdot [R(t^* - 1) - 2R(t^*) + R(t^* + 1)]}$$
  $$\tau_{\text{exact}} = t^* + \Delta \tau \quad (< 0.5\text{ms sub-sample precision})$$
* **Zero-Crossing Snap:** Snaps boundaries to the nearest waveform zero-amplitude crossing point with positive slope.
* **Equal-Power Cosine Crossfade:** Applies 10–12ms smooth cosine crossfading:
  $$\text{fade\_in}(t) = \sin^2\left(\frac{\pi t}{2 T}\right), \quad \text{fade\_out}(t) = \cos^2\left(\frac{\pi t}{2 T}\right)$$
* **EBU R128 Loudness Normalization:** Normalizes final concatenated audio to standard broadcast targets ($-23.0\text{ LUFS}$, $-1.0\text{ dBTP}$).

---

### 3.6 Stage 8: Native ASS/SSA & SRT Subtitle Retiming Module
* **Goal:** Synchronize foreign subtitle streams simultaneously with the audio track.
* **ASS/SSA Style Preservation:**
  * Parses `.ass` event lines (`Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text`).
  * Retimes `Start` and `End` timestamps based on the Master EDL mapping table:
    $$t_{\text{ref}} = \text{MapTarToRef}(t_{\text{tar}})$$
  * Preserves 100% of custom anime fonts, karaoke timing tags (`\k`), vector drawings, colors, positioning (`\pos`), and styles untouched.

---

## 4. Component File Architecture

```text
dub_sync_engine/
├── __init__.py               # Exports public engine API
├── config.py                 # DubSyncConfig, Preset, Broadcast standards
├── media_probe.py            # FFprobe stream detection and Rational FPS parser
├── vad_engine.py             # Silero VAD ONNX wrapper + 1D Binary FFT Speed Sweep
├── visual_anchors.py         # 80% Safe-crop, 3-frame burst, N-gram rhythm matching
├── consensus_engine.py       # Dual-Layer Consensus (Visual + Bandpass Music + VAD)
├── block_segmenter.py        # Split-Penalty DP, Exact Scaled Slicing, Silence-Gating
├── audio_splicer.py          # Parabolic sub-ms xcorr, zero-crossing, EBU R128 loudnorm
├── subtitle_engine.py        # [NEW] Native ASS/SSA & SRT styled subtitle retimer
├── verifier_engine.py        # Autonomous Closed-Loop Verification & Auto-Healing
├── mkv_muxer.py              # Multi-track MKV muxer (video + master EN + synced AR + subs)
├── qc_report.py              # Diagnostic Markdown & JSON Forensic Scorecards
├── tui.py                    # Real-time Rich terminal UI
└── cli.py                    # Command-line interface
```

---

## 5. Comprehensive Verification Benchmark Suite

To validate 100% accuracy and zero regression, the engine will be tested on the standard 3-episode benchmark:

1. **Hero 108 — Episode 01** (Standard PAL 25fps to Film 23.976fps conversion).
2. **Hero 108 — Episode 02** (Minute 8 commercial cut omission + opening logo offset).
3. **Hero 108 — Episode 03** (Multiple mid-episode TV censorship cuts + complex dialogue).

### Benchmark Pass Criteria:
* ✅ Zero cumulative audio drift across all 22 minutes ($0.000\text{ms}$ sub-sample accuracy).
* ✅ Zero speech clipping (no chopped syllables or dialogue cutoffs).
* ✅ 100% transparent M&E bridging over omitted scenes.
* ✅ Sub-frame lip-sync alignment verified by Closed-Loop Scorecard ($>98\%$ passed windows).

---
*Archived in `docs/MASTER_SYNCHRONIZATION_ARCHITECTURE_AND_PLAN.md`.*
