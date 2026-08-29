# DubSync Pro 🎙️🎬

**DubSync Pro** is a studio-grade, multi-modal automated audio synchronization engine designed to align foreign dubbed audio tracks (e.g., Arabic, Spanish, French, Japanese) with high-definition master reference videos (AMZN, Blu-ray, WEB-DL) with **sub-frame accuracy (0.000ms drift)**.

It solves complex synchronization challenges such as frame rate conversions (25fps PAL to 23.976/24fps Film), foreign TV commercial/censorship cut omissions, opening logo deltas, and cartoon background repetition.

---

## 🌟 Key Features

* **Dual-Layer Cross-Validation Architecture**:
  * **Layer 1 (Sequence-Verified Visual Cuts)**: Analyzes chains of consecutive camera cuts (N-gram rhythm verification) to find exact frame boundaries without being deceived by cartoon animation loops.
  * **Layer 2 (Acoustic Bandpass Transients & Silero VAD)**: Tracks 800Hz–3.5kHz background music beats and neural vocal cord probabilities to establish an immutable timeline offset spine.
* **Exact Scaled Target Slicing Math**:
  * Mathematically locks continuous dialogue acts to broadcast tempo standards (0.960000x PAL, 1.000000x Film).
  * Completely eliminates audio duration overflows, cumulative drift, and dialogue truncation.
* **Intelligent TV Cut & Censorship Bridging**:
  * Automatically detects omitted scenes and bridges them with vocal-filtered ambient Music & Effects (M&E) fallback audio from the master reference.
* **Closed-Loop Autonomous Self-Auditing & Healing**:
  * Probes sliding correlation windows across the final rendered MKV to verify acoustic alignment and heal any false fallbacks.
* **Rich TUI & Comprehensive Forensic Diagnostics**:
  * Real-time Rich terminal UI with step-by-step telemetry, interactive inspection modals, and automatic markdown/JSON diagnostic reports.

---

## 🏗️ Architecture & Pipeline

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               DUBSYNC PRO PIPELINE                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Media Ingestion & Audio Probe       -> Extract 48kHz internal PCM streams           │
│ 2. Safe-Zone Center-Crop Extraction    -> Scene cut keyframes (80% safe zone)          │
│ 3. Dual-Layer Multi-Modal Consensus    -> Visual chains + Music transients + VAD       │
│ 4. Macro-Block Clustering & Speed Cal  -> Independent broadcast slope estimation       │
│ 5. Adaptive EDL Generation             -> Exact scaled slicing & cut boundary assembly │
│ 6. Closed-Loop Auto-Audit Scorecard    -> Probed sub-frame verification & healing      │
│ 7. High-Fidelity Splicing & MKV Muxing -> Zero-crossing crossfade & mkvmerge/FFmpeg    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

For complete technical and mathematical explanations of all 11 algorithms, read [docs/SYNCHRONIZATION_METHODS.md](docs/SYNCHRONIZATION_METHODS.md).

---

## 📦 Prerequisites & Installation

### 1. External Tools
* **Python 3.9+**
* **FFmpeg** (must be installed and accessible in your system `PATH`)

### 2. Install Dependencies
```bash
git clone https://github.com/adam76064/DubSync.git
cd DubSync
pip install -r requirements.txt
```

---

## 🚀 Quick Start & CLI Usage

### Basic CLI Command
```bash
python -m dub_sync_engine.cli --ref "path/to/Master.Video.1080p.mkv" --tar "path/to/Foreign_Dub_Episode.mp4" --out "path/to/Master_Synced.mkv"
```

### Interactive Wrapper
```bash
python run_dub_sync.py
```

### Common Flags & Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--ref` | Path to reference master video/audio file | *(Required)* |
| `--tar` | Path to foreign dub target video/audio file | *(Required)* |
| `--out` | Output path for the synchronized MKV | `[ref_name]_Synced.mkv` |
| `--matcher-mode` | Matching mode: `hybrid`, `audio`, `visual`, `orb`, `vad` | `hybrid` |
| `--strategy` | Sync strategy: `hybrid`, `blocks`, `dtw` | `hybrid` |
| `--preset` | Tuning preset: `studio_ultra`, `balanced`, `fast` | `studio_ultra` |
| `--fallback-mode` | Fallback treatment: `vocal_filtered`, `full_reference`, `silence` | `vocal_filtered` |
| `--report` | Generate diagnostic Markdown and JSON forensic reports | `True` |

---

## 📄 Forensic Diagnostic Reports

After each synchronization run, DubSync Pro generates a detailed **Forensic Diagnostic Report** (`.md` and `.json`), detailing:
* Video frame rate conversions and audio internal sample rates
* Complete chronological Anchor Registry with confidence and hash distance
* Macro-block cluster breakdown and independent speed slopes
* Detected omitted cuts and M&E bridge durations
* Closed-loop verification audit scorecard (Mean alignment error, peak error, coverage)

---

## 📜 License

This project is open source and available under the MIT License.
