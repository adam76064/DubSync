# DubSync Pro — Engineering Review

**Date:** 2026-08-29 · **Commit reviewed:** `f84fdcc` ("feat: initial commit of DubSync Pro v3.5")
**Scope:** full read of `dub_sync_engine/` (17 modules, ~3.5 kLOC) + `tools/`, `docs/`, `README.md`, packaging.
**Method:** static review **plus** dynamic testing — the pipeline was executed end-to-end on a
synthetic reference/dub pair whose ground-truth timing map is known exactly, and the rendered
output was measured independently of the tool's own reporting.

This review is **read-only**: no engine behaviour was changed. Everything below is reproducible
with the harness in [`tests/`](tests/README.md).

---

## 1. Verdict

The architecture is genuinely thoughtful — the broadcast-slope estimation, the monotonic DP
lattice, and the Tier-1 visual matcher all work, and the EDL math is correct. But the project
is **not production-ready**, for one reason that outweighs everything else:

> **The tool cannot tell you whether it succeeded.** Its closed-loop audit reports a hard-coded
> `mean error 24.5 ms / 99.2 % verified` on every single run — including runs where the measured
> error was **7.8 seconds** and **0 %** of the timeline was in sync.

Combined with a broken default code path, that means the advertised "studio-grade, sub-frame
accurate, self-auditing" behaviour is not what a user gets today.

### Severity summary

| # | Finding | Severity |
|:--|:--------|:---------|
| F1 | Closed-loop audit metrics are hard-coded constants, not measurements | 🔴 Critical |
| F2 | Default (`auto`/`hybrid`) path collapses to 1 anchor → seconds-scale error | 🔴 Critical |
| F3 | Omitted-scene gaps are always placed at the *end* of an anchor interval | 🟠 High |
| F4 | `AcousticRefineEngine` / `SpectralFingerprintEngine` are constructed but never called | 🟠 High |
| F5 | README documents a CLI that does not exist; every documented command fails | 🟠 High |
| F6 | Zero-crossing snapping & cosine crossfades are dead code; segments are hard-cut | 🟡 Medium |
| F7 | `AudioSplicerEngine.build_edl` references a non-existent config field | 🟡 Medium |
| F8 | No tests, no CI, no packaging, no LICENSE; five different version strings | 🟡 Medium |
| F9 | Import-time FFmpeg dependency; ~2 GB of scratch left behind per feature | 🟡 Medium |
| F10 | Dead config knobs & brittle ffmpeg-stderr scraping | 🔵 Low |
| F11 | 7.3 MB vendored third-party `audalign` + unreferenced 514-line legacy duplicate | 🔵 Low |
| F12 | Duration not bit-exact; per-block speed calibration computed but never applied | 🔵 Low |
| F13 | Bare `except Exception: pass` hides whole layers failing | 🔵 Low |

---

## 2. How the measurements were taken

A 60 s / 24 fps reference was synthesised with scene cuts at 12 s, 24 s, 36 s, 48 s and a
deterministic M&E + "speech" audio bed. The dub was derived from it by **removing ref[24 s, 30 s]**
(a censorship cut) and **slowing the result to 0.96×**, letterboxed to a different scale.

Ground truth: `tar_time = (ref_time − 6 s if ref_time ≥ 30 s else ref_time) / 0.96`

The output MKV's dub track (stream `0:a:1`) was then cross-correlated against the original
reference audio in 4 s windows, ±8 s search. Lag `0 ms` = perfectly in sync. The vocal-filtered
fallback leaves a hard spectral notch at 1200 Hz / 2400 Hz, which was used to independently
classify each region as *real dub* or *English M&E bridge*.

### Measured results

| Run | Strategy | Duration error | Mean abs. error | Max error | Windows within ±120 ms | Gap reported (truth: 24–30 s) |
|:--|:--|--:|--:|--:|--:|:--|
| run1 | `--matcher visual` | +32 ms | 466 ms | 5 980 ms | **89.7 %** | 30.03–36.0 s ❌ |
| run2 | **default** (`auto` / `hybrid`) | +32 ms | 3 133 ms | 7 040 ms | **3.4 %** | 56.25–60.0 s ❌ |
| run3 | `--strategy dtw` | −96 ms | 4 003 ms | 7 820 ms | **0.0 %** | none ❌ |

**All three runs printed:** `Closed-Loop Audit Passed: 99.2% frames verified (Mean Error: 24.5ms).`

> Reproduce: `python3 tests/synthetic_media.py && ./tests/run_e2e.sh` — see [`tests/README.md`](tests/README.md).

---

## 3. Findings

### F1 🔴 The audit scorecard is fabricated — `verifier_engine.py:160-162`

```python
audit = VerificationAudit(
    ...
    mean_alignment_error_ms=24.5,      # constant
    max_alignment_error_ms=38.0,       # constant
    passed_windows_pct=99.2,           # constant
```

The docstring says it "probes sliding correlation windows across the final rendered MKV". It does
not: no window is ever probed, `total_probed_windows` only counts fallback segments, and the three
headline numbers are literals. They are printed to the terminal, written to the JSON report, and
rendered into the Markdown report's "Closed-Loop Auto-Verification Audit Scorecard" section.

**Impact:** the one mechanism meant to catch failure is the one thing guaranteed to report success.
A user has no signal at all. This also makes every other bug below invisible in normal use.

**Fix:** either measure it (the verification harness in `tests/verify_output.py` already does
exactly this in ~40 lines — port the windowed envelope cross-correlation into
`ClosedLoopVerifierEngine`), or delete the fields and stop claiming verification. Do not ship
placeholder numbers.

---

### F2 🔴 The default path collapses to a single anchor — `consensus_engine.py:51-52,175`

`--matcher auto --strategy hybrid` (the defaults) produced **1 anchor**, versus 4 correct anchors
from the plain visual tier, and was ~14× less accurate (3 133 ms vs 466 ms mean error).

Three defects compound:

1. **`config.fps_ratio` does not exist.** `consensus_engine.py:51` guards with
   `hasattr(self.config, "fps_ratio")`, but `DubSyncConfig` has no such field, so the guard is
   *always* false and the global speed is hard-coded to `0.96`. The acoustic layer then
   resamples the target by the wrong factor (true value here: `1.041667`), smearing the
   correlation and pushing peaks below the `0.40` acceptance threshold.
2. **Visual confidence can never reach the admission gate.** `visual_anchors.py:176`:
   `confidence = max(0.0, 1.0 - composite_dist/12.0) * (0.5 + 0.5*color_sim)` with
   `max_hash_dist = 14`. Any match with distance ≥ 12 scores exactly **0.0**; observed confidences
   in run1 were `0.488, 0.0, 0.200, 0.331`.
3. **The consensus gate requires `confidence >= 0.90`** (`consensus_engine.py:175`), which (2) makes
   essentially unreachable. So the visual layer is silently excluded from the "multi-modal" fusion.

Net effect: one degenerate anchor at `t=0`, one dub segment spanning the whole film at a clamped
speed of `56.25/60 = 0.9375`, and a spurious 3.75 s fallback at the tail.

**Fix:** add `fps_ratio` to `DubSyncConfig` and populate it from the probed streams (or better:
derive the slope from a cheap first pass — `BlockSegmenterEngine.calibrate_global_slope` already
does this correctly). Rescale the confidence formula to `max_hash_dist` and lower the gate to
something the formula can actually express. Add a regression test asserting N anchors ≥ N for a
known pair.

---

### F3 🟠 Omitted scenes are always bridged at the end of the interval — `block_segmenter.py:333-360`

Scenario B computes `dub_ref_len = dt / g_speed` and places the dub **first**, putting the
English M&E bridge at the tail:

```
dub:      ref[a1, a1 + dub_ref_len]
fallback: ref[a1 + dub_ref_len, a2]
```

With anchors only at scene cuts, the position of the missing material *inside* an interval is
genuinely ambiguous — the tool assumes it is always at the end. In run1 that inverted reality:
ground truth removed ref[24, 30]; the tool bridged ref[30.03, 36.0] instead. Result: **6 s of real
dub discarded and replaced with English audio, and 6 s of dub shifted by +5 980 ms.**

Region classification confirms the inversion:

```
              region    notch    classified         truth
    [ 24.0, 30.0)   0.4369           DUB      FALLBACK  MISMATCH
    [ 30.0, 36.0)   0.1050      FALLBACK           DUB  MISMATCH
```

**Fix:** this is resolvable with evidence the engine already computes. Correlate a short probe at
*both* ends of the interval against the dub (or use the VAD/envelope layer) and place the gap where
the dub actually stops matching. The `ClosedLoopVerifierEngine` is explicitly documented to do this
healing — but it re-derives the projection from `prev_dub.tar_end`, i.e. it re-asserts the same
assumption it is supposed to test, so it can never correct it (`verifier_engine.py:76-77`).

---

### F4 🟠 Three documented engines are constructed and never called — `pipeline.py:43,47`

| Engine | Constructed | Called? | Documented as |
|:--|:--|:--|:--|
| `AcousticRefineEngine.refine_anchors` | `pipeline.py:47` | ❌ never | "sub-millisecond acoustic refinement" (§1.9) |
| `SpectralFingerprintEngine.discover_spectral_anchors` | `pipeline.py:43` | ❌ never | "Tier 3 … active" (§1.5) |
| `SileroVADEngine.discover_speech_anchors` | — | ❌ never | "Tier 4" (§1.6) |

`docs/SYNCHRONIZATION_METHODS.md` lists 11 algorithms under "ACTIVE SYNCHRONIZATION METHODS";
three of them are dead code and the headline acoustic-refinement stage never runs. Note also that
`--matcher spectral`, `--matcher vad` and `--matcher audio` all fall through to the *same* `else`
branch in `pipeline.py:118-124`, so four of the five documented matcher modes are identical —
and `hybrid` (the documented default) is not even an accepted value.

**Fix:** either wire them in or delete them and correct the docs. Shipping unreachable code that
the README sells as a feature is worse than not having it.

---

### F5 🟠 The README documents a CLI that does not exist

| README says | Reality | Verified |
|:--|:--|:--|
| `--ref`, `--tar`, `--out` | positional args `ref_video foreign_video output_video` | `cli.py: error: unrecognized arguments: --out` |
| `--matcher-mode hybrid\|audio\|visual\|orb\|vad` | `--matcher auto\|visual\|orb\|spectral\|vad` | `error: unrecognized arguments: --matcher-mode` |
| `--preset studio_ultra\|balanced\|fast` | `--preset studio\|balanced\|fast` | `error: invalid choice: 'studio_ultra'` |
| `--fallback-mode` | `--fallback` | — |
| `--report` (default True) | does not exist; reports always written | — |

The Quick Start command in the README fails immediately on a clean checkout.

---

### F6 🟡 The "sample-accurate splicer" is dead code — `audio_splicer.py:196-203, 268-282`

`render_and_splice` allocates the output buffer and builds the crossfade curves:

```python
final_audio = np.zeros((total_ref_samples, channels), dtype=np.int16)
crossfade_samples = int((self.config.crossfade_duration_ms / 1000.0) * sr_ref)
cf_fade_in  = np.sin(np.linspace(0, np.pi / 2, crossfade_samples, ...)) ** 2
cf_fade_out = np.cos(np.linspace(0, np.pi / 2, crossfade_samples, ...)) ** 2
```

…then never touches any of them. Segments are rendered to individual WAVs and hard-joined with the
FFmpeg `concat` demuxer. `find_nearest_zero_crossing()` (line 152) has no callers. So every segment
boundary is a **hard cut with no crossfade** — the exact clicks the feature exists to prevent —
and the buffer (≈1 GB of RAM for a feature-length stereo 48 kHz timeline) is allocated and
discarded. `config.zero_crossing_snap` and `crossfade_duration_ms` do nothing.

---

### F7 🟡 `build_edl` reads a config field that does not exist — `audio_splicer.py:104`

```python
if r_dur < self.config.min_scene_duration_sec:   # AttributeError
```

`DubSyncConfig` defines no `min_scene_duration_sec`. The method is currently unreachable, but it is
exported public API (`__init__.py` → `AudioSplicerEngine`). The same latent class of bug is F2's
`fps_ratio`.

---

### F8 🟡 Project hygiene

* **No tests.** Zero test files; `pytest` is not in any requirements file.
* **No CI.** No `.github/workflows/` (the only workflows in the repo are from the vendored
  `audalign` copy). Nothing gates a push.
* **No packaging.** No `pyproject.toml` / `setup.py`; `run_dub_sync.py` works only via a
  `sys.path` hack. Dependencies are unpinned (`numpy>=1.22.0`).
* **No LICENSE file**, though the README states MIT — and `archive/external/audalign-1.2.4/`
  ships third-party MIT code, so licensing needs to be explicit anyway.
* **Version is five different values:** `__init__.py` `2.0.0`, TUI banner `2.0.0`, pipeline report
  `v2.0.0` (hard-coded string), docs `2.1.0`, `qc_report.py` fallback `v2.2.0`, README and commit
  message `v3.5`.

---

### F9 🟡 Operational rough edges

* **Import-time FFmpeg lookup** (`media_probe.py:50`): `FFMPEG_PATH = get_ffmpeg_path()` runs at
  import, so `import dub_sync_engine` raises `RuntimeError` on a machine without FFmpeg. This
  blocks tests, docs builds and `--help`. Make it lazy (it is also the only reason this review
  needed `imageio-ffmpeg` installed).
* **Scratch space is never cleaned.** A 60 s / 1.3 MB test pair produced a **44 MB** temp folder
  (two full PCM WAVs + per-segment WAVs + the rendered master) — roughly **2 GB for a feature**.
  It is written next to the output and only removed if the user answers an interactive
  `Confirm.ask` whose default is *keep*, and which is silently skipped when stdin is not a TTY.
* **Brittle probing.** `FFPROBE_PATH` is resolved and then never used (`media_probe.py:51`);
  all metadata comes from regex-scraping `ffmpeg -i` stderr, which breaks across ffmpeg versions
  and locales (e.g. channel count only recognises `stereo`/`5.1`/`mono`, so 7.1 or `7.1(wide)`
  silently report `None`).

---

### F10 🔵 Dead knobs and unused code

Read nowhere: `use_temporal_burst`, `burst_delta_frames`, `use_ransac_block_clustering`,
`max_speed_deformation`, `zero_crossing_snap`, `zero_crossing_window_ms`, `enable_acoustic_refine`,
`acoustic_window_ms`, `speech_band_attenuation`, `qc_report_html`, `verbose`. Users tuning these
will see no change. `QCReportGenerator` is exported but unused (only `ForensicReportGenerator` runs).

---

### F11 🔵 Repository weight

* `archive/` is **7.3 MB of 43 % of the repo**, most of it a vendored copy of `audalign 1.2.4`
  (MIT, © 2020 Ben Miller) that nothing imports. Consider a submodule, a download step, or deletion.
* `tools/dub_sync.py` (514 lines) is a complete unreferenced legacy duplicate of the engine
  (same FFmpeg discovery, same scene-cut approach, same `--matcher` ideas). It is not imported by
  anything and diverges from the package, which is a maintenance trap.
* `dub_sync_engine/models/silero_vad.onnx` (2.3 MB) is committed — reasonable for a self-contained
  tool, but undocumented and unverifiable; add a checksum and provenance note.

---

### F12 🔵 Accuracy details

* **Duration is not bit-exact**: +32 ms (visual), −96 ms (DTW) against a 60 s reference; the README
  promises "0.000ms drift". Cause is per-segment `-t` truncation plus AAC muxing, not the EDL math,
  which is exact (unit tests confirm coverage of 60.000 s to the millisecond).
* **Per-block speed calibration is computed, printed, reported — and discarded.**
  `cluster_into_blocks` derives an independent slope per macro-block (this is a headline feature of
  Stage 5), but `build_macro_edl` ignores `block.speed_factor` and uses one global slope for every
  segment. Blocks are effectively display-only.
* **RANSAC is not RANSAC**: `cluster_into_blocks` is greedy median-offset chaining with no
  sampling or consensus, despite `use_ransac_block_clustering` and §1.7.

---

### F13 🔵 Silent failure swallowing

`consensus_engine.py` wraps each entire layer in `except Exception: pass` (lines 113, 160, 184). If
the ONNX runtime fails to load, or scipy raises, the layer contributes nothing and the run
continues as if that modality simply found no anchors. There is no logging anywhere in the
codebase. At minimum, log the exception at DEBUG/WARNING and surface a per-layer candidate count.

---

## 4. What works well

* **MediaProbe parsing is correct.** Durations (60.000 s / 56.250 s), fps (24 / 25), codecs,
  sample rates, channel counts and languages were all parsed correctly from both files.
* **The Tier-1 visual matcher is genuinely good.** All four scene cuts were matched, and every
  anchor landed within ~1 frame of ground truth: `12↔12.44` (truth 12.5), `24↔24.92` (25.0),
  `36↔31.20` (31.25), `48↔43.68` (43.75). Excluding the F3 ambiguity region, **mean |error| was
  58 ms with 89.7 % of windows inside ±120 ms** — that is respectable sub-frame-ish performance.
* **The EDL / broadcast-slope math is correct and self-consistent.** Verified by unit test:
  slope detection snaps to the real standards (0.96 / 1.041667), `atempo` is applied in the right
  direction, coverage is exactly 60.000 s across 1:1, PAL-speedup and cut-omission cases, and the
  monotonic DP lattice + backtracking has no ordering or cycle defects.
* **Silero VAD is wired correctly.** Input shapes `[1,576]` / state `[2,1,128]` / scalar `sr` match
  the ONNX graph; 1 875 frames over 60.0 s with a sane probability range. (It contributed nothing
  on this synthetic signal, which is noise rather than speech — so its *effectiveness* is untested,
  not its plumbing.)
* **MKV output is correct.** Video stream copied, track 1 = reference audio tagged `eng`,
  track 2 = synced dub tagged `ara`.
* **No command-injection surface.** Every subprocess call is list-based; no `shell=True`.

---

## 5. Caveats on the measurements

* The synthetic dub is derived *from* the reference (identical M&E bed), which is **favourable** to
  the tool. Real dubs have different music/SFX mixes, so real-world accuracy should be expected to
  be **worse**, not better.
* The synthetic "speech" is band-limited noise, so the VAD layer's real-world contribution is
  untested here — F13 is precisely why that is hard to tell.
* One 60 s clip is a small sample. It is enough to establish F1 and F2 (which are not
  accuracy edge cases — they are a fabricated metric and a collapsed anchor set), but the exact
  millisecond figures will vary by content.

---

## 6. Suggested remediation order

**P0 — before anyone trusts an output**
1. **F1**: make the audit report real measurements, or remove the claim.
2. **F2**: add `fps_ratio` (or derive the slope), fix the confidence scale, lower the gate; assert
   a minimum anchor count before accepting a timeline.
3. **F5**: reconcile README with `cli.py`; add `--help` examples that actually run.

**P1 — correctness**
4. **F3**: place omitted-scene gaps using acoustic evidence at both ends of the interval.
5. **F4**: wire or delete the acoustic-refine / spectral / speech-anchor engines.
6. **F6**: implement the crossfade (or delete the dead code and the claim) and drop the unused
   buffer.
7. **F8**: add the pytest suite, a CI workflow, `pyproject.toml`, a LICENSE, and a single source of
   truth for the version.

**P2 — quality of life**
8. **F7**, **F9**, **F10**, **F11**, **F12**, **F13**: lazy FFmpeg lookup, temp-dir cleanup
   (default to clean unless `--keep-temp`), use `ffprobe -print_format json` instead of stderr
   scraping, prune dead config knobs, remove or submodule `archive/`, delete `tools/dub_sync.py`,
   replace bare `except Exception: pass` with logging.

---

## 7. Appendix — reproduction

```bash
pip install -r requirements.txt -r requirements-dev.txt

# 1. build the synthetic reference/dub pair with known ground truth
python3 tests/synthetic_media.py           # -> tests/_media/{ref.mp4,tar.mp4,ground_truth.json}

# 2. run every strategy and measure each output independently
./tests/run_e2e.sh

# 3. fast unit tests (no media required)
python3 -m pytest tests/ -q
```

`tests/verify_output.py` is standalone: `python3 tests/verify_output.py <out.mkv>`.
The `xfail` tests in `tests/test_known_defects.py` encode findings F1–F7; they will start passing
as each is fixed.
