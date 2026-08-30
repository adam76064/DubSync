# DubSync — Chat History Summary & Engineering Lessons

> Extracted from the full development transcript between the project owner and the
> "Antigravity" assistant. This document captures **what was tried, what worked, what
> broke, and why** — so future changes build on evidence instead of re-discovering
> the same failures.
>
> The current repo (`main` @ `9bd079a`) is the state the owner asked to restore at the
> end of the transcript: *"restore engine to checkpoint-v2.5-pre-pure-audio (full visual
> and acoustic multi-modal sync)"*.

---

## 1. The Problem Being Solved

Sync the **Arabic audio** of a low-quality cartoon TV rip onto the **English HQ master**
video (Blu-ray / WEB-DL), producing a dual-audio MKV. The two versions do **not** match:

* **Speed differences** — 25 fps PAL broadcast vs 24 fps film (`0.960000x` tempo ratio).
* **Censored / omitted scenes** — TV broadcasts cut action, jokes, intros, end credits.
* **Extra scenes** — TV bumpers, recaps, station intros that don't exist in the master.
* **Micro-trims** — 1–2 second internal trims (e.g. a single punch removed).
* **Aspect-ratio / quality differences** — squished 4:3 vs 16:9, letterboxing, watermarks,
  low resolution (416p vs 1080p), heavy compression.

**Why generic tools (audalign, subsync) fail**: they compute a *single global offset* and
correlate *raw waveform/dialogue*. Dubbed dialogue is a different language, and a single
offset cannot survive multiple cuts.

---

## 2. Timeline of Key Milestones

| Stage | What was built | Result on test files |
| :--- | :--- | :--- |
| v1 | Visual scene-anchor tool (pHash + monotonic DP) | **Monkie Kid: great** (669 anchors) |
| v2 | Modular "DubSync Pro" + TUI + sub-ms acoustic refine | Monkie Kid great; **Hero 108 failed** (1 anchor → slow-mo) |
| v2.1 | Multi-modal consensus + closed-loop verifier + block-RANSAC | Fixed over-slicing drift |
| v2.2 | Anchor-knot alignment + micro-trim clamping | Fixed minute-8 drift in Ep 2 |
| v2.3 | Adaptive anchor-knot slicing (no anchor discarding) | Ep 3 *worse* — false visual matches |
| v2.4 | Acoustic-gated multi-frame (audio-first, visual gated) | Improved, but still false cuts |
| v2.5 | N-gram sequence rhythm + dual-layer cross-validation + exact scaled slicing | **"wonderful results"** — owner saved checkpoint |
| (exp) | Pure-audio only (visual removed) | Fast but still flawed; **reverted** |
| (exp) | "Claude fixes" (chromaprint bootstrap, micro-DTW, EBU R128, KD-tree) | **"worst output ever" — reverted** |
| final | Restore to `checkpoint-v2.5-pre-pure-audio` | current state |

**Benchmark episodes** used throughout: `LEGO Monkie Kid S01E01` (clean 24fps WEB-DL vs
720p dub) and `Hero 108 S01E01/02/03` (PAL 25fps, multiple censored cuts, micro-trims).

---

## 3. What Actually Worked (the "golden" findings)

These are the techniques that, per the transcript, produced the **zero-drift
"PIECEWISE_SYNCED" / "wonderful results"** outputs:

1. **Global broadcast-speed lock (the "clock spine").**
   An episode runs at **one** hardware speed — `0.960000x` (PAL 25fps), `1.000000x`
   (film), `1.041667x` (PAL speedup), or `1.001001x` (NTSC). Never micro-stretch per shot.

2. **Anchor knots = verified milestones.**
   Every confirmed match is an immutable pin. Within a continuous act the audio plays
   at the locked speed; at a **real** cut it re-anchors to the next verified knot
   (zero downstream drift).

3. **Cut detection = a real timeline discontinuity.**
   A cut exists only when `tar` time stops advancing proportionally (speed ratio drops
   below ~`0.90x`). A mere *gap in anchors* is **not** a cut.

4. **Slope-continuity gate (anti-false-English).**
   If two adjacent blocks share the same speed slope and the Arabic timeline advanced
   proportionally (`Δt_tar ≈ Δt_ref × speed`), **merge them** — do not insert English.

5. **Exact scaled target slicing (anti-overflow / anti-cutoff).**
   `tar_end = tar_start + (ref_duration × speed)` — guarantees output duration == video
   duration to `0.000ms`, eliminating the `+2.3s` overflow drift and speech truncation.

6. **Localized (not global) outlier filtering.**
   Only compare the *opening* anchor against its neighbor to reject the black-frame /
   logo false match at `0:00`. A **global median filter destroyed Episode 3** by
   discarding all genuine second-half anchors after cuts shifted the offset.

7. **N-gram sequence rhythm verification (anti-cartoon-trap).**
   Never trust an isolated frame. Require a chain of ≥3 consecutive camera cuts whose
   inter-cut time ratios match the broadcast speed. A single pose/black frame may
   repeat; a 3-shot rhythm essentially never does.

8. **Acoustic M&E band (800 Hz–3.2 kHz) + Silero VAD as the primary spine.**
   Background music/SFX is language-independent and unique in time; speech *bursts and
   pauses* (not words) are identical because voice actors match the same lip flaps.

---

## 4. What Repeatedly Failed (and the root cause)

| Failure | Symptom | Root cause |
| :--- | :--- | :--- |
| **Micro-slicing** (276 cuts) | drift + robotic voice warble | `atempo` buffer latency (~20ms/slice) accumulates across hundreds of slices |
| **Blind duration-ratio stretch** | slow-mo audio (`0.918x`) | fallback `speed = tar_dur / ref_dur` when <2 anchors |
| **Global median outlier filter** | Ep 3 second half totally lost | cuts legitimately shift offset; global filter threw away real anchors |
| **Black-frame false anchor @ 0:00** | dialogue "teleported" to 0:00, intro replayed | opening logo black frame matched an in-scene black transition |
| **False visual matches in cuts** | Arabic speech *before* character appears, then English replay | a frame inside a cut zone matched a frame *after* the cut |
| **Ghost cut micro-blips** (1.7s / 4.4s) | tiny Arabic blips + English islands | weak noise correlation (≥0.35) accepted inside silent cut zones |
| **Speed float** (`1.045x`) | subtle end-of-episode drift | per-interval speed allowed to float instead of locking to broadcast standard |
| **Slope-continuity over-merge** | minute-8 drift returned | merged across a *real* 1.75s trim instead of re-anchoring |
| **"Claude fixes" additions** | crash (`broadcast_snap` missing) + worst output | un-integrated modules added without regression testing |
| **Verifier metrics** | misleading "99.2% passed" | metrics were **hardcoded**, not measured |
| **Self-healing** | never fired | cross-correlated foreign dub vs English (different languages) |

---

## 5. The Meta-Lesson (most important)

**Every "accuracy fix" in this project introduced a regression elsewhere**, and each
regression was only caught by the owner manually re-watching the output:

* The global median filter fixed Ep 2's intro but destroyed Ep 3.
* The slope-continuity gate fixed false English but over-merged real trims.
* The "no-anchor-discard" rule fixed Ep 3's lost anchors but trusted false visual matches.
* The "Claude fixes" looked good in isolation but crashed and degraded quality.

**Conclusion: the single highest-leverage improvement is automated regression testing**
against the known benchmark episodes — so a change that fixes Monkie Kid can be checked
for regressions against Hero 108 in seconds, instead of a manual "it's worse now" loop.

---

## 6. The Architecture the Owner Converged On

```
Level 1 — ACOUSTIC MASTER SPINE  (ground truth)
   bandpass M&E (800Hz–3.2kHz) + Silero VAD speech envelopes
   → locks global broadcast speed + identifies macro acts

Level 2 — GATED MULTI-FRAME VISUALS  (cut snappers, helper only)
   N-gram (≥3) shot sequences, searched only within ±2s of an acoustic anchor
   → snaps boundaries to exact video frames, immune to cartoon repetition

Level 3 — ADAPTIVE ANCHOR-KNOT SPLICING + CLOSED-LOOP VERIFICATION
   exact scaled slicing · slope-continuity gate · localized opening check
   near-duplicate anchor snapping · broadcast speed lock · M&E cut bridging
```

The owner's final stated principle (near end of transcript): **"Audio First, Video
Second"** — audio tells you *which scene* you're in; video tells you the *exact frame*.

---

## 7. Current Code vs. This History

The restored `checkpoint-v2.5-pre-pure-audio` code already contains most of the
"golden" fixes (§3), but the transcript also reveals several were **never fully wired**:

| History finding | Status in restored code |
| :--- | :--- |
| Sub-ms acoustic refinement ("Stage 5") | **Dead code** — `refine_anchors()` never called |
| Closed-loop verification metrics | **Hardcoded** (`24.5ms`/`38.0ms`/`99.2%`) |
| Self-healing of false fallbacks | Correlates **dub vs English** (never fires) |
| Equal-power crossfade + zero-crossing snap | Buffers allocated but **never used**; segments hard-concatenated |
| Silero VAD input convention | v4 (576-sample) context vs bundled **v5** model |
| `min_scene_duration_sec` | Referenced but **undefined** (latent `AttributeError`) |
| CLI flags / README | Mismatched (`--ref/--tar/--out` in README vs positional args) |

> These were the exact issues fixed in the follow-up PR ("Fix dead-code paths and fake
> metrics so DubSync actually works").

---

## 8. Remaining Improvement Opportunities (prioritized)

1. **Regression test harness** — golden-file tests on Monkie Kid + Hero 108 (the two
   benchmark episodes) so no future "fix" silently regresses the other.
2. **Cut-aware global slope calibration** — estimate the broadcast speed using only
   *continuous* intervals (exclude spans crossing cuts), not a naive median.
3. **Strict broadcast-speed locking** — never let a continuous act float to arbitrary
   values (e.g. `1.045x`); snap to standards.
4. **Micro-chopping guard** — minimum dub-act duration + raised correlation threshold
   so silent cut zones don't fragment into 1.7s Arabic blips + English islands.
5. **Audio-gated visual acceptance** — strengthen the consensus gate so an isolated
   visual match with no acoustic support cannot pass (the recurring false-match bug).
