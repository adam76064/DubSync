# subsync — Code Analysis & What We Can Borrow for DubSync

> Research + code review of the `subsync` project(s), with a concrete mapping of
> transferable algorithms onto DubSync's engine.
>
> Repos examined (cloned to `/home/user/subsync-analysis/`):
> - `sc0ty/subsync` — the canonical "synchronize subtitles using ML" tool (GPL-3.0).
> - `tympanix/subsync` — a pure-Python fork/successor (Apache-2.0).

---

## 1. What subsync is (and isn't)

`subsync` fixes **subtitle** timing against a video's **audio**. It is *not* a dub
audio replacer. But underneath, it solves a problem that is structurally identical to
the core of DubSync:

> Given two sparse, independently-timed event streams over the same content, recover
> the (piecewise) affine time transform `tar_time = a · ref_time + b` that maps one
> timeline onto the other, robustly ignoring false/missing events.

For subsync the two event streams are *subtitle words* and *speech-recognized words*.
For DubSync they are *reference anchors* and *target anchors* (visual + acoustic +
VAD). The geometry is the same; only the feature source differs. This is exactly why
its math is directly reusable.

---

## 2. How the two implementations work

### 2.1 `sc0ty/subsync` (the important one — the algorithm)

Pipeline (`subsync/synchro/` + the compiled `gizmo/` C++ core):

1. **Reference audio → words** (`subsync/synchro/speech.py`, `pipeline.py`):
   demux audio → pocketsphinx/ffmpeg speech recognition → a stream of
   `(text, time, duration, score)` words.
2. **Subtitle file → words** (`SubtitlePipeline`): parse SRT/SSA/ASS → words. For
   languages without whitespace (CJK), it splits each line into **N-grams**
   (`gizmo/text/ngrams.cpp`) and distributes the line duration across them.
3. **Optional cross-language dictionary** (`gizmo/text/translator.cpp`): if sub and
   ref languages differ, map words via a bilingual dictionary before matching.
4. **Word similarity gate** (`gizmo/text/dictionary.cpp`, `compareWords`): a
   position-weighted, case-insensitive string similarity. A candidate match between
   a sub word and a ref word is only accepted if `sim >= minWordsSim`.
5. **Point-cloud + RANSAC line fit** (`gizmo/synchro/synchronizer.cpp`,
   `gizmo/math/linefinder.cpp`):
   - Every accepted match emits a 2-D point `(sub_time, ref_time)`.
   - `LineFinder` incrementally maintains the best line `y = a·x + b` through the
     point cloud, constrained to be monotonically increasing (`0.1 ≤ a ≤ 10`) and to
     stay within `maxDistance` of the identity at both ends of the span.
   - It evaluates candidate lines from each new point against all prior points using
     **spatial quadrant bucketing** (60 s cells) so each evaluation is near O(1)
     instead of O(N). The line with the most inliers (within `maxError = 5 s`) wins.
   - `Line::interpolate()` is an ordinary least-squares fit; it returns
     `(Σxy …)` which is exactly the **squared Pearson correlation r²**, used as the
     "factor" quality metric.
6. **Iterative RANSAC outlier removal** (`Synchronizer::correlate`): while
   `factor < minCorrelation` or `maxDistance` exceeded, remove the furthest point and
   refit — discarding false matches automatically.
7. **Coverage constraint** (`countBuckets`): matched points must fall in *distinct
   subtitle lines* (buckets), preventing one dense region from dominating the fit.

Result: a global linear formula `sub_time → audio_time` (slope `a` = speed ratio,
intercept `b` = offset).

### 2.2 `tympanix/subsync` (pure Python, ML-flavored)

`subsync/media.py`:
1. **MFCC** (`librosa`) over the audio → 13-D frames at 16 kHz.
2. A **neural net** (`net.py`, Keras/TensorFlow) predicts per-frame speech
   probability.
3. Subtitle text → binary "speech present" label sequence over the same frame grid.
4. **Shift search**: for a margin window, roll the label sequence and compute
   **log-loss** against the predicted speech probability; the shift with minimum
   log-loss wins (this is a normalized cross-correlation in disguise).
5. **Recursive per-sentence sync** (`sync_all`): fit globally, then bisect the
   subtitle list and refit each half with a shrinking margin — hierarchical local
   refinement.

---

## 3. Licensing (critical constraint)

| Repo | License | Implication for DubSync (MIT) |
| :--- | :--- | :--- |
| `sc0ty/subsync` | **GPL-3.0** | **Cannot copy the C++ `gizmo` code verbatim** into an MIT project. |
| `tympanix/subsync` | **Apache-2.0** | Compatible, but still prefer reimplementation for cleanliness. |

**Policy:** we re-implement the *ideas* (RANSAC line fitting, Pearson r² quality
metric, similarity-gated sparse point matching, recursive refinement) as original
Python in DubSync. These are standard, well-documented algorithms, not GPL
expression. No code or strings may be copied from `gizmo/`.

---

## 4. What to borrow — mapped onto DubSync's files

### 4.1 ⭐ Replace the "fake RANSAC" with a real LineFinder (`block_segmenter.py`)

DubSync's `cluster_into_blocks()` says "using RANSAC" but is actually greedy
offset-difference clustering. subsync's `LineFinder` is the real thing and is a
drop-in conceptual upgrade:

- **Candidate generation:** for each anchor, try the line through it and each prior
  anchor (or the `a = 1` line), count inliers within a tolerance (`maxError`).
- **Monotonic + global-consistency constraints:** slope in `[0.1, 10]`, and the line
  must land within `maxDistance` of identity at `minX`/`maxX` — this is a principled
  version of DubSync's ad-hoc "offset must be consistent across the span" checks and
  directly kills the *false black-frame / stray visual match* failure mode (a wrong
  anchor can't sit on the consensus line).
- **Spatial bucketing** (60 s quadrants) to keep per-anchor evaluation cheap, so the
  O(N²) candidate scan stays fast on hundreds of anchors.

This one change replaces the hand-tuned `discontinuity_threshold_sec`, the
frozen-anchor cleanup, and the opening-black-frame hack in `build_macro_edl` with a
single robust mechanism: *anchors that fit the consensus line are inliers; everything
else is a cut boundary or a false match.*

### 4.2 ⭐ Use Pearson r² as the real "confidence" (`block_segmenter.py`, `pipeline.py`)

`Line::interpolate()` returns the squared Pearson correlation of the fit. DubSync
currently reports `confidence = mean(anchor.confidence)` — a fuzzy number. Adopt:

```
per-block confidence = r² of the least-squares line through that block's anchors
                       (plus the global RANSAC inlier ratio)
```

This is a **measured** quality metric (consistent with the PR #1 philosophy of "no
fake metrics") and directly comparable across blocks. It slots into `ContinuousBlock`
and the forensic report.

### 4.3 ⭐ Similarity-gated sparse point matching (`consensus_engine.py`)

subsync gates every candidate point by `compareWords(sim) >= minWordsSim`. DubSync's
analog is the acoustic gate + N-gram `seq_len` boost (Phase 5). The refinement to
steal: instead of a **binary** acoustic gate, compute a **continuous similarity**
between the visual match and the acoustic anchor at that time (e.g. normalized M&E
correlation at the visual cut, weighted by `seq_len`), and feed that score in as the
**point weight** for the RANSAC fit. Bad-but-passing matches then contribute low
weight instead of being either hard-accepted or hard-rejected — smoother than the
current gate and less likely to drop real anchors (the exact regression the history
warned about).

### 4.4 Coverage/diversity constraint (`block_segmenter.py`)

subsync's `countBuckets` requires inliers to span *distinct* subtitle lines, so one
dense region can't hijack the fit. DubSync's equivalent: require RANSAC inliers to
span distinct **scenes/act windows** (e.g. at least `min_points` across `min_span`
seconds of ref time), not just `min_points` total. This prevents a single strongly
matched opening from over-determining the global slope.

### 4.5 Recursive refinement (`block_segmenter.py` or a new pass)

tympanix's `sync_all` (global fit → bisect → refit each half with shrinking margin)
is a clean recipe for **piecewise-linear** alignment: after the global line, split at
the largest inlier gap and refit each side independently. This is a more principled
version of DubSync's per-block speed calibration and would let `cluster_into_blocks`
recover multi-speed episodes (PAL acts mixed with 1.0× acts) without hand-tuned
discontinuity thresholds.

---

## 5. What NOT to borrow

- **Speech recognition / ASR** (`pocketsphinx`, `librosa` + neural speech prob,
  bilingual dictionary). DubSync is deliberately **language-independent** — it syncs
  on M&E + visual structure, not on recognized words. Importing ASR would reintroduce
  language models, huge deps, and the very fragility the project avoids. (This is the
  *opposite* direction: subsync needs the words because it has no video; DubSync has
  video and doesn't need the words.)
- **The C++ `gizmo` build** (GPL + heavy native deps: ffmpeg/pocketsphinx). Keep
  DubSync pure-Python/numpy.

---

## 6. Recommended sequence (if we proceed)

1. **`block_segmenter.py`:** add a `LineFinder`-style RANSAC fitter (pure numpy) and
   use it in `cluster_into_blocks` for the global slope + block inliers. Return r² as
   `confidence`. Keep the existing `sanitize_edl`/EDL builder as the downstream
   consumer.
2. **`consensus_engine.py`:** emit a continuous similarity score per anchor (weight)
   and pass it into the fitter.
3. **`pipeline.py`:** report `confidence = r²` and `inlier_ratio` in the forensic
   payload.
4. **Tests:** add synthetic scenarios with planted false anchors (already have
   `black_frame`, `intro_gap`) and assert the RANSAC fit recovers the true slope while
   discarding the outliers — this is exactly the regression matrix from the plan.

Every step lands behind the existing `pytest` matrix and a git checkpoint, per the
`IMPLEMENTATION_PLAN.md` discipline.

---

## 7. Key source references (for the implementer)

| Concept | Location |
| :--- | :--- |
| Sparse point cloud + similarity gate | `gizmo/synchro/synchronizer.cpp` (`addSubWord`/`addRefWord`) |
| RANSAC line fitting w/ quadrant bucketing | `gizmo/math/linefinder.cpp` |
| OLS fit + Pearson r² "factor" | `gizmo/math/line.cpp` (`Line::interpolate`) |
| Iterative furthest-outlier removal | `gizmo/synchro/synchronizer.cpp` (`correlate`) |
| Line coverage / bucket constraint | `gizmo/synchro/synchronizer.cpp` (`countBuckets`) |
| Word similarity | `gizmo/text/dictionary.cpp` (`compareWords`) |
| CJK N-gram splitting | `gizmo/text/ngrams.cpp` |
| MFCC + logloss shift search | `tympanix/subsync/subsync/media.py` |
| Recursive bisect refinement | `tympanix/subsync/subsync/media.py` (`sync_all`) |
