# DubSync ← subsync Integration Plan

> Re-implementing subsync's transferable algorithms (documented in
> [`SUBSYNC_ANALYSIS.md`](./SUBSYNC_ANALYSIS.md)) as original Python/numpy inside
> DubSync. GPL-3.0 code is never copied; only the algorithms are reimplemented.

## To-do list (ordered; each item = dig deep → integrate → test → checkpoint)

1. **RANSAC line fitter** — implement a `LineFinder`-style robust fitter in a new
   `dub_sync_engine/line_fit.py` (pure numpy): incremental point-cloud line fit with
   inlier counting, monotonic-slope + global-consistency constraints, and iterative
   furthest-outlier removal. Returns slope `a`, intercept `b`, r², and inlier set.
   - Dig deep: exact math, quadrant bucketing, constraints, edge cases.
   - Integrate: unit-testable standalone module first; then wire into
     `block_segmenter.calibrate_global_slope()` and `cluster_into_blocks()`.

2. **Pearson r² as real confidence** — replace the fuzzy `mean(anchor.confidence)`
   with the fit's r² + inlier ratio in `ContinuousBlock.confidence` and the forensic
   report (`pipeline.py`).
   - Dig deep: what r² means here, degenerate cases (few points, collinear),
     calibration to a 0–1 confidence.

3. **Continuous similarity weights** — `consensus_engine.py` currently *binary-gates*
   visual matches on acoustic support. Emit a continuous `weight` (normalized M&E
   correlation at the cut × seq_len boost) and feed it into the fitter.
   - Dig deep: what signal to use, how to normalize, how to avoid dropping real
     anchors (the historical regression risk).

4. **Coverage / diversity constraint** — require RANSAC inliers to span distinct
   act/scene windows (min points across min ref-time span), not just a raw count.
   - Dig deep: how to bucket by ref time, thresholds, interaction with real cuts.

5. **Recursive piecewise refinement** — after the global line, bisect at the largest
   inlier gap and refit each side (tympanix `sync_all` idea) to recover multi-speed
   episodes without hand-tuned discontinuity thresholds.
   - Dig deep: split criterion, margin shrinking, termination, interaction with
     broadcast-speed snapping.

6. **Full validation** — extend the synthetic fixture matrix with planted-outlier
   cases; run the whole suite; end-to-end smoke; commit + tag per completed idea.

## Ground rules

- Every idea lands as its own commit + `checkpoint-subsync-<n>` tag.
- `pytest -q` must stay green at every step; add tests *before/with* each change.
- No regressions on the existing 32 tests; no end-to-end (owner constraint:
  "no_static") — static + synthetic validation only.
- GPL-3.0 source is a reference for *understanding*, never for copying.

---

## Deep dives (findings, recorded per idea before integration)

### Idea 1 — RANSAC line fitter
Prototype (`/tmp/proto_linefit.py`) verified against synthetic anchor clouds:
- **Clean 1.0×:** recovers `a=1.0, b=0.001, r²=1.0`, 50/50 inliers. ✓
- **PAL 0.96×:** recovers `a=0.960, b=2.0`, 50/50. ✓
- **8 planted false anchors:** recovers `a=0.960, b=2.0`, keeps 50/50 true
  inliers and **0/8 false** — RANSAC discards scattered false matches
  automatically, which is exactly the Hero-108 failure mode.
- **Two-speed episode:** fits the second act only (29/50 inliers) — a clean
  signal that a *single* line is wrong (feeds idea 5).

Adaptations from subsync's C++ for DubSync:
- **Global-consistency constraint must change.** subsync forces `|a·x+b − x| ≤
  maxDistance` at both endpoints (offset ≈ 0), which is wrong for DubSync (real
  logo gaps / censored cuts produce large legit offsets). We instead constrain
  the line to map `[0, ref_duration]` into `[−margin, tar_duration + margin]`.
- **Slope range:** subsync allows `[0.1, 10]` (subtitle timing); DubSync should
  use `[0.85, 1.15]` (or configurable) around broadcast standards.
- **Inlier tolerance:** subsync uses 5 s (`maxError`); DubSync anchors are
  sub-ms acoustically snapped, so a configurable `ransac_inlier_tolerance_sec`
  (default ~1.0 s) is appropriate.
- **Quadrant bucketing is unnecessary** at DubSync's scale (consensus chain is
  tens–hundreds of anchors); plain O(N²) pair enumeration is fine. Bucketing can
  be added later if anchor counts explode.

### Idea 2 — Pearson r² as confidence
**Critical finding:** r² is computed on *inliers after outlier removal*, so it
can read `1.0` even when most points were thrown away (two-speed case: r²=1.0 on
29/50). r² alone is misleading. Confidence must be a composite:

```
confidence = r² × inlier_ratio        # both in [0,1]; inlier_ratio = inliers / N
```

This is a *measured* metric (aligns with PR #1's "no fake metrics" rule) and
drops naturally for multi-speed episodes, signaling the need for recursive split
(idea 5).

### Idea 3 — Continuous similarity weights
`consensus_engine.py` STEP 3 gates visual matches *binarily*
(`is_acoustically_confirmed`). The subsync analog is a continuous
`compareWords(sim) ≥ minWordsSim` per-point similarity. We will:
- Add `weight: float = 1.0` to `AnchorMatch` (separate from `confidence`, which
  `build_macro_edl` already uses as `min(…)`).
- In STEP 3, set `weight` = strength of the acoustic confirmation (max `peak` of
  acoustic/VAD candidates within the gate window) × `seq_len` boost, normalized.
- Pure acoustic/VAD candidates carry `weight = their peak`.
- The RANSAC fitter (idea 1) uses `weight` when counting inliers and in the OLS
  fit — a weakly-confirmed match contributes little, instead of being hard
  accepted/rejected (avoids the historical "threshold too strict → dropped real
  anchors" regression).

### Idea 4 — Coverage / diversity constraint
subsync's `countBuckets` requires inliers to land in *distinct* subtitle lines so
one dense region can't dominate. DubSync equivalent: bucket inliers by
`floor(ref_time / bucket_sec)` and require `≥ min_buckets` distinct buckets
(default bucket ~60 s, min ~3). A fit that only covers a single dense opening is
downgraded/rejected, preventing a false early anchor from over-determining the
global slope.

### Idea 5 — Recursive piecewise refinement
tympanix `sync_all`: global fit → bisect → refit each half with shrinking margin.
For DubSync:
- After the global RANSAC fit, if `inlier_ratio < target` (e.g. 0.8), find the
  largest gap between consecutive inliers and split the anchor set there.
- Refit each side independently (each returns its own slope/offset/inliers),
  recursing until `inlier_ratio ≥ target`, span `< min_span`, or depth limit.
- Each resulting leaf line becomes one `ContinuousBlock` (slope snapped to a
  broadcast standard). This replaces the greedy
  `discontinuity_threshold_sec` clustering with a principled, inlier-driven
  segmentation.
