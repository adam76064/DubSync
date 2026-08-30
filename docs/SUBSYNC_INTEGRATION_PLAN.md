# DubSync ← subsync Integration Plan

> Re-implementing subsync's transferable algorithms (documented in
> [`SUBSYNC_ANALYSIS.md`](./SUBSYNC_ANALYSIS.md)) as original Python/numpy inside
> DubSync. GPL-3.0 code is never copied; only the algorithms are reimplemented.

## To-do list (ordered; each item = dig deep → integrate → test → checkpoint)

1. **RANSAC line fitter** ✅ `checkpoint-subsync-1` — `dub_sync_engine/line_fit.py`.
2. **Pearson r² as real confidence** ✅ `checkpoint-subsync-2` — `_block_confidence`,
   `calibrate_global_fit`.
3. **Continuous similarity weights** ✅ `checkpoint-subsync-3` — `AnchorMatch.weight`,
   consensus-engine confirmation strength.
4. **Coverage / diversity constraint** ✅ `checkpoint-subsync-4` — `coverage_ratio`.
5. **Recursive piecewise refinement** ✅ `checkpoint-subsync-5` (+`-5b` for the
   `build_macro_edl` per-act-speed wiring) — `fit_piecewise_lines`, residual-minimizing
   breakpoint split.
6. **Full validation** ✅ — planted-outlier fixtures, full suite green (55 tests),
   imports/compile clean. End-to-end smoke is out of scope per the owner's
   "no_static" constraint.

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
- After the global RANSAC fit, if `inlier_ratio < target` (e.g. 0.8), split at
  the breakpoint that **minimizes the combined weighted residual of two
  independent least-squares fits** (segmented-regression breakpoint estimate).
  *(Inlier-gap heuristics were tried and rejected: they are not robust for
  contiguous multi-speed acts and depend on the random anchor realization.)*
- Refit each side independently, recursing until `inlier_ratio ≥ target`,
  too few points, or depth limit.
- Each resulting leaf line becomes one `ContinuousBlock` (slope snapped to a
  broadcast standard), and `build_macro_edl` applies the per-act block speed
  via `_speed_at`. This replaces the greedy `discontinuity_threshold_sec`
  clustering with a principled, inlier-driven segmentation.
