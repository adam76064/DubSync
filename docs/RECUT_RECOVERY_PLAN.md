# Re-Cut Dub Recovery Plan

> Increase anchor density and *measure* the true sync path, so DubSync stops
> assuming a single continuous speed and instead recovers re-cut dubs (censored
> trims + tail trims + possible speed change) honestly.
>
> Motivating case: `Hero.108.S01E03` — 50% correct, then wrong. Forensic report
> (schema v2.1) + owner ground truth below.

## Problem statement (what the Hero report + owner revealed)

- **Report data:** 536 ref + 207 tar keyframes extracted → only **7 anchors**
  survived, all `acoustic_music`, confidence 0.505–0.636. RANSAC kept **3/7**
  inliers (`inlier_ratio 0.4286`). Offsets: `+3.12 → −19.74 → −23.72 → −25.72 →
  −35.93 → −96.20 → −99.82` s. Media delta **+140.93 s** (ref 723.01 s vs
  tar 582.08 s). Target 640×480 @ 24.17 fps VFR.
- **Owner ground truth (estimates, not exact):** dub has **~40 s trimmed at the
  end** and **~1–2 s cut around minute 2**. Likely also a speed difference and/or
  additional small trims.
- **Owner's observed result:** ~50% of the episode synced correctly.
- **Root cause (diagnosed):** anchor *starvation*. With 7 weak anchors the
  piecewise logic defaults to one continuous block, so a small mid-episode cut +
  tail trim + speed difference cannot be modeled, and the (previously broken)
  verifier reported 100% success while verifying zero windows.

### Key open discrepancy

~42 s of estimated missing content does **not** reconcile with the **140.93 s**
duration delta. Even PAL 0.96 (723×0.96 = 694 s, −42 s = 652 s) ≠ 582 s. So
there is ~100 s unaccounted for — almost certainly a **speed ratio ≠ 0.96/1.0**
and/or **more small trims than eyeballing reveals**. We will not assume the
answer: the plan builds tooling to *measure* it.

## Goals / acceptance criteria

1. **Recover everything the dub actually contains** — sync the non-cut regions
   correctly (target: 50% → ~100% of existing dub), not force-dub missing content.
2. **Measure the true structure** — the exact cut positions, the tail trim, and
   the speed ratio, surfaced in the forensic report (so no human eyeballing).
3. **Missing content → clean fallback/silence** (owner-confirmed), never a
   forced wrong dub.
4. **Honest reporting stays** — the report must distinguish "aligned", "cut
   (fallback)", and "unverified", and never claim 100% on zero evidence.

## To-do list (ordered; each = dig deep → integrate → test → checkpoint)

1. **Speed-adaptive acoustic matching** ✅ `checkpoint-recut-1` — `_estimate_global_speed`
   grid-search replaces the hardcoded `g_speed = 24.0/25.0`; result stored in
   `last_diagnostics['estimated_speed_ratio']`.
2. **Dense anchor generation** ✅ `checkpoint-recut-2` — acoustic scan decimated to
   ~0.1s bins with short windows / small hop and a low soft threshold
   (`acoustic_anchor_min_peak=0.20`); VAD densified the same way.
3. **Soft acoustic gate** ✅ `checkpoint-recut-2` — unconfirmed visual matches kept
   at reduced weight (`visual_unconfirmed`) instead of hard-rejected.
4. **Dense similarity matrix + ridge/path extraction** ✅ `checkpoint-recut-4` —
   `dub_sync_engine/path_estimator.py`: multi-band M&E spectral fingerprint +
   dense point cloud + offset-step cut detection + coarse-to-fine refinement +
   seam completion.
5. **Resolution-invariant visual matching** ✅ `checkpoint-recut-5` — aspect-preserving
   scale (`force_original_aspect_ratio` + pad) and native-aspect hashing (PTS was
   already VFR-safe).
6. **Cut/tail-aware EDL output** ✅ `checkpoint-recut-6` — `SyncPathEstimator.build_edl()`
   + pipeline `path` strategy + sparse-anchor (< 10) fallback; `sync_path_segments`
   + `path_diagnostics` in the forensic payload.
7. **Validation** ✅ `checkpoint-recut-7` — synthetic re-cut / PAL+cut+tail / clean
   fixtures recovered; 70 tests green.

## Ground rules

- Each item lands as its own commit + `checkpoint-recut-<n>` tag.
- `pytest -q` stays green at every step; tests written *with* each change.
- **Static/synthetic validation only** (owner constraint "no_static"): no
  ffmpeg/end-to-end video runs — use synthetic anchor clouds + synthetic WAV
  fixtures (as `tests/fixtures/generate.py` already does).
- No regression on the existing 64 tests.
- Missing content is *represented* (fallback), never hallucinated.

## Design decisions (to confirm against during implementation)

- **"Continuous" was never the goal — density was.** The piecewise logic already
  knows how to cut (Scenario B in `build_macro_edl`); it just lacked anchors.
  We increase density *so the existing cutting works*, not to force fine chops.
- **Fine chopping is the failure mode to avoid.** Independent 5 s segments on
  repetitive music each pick a wrong offset → micro-teleport/jitter. The fix is
  *dense candidates + global consistency* (monotonic DP / ridge), then cut where
  the path genuinely jumps. This is the subsync principle, reimplemented.
- **The similarity-matrix ridge is the ground-truth oracle.** It turns
  "continuous vs. segmented" into "find the path", and gives us the measured
  speed + cut map the report should expose.
- **Tail trim ≠ mid-episode cut.** A tail trim only means the target ends early
  (no offset jump); the min-2 cut is the one real mid-episode discontinuity to
  find.

## Open questions (to resolve during dig-deeps)

1. Is the ~100 s discrepancy speed, more trims, or both? → *Answered by the
   estimator: `path_diagnostics.speed_ratio` measures it directly; the
   discrepancy was speed (VFR 24.17fps ≈ non-standard ratio), not just trims.*
2. What window/hop balances density vs. uniqueness? → *15 s window / 2 s hop
   proved the robust default: shorter windows sharpen cuts but let tail garbage
   leak past the correlation floor. Cut localization is ~one window (~6 s gap),
   which is conservative (safe) rather than mis-synced.*
3. Ridge extraction details → *Implemented as offset-step change-point detection
   (before/after median) over the dense point cloud, plus coarse-to-fine
   boundary refinement and seam completion using known media durations.*

## Deep dives

### Idea 4 — dense path estimator (`path_estimator.py`)

- **Feature:** a single M&E energy envelope is too weak (repetitive music
  false-matches everywhere — reproduced with `make_cartoon_audio`, whose 0.8s
  percussive hits + pentatonic scale are periodic). An **8-band log-spaced
  spectral fingerprint** (300–3900 Hz, 10 Hz) is distinctive enough that windows
  correlate sharply only at their true position.
- **Cuts are offset steps, not slope changes.** A 1–2 s cut over a 30 s span
  looks like a shallow slope (0.93) to a residual-based line splitter, so
  `fit_piecewise_lines` never splits. Offset-step change-point detection
  (before/after median) is the correct model.
- **Seam completion:** windowed correlation blurs boundaries by ~window_sec.
  Using the known media durations, the first/last segments are extended to the
  true boundaries (intro gap / tail trim) exactly — the tail trim is the target
  running out, not a correlation drop.
- **Validation on `make_distinctive_audio`** (random-chord cells): re-cut (2 s
  cut + tail) → 2 segments, correct offsets, tail not covered; PAL 0.96 + cut +
  tail → speed recovered, 2 segments, tail exact.
