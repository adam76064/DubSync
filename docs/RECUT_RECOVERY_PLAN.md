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

1. **Speed-adaptive acoustic matching** — remove the hardcoded `g_speed =
   24.0/25.0` in `consensus_engine` STEP 1; resample the target over a grid of
   broadcast/ratio candidates (reuse the drift profiler's `CANDIDATE_SPEED_RATIOS`
   idea) and correlate per ratio. Expected to recover most of the acoustic
   anchors *and* report the measured speed ratio.
2. **Dense anchor generation** — shorten the acoustic scan window (12 s → 2–5 s)
   and hop (10 s → 0.5–1 s); keep candidates above a *low* threshold with soft
   scores instead of hard `min_acoustic_peak = 0.50` rejection. Same treatment
   for VAD (align dense probability curves, not just `min_vad_peak = 0.55` bursts).
3. **Soft acoustic gate** — replace the hard visual-reject gate (`acoustic_gate_window_sec`
   / `acoustic_gate_offset_sec`) with a continuous confirmation *weight*; let
   RANSAC + the monotonic DP lattice drop bad matches instead of discarding all
   unconfirmed visuals.
4. **Dense similarity matrix + ridge/path extraction** — the core "measure the
   truth" piece. Build a coarse ref×tar similarity matrix (~1 s resolution) from
   spectral/envelope/VAD features; extract ridge segments (= continuous alignment
   paths) and the jumps between them (= cuts). This reveals cut positions, tail
   trim, *and* speed in one shot, resolving the 140.93 s discrepancy.
5. **Resolution-invariant visual matching** — downscale both sources to a
   canonical resolution (e.g. 320×180) before hashing; use the target's actual
   PTS (not frame index) for VFR. Restores the most trustworthy anchor type for
   pinning non-cut regions.
6. **Cut/tail-aware EDL output** — wire the measured path into the block
   segmentation so a mid-episode cut + tail trim produce multiple blocks (and
   fallback segments), not one continuous block. Keep the existing tail-trim
   logic; add explicit cut-step handling.
7. **Validation** — synthetic fixtures modeling the Hero structure (1–2 s cut at
   min 2, ~40 s tail trim, optional PAL/NTSC speed) must be recovered: correct
   segment count, cut position within tolerance, measured speed, and fallback
   spans only where content is genuinely absent. Full suite green at every step.

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

1. Is the ~100 s discrepancy speed, more trims, or both? (The matrix/ridge
   measurement answers this — no manual guess.)
2. What exact acoustic/VAD window size + hop best balances density vs. uniqueness
   for cartoon/music-heavy dubs? (Prototype on synthetic fixtures first.)
3. Ridge extraction details: 1-D vs 2-D peaks, minimum segment length, jump
   penalty, and how to surface multiple ridges as "cuts" in the report.

## Deep dives

*(Filled in per idea during implementation, as in `SUBSYNC_INTEGRATION_PLAN.md`.)*
