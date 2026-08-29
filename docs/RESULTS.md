# DubSync — Implementation Results & Regression Matrix

> Generated as part of executing
> [`docs/IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md). Each phase below is
> checked off only after its acceptance gate passes and the synthetic regression
> matrix is green. See `docs/CHAT_HISTORY_SUMMARY.md` for the failure patterns
> this work targets.

---

## Regression matrix (synthetic fixtures)

Fixture → defect class it exercises, and the current (post-fix) engine behavior.

| Fixture | Defect class | Result |
| :--- | :--- | :--- |
| `clean_1x` | none (baseline) | ✅ 1 block, speed 1.0, 0 fallback |
| `pal_speed` | PAL 25→24fps | ✅ 1 block, speed snapped to `24/25` (0.96) |
| `single_cut` | 22–50s censored cut | ✅ 2 blocks, exactly 1 fallback ≈9s at 300–309 |
| `micro_trim` | 1.75–2.3s micro-trim | ✅ cut detected, re-anchored, 0 downstream drift |
| `extra_scene` | inserted bumper | ✅ bumper trimmed, 0 drift after |
| `intro_gap` | +4s logo gap | ✅ opening fallback ≈4s, no displaced dialogue |
| `black_frame` | false 0:00 anchor | ✅ false (0.0, 12.8) anchor rejected; real anchor at 32s |

Full matrix: **32 tests pass** (`pytest -q`), < 6s.

---

## Phase checklist

### Phase 1 — Regression test harness ✅
- `tests/fixtures/generate.py`: deterministic synthetic scenarios + audio builder.
- `tests/test_config.py`, `test_block_segmenter.py`, `test_audio_splicer.py`,
  `test_verifier.py`, `test_visual_anchors.py`.
- `tests/fixtures/manifest.example.json` for owner-supplied real media
  (`test_files/` is gitignored).
- Gate: `pytest -q` green, fast. **PASS.**

### Phase 2 — Cut-aware global slope calibration ✅
`block_segmenter.calibrate_global_slope()` now uses a length-weighted,
cut-filtered histogram-peak estimator (skips `dr < 3.0s` intervals, discards
intervals outside `0.94–1.06`, weights by span) and snaps to a broadcast
standard. Gate: `pal_speed`/`single_cut` slope within ±0.005. **PASS.**

### Phase 3 — Strict broadcast-speed locking ✅
- `config.py`: `BROADCAST_STANDARDS`, `snap_to_broadcast_speed()`, `strict_speed`.
- `block_segmenter` and `audio_splicer` route speed through the helper; CLI gains
  `--strict-speed` / `--no-strict-speed`.
- Gate: no floating speeds; every segment's `speed_factor` ∈ standards. **PASS.**

### Phase 4 — Micro-chopping guard ✅
- `config.py`: `min_acoustic_peak` (0.50), `min_vad_peak` (0.55),
  `min_dub_act_sec` (5.0), `micro_fallback_merge_sec` (0.5).
- `consensus_engine.py`: thresholds sourced from config.
- `audio_splicer.sanitize_edl()`: reclassify tiny dub fragments between fallbacks,
  merge adjacent fallbacks, absorb micro-fallbacks with contiguous target audio,
  merge contiguous matching-speed dubs. Used by `block_segmenter` and `verifier`.
- Gate: exactly one contiguous fallback per real cut. **PASS.**

### Phase 5 — Strengthen audio-gating of visual matches ✅
- `consensus_engine.py`: acoustic gate uses `acoustic_gate_window_sec` /
  `acoustic_gate_offset_sec`; no high-confidence bypass — isolated visual matches
  with no acoustic support are always rejected.
- `visual_anchors.py`: `AnchorMatch.seq_len` surfaced; tiered N-gram boost
  (`seq_len≥2` +0.10, `seq_len≥3` +0.20) so consecutive-cut chains win.
- Gate: `black_frame`/`intro_gap` produce zero wrong-offset anchors at 0:00. **PASS.**

### Phase 6 — Verification & rollout ✅
- Verifier metrics remain *measured* (not hardcoded) — PR #1 behavior preserved.
- `--qc` mode added (`cli.py` + `verifier.measure_drift_profile`): estimates the
  global broadcast speed ratio (scale-aware search over known standards) and the
  residual per-window drift, without re-rendering.
- This `RESULTS.md`.

---

## Notable bugs found & fixed during the loop

1. **Frozen-anchor false-drop** — the near-duplicate cleanup in
   `block_segmenter` dropped the legitimate first anchor after a real cut (a
   "frozen" target time that is actually correct), placing the `single_cut`
   fallback one block late (315–324 instead of 300–309). Fixed by only dropping a
   frozen anchor when it is an *offset outlier* (neighbors agree, it deviates).
2. **`sanitize_edl` gap left behind** — absorbing a micro-fallback dropped it but
   left a 0.3s ref gap, so surrounding dubs didn't merge. Fixed by merging prev+next
   across the absorbed fallback.
3. **`_normalized_correlation` lag math** — lag was measured against the *search
   window start*, which is wrong when clamped at file boundaries (intro gaps).
   Fixed to measure lag relative to the expected target position.
4. **`--qc` scale detection** — a fixed-window cross-correlation smears under a
   4% PAL speed mismatch. Fixed with a scale-aware search over candidate broadcast
   ratios before per-window offset measurement.
5. **Fixture realism** — the first synthetic "cartoon" bed (band-limited noise) was
   too white to produce sharp correlation peaks; replaced with a tonal, rhythmic,
   melodic bed that is distinctive in time (and thus meaningful for the verifier).

---

## Remaining work (out of scope, as agreed)

- Drop-list item C9 (golden-test harness driving `DubSyncPipeline.execute()` on
  real media) — **explicitly excluded** by the owner.
- Re-introducing any of the reverted "Claude fixes" (chromaprint bootstrap,
  micro-DTW, EBU R128, KD-tree) — kept out; only the micro-fallback merger idea was
  re-imported, behind tests.
- Subtitle retiming, batch `SxxExx` matching, GUI — queued separately.
