# DubSync — Concrete Implementation Plan

> Grounded in the failure patterns documented in
> [`docs/CHAT_HISTORY_SUMMARY.md`](./CHAT_HISTORY_SUMMARY.md) and the actual code at
> `checkpoint-v2.5-pre-pure-audio` (+ the dead-code/metrics fixes in PR #1).
>
> **Guiding principle from the history:** every past "fix" fixed one benchmark and
> broke another, and each regression was only caught by manual re-watching. This plan
> therefore puts **regression testing first** and makes every change reversible via
> git checkpoints.

---

## 0. Goals & non-goals

**Goals**
1. Make every sync change provably non-regressive against the two known benchmarks
   (Monkie Kid = clean; Hero 108 = PAL + cuts + micro-trims).
2. Eliminate the five remaining defect classes the history identified.
3. Preserve the "wonderful results" behavior — never regress a working episode.

**Non-goals (explicitly out of scope for now)**
- Re-adding the reverted "Claude fixes" (chromaprint bootstrap, micro-DTW, EBU R128,
  KD-tree). These were reverted for good reason; only *re-introduce ideas* (e.g. the
  micro-fallback merger) behind tests, one at a time.
- New features (subtitle retiming, batch `SxxExx` matching, GUI). Queue separately.

---

## 1. Phase 1 — Regression test harness (highest leverage, do first)

**Why:** the meta-lesson of the entire transcript. No future change should ship without
a fast way to detect "fixed Hero 108, broke Monkie Kid."

### 1.1 Synthetic golden fixtures (`tests/fixtures/generate.py`)
Deterministic generator that produces a reference WAV/MKV + a target WAV/MKV with a
**known ground-truth EDL**. This works in CI with only `numpy` + `ffmpeg` (no real media).

Generate, per scenario, a shared "M&E bed + dialogue bursts" signal and mutate the target:

| Scenario | Mutation applied | Expected engine behavior |
| :--- | :--- | :--- |
| `clean_1x` | identity (same length) | 1 block, speed 1.0, 0 fallback |
| `pal_speed` | target = ref time-scaled by 24/25 | 1 block, speed ≈ 0.96, 0 fallback |
| `single_cut` | delete a 9s span from target | 2 blocks, 1 fallback of ≈9s |
| `micro_trim` | delete a 1.75s span | cut detected, re-anchored, 0 downstream drift |
| `extra_scene` | insert a 10s bumper in target | bumper trimmed, 0 drift after |
| `intro_gap` | shift target start by +4s (logo) | opening fallback ≈4s, no displaced dialogue |
| `black_frame` | identical black frame at 0:00 vs mid-file | no false anchor @ 0:00 |

Ground truth is emitted as JSON: list of `(ref_start, ref_end, tar_start, tar_end,
speed, type)`.

### 1.2 Test driver (`tests/test_sync.py`)
For each fixture, run `DubSyncPipeline.execute()` (or the engine directly, for speed)
and assert:

1. **Speed accuracy** — each block's `speed_factor` within `±0.01` of ground truth.
2. **Cut detection** — fallback segments' `(ref_start, ref_end)` within `±0.5s`.
3. **No false English** — in ground-truth-continuous regions, output contains **zero**
   fallback segments; in cut regions, **no** dub segments.
4. **No drift** — final rendered audio length == ref length within `±50ms`.
5. **No displaced speech** — for `intro_gap`/`black_frame`, first dub segment's
   `tar_start` matches ground truth (not `12.8s`, etc.).

### 1.3 Real-fixture convention (owner-supplied, gitignored)
Add `test_files/` to `.gitignore` (already ignored) and a manifest `test_files/manifest.json`:
```json
{ "monkie_kid": {"ref": "...", "tar": "...", "known_cuts": [[616.7, 626.2]]},
  "hero108_ep3": {"ref": "...", "tar": "...", "known_cuts": [[...]]} }
```
The same assertions run against real files when present, skipped otherwise.

**Acceptance:** `pytest -q` green on all synthetic fixtures; same harness runs in < 60s.

**Checkpoint:** `git tag checkpoint-v2.5-tests-green` before any code change.

---

## 2. Phase 2 — Cut-aware global slope calibration

**Why (history):** `calibrate_global_slope()` takes a plain **median** of all pairwise
intervals. Intervals that cross a real cut drag the estimate; small (1–2s) trims don't
get filtered by the `0.90 ≤ s ≤ 1.10` gate and pollute the median. The wrong global
speed then feeds `build_macro_edl`.

**File:** `dub_sync_engine/block_segmenter.py` → `calibrate_global_slope()`.

**Concrete change** — replace the plain median with a **length-weighted, cut-filtered,
broadcast-snapped estimator**:

```python
def calibrate_global_slope(self, matches):
    weights, slopes = [], []
    for i in range(len(matches) - 1):
        dr = matches[i+1].ref_time - matches[i].ref_time
        dt = matches[i+1].tar_time - matches[i].tar_time
        if dr < 3.0:                       # skip tiny intervals (frame jitter)
            continue
        s = dt / dr
        if not (0.94 <= s <= 1.06):        # discard intervals crossing real cuts
            continue
        slopes.append(s)
        weights.append(dr)                 # long continuous spans are more reliable
    if not slopes:
        return 1.0
    # weighted histogram peak (not arithmetic median) → robust to a minority of trims
    hist, edges = np.histogram(slopes, bins=200, range=(0.94, 1.06), weights=weights)
    raw = edges[np.argmax(hist)]
    return snap_to_broadcast_speed(raw)    # shared helper, see Phase 3
```

**Acceptance:** `pal_speed` and `single_cut` fixtures yield a global slope within
`±0.005` of `24/25`, whereas the *old* median (add a debug print in the test) visibly
under/over-estimates on the `micro_trim` + `single_cut` fixture.

---

## 3. Phase 3 — Strict broadcast-speed locking

**Why (history):** a continuous act was allowed to float to `1.0458x`, introducing a
~500ms end-of-episode drift. Continuous acts must snap to a broadcast standard, never
float to an arbitrary value.

**Files:** `config.py` (new shared helper), `block_segmenter.py`, `audio_splicer.py`.

**Concrete change:**

1. Add a shared, pure function (module-level in `config.py`):
```python
BROADCAST_STANDARDS = (1.0, 24/25, 25/24, 24/23.976, 23.976/24)  # 1.0, 0.96, 1.041667, 1.001001, 0.999

def snap_to_broadcast_speed(raw: float, tol: float = 0.006) -> float:
    for std in BROADCAST_STANDARDS:
        if abs(raw - std) <= tol:
            return std
    return max(0.90, min(1.10, raw))
```
2. Route **every** place that computes or clamps a speed through this helper:
   - `block_segmenter.cluster_into_blocks()` (already snaps per-block — reuse helper)
   - `block_segmenter.build_macro_edl()` Scenario A/C (`g_speed` already snapped; keep)
   - `audio_splicer.render_and_splice()` — replace `speed = max(0.90, min(1.10, speed))`
     with `speed = snap_to_broadcast_speed(speed)`.
3. Add a `--strict-speed` config flag (default on) so this is testable/disable-able.

**Acceptance:** synthetic `micro_trim` fixture shows no floating speeds; every rendered
segment's `speed_factor` ∈ `BROADCAST_STANDARDS`. Real Hero 108 run: no `1.045x` segment.

---

## 4. Phase 4 — Micro-chopping guard

**Why (history):** weak correlation peaks (≥0.35) inside silent cut zones produced
1.7s / 4.4s Arabic "blips" interleaved with 3–5s English islands. A single 30s cut
became 7 fragments.

**Files:** `consensus_engine.py`, `config.py`, `verifier_engine.py`.

**Concrete change:**

1. **Raise and centralize thresholds** in `consensus_engine.py`:
   - acoustic music gate `peak >= 0.40` → configurable `config.min_acoustic_peak` (default 0.50)
   - VAD gate `norm_peak >= 0.48` → `config.min_vad_peak` (default 0.55)
2. **Minimum dub-act duration** (`config.min_dub_act_sec`, default 5.0):
   post-process the EDL — any `dub` segment shorter than `min_dub_act_sec` that is
   sandwiched between `fallback` segments is reclassified as `fallback` (it's ambient
   noise inside a real cut), and adjacent `fallback`s are merged.
3. **Micro-fallback merger** (this is the one idea worth re-importing from the reverted
   "Claude fixes", done *behind tests* this time): in `verifier_engine.audit_and_heal_edl`,
   merge any fallback gap `< 0.5s` into its neighboring dub segment.

**Acceptance:** synthetic `single_cut`/`micro_trim` fixtures produce **exactly one**
contiguous fallback per real cut — no dub blips inside a cut, no fallback islands inside
continuous regions.

---

## 5. Phase 5 — Strengthen audio-gating of visual matches

**Why (history):** the single most recurring bug — a visual frame inside a cut zone
matched a frame *after* the cut (or a black frame at 0:00 matched mid-file), causing
displaced speech + duplicated English. The consensus engine already gates visuals on
acoustic confirmation, but the gate must be made airtight and configurable.

**Files:** `consensus_engine.py`, `config.py`.

**Concrete change:**

1. **Remove any high-confidence bypass.** Audit `discover_consensus_anchors()` STEP 3 so
   that *no* visual match is added without an acoustic anchor within the gate window.
   (The reverted history explicitly blamed an `or m.confidence >= 0.85` style bypass.)
2. **Make the gate explicit & configurable:**
   ```python
   # config
   acoustic_gate_window_sec: float = 4.0   # ±window around the acoustic anchor
   acoustic_gate_offset_sec: float = 2.0   # max allowed |Δoffset| between visual & acoustic
   min_acoustic_peak: float = 0.50
   ```
   Replace the hardcoded `<= 4.0 / <= 2.0 / >= 0.40` magic numbers.
3. **N-gram verification is already present** in `visual_anchors.match_anchors()`
   (seq_len ≥ 2 boost) — increase the minimum to ≥ 3 consecutive cuts as the history
   recommended, and surface `seq_len` in the anchor telemetry so the report shows it.
4. (Stretch) **Local gated search**: instead of the visual matcher comparing every ref
   anchor against every tar anchor globally (`O(N²)`), pre-cluster by acoustic offset
   and only compare within `±(acoustic_gate_window_sec)` — kills false matches *and*
   speeds up matching. Implement as an optional `gated=True` path.

**Acceptance:** `black_frame` and `intro_gap` fixtures produce **zero** anchors at 0:00
with wrong offsets; no dialogue displaced. Real Ep 2 run: no `12.8s` opening anchor.

---

## 6. Phase 6 — Verification & rollout

1. **Wire the closed-loop verifier to report truth** (already fixed in PR #1 — do not
   regress): metrics must be *measured*, not hardcoded.
2. **Add a `--qc` mode** that runs the verifier on an existing synced MKV and emits the
   per-segment drift table without re-rendering — lets the owner audit any output fast.
3. **Run the full matrix** before/after each phase and record results in
   `docs/RESULTS.md` (a table: fixture × phase × drift/fallback counts). This turns the
   "it's worse now" loop into a glanceable diff.

---

## 7. Sequencing & checkpoints

| Step | Change | Gate | Checkpoint tag |
| :--- | :--- | :--- | :--- |
| 1 | Test harness (synthetic + manifest) | all green | `checkpoint-tests-green` |
| 2 | Cut-aware slope calibration | fixtures + no Monkie/Hero regression | `checkpoint-slope` |
| 3 | Broadcast speed locking | no `1.045x`, fixtures green | `checkpoint-speed-lock` |
| 4 | Micro-chopping guard | one fallback per cut | `checkpoint-micro-guard` |
| 5 | Audio-gating of visuals | black-frame/intro fixtures green | `checkpoint-gating` |
| 6 | `--qc` mode + RESULTS.md | — | `checkpoint-qc` |

Each phase lands as its own commit on a feature branch and is merged only after the
regression matrix is clean. Reverting any phase = `git revert` of one commit.

---

## 8. Risks & mitigations

| Risk | Mitigation |
| :--- | :--- |
| Stricter thresholds (Ph.4/5) drop too many *real* anchors (like the global-median filter did) | Thresholds are configurable; regression fixtures encode the "must-keep" anchors; test real episodes in manifest |
| Speed locking hides a genuine non-standard master | `snap_to_broadcast_speed` falls back to clamped raw value if no standard matches within tolerance |
| Gated local search (Ph.5 stretch) changes matching semantics | Ship as opt-in `gated=True`; keep global path default until fixtures confirm parity |
| No real media in this environment | Synthetic fixtures cover every failure mode; real-fixture manifest runs on owner's machine |
