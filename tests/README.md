# Test harness

Everything in [`../REVIEW.md`](../REVIEW.md) is reproducible from this directory.

There are two layers:

| Layer | Command | Needs media? | Purpose |
|:--|:--|:--|:--|
| Unit | `python3 -m pytest tests/ -q` | no | pins the EDL / slope math that is correct |
| End-to-end | `./tests/run_e2e.sh` | yes | measures real output against known ground truth |

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

FFmpeg is required. A system `ffmpeg` on `PATH` is preferred; otherwise the harness
falls back to the bundled `imageio-ffmpeg` binary (note that `dub_sync_engine.media_probe`
resolves FFmpeg at *import* time, so a missing FFmpeg makes even `--help` fail — see
REVIEW.md F9).

## End-to-end

```bash
python3 tests/synthetic_media.py     # build the fixture -> tests/_media/
./tests/run_e2e.sh                   # run all strategies -> tests/_results/
```

`synthetic_media.py` builds a 60 s reference with scene cuts at 12/24/36/48 s, then derives
the dub by **removing ref[24 s, 30 s]** and **slowing to 0.96×**, letterboxed to a different
picture geometry. Ground truth is recorded in `tests/_media/ground_truth.json`:

```
tar_time = (ref_time - 6.0 if ref_time >= 30.0 else ref_time) / 0.96
```

`verify_output.py` re-measures the rendered MKV from scratch — it never reads the tool's own
report:

1. output duration vs reference duration;
2. per-window sync error by envelope cross-correlation against the *original* reference audio
   (lag `0 ms` = perfectly in sync);
3. per-region classification as real dub vs English M&E bridge, via the spectral notch the
   `vocal_filtered` fallback leaves at 1200 Hz / 2400 Hz.

It then prints what the tool's own audit claimed, so the two can be compared directly.

### Baseline

`f84fdcc` (as first reviewed) — every run reported `99.2% frames verified (Mean Error: 24.5ms)`:

| Strategy | Mean abs. error | Windows within ±120 ms |
|:--|--:|--:|
| `--matcher visual` | 466 ms | 89.7 % |
| default (`auto`/`hybrid`) | 3 133 ms | 3.4 % |
| `--strategy dtw` | 4 003 ms | 0.0 % |

Current branch, after the F1/F2/F5 fixes (the audit now measures, and agrees with this checker
to within ~1 point of coverage):

| Strategy | Anchors | Omission placed at (truth 24–30 s) | Self-reported | Independent |
|:--|--:|:--|--:|--:|
| `--matcher visual` | 4 | 30.03–36.0 ❌ | 1 466 ms / 76.7 % | 403 ms / 85.7 % |
| default (`auto`/`hybrid`) | 11 | 23.99–30.0 ✅ | 204 ms / 88.1 % | 558 ms / 89.3 % |
| `--strategy dtw` | 11 | not detected ❌ | 3 189 ms / 29.4 % | 1 880 ms / 3.6 % |

## Known-defect specs

`test_known_defects.py` encodes findings F1–F7 and F13 as `xfail` tests. They document
behaviour the engine *should* have; they currently fail for exactly the documented reason and
will flip to XPASS once fixed (remove the marker then).

```bash
python3 -m pytest tests/test_known_defects.py -q -rX      # summary
python3 -m pytest tests/test_known_defects.py --runxfail  # see each failure message
```

## Caveats

* The synthetic dub is derived *from* the reference (identical M&E bed), which flatters the
  tool — real dubs with different music/SFX mixes should be expected to do worse.
* The synthetic "speech" is band-limited noise, not speech, so the Silero VAD layer
  contributes nothing here. Its plumbing is exercised; its effectiveness is not.
* `tests/_media/` and `tests/_results/` are gitignored.
