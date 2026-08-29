"""Phase 3: broadcast speed locking helper."""
from dub_sync_engine.config import snap_to_broadcast_speed, BROADCAST_STANDARDS, DubSyncConfig


def test_snap_exact_standards():
    assert snap_to_broadcast_speed(1.0) == 1.0
    assert snap_to_broadcast_speed(24.0 / 25.0) == round(24.0 / 25.0, 6)   # 0.96
    assert snap_to_broadcast_speed(25.0 / 24.0) == round(25.0 / 24.0, 6)   # 1.041667
    # NTSC pulldown (1.001001) folds into 1.0 (near-unity, historical behavior).
    assert snap_to_broadcast_speed(24.0 / 23.976) == 1.0


def test_snap_within_tolerance():
    # 1.0458x (the historical drift culprit) is NOT a standard -> clamped, not snapped.
    v = snap_to_broadcast_speed(1.0458)
    assert v not in BROADCAST_STANDARDS
    assert 0.90 <= v <= 1.10
    # A near-PAL value within tolerance snaps to PAL (absorbs frame quantization noise).
    assert snap_to_broadcast_speed(0.9630) == round(24.0 / 25.0, 6)


def test_snap_clamps_out_of_range():
    assert snap_to_broadcast_speed(0.5) == 0.90
    assert snap_to_broadcast_speed(1.5) == 1.10


def test_config_has_new_fields():
    c = DubSyncConfig()
    for f in ("min_acoustic_peak", "min_vad_peak", "min_dub_act_sec",
              "micro_fallback_merge_sec", "acoustic_gate_window_sec",
              "acoustic_gate_offset_sec", "strict_speed"):
        assert hasattr(c, f), f"missing config field {f}"
