#!/usr/bin/env bash
# End-to-end check: run every sync strategy over the synthetic fixture and
# measure each result independently of the tool's own reporting.
#
#   ./tests/run_e2e.sh
#
# Requires the fixture to exist:  python3 tests/synthetic_media.py
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDIA="$ROOT/tests/_media"
RESULTS="$ROOT/tests/_results"

[ -f "$MEDIA/ref.mp4" ] || { echo "fixture missing - run: python3 tests/synthetic_media.py" >&2; exit 1; }

mkdir -p "$RESULTS"
rm -rf "$RESULTS"/*

run_strategy () {
    local name="$1"; shift
    local out="$RESULTS/$name.mkv"
    echo "=================================================================="
    echo "  $name"
    echo "=================================================================="
    ( cd "$ROOT" && python3 run_dub_sync.py \
        "$MEDIA/ref.mp4" "$MEDIA/tar.mp4" "$out" "$@" < /dev/null ) \
        | grep -E "\[OK\]|-> Block|->\s" || true
    python3 "$ROOT/tests/verify_output.py" "$out" "$MEDIA" "$name" || true
    echo
}

run_strategy visual   --preset studio --matcher visual --strategy hybrid
run_strategy default  --preset studio --matcher auto   --strategy hybrid
run_strategy dtw      --preset studio --matcher auto   --strategy dtw

echo "Artifacts (mkv + forensic reports) are in: $RESULTS"
