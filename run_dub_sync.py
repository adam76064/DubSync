#!/usr/bin/env python3
"""
DubSync Pro: Studio-Grade Cartoon & Anime Dub Audio Synchronizer
Root execution entrypoint.
"""

import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure package directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dub_sync_engine.cli import main

if __name__ == "__main__":
    main()
