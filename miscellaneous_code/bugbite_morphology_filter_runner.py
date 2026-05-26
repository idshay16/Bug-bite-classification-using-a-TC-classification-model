#!/usr/bin/env python3
"""Tiny harness for a quick dry-run of the bug bite filter."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("bugbite_morphology_filter.py")

cmd = [
    sys.executable,
    str(SCRIPT),
    "--input", "cyclone_dataset/*",
    "--out", "bugbite_filter_output",
    "--dry-run",
    "--max-files", "5",
]

print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd))
