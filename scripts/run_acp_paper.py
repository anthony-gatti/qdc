#!/usr/bin/env python3
"""Compatibility wrapper for the clean two-node ACP paper milestone."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_module("experiments.run_single_pair_paper", run_name="__main__")
