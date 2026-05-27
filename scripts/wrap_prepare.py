#!/usr/bin/env python3
"""Wrapper to run src/data/prepare.py as a script."""
import runpy
from pathlib import Path

runpy.run_path(Path(__file__).resolve().parents[1] / 'src' / 'data' / 'prepare.py', run_name='__main__')
