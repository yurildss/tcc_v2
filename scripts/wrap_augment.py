#!/usr/bin/env python3
"""Wrapper to run src/data/augment.py as a script."""
import runpy
from pathlib import Path

runpy.run_path(Path(__file__).resolve().parents[1] / 'src' / 'data' / 'augment.py', run_name='__main__')
