#!/usr/bin/env python3
"""Wrapper to run src/data/ingest.py as a script."""
import runpy
from pathlib import Path

runpy.run_path(Path(__file__).resolve().parents[1] / 'src' / 'data' / 'ingest.py', run_name='__main__')
