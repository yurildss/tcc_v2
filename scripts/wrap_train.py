#!/usr/bin/env python3
"""Wrapper to run src/models/train_custom.py as a script."""
import runpy
from pathlib import Path

runpy.run_path(Path(__file__).resolve().parents[1] / 'src' / 'models' / 'train_custom.py', run_name='__main__')
