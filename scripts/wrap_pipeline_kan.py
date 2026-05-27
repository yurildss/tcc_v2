#!/usr/bin/env python3
"""Wrapper to run src/models/pipeline_kan.py as a script."""
import runpy
from pathlib import Path

runpy.run_path(Path(__file__).resolve().parents[1] / 'src' / 'models' / 'pipeline_kan.py', run_name='__main__')
