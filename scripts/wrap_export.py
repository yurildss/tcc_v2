#!/usr/bin/env python3
"""Wrapper to run src/export/export_all_esp32.py (or other export scripts)."""
import runpy
from pathlib import Path

# prefer specific export script if present
export_script = Path(__file__).resolve().parents[1] / 'src' / 'export' / 'export_all_esp32.py'
if export_script.exists():
    runpy.run_path(export_script, run_name='__main__')
else:
    runpy.run_path(Path(__file__).resolve().parents[1] / 'src' / 'export' / 'export_kan_cpp.py', run_name='__main__')
