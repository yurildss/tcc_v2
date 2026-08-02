#!/usr/bin/env python3

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

runpy.run_module(
    "src.models.pipeline_kan",
    run_name="__main__",
)
