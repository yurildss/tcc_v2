#!/usr/bin/env python3

import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_module(
    "src.models.train_custom",
    run_name="__main__",
)
