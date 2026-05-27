#!/usr/bin/env bash
set -euo pipefail


echo "1/7 - Ingest raw data"
python3 scripts/wrap_ingest.py

echo "2/7 - Prepare: cortar batimentos"
python3 scripts/wrap_prepare.py

echo "3/7 - Augment"
python3 scripts/wrap_augment.py

echo "4/7 - Split"
python3 scripts/wrap_split.py

echo "5/7 - Extract + build pipeline"
python3 scripts/wrap_pipeline_kan.py

echo "6/7 - Train (KAN)"
python3 scripts/wrap_train.py

echo "7/7 - Export headers"
python3 scripts/wrap_export.py

echo "Pipeline finalizado."
