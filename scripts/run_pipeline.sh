#!/usr/bin/env bash
set -euo pipefail

# Mostra quanto tempo cada etapa levou (para achar gargalos)
INICIO_TOTAL=$SECONDS
etapa_ini=$SECONDS
tempo() { echo "   ⏱  $1: $(( (SECONDS - etapa_ini) / 60 )) min $(( (SECONDS - etapa_ini) % 60 )) s"; etapa_ini=$SECONDS; }


echo "1/7 - Ingest raw data"
python3 scripts/wrap_ingest.py
tempo "ingest"

echo "2/7 - Prepare: cortar batimentos"
python3 scripts/wrap_prepare.py
tempo "prepare"

echo "3/7 - Split inter-paciente (antes do augmentation)"
python3 scripts/wrap_split.py
tempo "split"

echo "4/7 - Augment (somente treino)"
python3 scripts/wrap_augment.py
tempo "augment"

echo "5/7 - Extract + build pipeline"
python3 scripts/wrap_pipeline_kan.py
tempo "extração de features"

echo "6/7 - Train (KAN)"
python3 scripts/wrap_train.py
tempo "treino KAN"

echo "=================================================="
echo "6.5/7 - Treinar e Avaliar Benchmark (CNN1D, LSTM, RF, XGBoost)"
echo "=================================================="
python3 -m src.models.benchmark
tempo "benchmark"

echo "7/7 - Export headers"
python3 scripts/wrap_export.py
tempo "export"

echo "Pipeline finalizado em $(( (SECONDS - INICIO_TOTAL) / 60 )) min."
