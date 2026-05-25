# =============================================================================
# gerar_dataset_h.py
# =============================================================================
# Gera ArduinoCode/dataset.h com features JA computadas e escalonadas
# pelo Python — usando o mesmo scaler.pkl do treino.
#
# Usa load_signal_with_structure() + extract_features() do
# features_estruturais.py (46 features, anotacoes do CSV).
#
# O ESP32 usa esse arquivo diretamente no predict_from_array(),
# sem nenhuma extracao de features local.
#
# Uso:
#   python gerar_dataset_h.py
#   python gerar_dataset_h.py --split test --per_class 50 --seed 42
# =============================================================================

import os
import random
import argparse
import textwrap
import numpy as np
import joblib

from features_estruturais import (
    load_signal_with_structure,
    extract_features,
    FS,
)

# =============================================================================
# ARGUMENTOS
# Compativel com Jupyter (exec/run) e linha de comando
# Para alterar os valores dentro do Jupyter, edite a secao "Modo Jupyter"
# =============================================================================

def _is_jupyter():
    try:
        return get_ipython().__class__.__name__ in (
            "ZMQInteractiveShell", "TerminalInteractiveShell"
        )
    except NameError:
        return False

if _is_jupyter():
    # ---- Altere aqui os valores quando rodar no Jupyter ----
    class args:
        dataset   = "dataset_split/test"   # split a usar: train, val ou test
        per_class = 100                     # amostras por classe
        seed      = 42
        output    = "ArduinoCode/dataset.h"
    print("Modo Jupyter — valores em uso:")
    print(f"  dataset   = {args.dataset}")
    print(f"  per_class = {args.per_class}")
    print(f"  output    = {args.output}")
else:
    parser = argparse.ArgumentParser(
        description="Gera dataset.h para ESP32 com features pre-computadas"
    )
    parser.add_argument("--dataset",   default="dataset_split/test")
    parser.add_argument("--per_class", type=int, default=50)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default="ArduinoCode/dataset.h")
    args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# =============================================================================
# CARREGA SCALER E ENCODER
# =============================================================================
SCALER_PATH  = "ArduinoCode/scaler.pkl"
ENCODER_PATH = "ArduinoCode/encoder.pkl"

if not os.path.exists(SCALER_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        "scaler.pkl ou encoder.pkl nao encontrado em ArduinoCode/.\n"
        "Execute kolmo_corrigido.py primeiro."
    )

scaler      = joblib.load(SCALER_PATH)
encoder     = joblib.load(ENCODER_PATH)
class_names = list(encoder.classes_)
num_classes = len(class_names)

print(f"Scaler  carregado: {scaler.n_features_in_} features")
print(f"Encoder carregado: {num_classes} classes -> {class_names}")

assert scaler.n_features_in_ == 46, (
    f"scaler.pkl tem {scaler.n_features_in_} features, esperado 46. "
    f"Regere com kolmo_corrigido.py atualizado."
)

# =============================================================================
# CARREGA E PROCESSA O DATASET
# Usa a MESMA funcao extract_features() do treino — garantia de consistencia
# =============================================================================
def load_dataset(dataset_path, per_class):
    classes = sorted([
        c for c in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, c))
    ])

    X_raw, y_raw = [], []
    print(f"\nCarregando {per_class} amostras por classe de '{dataset_path}'...")

    for label_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith(".csv")]
        random.shuffle(files)

        loaded = 0
        skipped = 0
        for fname in files:
            if loaded >= per_class:
                break
            data = load_signal_with_structure(os.path.join(class_path, fname))
            if data is None:
                skipped += 1
                continue
            try:
                feat = extract_features(data, FS)
                X_raw.append(feat)
                y_raw.append(label_idx)
                loaded += 1
            except Exception:
                skipped += 1

        print(f"  [{label_idx}] {class_name:>3s}: {loaded} carregadas"
              + (f"  ({skipped} ignoradas)" if skipped else ""))

    return (np.array(X_raw, dtype=np.float32),
            np.array(y_raw, dtype=np.int32),
            classes)


X_raw, y_raw, classes = load_dataset(args.dataset, args.per_class)
n_samples  = len(X_raw)
n_features = X_raw.shape[1]
print(f"\nTotal: {n_samples} amostras x {n_features} features")

# Escalonamento com o MESMO scaler do treino
X_scaled = scaler.transform(X_raw).astype(np.float32)
print("Escalonamento aplicado.")

# =============================================================================
# ESTIMATIVA DE TAMANHO NA FLASH
# =============================================================================
total_kb = (n_samples * n_features * 4 + n_samples * 4) / 1024
print(f"\nEstimativa de tamanho na flash: ~{total_kb:.1f} KB")

LIMIT_KB = 256
if total_kb > LIMIT_KB:
    max_pc = int(LIMIT_KB * 1024 / (n_features * 4 * num_classes))
    print(f"  AVISO: excede {LIMIT_KB} KB. Reduza --per_class para ~{max_pc}")
else:
    print(f"  OK — dentro do limite de {LIMIT_KB} KB.")

# =============================================================================
# GERA O dataset.h
# =============================================================================
os.makedirs(os.path.dirname(args.output), exist_ok=True)
class_map = ", ".join([f"{i}={c}" for i, c in enumerate(classes)])

with open(args.output, "w") as f:
    f.write(textwrap.dedent(f"""\
        // =====================================================================
        // dataset.h — Features pre-computadas pelo Python (46 features)
        // Gerado por : gerar_dataset_h.py
        // Dataset    : {args.dataset}
        // Amostras   : {n_samples} ({args.per_class} por classe, seed={args.seed})
        // Features   : {n_features} (StandardScaler do treino aplicado)
        // Classes    : {class_map}
        // Flash      : ~{total_kb:.1f} KB
        //
        // USO NO ESP32:
        //   #include "dataset.h"
        //   float buf[NUM_FEATURES];
        //   for (int i = 0; i < NUM_SAMPLES; i++) {{
        //       for (int j = 0; j < NUM_FEATURES; j++)
        //           buf[j] = pgm_read_float_near(&dataset[i][j]);
        //       int pred = predict_from_array(buf);
        //       bool ok  = (pred == (int)pgm_read_dword_near(&labels[i]));
        //   }}
        // =====================================================================

        #ifndef DATASET_H
        #define DATASET_H

        #include <pgmspace.h>

        #define NUM_SAMPLES  {n_samples}
        #define NUM_FEATURES {n_features}
        #define NUM_CLASSES  {num_classes}

        // Classes: {class_map}
        const char* CLASS_NAMES_DS[] = {{
    """))

    for c in classes:
        f.write(f'    "{c}",\n')
    f.write("};\n\n")

    f.write("// Features escalonadas — prontas para predict_from_array()\n")
    f.write(f"const float dataset[NUM_SAMPLES][NUM_FEATURES] PROGMEM = {{\n")
    for i, row in enumerate(X_scaled):
        vals = ", ".join(f"{v:.6f}f" for v in row)
        f.write(f"    /* [{i:4d}] cls={y_raw[i]} */ {{ {vals} }},\n")
    f.write("};\n\n")

    f.write("// Rotulos verdadeiros\n")
    f.write("const int labels[NUM_SAMPLES] PROGMEM = {\n    ")
    f.write(",\n    ".join(
        ", ".join(map(str, y_raw[i:i+10])) for i in range(0, len(y_raw), 10)
    ))
    f.write("\n};\n\n")
    f.write("#endif // DATASET_H\n")

print(f"\ndataset.h gerado em '{args.output}'")
print(f"  {n_samples} amostras x {n_features} features")
print(f"  Flash estimada: {total_kb:.1f} KB")
print()
print("Proximos passos:")
print("  1. Copie ArduinoCode/dataset.h  -> pasta do projeto Arduino")
print("  2. Copie ArduinoCode/kan_model.h -> pasta do projeto Arduino")
print("  3. No .ino: USE_GROUND_TRUTH = 1")
print("  4. Upload para o ESP32 e abra o Serial Monitor em 115200 baud")
