# =============================================================================
# src/export/generate_dataset_h.py
# =============================================================================
# Gera dataset.h com amostras do CONJUNTO DE TESTE (pacientes nunca vistos no
# treino), com as features já escalonadas, para avaliar o modelo no ESP32.
#
# As features vêm de X_test.npy — exatamente os mesmos números que o Python
# usou para calcular as métricas de teste. Assim a comparação Python x ESP32
# mede só a fidelidade da exportação, sem diferença de pré-processamento.
#
# Formato idêntico ao original (gerar_dataset_h.py, commit 123a194):
#   dataset[NUM_SAMPLES][NUM_FEATURES] e labels[NUM_SAMPLES] em PROGMEM.
# =============================================================================
import os
import numpy as np

LIMIT_KB = 256


def amostrar_estratificado(y, per_class, seed=42):
    """Índices com até `per_class` amostras de cada classe, em ordem de classe."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        cand = np.flatnonzero(y == c)
        idx.extend(sorted(rng.choice(cand, size=min(per_class, len(cand)), replace=False)))
    return np.array(idx, dtype=np.int64)


def generate_dataset_h(X_scaled, y, class_names, per_class=50, seed=42,
                       output="firmware/generated_headers/dataset.h",
                       origem="data/processed/split/X_test.npy"):
    idx = amostrar_estratificado(y, per_class, seed)
    X = np.asarray(X_scaled, dtype=np.float32)[idx]
    labels = np.asarray(y, dtype=np.int64)[idx]
    n_samples, n_features = X.shape
    num_classes = len(class_names)

    total_kb = (n_samples * n_features * 4 + n_samples * 4) / 1024
    if total_kb > LIMIT_KB:
        print(f"  AVISO: dataset.h com ~{total_kb:.1f} KB excede {LIMIT_KB} KB — reduza per_class")

    class_map = ", ".join(f"{i}={c}" for i, c in enumerate(class_names))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(
            "// =====================================================================\n"
            f"// dataset.h — {n_samples} amostras do TESTE inter-paciente (46 features)\n"
            "// Gerado por : src/export/generate_dataset_h.py\n"
            f"// Origem     : {origem}\n"
            f"// Amostras   : {n_samples} (até {per_class} por classe, seed={seed})\n"
            f"// Features   : {n_features} (StandardScaler do treino já aplicado)\n"
            f"// Classes    : {class_map}\n"
            f"// Flash      : ~{total_kb:.1f} KB\n"
            "//\n"
            "// USO NO ESP32:\n"
            "//   #include \"dataset.h\"\n"
            "//   float buf[NUM_FEATURES];\n"
            "//   for (int i = 0; i < NUM_SAMPLES; i++) {\n"
            "//       for (int j = 0; j < NUM_FEATURES; j++)\n"
            "//           buf[j] = pgm_read_float_near(&dataset[i][j]);\n"
            "//       int pred = predict_from_array(buf);\n"
            "//       bool ok  = (pred == (int)pgm_read_dword_near(&labels[i]));\n"
            "//   }\n"
            "// =====================================================================\n\n"
            "#ifndef DATASET_H\n#define DATASET_H\n\n#include <pgmspace.h>\n\n"
            f"#define NUM_SAMPLES  {n_samples}\n"
            f"#define NUM_FEATURES {n_features}\n"
            f"#define NUM_CLASSES  {num_classes}\n\n"
            f"// Classes: {class_map}\n"
            "const char* CLASS_NAMES_DS[] = {\n"
        )
        for c in class_names:
            f.write(f'    "{c}",\n')
        f.write("};\n\n// Features escalonadas — prontas para predict_from_array()\n")
        f.write("const float dataset[NUM_SAMPLES][NUM_FEATURES] PROGMEM = {\n")
        for i, row in enumerate(X):
            vals = ", ".join(f"{v:.6f}f" for v in row)
            f.write(f"    /* [{i:4d}] cls={labels[i]} */ {{ {vals} }},\n")
        f.write("};\n\n// Rotulos verdadeiros\nconst int labels[NUM_SAMPLES] PROGMEM = {\n    ")
        f.write(",\n    ".join(", ".join(map(str, labels[i:i + 10])) for i in range(0, len(labels), 10)))
        f.write("\n};\n\n#endif // DATASET_H\n")

    contagem = ", ".join(f"{class_names[c]}={int((labels == c).sum())}" for c in range(num_classes))
    print(f"  dataset.h gerado em {output}: {n_samples} amostras ({contagem}), ~{total_kb:.1f} KB")
    # Devolve as amostras para a verificação numérica usar exatamente as mesmas
    return X, labels
