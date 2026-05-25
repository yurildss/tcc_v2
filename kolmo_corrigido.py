# =============================================================================
# KAN - ECG CLASSIFICATION + TINYML EXPORT (ESP32)
# Versao unificada e corrigida — 46 features estruturais
# =============================================================================

import os
import re
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from kan import KAN

# Importa a extracao de features estruturais (46 features)
# que usa as anotacoes do CSV diretamente, sem find_peaks
from features_estruturais import (
    extract_features,
    load_signal_with_structure,
    FS,
)

N_FEATURES = 46   # numero de features — atualizar se features_estruturais.py mudar


# =============================================================================
# BLOCO 1 — BUILD DATASET usando features estruturais
# =============================================================================

def build_dataset(root_dir, fs=FS):
    """
    Percorre root_dir/<classe>/*.csv e extrai as 46 features estruturais.
    Usa a posicao EXATA dos picos anotados (coluna 'type') — sem find_peaks.
    Ignora arquivos sem exatamente 3 picos anotados.
    """
    X, y = [], []
    skipped = 0

    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        loaded = 0
        for file in os.listdir(class_path):
            if not file.endswith(".csv"):
                continue

            data = load_signal_with_structure(os.path.join(class_path, file))
            if data is None:
                skipped += 1
                continue

            try:
                feat = extract_features(data, fs)
                X.append(feat)
                y.append(class_name)
                loaded += 1
            except Exception:
                skipped += 1

        print(f"  {class_name}: {loaded} amostras")

    if skipped > 0:
        print(f"  (ignorados: {skipped} arquivos)")

    return np.array(X, dtype=np.float32), np.array(y)


# =============================================================================
# BLOCO 2 — CARREGAMENTO DOS DADOS E PRE-PROCESSAMENTO
# =============================================================================

DATASET_ROOT = "dataset_split"   # gerado pelo split_dataset.py

print("=" * 60)
print("Carregando datasets...")
print("=" * 60)

print("Train:")
X_train, y_train = build_dataset(os.path.join(DATASET_ROOT, "train"))
print("Val:")
X_val,   y_val   = build_dataset(os.path.join(DATASET_ROOT, "val"))
print("Test:")
X_test,  y_test  = build_dataset(os.path.join(DATASET_ROOT, "test"))

print(f"\nTreino    : {X_train.shape[0]:>6} amostras | {X_train.shape[1]} features")
print(f"Validacao : {X_val.shape[0]:>6} amostras | {X_val.shape[1]} features")
print(f"Teste     : {X_test.shape[0]:>6} amostras | {X_test.shape[1]} features")

# Confirma que o numero de features bate com o esperado
assert X_train.shape[1] == N_FEATURES, \
    f"Esperado {N_FEATURES} features, obtido {X_train.shape[1]}. " \
    f"Atualize N_FEATURES no topo do arquivo."

# Codificacao dos rotulos
encoder      = LabelEncoder()
y_train_enc  = encoder.fit_transform(y_train)
y_val_enc    = encoder.transform(y_val)
y_test_enc   = encoder.transform(y_test)
num_classes  = len(encoder.classes_)
class_names  = list(encoder.classes_)

print(f"\nClasses ({num_classes}): {class_names}")

# Escalonamento — fit APENAS no treino
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Serializa para reutilizacao
os.makedirs("ArduinoCode", exist_ok=True)
joblib.dump(scaler,  "ArduinoCode/scaler.pkl")
joblib.dump(encoder, "ArduinoCode/encoder.pkl")
print("Scaler e encoder salvos em ArduinoCode/")

# Tensores PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_train_t = torch.tensor(y_train_enc,    dtype=torch.long)
y_val_t   = torch.tensor(y_val_enc,      dtype=torch.long)
y_test_t  = torch.tensor(y_test_enc,     dtype=torch.long)


# =============================================================================
# BLOCO 3 — DEFINICAO DA KAN
# =============================================================================

n_features = X_train_t.shape[1]   # 46

model = KAN(
    width=[n_features, 32, 16, num_classes],
    grid=5,
    k=3,
)

print(f"\nArquitetura KAN: {n_features} -> 32 -> 16 -> {num_classes}")

# dataset dict para o treino customizado (treino_customizado.py usa isso)
dataset = {
    'train_input': X_train_t,
    'train_label': y_train_t,
    'test_input':  X_val_t,
    'test_label':  y_val_t,
}

print("\nModelo criado. Execute treino_customizado.py para treinar.")
print("Variaveis disponiveis no escopo:")
print("  model, dataset, encoder, scaler, class_names, num_classes")
print("  X_train_t, X_val_t, X_test_t")
print("  y_train_t, y_val_t, y_test_t")
print("  y_train_enc, y_val_enc, y_test_enc")
print("  y_train, y_val, y_test")


# =============================================================================
# BLOCO 4 — AVALIACAO (executar apos o treino)
# =============================================================================

def avaliar(model, X_t, y_enc, y_labels, split_name="val"):
    """
    Avalia o modelo em um split e imprime matriz de confusao + relatorio.
    """
    model.eval()
    with torch.no_grad():
        y_pred_idx    = torch.argmax(model(X_t), dim=1).numpy()
    y_pred_labels = encoder.inverse_transform(y_pred_idx)

    print(f"\n{'='*60}")
    print(f"AVALIACAO — {split_name.upper()}")
    print(f"{'='*60}")
    print(classification_report(y_labels, y_pred_labels,
                                 target_names=class_names, digits=4))

    cm = confusion_matrix(y_labels, y_pred_labels, labels=class_names)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusao — {split_name.upper()}")
    plt.tight_layout()
    path = f"ArduinoCode/confusion_matrix_{split_name}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Salvo em {path}")

    return y_pred_labels


# Para avaliar apos o treino, chame:
#   avaliar(model, X_val_t,  y_val,  y_val,  "val")
#   avaliar(model, X_test_t, y_test, y_test, "test")


# =============================================================================
# BLOCO 5 — EXPORTACAO PARA C++ (sem prune/symbolic)
# =============================================================================

def exportar_para_esp32():
    """
    Exporta o modelo para C++ via exportar_kan_cpp.py.
    Chame esta funcao apos o treino e verificacao da acuracia.
    """
    print("\nExportando modelo para C++...")
    print("Carregando melhor checkpoint...")

    best_state = torch.load("ArduinoCode/kan_best_state.pt")
    model.load_state_dict(best_state)
    print("Melhor modelo carregado.")

    from exportar_kan_cpp import export_kan_to_cpp, verify_export
    export_kan_to_cpp(
        model,
        output_path="ArduinoCode/kan_model.h",
        class_names=class_names,
        use_progmem=True,
    )
    verify_export(model, X_val_t, y_val_enc, n_samples=200)


# Para exportar apos o treino, chame:
#   exportar_para_esp32()


# =============================================================================
# BLOCO 6 — GERACAO DO dataset.h PARA VALIDACAO NO ESP32
# =============================================================================

def gerar_dataset_h(split="test", samples_per_class=50):
    """
    Gera dataset.h com features pre-computadas do split indicado.
    Usa o split de TEST por padrao (dados que o modelo nunca viu).
    """
    from gerar_dataset_h import generate_dataset_h

    generate_dataset_h(
        dataset_path=os.path.join(DATASET_ROOT, split),
        scaler=scaler,
        encoder=encoder,
        samples_per_class=samples_per_class,
        output="ArduinoCode/dataset.h",
        fs=FS,
    )

# Para gerar, chame:
#   gerar_dataset_h(split="test", samples_per_class=50)
