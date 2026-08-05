# =============================================================================
# kolmo_corrigido.py (moved to src/models/pipeline_kan.py)
# =============================================================================

import os
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from kan import KAN
from src.features.extract import (
    extract_features,
    load_signal_with_structure,
    FS,
)

N_FEATURES = 46


def build_dataset(root_dir, fs=FS):
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


DATASET_ROOT = "data/processed/split"

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

assert X_train.shape[1] == N_FEATURES, (
    f"Esperado {N_FEATURES} features, obtido {X_train.shape[1]}."
)

encoder      = LabelEncoder()
y_train_enc  = encoder.fit_transform(y_train)
y_val_enc    = encoder.transform(y_val)
y_test_enc   = encoder.transform(y_test)
num_classes  = len(encoder.classes_)
class_names  = list(encoder.classes_)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

os.makedirs("firmware/generated_headers", exist_ok=True)
joblib.dump(scaler,  "firmware/generated_headers/scaler.pkl")
joblib.dump(encoder, "firmware/generated_headers/encoder.pkl")
print("Scaler e encoder salvos em firmware/generated_headers/")

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_train_t = torch.tensor(y_train_enc,    dtype=torch.long)
y_val_t   = torch.tensor(y_val_enc,      dtype=torch.long)
y_test_t  = torch.tensor(y_test_enc,     dtype=torch.long)


# KAN
n_features = X_train_t.shape[1]

model = KAN(
    width=[n_features, 32, 16, num_classes],
    grid=5,
    k=3,
)

print(f"\nArquitetura KAN: {n_features} -> 32 -> 16 -> {num_classes}")

dataset = {
    'train_input': X_train_t,
    'train_label': y_train_t,
    'test_input':  X_val_t,
    'test_label':  y_val_t,
}

# Salva arrays .npy para o benchmark.py consumir
import numpy as np

npy_dir = DATASET_ROOT
os.makedirs(npy_dir, exist_ok=True)

np.save(os.path.join(npy_dir, "X_train.npy"), X_train_scaled)
np.save(os.path.join(npy_dir, "y_train.npy"), y_train_enc)
np.save(os.path.join(npy_dir, "X_val.npy"),   X_val_scaled)
np.save(os.path.join(npy_dir, "y_val.npy"),   y_val_enc)
np.save(os.path.join(npy_dir, "X_test.npy"),  X_test_scaled)
np.save(os.path.join(npy_dir, "y_test.npy"),  y_test_enc)

print(f"\nArrays .npy salvos em {npy_dir}/")
print(f"  X_train : {X_train_scaled.shape}")
print(f"  X_val   : {X_val_scaled.shape}")
print(f"  X_test  : {X_test_scaled.shape}")

print("\nModelo criado. Execute src/models/train_custom.py para treinar.")
print("Variaveis disponiveis no escopo:")
print("  model, dataset, encoder, scaler, class_names, num_classes")


def avaliar(model, X_t, y_enc, y_labels, split_name="val"):
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
    path = f"firmware/generated_headers/confusion_matrix_{split_name}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Salvo em {path}")

    return y_pred_labels


def exportar_para_esp32():
    print("\nExportando modelo para C++...")
    best_state = torch.load("firmware/generated_headers/kan_best_state.pt")
    model.load_state_dict(best_state)
    from src.export.export_kan_cpp import export_kan_to_cpp, verify_export
    export_kan_to_cpp(
        model,
        output_path="firmware/generated_headers/kan_model.h",
        class_names=class_names,
        use_progmem=True,
    )
    verify_export(model, X_val_t, y_val_enc, n_samples=200)


def gerar_dataset_h(split="test", samples_per_class=50):
    from src.export.generate_dataset_h import generate_dataset_h

    generate_dataset_h(
        dataset_path=os.path.join(DATASET_ROOT, split),
        scaler=scaler,
        encoder=encoder,
        samples_per_class=samples_per_class,
        output="firmware/generated_headers/dataset.h",
        fs=FS,
    )
