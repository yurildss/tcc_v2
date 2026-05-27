# =============================================================================
# benchmark_modelos.py (moved to src/models/benchmark.py)
# =============================================================================
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features.extract import load_signal_with_structure, extract_features, FS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 65)
print(f"Benchmark de Modelos — ECG MIT-BIH")
print(f"Device PyTorch : {device}")
if device.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    xgb_device = "cuda"
else:
    xgb_device = "cpu"

DATASET_ROOT     = "data/processed/split"
OUTPUT_DIR       = "results/benchmark"
SEED             = 42
SIGNAL_LEN_FIXED = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

def carregar_sinais_e_features(split):
    sinais, features, rotulos = [], [], []
    skipped = 0
    for cls in sorted(os.listdir(os.path.join(DATASET_ROOT, split))):
        cls_path = os.path.join(DATASET_ROOT, split, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if not fname.endswith(".csv"):
                continue
            data = load_signal_with_structure(os.path.join(cls_path, fname))
            if data is None:
                skipped += 1
                continue
            try:
                sig = data['signal'].astype(np.float32)
                sig_interp = np.interp(
                    np.linspace(0, len(sig)-1, SIGNAL_LEN_FIXED),
                    np.arange(len(sig)), sig
                ).astype(np.float32)
                feat = extract_features(data, FS)
                sinais.append(sig_interp)
                features.append(feat)
                rotulos.append(cls)
            except Exception:
                skipped += 1
    if skipped:
        print(f"  ({skipped} ignorados)")
    return (np.array(sinais, dtype=np.float32), np.array(features, dtype=np.float32), np.array(rotulos))

print("\nCarregando dados...")
print("Train:"); S_train, X_train, y_train = carregar_sinais_e_features("train")
print("Val:");   S_val,   X_val,   y_val   = carregar_sinais_e_features("val")
print("Test:");  S_test,  X_test,  y_test  = carregar_sinais_e_features("test")
print(f"\nTrain:{len(y_train):,}  Val:{len(y_val):,}  Test:{len(y_test):,}")

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_val_enc   = encoder.transform(y_val)
y_test_enc  = encoder.transform(y_test)
class_names = list(encoder.classes_)
num_classes = len(class_names)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s   = scaler.transform(X_val).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

def norm_sinal(S):
    mu  = S.mean(axis=1, keepdims=True)
    std = S.std(axis=1,  keepdims=True) + 1e-8
    return (S - mu) / std

S_train_n = norm_sinal(S_train)
S_val_n   = norm_sinal(S_val)
S_test_n  = norm_sinal(S_test)

X_tr_t = torch.tensor(X_train_s)
X_va_t = torch.tensor(X_val_s)
X_te_t = torch.tensor(X_test_s)

S_tr_cnn  = torch.tensor(S_train_n[:, None, :])
S_va_cnn  = torch.tensor(S_val_n[:,   None, :])
S_te_cnn  = torch.tensor(S_test_n[:,  None, :])
S_tr_lstm = torch.tensor(S_train_n[:, :, None])
S_va_lstm = torch.tensor(S_val_n[:,   :, None])
S_te_lstm = torch.tensor(S_test_n[:,  :, None])

y_tr_t = torch.tensor(y_train_enc, dtype=torch.long)
y_va_t = torch.tensor(y_val_enc,   dtype=torch.long)
y_te_t = torch.tensor(y_test_enc,  dtype=torch.long)

results = {}

def avaliar_e_salvar(nome, y_true, y_pred, t_treino, t_inf_ms, tam_kb):
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm     = confusion_matrix(y_true, y_pred, labels=class_names)
    recalls = {}
    for cls in class_names:
        mask = y_true == cls
        recalls[cls] = (y_pred[mask] == cls).mean() if mask.sum() > 0 else 0.0
    print(f"\n{'='*55}\n  {nome}\n{'='*55}")
    print(f"  Acuracia  : {acc*100:.2f}%  |  F1 Macro: {f1*100:.2f}%")
    with open(os.path.join(OUTPUT_DIR, f"{nome}_report.txt"), "w") as f:
        f.write(f"Modelo: {nome}\nAcc:{acc*100:.2f}% F1:{f1*100:.2f}%\nTreino:{t_treino:.1f}s Inf:{t_inf_ms:.3f}ms Tam:{tam_kb:.1f}KB\n\n{report}")
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title(f"Matriz de Confusao — {nome}")
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome}_confusion.png"), dpi=120)
    plt.close()
    results[nome] = {'acc': acc, 'f1': f1, 'recalls': recalls, 'treino_s': t_treino, 'inf_ms': t_inf_ms, 'tam_kb': tam_kb}

def medir_inferencia(predict_fn, X, n=200):
    X_sub = X[:n]
    t0 = time.perf_counter()
    predict_fn(X_sub)
    return (time.perf_counter() - t0) / n * 1000

def inferencia_em_batches(model, X, batch_size=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device)
            preds.append(torch.argmax(model(batch), dim=1).cpu())
    return torch.cat(preds).numpy()

def treinar_pytorch(model, X_tr, y_tr, X_va, y_va, epochs, bs, lr, nome):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True, pin_memory=False)
    best_acc, best_state = 0.0, None
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(X_va), bs):
                Xvb = X_va[i:i+bs].to(device)
                yvb = y_va[i:i+bs].to(device)
                correct += (torch.argmax(model(Xvb), dim=1) == yvb).sum().item()
                total   += len(yvb)
        val_acc = correct / total
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  [{nome}] Ep {epoch:3d}/{epochs} | val_acc={val_acc*100:.1f}%")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return time.perf_counter() - t0

# (Remaining code for models and export unchanged, can be executed in this module)
