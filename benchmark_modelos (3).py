# =============================================================================
# benchmark_modelos.py
# =============================================================================
# Compara KAN, CNN1D, LSTM, XGBoost e Random Forest no mesmo dataset.
# Suporte a CUDA — CNN1D e LSTM usam GPU automaticamente se disponivel.
# XGBoost usa GPU via tree_method='hist' com device='cuda' se disponivel.
# Random Forest e scikit-learn nao suportam CUDA (roda na CPU normalmente).
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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from features_estruturais import load_signal_with_structure, extract_features, FS

# =============================================================================
# DEVICE
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 65)
print(f"Benchmark de Modelos — ECG MIT-BIH")
print(f"Device PyTorch : {device}")
if device.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    xgb_device = "cuda"
    print(f"XGBoost device : cuda (tree_method=hist)")
else:
    xgb_device = "cpu"
    print("XGBoost device : cpu")
print("Random Forest  : cpu (sklearn nao suporta CUDA)")
print("=" * 65)

DATASET_ROOT     = "dataset_split"
OUTPUT_DIR       = "benchmark_resultados"
SEED             = 42
SIGNAL_LEN_FIXED = 400
CNN_EPOCHS       = 50
LSTM_EPOCHS      = 50
BATCH_SIZE       = 256
LR               = 1e-3

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================

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
    return (np.array(sinais, dtype=np.float32),
            np.array(features, dtype=np.float32),
            np.array(rotulos))

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

# Tensores de features (CPU — para sklearn/XGBoost e KAN)
X_tr_t = torch.tensor(X_train_s)
X_va_t = torch.tensor(X_val_s)
X_te_t = torch.tensor(X_test_s)

# Tensores de sinal para CNN e LSTM
# IMPORTANTE: ficam na CPU — sao movidos para o device batch a batch
# Mover o dataset inteiro para GPU causaria OOM em GPUs com < 8GB VRAM
S_tr_cnn  = torch.tensor(S_train_n[:, None, :])   # (N, 1, L) — CPU
S_va_cnn  = torch.tensor(S_val_n[:,   None, :])
S_te_cnn  = torch.tensor(S_test_n[:,  None, :])
S_tr_lstm = torch.tensor(S_train_n[:, :, None])   # (N, L, 1) — CPU
S_va_lstm = torch.tensor(S_val_n[:,   :, None])
S_te_lstm = torch.tensor(S_test_n[:,  :, None])

# Labels ficam no device para comparacao rapida na validacao
y_tr_t = torch.tensor(y_train_enc, dtype=torch.long)   # CPU — movido no loader
y_va_t = torch.tensor(y_val_enc,   dtype=torch.long)
y_te_t = torch.tensor(y_test_enc,  dtype=torch.long)

# =============================================================================
# UTILITARIOS
# =============================================================================

results = {}

def avaliar_e_salvar(nome, y_true, y_pred, t_treino, t_inf_ms, tam_kb):
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    cm     = confusion_matrix(y_true, y_pred, labels=class_names)
    recalls = {}
    for cls in class_names:
        mask = y_true == cls
        recalls[cls] = (y_pred[mask] == cls).mean() if mask.sum() > 0 else 0.0

    print(f"\n{'='*55}\n  {nome}\n{'='*55}")
    print(f"  Acuracia  : {acc*100:.2f}%  |  F1 Macro: {f1*100:.2f}%")
    print(f"  Treino    : {t_treino:.1f}s  |  Inf: {t_inf_ms:.3f}ms/amostra  |  {tam_kb:.1f}KB")
    print(report)

    with open(os.path.join(OUTPUT_DIR, f"{nome}_report.txt"), "w") as f:
        f.write(f"Modelo: {nome}\nAcc:{acc*100:.2f}% F1:{f1*100:.2f}%\n"
                f"Treino:{t_treino:.1f}s Inf:{t_inf_ms:.3f}ms Tam:{tam_kb:.1f}KB\n\n{report}")

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title(f"Matriz de Confusao — {nome}")
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome}_confusion.png"), dpi=120)
    plt.close()

    results[nome] = {'acc': acc, 'f1': f1, 'recalls': recalls,
                     'treino_s': t_treino, 'inf_ms': t_inf_ms, 'tam_kb': tam_kb}


def medir_inferencia(predict_fn, X, n=200):
    X_sub = X[:n]
    t0 = time.perf_counter()
    predict_fn(X_sub)
    return (time.perf_counter() - t0) / n * 1000


def inferencia_em_batches(model, X, batch_size=128):
    """
    Faz inferencia em batches para evitar estouro de memoria.
    X fica na CPU — cada batch e movido para o device na hora.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device)
            preds.append(torch.argmax(model(batch), dim=1).cpu())
    return torch.cat(preds).numpy()


def treinar_pytorch(model, X_tr, y_tr, X_va, y_va, epochs, bs, lr, nome):
    """
    Loop de treino generico para modelos PyTorch.
    Dados ficam na CPU e sao movidos para o device batch a batch,
    evitando OOM em GPUs com pouca VRAM.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-5
    )
    loss_fn = nn.CrossEntropyLoss()
    loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=bs, shuffle=True, pin_memory=False
    )
    best_acc, best_state = 0.0, None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in loader:
            # Move batch para o device — dados ficam na CPU fora do loop
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validacao em batches para nao estourar VRAM
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

# =============================================================================
# MODELO 1 — CNN 1D
# =============================================================================

class CNN1D(nn.Module):
    def __init__(self, n_classes, signal_len):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=15, padding=7),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=9,  padding=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5,  padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.fc(self.convs(x))

print(f"\n{'='*65}\nMODELO 1 — CNN 1D  [{device}]\n{'='*65}")
cnn = CNN1D(num_classes, SIGNAL_LEN_FIXED).to(device)
t_cnn = treinar_pytorch(cnn, S_tr_cnn, y_tr_t, S_va_cnn, y_va_t,
                         CNN_EPOCHS, BATCH_SIZE, LR, "CNN1D")

# Inferencia em batches
y_cnn_idx = inferencia_em_batches(cnn, S_te_cnn, batch_size=256)
y_cnn = encoder.inverse_transform(y_cnn_idx)

# Tempo de inferencia (1 amostra por vez, movendo para o device)
cnn.eval()
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(min(200, len(S_te_cnn))):
        cnn(S_te_cnn[i:i+1].to(device))
t_cnn_inf = (time.perf_counter() - t0) / min(200, len(S_te_cnn)) * 1000

torch.save({k: v.cpu() for k, v in cnn.state_dict().items()},
           os.path.join(OUTPUT_DIR, "cnn1d.pt"))
tam_cnn = os.path.getsize(os.path.join(OUTPUT_DIR, "cnn1d.pt")) / 1024
avaliar_e_salvar("CNN1D", y_test, y_cnn, t_cnn, t_cnn_inf, tam_cnn)

# =============================================================================
# MODELO 2 — LSTM BIDIRECIONAL
# =============================================================================

class LSTM_ECG(nn.Module):
    """
    LSTM bidirecional com hidden reduzido para caber na RAM.
    hidden=64 (era 128) reduz consumo de memoria em 4x mantendo boa acuracia.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(64*2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def inferencia_em_batches_lstm_placeholder(): pass  # removido — funcao ja definida acima


print(f"\n{'='*65}\nMODELO 2 — LSTM Bidirecional  [{device}]\n{'='*65}")
lstm = LSTM_ECG(num_classes).to(device)

# LSTM usa batch menor para nao estourar a RAM
LSTM_BATCH = min(BATCH_SIZE, 128)
t_lstm = treinar_pytorch(lstm, S_tr_lstm, y_tr_t, S_va_lstm, y_va_t,
                          LSTM_EPOCHS, LSTM_BATCH, LR, "LSTM")

# Inferencia em batches — evita alocar o dataset inteiro de uma vez
y_lstm_idx = inferencia_em_batches(lstm, S_te_lstm, batch_size=128)
y_lstm = encoder.inverse_transform(y_lstm_idx)

# Mede tempo de inferencia com batch pequeno (1 amostra por vez)
lstm.eval()
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(min(200, len(S_te_lstm))):
        lstm(S_te_lstm[i:i+1].to(device))
t_lstm_inf = (time.perf_counter() - t0) / min(200, len(S_te_lstm)) * 1000

torch.save({k: v.cpu() for k, v in lstm.state_dict().items()},
           os.path.join(OUTPUT_DIR, "lstm.pt"))
tam_lstm = os.path.getsize(os.path.join(OUTPUT_DIR, "lstm.pt")) / 1024
avaliar_e_salvar("LSTM", y_test, y_lstm, t_lstm, t_lstm_inf, tam_lstm)

# =============================================================================
# MODELO 3 — XGBOOST
# =============================================================================

print(f"\n{'='*65}\nMODELO 3 — XGBoost  [device={xgb_device}]\n{'='*65}")
n_total = len(y_train_enc)
sample_weights = np.array([
    n_total / (num_classes * np.sum(y_train_enc == c))
    for c in y_train_enc
])

# XGBoost suporta CUDA via device='cuda' (versao >= 2.0) ou gpu_hist
xgb_params = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=SEED, n_jobs=-1,
    tree_method='hist',
)
if xgb_device == "cuda":
    xgb_params['device'] = 'cuda'

t0 = time.perf_counter()
xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train_s, y_train_enc, sample_weight=sample_weights,
        eval_set=[(X_val_s, y_val_enc)], verbose=False)
t_xgb = time.perf_counter() - t0

y_xgb = encoder.inverse_transform(xgb.predict(X_test_s))
t_xgb_inf = medir_inferencia(xgb.predict, X_test_s)
joblib.dump(xgb, os.path.join(OUTPUT_DIR, "xgboost.pkl"))
tam_xgb = os.path.getsize(os.path.join(OUTPUT_DIR, "xgboost.pkl")) / 1024
avaliar_e_salvar("XGBoost", y_test, y_xgb, t_xgb, t_xgb_inf, tam_xgb)

# =============================================================================
# MODELO 4 — RANDOM FOREST  (CPU — sklearn nao suporta CUDA)
# =============================================================================

print(f"\n{'='*65}\nMODELO 4 — Random Forest  [cpu]\n{'='*65}")
t0 = time.perf_counter()
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                             class_weight='balanced', random_state=SEED, n_jobs=-1)
rf.fit(X_train_s, y_train_enc)
t_rf = time.perf_counter() - t0
y_rf = encoder.inverse_transform(rf.predict(X_test_s))
t_rf_inf = medir_inferencia(rf.predict, X_test_s)
joblib.dump(rf, os.path.join(OUTPUT_DIR, "random_forest.pkl"))
tam_rf = os.path.getsize(os.path.join(OUTPUT_DIR, "random_forest.pkl")) / 1024
avaliar_e_salvar("RandomForest", y_test, y_rf, t_rf, t_rf_inf, tam_rf)

# =============================================================================
# MODELO 5 — KAN
# =============================================================================

print(f"\n{'='*65}\nMODELO 5 — KAN  [{device}]\n{'='*65}")
try:
    from kan import KAN
    kan_model = KAN(width=[46, 32, 16, num_classes], grid=5, k=3).to(device)
    best_state = torch.load("ArduinoCode/kan_best_state.pt", map_location=device)
    kan_model.load_state_dict(best_state)
    kan_model.eval()

    X_te_dev = X_te_t.to(device)
    with torch.no_grad():
        y_kan = encoder.inverse_transform(
            torch.argmax(kan_model(X_te_dev), dim=1).cpu().numpy()
        )
    t_kan_inf = medir_inferencia(lambda x: kan_model(x), X_te_dev)
    tam_kan   = os.path.getsize("ArduinoCode/kan_best_state.pt") / 1024
    avaliar_e_salvar("KAN", y_test, y_kan, 0.0, t_kan_inf, tam_kan)
except FileNotFoundError:
    print("  kan_best_state.pt nao encontrado — execute treino_customizado.py primeiro.")

# =============================================================================
# TABELA COMPARATIVA FINAL
# =============================================================================

print("\n\n" + "=" * 75)
print("COMPARACAO FINAL — TODOS OS MODELOS (split TEST)")
print("=" * 75)
print(f"\n{'Modelo':<14} {'Acc%':>6} {'F1%':>6} {'Inf(ms)':>9} {'Tam(KB)':>9}"
      + "".join(f" {'R_'+c:>6}" for c in class_names))
print("-" * (40 + 7 * num_classes))

rows = []
for nome, r in results.items():
    row = {'Modelo': nome, 'Acc%': round(r['acc']*100,2),
           'F1%': round(r['f1']*100,2), 'Inf_ms': round(r['inf_ms'],3),
           'Tam_KB': round(r['tam_kb'],1)}
    for c in class_names:
        row[f'R_{c}'] = round(r['recalls'].get(c,0)*100,1)
    rows.append(row)

    line = (f"{nome:<14} {r['acc']*100:>6.2f} {r['f1']*100:>6.2f}"
            f" {r['inf_ms']:>9.3f} {r['tam_kb']:>9.1f}")
    for c in class_names:
        line += f" {r['recalls'].get(c,0)*100:>6.1f}"
    print(line)

pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "comparacao_final.csv"), index=False)
print(f"\nTabela salva em {OUTPUT_DIR}/comparacao_final.csv")

# =============================================================================
# GRAFICOS
# =============================================================================

modelos = list(results.keys())
cores   = plt.cm.Set2(np.linspace(0, 1, len(modelos)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Comparacao de Modelos — ECG MIT-BIH  [{device}]", fontsize=13)

accs = [results[m]['acc']*100 for m in modelos]
f1s  = [results[m]['f1']*100  for m in modelos]

for ax, vals, title in zip(axes[:2], [accs, f1s], ["Acuracia (%)", "F1 Macro (%)"]):
    bars = ax.bar(modelos, vals, color=cores)
    ax.set_title(title); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}",
                ha='center', va='bottom', fontsize=9)

ax = axes[2]
for nome, cor in zip(modelos, cores):
    r = results[nome]
    ax.scatter(r['tam_kb'], r['inf_ms'], s=200, color=cor, label=nome, zorder=3)
    ax.annotate(nome, (r['tam_kb'], r['inf_ms']),
                textcoords="offset points", xytext=(6,4), fontsize=9)
ax.set_title("Tamanho vs Inferencia"); ax.set_xlabel("KB"); ax.set_ylabel("ms/amostra")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_final.png"), dpi=130)
plt.close()

# Recall por classe
fig, ax = plt.subplots(figsize=(14, 6))
x     = np.arange(len(class_names))
width = 0.8 / len(modelos)
for i, (nome, cor) in enumerate(zip(modelos, cores)):
    rec = [results[nome]['recalls'].get(c,0)*100 for c in class_names]
    ax.bar(x + (i - len(modelos)/2 + 0.5)*width, rec, width, label=nome, color=cor)
ax.set_title("Recall por Classe"); ax.set_xticks(x); ax.set_xticklabels(class_names)
ax.set_ylim(0, 110); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
ax.axhline(80, color='red', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_recall.png"), dpi=130)
plt.close()
print(f"Graficos salvos em {OUTPUT_DIR}/")
