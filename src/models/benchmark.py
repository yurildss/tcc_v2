# =============================================================================
# benchmark.py
# =============================================================================
# Compara KAN, CNN1D, LSTM, XGBoost e Random Forest no mesmo dataset.
# Avalia Acurácia, F1-Score, Recall por classe, Tamanho (KB) e Latência (ms).
# =============================================================================

import os
import time
import yaml
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score
from xgboost import XGBClassifier

# Tenta importar KAN da biblioteca pykan
try:
    from kan import KAN
    HAS_KAN = True
except ImportError:
    HAS_KAN = False

# =============================================================================
# 1. CONFIGURAÇÕES E DIREÇÃO DE PASTAS
# =============================================================================
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    SPLIT_DIR = config.get('data', {}).get('processed', {}).get('split', 'data/processed/split')
    OUTPUT_DIR = "benchmark_resultados"
    KAN_STATE_PATH = config.get('models', {}).get('kan_state', 'firmware/generated_headers/kan_best_state.pt')
else:
    SPLIT_DIR = "data/processed/split"
    OUTPUT_DIR = "benchmark_resultados"
    KAN_STATE_PATH = "firmware/generated_headers/kan_best_state.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print(f"Benchmark de Modelos ECG — TinyML (Device: {device})")
print("=" * 65)

# =============================================================================
# 2. DEFINIÇÃO DAS ARQUITETURAS DE DEEP LEARNING (PyTorch)
# =============================================================================

class CNN1D(nn.Module):
    def __init__(self, num_classes, input_length):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        
        final_len = input_length // 4
        self.fc1 = nn.Linear(32 * max(1, final_len), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleLSTM(nn.Module):
    def __init__(self, num_classes, input_length, hidden_size=64, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# =============================================================================
# 3. FUNÇÃO DE TREINO DEEP LEARNING
# =============================================================================

def train_pytorch_model(model, X_train, y_train, X_val, y_val, model_name, epochs=30, batch_size=64):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    print(f"\n[{model_name}] A treinar...")
    best_loss = float('inf')
    best_state = None
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            vout = model(val_x_t)
            vloss = criterion(vout, val_y_t).item()
            if vloss < best_loss:
                best_loss = vloss
                best_state = model.state_dict()
                
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {vloss:.4f}")
            
    t_train = time.time() - t0
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    save_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}.pt")
    torch.save(model.state_dict(), save_path)
    return model, t_train, save_path

# =============================================================================
# 4. EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    print("A carregar matrizes de dados pré-processadas...")
    X_train_path = os.path.join(SPLIT_DIR, "X_train.npy")
    y_train_path = os.path.join(SPLIT_DIR, "y_train.npy")
    X_val_path   = os.path.join(SPLIT_DIR, "X_val.npy")
    y_val_path   = os.path.join(SPLIT_DIR, "y_val.npy")
    X_test_path  = os.path.join(SPLIT_DIR, "X_test.npy")
    y_test_path  = os.path.join(SPLIT_DIR, "y_test.npy")

    if not os.path.exists(X_train_path):
        print(f"ERRO: Ficheiros não encontrados em '{SPLIT_DIR}'. Executa a etapa de split/pipeline primeiro.")
        return

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val   = np.load(X_val_path)
    y_val   = np.load(y_val_path)
    X_test  = np.load(X_test_path)
    y_test  = np.load(y_test_path)

    class_names = ['A', 'B', 'F', 'L', 'N', 'R', 'V', 'f']
    num_classes = len(np.unique(y_train))
    input_length = X_train.shape[1]

    results = {}

    def registrar_modelo(nome, y_true, y_pred, t_inf_total, file_path):
        acc = accuracy_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        rec_cls = recall_score(y_true, y_pred, average=None) * 100
        inf_ms = (t_inf_total / len(y_true)) * 1000
        tam_kb = os.path.getsize(file_path) / 1024.0 if (file_path and os.path.exists(file_path)) else 0.0

        results[nome] = {
            'acc': acc,
            'f1': f1,
            'recall_cls': rec_cls,
            'inf_ms': inf_ms,
            'tam_kb': tam_kb
        }

        # Relatório de Classificação e Matriz de Confusão
        rep = classification_report(y_true, y_pred, target_names=class_names)
        with open(os.path.join(OUTPUT_DIR, f"{nome}_report.txt"), "w") as f:
            f.write(rep)
            f.write(f"\nInferencia: {inf_ms:.4f} ms/amostra | Tamanho: {tam_kb:.2f} KB\n")

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matriz de Confusão - {nome}')
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Prevista')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{nome}_confusion.png"))
        plt.close()

    # --- 4.1 Avaliar KAN (se os pesos salvos existirem) ---
    if HAS_KAN and os.path.exists(KAN_STATE_PATH):
        print("\n[KAN] A carregar modelo treinado...")
        try:
            kan_model = KAN(width=[input_length, 16, num_classes], grid=5, k=3, device=device)
            kan_model.load_state_dict(torch.load(KAN_STATE_PATH, map_location=device))
            kan_model.eval()

            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            t0 = time.time()
            with torch.no_grad():
                out_kan = kan_model(X_test_t)
                preds_kan = torch.argmax(out_kan, dim=1).cpu().numpy()
            t_inf_kan = time.time() - t0

            registrar_modelo("KAN", y_test, preds_kan, t_inf_kan, KAN_STATE_PATH)
            print("✅ KAN avaliada com sucesso.")
        except Exception as e:
            print(f"⚠️ Não foi possível avaliar a KAN: {e}")
    else:
        print("\n⚠️ Ficheiro de pesos da KAN não encontrado ou pykan ausente. A saltar KAN...")

    # --- 4.2 XGBoost ---
    print("\n[XGBoost] A treinar...")
    xgb_path = os.path.join(OUTPUT_DIR, "xgboost.pkl")
    xgb = XGBClassifier(n_estimators=100, max_depth=5, tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu')
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, xgb_path)

    t0 = time.time()
    preds_xgb = xgb.predict(X_test)
    t_inf_xgb = time.time() - t0
    registrar_modelo("XGBoost", y_test, preds_xgb, t_inf_xgb, xgb_path)

    # --- 4.3 Random Forest ---
    print("\n[RandomForest] A treinar...")
    rf_path = os.path.join(OUTPUT_DIR, "random_forest.pkl")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, rf_path)

    t0 = time.time()
    preds_rf = rf.predict(X_test)
    t_inf_rf = time.time() - t0
    registrar_modelo("RandomForest", y_test, preds_rf, t_inf_rf, rf_path)

    # --- 4.4 CNN1D ---
    cnn = CNN1D(num_classes, input_length)
    cnn, _, cnn_path = train_pytorch_model(cnn, X_train, y_train, X_val, y_val, "CNN1D")
    
    t0 = time.time()
    cnn.eval()
    with torch.no_grad():
        preds_cnn = torch.argmax(cnn(torch.tensor(X_test, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
    t_inf_cnn = time.time() - t0
    registrar_modelo("CNN1D", y_test, preds_cnn, t_inf_cnn, cnn_path)

    # --- 4.5 LSTM ---
    lstm = SimpleLSTM(num_classes, input_length)
    lstm, _, lstm_path = train_pytorch_model(lstm, X_train, y_train, X_val, y_val, "LSTM")

    t0 = time.time()
    lstm.eval()
    with torch.no_grad():
        preds_lstm = torch.argmax(lstm(torch.tensor(X_test, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
    t_inf_lstm = time.time() - t0
    registrar_modelo("LSTM", y_test, preds_lstm, t_inf_lstm, lstm_path)

    # =============================================================================
    # 5. GERAÇÃO DOS PAINÉIS COMPARATIVOS (IDÊNTICO AO TEU CÓDIGO ORIGINAL)
    # =============================================================================
    print("\n[Gráficos] A gerar painéis comparativos finais...")
    modelos = list(results.keys())
    cores = sns.color_palette("Set2", len(modelos))

    # --- 5.1 Painel Principal: Acurácia, F1-Score e Tamanho vs Inferência ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric_key, title in zip(
        axes[:2],
        ['acc', 'f1'],
        ['Acuracia (%)', 'F1-Score Macro (%)']
    ):
        vals = [results[m][metric_key] for m in modelos]
        bars = ax.bar(modelos, vals, color=cores)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)

    ax_scatter = axes[2]
    for nome, cor in zip(modelos, cores):
        r = results[nome]
        ax_scatter.scatter(r['tam_kb'], r['inf_ms'], s=200, color=cor, label=nome, zorder=3)
        ax_scatter.annotate(nome, (r['tam_kb'], r['inf_ms']), textcoords="offset points", xytext=(6,4), fontsize=9)
    
    ax_scatter.set_title("Tamanho vs Inferência (TinyML)")
    ax_scatter.set_xlabel("Tamanho (KB)")
    ax_scatter.set_ylabel("Latência (ms/amostra)")
    ax_scatter.legend()
    ax_scatter.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_final.png"), dpi=130)
    plt.close()

    # --- 5.2 Gráfico de Recall por Classe ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.8 / len(modelos)

    for i, (nome, cor) in enumerate(zip(modelos, cores)):
        recalls = results[nome]['recall_cls']
        ax.bar(x + i*width, recalls, width, label=nome, color=cor)

    ax.set_title("Recall por Classe de Arritmia (%)")
    ax.set_xticks(x + width * (len(modelos)-1) / 2)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Recall (%)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_recall.png"), dpi=130)
    plt.close()

    # Tabela Sumário em CSV
    df_summary = pd.DataFrame(results).T.drop(columns=['recall_cls'])
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "comparacao_final.csv"))

    print(f"\n✅ Benchmark concluído com sucesso! Resultados salvos na pasta '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
