# =============================================================================
# benchmark.py
# =============================================================================
# Compara CNN1D, LSTM, XGBoost e Random Forest no dataset pré-processado.
# Suporte a CUDA — CNN1D e LSTM usam GPU automaticamente se disponível.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from xgboost import XGBClassifier

# =============================================================================
# 1. CONFIGURAÇÕES E PASTAS 
# =============================================================================
# Tenta carregar o config.yaml da raiz do projeto
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    SPLIT_DIR = config['data']['processed']['split']
else:
    # Fallback caso o script seja rodado fora da raiz
    SPLIT_DIR = "data/processed/split"

X_TRAIN_PATH = os.path.join(SPLIT_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(SPLIT_DIR, "y_train.npy")
X_VAL_PATH   = os.path.join(SPLIT_DIR, "X_val.npy")
Y_VAL_PATH   = os.path.join(SPLIT_DIR, "y_val.npy")
X_TEST_PATH  = os.path.join(SPLIT_DIR, "X_test.npy")
Y_TEST_PATH  = os.path.join(SPLIT_DIR, "y_test.npy")

OUTPUT_DIR = "benchmark_resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 65)
print(f"Benchmark de Modelos — ECG MIT-BIH (Device: {device})")
print("=" * 65)

# =============================================================================
# 2. DEFINIÇÃO DOS MODELOS DE DEEP LEARNING (PyTorch)
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
        
        # O input_length cai pela metade em cada MaxPool1d(2)
        final_len = input_length // 4
        self.fc1 = nn.Linear(32 * final_len, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Transforma de (batch, seq) para (batch, channel, seq)
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
        # Transforma de (batch, seq) para (batch, seq, feature)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Extrai o último state temporal
        out = self.fc(out)
        return out

# =============================================================================
# 3. FUNÇÕES AUXILIARES DE TREINO E AVALIAÇÃO
# =============================================================================

def train_pytorch_model(model, X_train, y_train, X_val, y_val, model_name, epochs=30, batch_size=64):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Prepara os tensores e DataLoaders
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    print(f"\n[{model_name}] Iniciando treinamento...")
    best_loss = float('inf')
    best_model_state = None
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validação
        model.eval()
        with torch.no_grad():
            val_out = model(val_x_t)
            val_loss = criterion(val_out, val_y_t).item()
            
            # Checkpoint (Salva a melhor época)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            
    train_time = time.time() - start_time
    
    # Restaura o melhor modelo e salva
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{model_name.lower()}.pt"))
    print(f"[{model_name}] Treinamento concluído em {train_time:.2f}s e salvo em {OUTPUT_DIR}/")
    
    return model, train_time

def evaluate_model(y_true, y_pred, model_name, class_names, inference_time):
    report = classification_report(y_true, y_pred, target_names=class_names)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"{model_name} -> Acurácia: {acc:.4f} | F1-Macro: {f1:.4f}")
    
    # Salva relatório em TXT
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nInference Time (total for test set): {inference_time:.4f}s\n")
        
    # Salva a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion.png"))
    plt.close()
    
    return acc, f1

# =============================================================================
# 4. PIPELINE PRINCIPAL DE BENCHMARK
# =============================================================================

def main():
    # 4.1 Carregar os tensores salvos pela etapa de Split/Extract
    print("Lendo tensores pré-processados...")
    try:
        X_train = np.load(X_TRAIN_PATH)
        y_train = np.load(Y_TRAIN_PATH)
        X_val   = np.load(X_VAL_PATH)
        y_val   = np.load(Y_VAL_PATH)
        X_test  = np.load(X_TEST_PATH)
        y_test  = np.load(Y_TEST_PATH)
    except FileNotFoundError as e:
        print(f"ERRO FATAL: Os dados em {SPLIT_DIR} não foram encontrados.")
        print("Certifique-se de ter rodado o wrap_split.py ou wrap_pipeline_kan.py antes do benchmark!")
        return
        
    print(f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shapes -> X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    
    class_names = ['A', 'B', 'F', 'L', 'N', 'R', 'V', 'f'] 
    num_classes = len(np.unique(y_train))
    input_length = X_train.shape[1]
    
    resultados_finais = {}

    # 4.2 Machine Learning
    print("\n[XGBoost] Iniciando treinamento...")
    xgb = XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        eval_metric='mlogloss'
    )
    t0 = time.time()
    xgb.fit(X_train, y_train)
    t_train_xgb = time.time() - t0
    
    t0 = time.time()
    preds_xgb = xgb.predict(X_test)
    t_inf_xgb = time.time() - t0
    acc_xgb, f1_xgb = evaluate_model(y_test, preds_xgb, "XGBoost", class_names, t_inf_xgb)
    joblib.dump(xgb, os.path.join(OUTPUT_DIR, "xgboost.pkl"))
    resultados_finais['XGBoost'] = {'Acuracia': acc_xgb, 'F1': f1_xgb, 'Treino(s)': t_train_xgb}

    print("\n[RandomForest] Iniciando treinamento...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    t_train_rf = time.time() - t0
    
    t0 = time.time()
    preds_rf = rf.predict(X_test)
    t_inf_rf = time.time() - t0
    acc_rf, f1_rf = evaluate_model(y_test, preds_rf, "RandomForest", class_names, t_inf_rf)
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "random_forest.pkl"))
    resultados_finais['RandomForest'] = {'Acuracia': acc_rf, 'F1': f1_rf, 'Treino(s)': t_train_rf}

    # 4.3 Deep Learning
    cnn = CNN1D(num_classes, input_length)
    cnn, t_train_cnn = train_pytorch_model(cnn, X_train, y_train, X_val, y_val, "CNN1D")
    t0 = time.time()
    cnn.eval()
    with torch.no_grad():
        preds_cnn = torch.argmax(cnn(torch.tensor(X_test, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
    t_inf_cnn = time.time() - t0
    acc_cnn, f1_cnn = evaluate_model(y_test, preds_cnn, "CNN1D", class_names, t_inf_cnn)
    resultados_finais['CNN1D'] = {'Acuracia': acc_cnn, 'F1': f1_cnn, 'Treino(s)': t_train_cnn}

    lstm = SimpleLSTM(num_classes, input_length)
    lstm, t_train_lstm = train_pytorch_model(lstm, X_train, y_train, X_val, y_val, "LSTM")
    t0 = time.time()
    lstm.eval()
    with torch.no_grad():
        preds_lstm = torch.argmax(lstm(torch.tensor(X_test, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
    t_inf_lstm = time.time() - t0
    acc_lstm, f1_lstm = evaluate_model(y_test, preds_lstm, "LSTM", class_names, t_inf_lstm)
    resultados_finais['LSTM'] = {'Acuracia': acc_lstm, 'F1': f1_lstm, 'Treino(s)': t_train_lstm}

    # 4.4 Consolidação e Gráfico Final
    df_res = pd.DataFrame(resultados_finais).T
    print("\n=========================================")
    print("        RESULTADOS FINAIS BENCHMARK      ")
    print("=========================================")
    print(df_res)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "comparacao_final.csv"))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_res.index, y=df_res['Acuracia'], palette="viridis")
    plt.title("Comparação de Acurácia entre Modelos")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1.05)
    for i, v in enumerate(df_res['Acuracia']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacao_final.png"), dpi=130)
    plt.close()
    
    print(f"\n✅ Benchmark concluído! Todos os pesos (.pkl, .pt) e gráficos estão em '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
