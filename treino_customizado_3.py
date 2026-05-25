# =============================================================================
# treino_customizado.py
# =============================================================================
# Loop de treino manual para MultKAN com loss customizada focada em
# reduzir Falsos Negativos nas classes problemáticas (A, B, f).
#
# Estratégias combinadas:
#   1. FBeta Loss (β=2)       — recall vale 4x mais que precision
#   2. Focal per-class        — gamma maior nas classes com mais FN
#   3. Weights por FN         — mais peso nas classes que mais erram
#   4. Label Smoothing        — melhora calibração
#   5. Regularização L1/L2    — controla amplitude das splines
#   6. Regularização entropia — predições mais confiantes
#
# Uso:
#   exec(open('treino_customizado.py').read())
#   (requer model, X_train_t, y_train_t, X_val_t, y_val_t,
#           encoder, class_names, num_classes no escopo)
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# =============================================================================
# SEÇÃO 1 — HIPERPARÂMETROS
# Todos os ajustes ficam aqui — não precisa mexer em mais nada
# =============================================================================

EPOCHS           = 150    # número de épocas
BATCH_SIZE       = 512    # tamanho do mini-batch
LR               = 1e-3   # learning rate inicial
LR_PATIENCE      = 30     # épocas sem melhora para reduzir o LR
LR_FACTOR        = 0.5    # fator de redução do LR
LR_MIN           = 1e-5   # LR mínimo

BETA             = 2.0    # β da FBeta Loss  (>1 = prioriza recall)
LABEL_SMOOTH     = 0.1    # suavização dos rótulos
LAMBDA_L1        = 1e-4   # regularização L1 nas splines
LAMBDA_L2        = 1e-4   # regularização L2 nas splines
LAMBDA_ENT       = 5e-3   # regularização de entropia

# Pesos da combinação final das losses principais (devem somar 1.0)
#   FBeta  — otimiza recall diretamente via gradiente suave
#   Focal  — foca nos exemplos difíceis por classe
#   BCE    — trata cada classe como problema binário independente
#            (sem competição via softmax — gradiente por classe separado)
W_FBETA          = 0.34
W_FOCAL          = 0.33
W_BCE            = 0.33

UPDATE_GRID      = True   # atualiza a grade das splines periodicamente
GRID_UPDATE_FREQ = 50     # a cada N épocas

PRINT_EVERY      = 20     # imprime log a cada N épocas
SAVE_BEST        = True   # salva o melhor modelo

# -----------------------------------------------------------------------------
# FN observados no último treino — usados para calcular pesos e gammas
# Ordem das classes: ['A', 'B', 'F', 'L', 'N', 'R', 'V', 'f']
#   índice:            0     1     2     3     4     5     6     7
# Ajuste esses valores com os FN da sua última avaliação
# -----------------------------------------------------------------------------
FN_OBSERVADOS = {
    0: 26,   # A — recall 74%  ← mais crítico
    1: 21,   # B — recall 79%
    2: 13,   # F — recall 87%
    3: 11,   # L — recall 89%
    4:  4,   # N — recall 96%  ← já bom
    5: 12,   # R — recall 88%
    6:  8,   # V — recall 92%
    7: 14,   # f — recall 86%
}

# Gamma focal por classe — maior = mais foco nos exemplos difíceis dessa classe
# Baseado diretamente nos FN: classes com mais FN recebem gamma maior
GAMMA_POR_CLASSE = [
    3.0,   # A — FN=26 (maior problema)
    2.5,   # B — FN=21
    2.0,   # F — FN=13
    2.0,   # L — FN=11
    1.0,   # N — FN=4  (já bom, gamma baixo)
    2.0,   # R — FN=12
    1.5,   # V — FN=8
    2.5,   # f — FN=14
]

# =============================================================================
# SEÇÃO 2 — CLASS WEIGHTS BASEADOS NOS FN OBSERVADOS
# Dá mais peso às classes que mais geram falsos negativos
# =============================================================================

def weights_from_fn(fn_dict, num_classes):
    """
    Cria pesos proporcionais aos FN observados.
    Classe com FN=26 recebe peso ~1.26, classe com FN=4 recebe ~1.04.
    Normalizado para que a média seja 1.
    """
    w = np.ones(num_classes, dtype=np.float32)
    for cls_idx, fn in fn_dict.items():
        w[cls_idx] = 1.0 + fn / 100.0
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

class_weights = weights_from_fn(FN_OBSERVADOS, num_classes)
print("Class weights (baseados nos FN):")
for i, (c, w) in enumerate(zip(class_names, class_weights.numpy())):
    fn = FN_OBSERVADOS.get(i, 0)
    print(f"  [{i}] {c}: {w:.3f}  (FN={fn})")

# =============================================================================
# SEÇÃO 3 — LOSSES
# =============================================================================

# -----------------------------------------------------------------------------
# 3a. FBeta Loss com Label Smoothing
#     β > 1 → recall tem mais peso que precision na otimização
#     Matematicamente:  F_β = (1+β²)·TP / ((1+β²)·TP + β²·FN + FP)
#     Minimizamos 1 - F_β para que o gradiente reduza FN
# -----------------------------------------------------------------------------
class FBetaLoss(nn.Module):
    def __init__(self, num_classes, beta=2.0, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.beta2       = beta ** 2
        self.smoothing   = smoothing
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        B, C = logits.shape

        # Probabilidades com softmax
        prob = F.softmax(logits, dim=1)                         # (B, C)

        # One-hot com label smoothing
        y = torch.zeros_like(prob)
        y.scatter_(1, targets.unsqueeze(1), 1.0)
        y = y * (1.0 - self.smoothing) + self.smoothing / C    # (B, C)

        # TP, FP, FN por classe (versão soft — diferenciável)
        tp = (prob * y).sum(dim=0)                              # (C,)
        fp = (prob * (1.0 - y)).sum(dim=0)                     # (C,)
        fn = ((1.0 - prob) * y).sum(dim=0)                     # (C,)

        # F-beta por classe
        num   = (1.0 + self.beta2) * tp
        denom = (1.0 + self.beta2) * tp + self.beta2 * fn + fp + 1e-8
        f_beta = num / denom                                    # (C,)

        # Aplica class weights
        if self.weight is not None:
            f_beta = f_beta * self.weight

        # Minimiza 1 - F_beta (médio entre classes)
        return 1.0 - f_beta.mean()


# -----------------------------------------------------------------------------
# 3b. Focal Loss per-class com Label Smoothing
#     Cada classe tem seu próprio gamma — classes com mais FN têm gamma maior
# -----------------------------------------------------------------------------
class FocalLossPerClass(nn.Module):
    def __init__(self, num_classes, gamma_per_class, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.register_buffer('gamma',  torch.tensor(gamma_per_class, dtype=torch.float32))
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        B, C = logits.shape

        # Label smoothing
        y_smooth = torch.full((B, C), self.smoothing / C, device=logits.device)
        y_smooth.scatter_(1, targets.unsqueeze(1),
                          1.0 - self.smoothing + self.smoothing / C)

        log_prob = F.log_softmax(logits, dim=1)                 # (B, C)
        prob     = torch.exp(log_prob)                          # (B, C)

        # p_t = probabilidade da classe verdadeira de cada amostra
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)

        # Gamma específico para cada amostra baseado na sua classe
        gamma_t = self.gamma[targets]                           # (B,)

        # Fator focal e cross-entropy suavizada
        focal_factor = (1.0 - p_t) ** gamma_t                  # (B,)
        ce           = -(y_smooth * log_prob).sum(dim=1)        # (B,)
        loss         = focal_factor * ce                        # (B,)

        # Class weights
        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss.mean()


# -----------------------------------------------------------------------------
# 3c. Binary Cross-Entropy (BCE) para classificação multiclasse
#
#     A BCE trata cada uma das C classes como um classificador binário
#     independente: para cada classe c, calcula sigmoid(logit_c) e compara
#     com y_c ∈ {0, 1}.
#
#     Diferença fundamental em relação à CrossEntropy (softmax):
#       - CrossEntropy: as probabilidades competem entre si (soma = 1)
#                       → aumentar p(A) obriga reduzir p(B), p(C)...
#       - BCE:          cada classe tem seu próprio gradiente independente
#                       → o modelo pode aprender "é A" e "é B" separadamente
#
#     Isso reduz FN porque o gradiente de cada classe flui sem ser
#     "bloqueado" pela competição com as outras classes via softmax.
#
#     Com label smoothing e class weights por FN, o efeito é amplificado
#     nas classes problemáticas.
# -----------------------------------------------------------------------------
class BCEMulticlassLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        B, C = logits.shape

        # Converte targets para one-hot  (B, C)
        y_onehot = torch.zeros(B, C, device=logits.device)
        y_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Label smoothing: 1 → (1 - ε),  0 → ε/C
        y_smooth = y_onehot * (1.0 - self.smoothing) + self.smoothing / C

        # BCE com sigmoid — cada logit avaliado independentemente
        # F.binary_cross_entropy_with_logits é numericamente estável
        bce_per_sample_class = F.binary_cross_entropy_with_logits(
            logits,       # (B, C) — sem softmax, usa sigmoid internamente
            y_smooth,     # (B, C) — alvo suavizado
            reduction='none',
        )                 # → (B, C)

        # Aplica class weights por coluna (peso da classe c para todas amostras)
        if self.weight is not None:
            bce_per_sample_class = bce_per_sample_class * self.weight.unsqueeze(0)

        # Média sobre classes e amostras
        return bce_per_sample_class.mean()


# -----------------------------------------------------------------------------
# Loss combinada: FBeta + Focal per-class + BCE
# Pesos controlados por W_FBETA, W_FOCAL, W_BCE na Seção 1
# -----------------------------------------------------------------------------
fbeta_loss = FBetaLoss(
    num_classes=num_classes,
    beta=BETA,
    smoothing=LABEL_SMOOTH,
    weight=class_weights,
)

focal_loss = FocalLossPerClass(
    num_classes=num_classes,
    gamma_per_class=GAMMA_POR_CLASSE,
    smoothing=LABEL_SMOOTH,
    weight=class_weights,
)

bce_loss = BCEMulticlassLoss(
    num_classes=num_classes,
    smoothing=LABEL_SMOOTH,
    weight=class_weights,
)

print(f"\nLoss configurada:")
print(f"  FBeta β          : {BETA}  (recall pesa {BETA**2:.0f}x mais que precision)")
print(f"  Label smoothing  : {LABEL_SMOOTH}")
print(f"  Gamma por classe : {dict(zip(class_names, GAMMA_POR_CLASSE))}")
print(f"  Pesos das losses : FBeta={W_FBETA} | Focal={W_FOCAL} | BCE={W_BCE}")
print(f"  λ L1 splines     : {LAMBDA_L1}")
print(f"  λ L2 splines     : {LAMBDA_L2}")
print(f"  λ Entropia       : {LAMBDA_ENT}")

# =============================================================================
# SEÇÃO 4 — REGULARIZAÇÕES NAS SPLINES
# =============================================================================

def spline_l1_loss(model):
    """L1 nos coef das splines — incentiva esparsidade."""
    l1 = torch.tensor(0.0)
    for layer in model.act_fun:
        l1 = l1 + layer.coef.abs().mean()
    return l1

def spline_l2_loss(model):
    """L2 nos coef das splines — controla amplitude."""
    l2 = torch.tensor(0.0)
    for layer in model.act_fun:
        l2 = l2 + (layer.coef ** 2).mean()
    return l2

def entropy_reg_loss(logits):
    """Penaliza distribuições uniformes — força predições confiantes."""
    prob     = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    return -(prob * log_prob).sum(dim=1).mean()

def total_loss(logits, targets, model):
    """Combina todas as componentes."""
    l_fbeta  = fbeta_loss(logits, targets)
    l_focal  = focal_loss(logits, targets)
    l_bce    = bce_loss(logits, targets)
    l_main   = W_FBETA * l_fbeta + W_FOCAL * l_focal + W_BCE * l_bce
    l_l1     = spline_l1_loss(model)    * LAMBDA_L1
    l_l2     = spline_l2_loss(model)    * LAMBDA_L2
    l_ent    = entropy_reg_loss(logits) * LAMBDA_ENT
    total    = l_main + l_l1 + l_l2 + l_ent
    return total, {
        'fbeta': l_fbeta.item(),
        'focal': l_focal.item(),
        'bce':   l_bce.item(),
        'l1':    l_l1.item(),
        'l2':    l_l2.item(),
        'ent':   l_ent.item(),
        'total': total.item(),
    }

# =============================================================================
# SEÇÃO 5 — DATALOADER, OPTIMIZER E SCHEDULER
# =============================================================================

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader  = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=LR_PATIENCE,
    factor=LR_FACTOR,
    min_lr=LR_MIN,
)

# =============================================================================
# SEÇÃO 6 — LOOP DE TREINO
# =============================================================================

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits   = model(X)
        loss_val, _ = total_loss(logits, y, model)
        preds    = torch.argmax(logits, dim=1)
        acc      = (preds == y).float().mean().item()

        # Recall por classe (para monitorar FN durante treino)
        recalls = []
        for c in range(num_classes):
            mask     = (y == c)
            if mask.sum() == 0:
                recalls.append(0.0)
                continue
            correct  = (preds[mask] == c).float().sum()
            recalls.append((correct / mask.sum()).item())
    return acc, loss_val.item(), recalls

history = {
    'epoch': [], 'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'lr': [],
    'fbeta': [], 'focal': [], 'bce': [], 'l1': [], 'l2': [], 'ent': [],
    'val_recalls': [],   # recall por classe ao longo do treino
}

best_val_acc = 0.0
best_epoch   = 0
best_state   = None

print("\n" + "=" * 75)
print("Iniciando loop de treino com foco em redução de Falsos Negativos")
print("=" * 75)
print(f"{'Época':>6} | {'T-Loss':>8} | {'T-Acc':>7} | "
      f"{'V-Loss':>8} | {'V-Acc':>7} | {'LR':>8} | Recall A  B  f")
print("-" * 75)

for epoch in range(1, EPOCHS + 1):

    # Atualiza grade das splines periodicamente
    if UPDATE_GRID and epoch % GRID_UPDATE_FREQ == 0:
        model.train()
        with torch.no_grad():
            model.update_grid(X_train_t)

    # Época de treino
    model.train()
    epoch_losses = {'fbeta': 0, 'focal': 0, 'bce': 0, 'l1': 0, 'l2': 0, 'ent': 0, 'total': 0}
    n_batches = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss, components = total_loss(logits, y_batch, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for k_name, v in components.items():
            epoch_losses[k_name] += v
        n_batches += 1

    for k_name in epoch_losses:
        epoch_losses[k_name] /= n_batches

    # Avaliação
    train_acc, train_loss, _        = evaluate(model, X_train_t, y_train_t)
    val_acc,   val_loss,   val_rec  = evaluate(model, X_val_t,   y_val_t)

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_acc)

    # Salva melhor modelo
    if SAVE_BEST and val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # Histórico
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    history['val_recalls'].append(val_rec)
    for k_name in ['fbeta', 'focal', 'bce', 'l1', 'l2', 'ent']:
        history[k_name].append(epoch_losses[k_name])

    # Log — mostra recall das 3 classes mais problemáticas
    if epoch % PRINT_EVERY == 0 or epoch == 1:
        marker = " ◄" if epoch == best_epoch else ""
        rec_A = val_rec[0] * 100
        rec_B = val_rec[1] * 100
        rec_f = val_rec[7] * 100
        print(f"{epoch:>6} | {train_loss:>8.4f} | {train_acc*100:>6.2f}% | "
              f"{val_loss:>8.4f} | {val_acc*100:>6.2f}% | "
              f"{current_lr:>8.2e} | "
              f"{rec_A:>5.1f}% {rec_B:>5.1f}% {rec_f:>5.1f}%{marker}")

print("-" * 75)
print(f"Melhor val_acc: {best_val_acc*100:.2f}% na época {best_epoch}")

# Restaura e salva melhor modelo
if SAVE_BEST and best_state is not None:
    model.load_state_dict(best_state)
    print("Modelo restaurado para o melhor checkpoint.")
    os.makedirs("ArduinoCode", exist_ok=True)
    torch.save(best_state, "ArduinoCode/kan_best_state.pt")
    print("Salvo em ArduinoCode/kan_best_state.pt")

# =============================================================================
# SEÇÃO 7 — AVALIAÇÃO FINAL COM RELATÓRIO COMPLETO
# =============================================================================

model.eval()
with torch.no_grad():
    y_pred_idx    = torch.argmax(model(X_val_t), dim=1).numpy()

y_pred_labels = encoder.inverse_transform(y_pred_idx)

print("\n" + "=" * 65)
print("RELATÓRIO FINAL — CONJUNTO DE VALIDAÇÃO")
print("=" * 65)
print(classification_report(y_val, y_pred_labels,
                             target_names=class_names, digits=4))

# =============================================================================
# SEÇÃO 8 — GRÁFICOS
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Treino Customizado — KAN ECG (foco em redução de FN)", fontsize=13)

epochs_list = history['epoch']

# Acurácia geral
ax = axes[0, 0]
ax.plot(epochs_list, [a*100 for a in history['train_acc']], label='Treino')
ax.plot(epochs_list, [a*100 for a in history['val_acc']],   label='Validação')
ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
           label=f'Melhor ({best_val_acc*100:.1f}%)')
ax.set_title("Acurácia (%)")
ax.set_xlabel("Época")
ax.legend(); ax.grid(alpha=0.3)

# Loss total
ax = axes[0, 1]
ax.plot(epochs_list, history['train_loss'], label='Treino')
ax.plot(epochs_list, history['val_loss'],   label='Validação')
ax.set_title("Loss Total")
ax.set_xlabel("Época")
ax.set_yscale('log')
ax.legend(); ax.grid(alpha=0.3)

# Recall das classes problemáticas ao longo do treino
ax = axes[0, 2]
recalls_matrix = np.array(history['val_recalls'])  # (epochs, classes)
cores = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c', '#f39c12', '#34495e']
for c_idx, (c_name, cor) in enumerate(zip(class_names, cores)):
    fn = FN_OBSERVADOS.get(c_idx, 0)
    lw = 2.5 if fn >= 15 else (1.5 if fn >= 10 else 1.0)
    ax.plot(epochs_list, recalls_matrix[:, c_idx] * 100,
            label=f'{c_name} (FN={fn})', color=cor, linewidth=lw)
ax.set_title("Recall por classe (validação)")
ax.set_xlabel("Época")
ax.set_ylabel("Recall (%)")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

# Componentes da loss
ax = axes[1, 0]
ax.plot(epochs_list, history['fbeta'], label=f'FBeta (β={BETA}, w={W_FBETA})')
ax.plot(epochs_list, history['focal'], label=f'Focal per-class (w={W_FOCAL})')
ax.plot(epochs_list, history['bce'],   label=f'BCE multiclasse (w={W_BCE})')
ax.set_title("Componentes principais da loss")
ax.set_xlabel("Época")
ax.set_yscale('log')
ax.legend(); ax.grid(alpha=0.3)

# Regularizações
ax = axes[1, 1]
ax.plot(epochs_list, history['l1'],  label=f'L1 (λ={LAMBDA_L1})')
ax.plot(epochs_list, history['l2'],  label=f'L2 (λ={LAMBDA_L2})')
ax.plot(epochs_list, history['ent'], label=f'Entropia (λ={LAMBDA_ENT})')
ax.set_title("Regularizações")
ax.set_xlabel("Época")
ax.set_yscale('log')
ax.legend(); ax.grid(alpha=0.3)

# Learning rate
ax = axes[1, 2]
ax.plot(epochs_list, history['lr'])
ax.set_title("Learning Rate")
ax.set_xlabel("Época")
ax.set_yscale('log')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ArduinoCode/treino_customizado.png", dpi=150)
plt.show()
print("Gráfico salvo em ArduinoCode/treino_customizado.png")
