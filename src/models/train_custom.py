# =============================================================================
# treino_customizado.py (moved to src/models/train_custom.py)
# =============================================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.models.pipeline_kan import model, X_train_t, y_train_t, X_val_t, y_val_t, encoder, class_names, num_classes

# =============================================================================
# HYPERPARAMS (kept as in original)
# =============================================================================

EPOCHS           = 150
BATCH_SIZE       = 512
LR               = 1e-3
LR_PATIENCE      = 30
LR_FACTOR        = 0.5
LR_MIN           = 1e-5

BETA             = 2.0
LABEL_SMOOTH     = 0.1
LAMBDA_L1        = 1e-4
LAMBDA_L2        = 1e-4
LAMBDA_ENT       = 5e-3

W_FBETA          = 0.34
W_FOCAL          = 0.33
W_BCE            = 0.33

UPDATE_GRID      = True
GRID_UPDATE_FREQ = 50

PRINT_EVERY      = 20
SAVE_BEST        = True

FN_OBSERVADOS = {
    0: 26, 1:21, 2:13, 3:11, 4:4, 5:12, 6:8, 7:14
}

GAMMA_POR_CLASSE = [3.0,2.5,2.0,2.0,1.0,2.0,1.5,2.5]

def weights_from_fn(fn_dict, num_classes):
    w = np.ones(num_classes, dtype=np.float32)
    for cls_idx, fn in fn_dict.items():
        w[cls_idx] = 1.0 + fn / 100.0
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

class_weights = weights_from_fn(FN_OBSERVADOS, num_classes)


class FBetaLoss(nn.Module):
    def __init__(self, num_classes, beta=2.0, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.beta2       = beta ** 2
        self.smoothing   = smoothing
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        B, C = logits.shape
        prob = F.softmax(logits, dim=1)
        y = torch.zeros_like(prob)
        y.scatter_(1, targets.unsqueeze(1), 1.0)
        y = y * (1.0 - self.smoothing) + self.smoothing / C
        tp = (prob * y).sum(dim=0)
        fp = (prob * (1.0 - y)).sum(dim=0)
        fn = ((1.0 - prob) * y).sum(dim=0)
        num   = (1.0 + self.beta2) * tp
        denom = (1.0 + self.beta2) * tp + self.beta2 * fn + fp + 1e-8
        f_beta = num / denom
        if self.weight is not None:
            f_beta = f_beta * self.weight
        return 1.0 - f_beta.mean()


class FocalLossPerClass(nn.Module):
    def __init__(self, num_classes, gamma_per_class, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.register_buffer('gamma',  torch.tensor(gamma_per_class, dtype=torch.float32))
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        B, C = logits.shape
        y_smooth = torch.full((B, C), self.smoothing / C, device=logits.device)
        y_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / C)
        log_prob = F.log_softmax(logits, dim=1)
        prob     = torch.exp(log_prob)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        gamma_t = self.gamma[targets]
        focal_factor = (1.0 - p_t) ** gamma_t
        ce           = -(y_smooth * log_prob).sum(dim=1)
        loss         = focal_factor * ce
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


class BCEMulticlassLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        B, C = logits.shape
        y_onehot = torch.zeros(B, C, device=logits.device)
        y_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        y_smooth = y_onehot * (1.0 - self.smoothing) + self.smoothing / C
        bce_per_sample_class = F.binary_cross_entropy_with_logits(
            logits,
            y_smooth,
            reduction='none',
        )
        if self.weight is not None:
            bce_per_sample_class = bce_per_sample_class * self.weight.unsqueeze(0)
        return bce_per_sample_class.mean()


fbeta_loss = FBetaLoss(num_classes=num_classes, beta=BETA, smoothing=LABEL_SMOOTH, weight=class_weights)
focal_loss = FocalLossPerClass(num_classes=num_classes, gamma_per_class=GAMMA_POR_CLASSE, smoothing=LABEL_SMOOTH, weight=class_weights)
bce_loss = BCEMulticlassLoss(num_classes=num_classes, smoothing=LABEL_SMOOTH, weight=class_weights)

def spline_l1_loss(model):
    l1 = torch.tensor(0.0)
    for layer in model.act_fun:
        l1 = l1 + layer.coef.abs().mean()
    return l1

def spline_l2_loss(model):
    l2 = torch.tensor(0.0)
    for layer in model.act_fun:
        l2 = l2 + (layer.coef ** 2).mean()
    return l2

def entropy_reg_loss(logits):
    prob     = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    return -(prob * log_prob).sum(dim=1).mean()

def total_loss(logits, targets, model):
    l_fbeta  = fbeta_loss(logits, targets)
    l_focal  = focal_loss(logits, targets)
    l_bce    = bce_loss(logits, targets)
    l_main   = W_FBETA * l_fbeta + W_FOCAL * l_focal + W_BCE * l_bce
    l_l1     = spline_l1_loss(model)    * LAMBDA_L1
    l_l2     = spline_l2_loss(model)    * LAMBDA_L2
    l_ent    = entropy_reg_loss(logits) * LAMBDA_ENT
    total    = l_main + l_l1 + l_l2 + l_ent
    return total, {'fbeta': l_fbeta.item(), 'focal': l_focal.item(), 'bce': l_bce.item(), 'l1': l_l1.item(), 'l2': l_l2.item(), 'ent': l_ent.item(), 'total': total.item()}


train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=LR_PATIENCE, factor=LR_FACTOR, min_lr=LR_MIN)

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits   = model(X)
        loss_val, _ = total_loss(logits, y, model)
        preds    = torch.argmax(logits, dim=1)
        acc      = (preds == y).float().mean().item()
        recalls = []
        for c in range(num_classes):
            mask     = (y == c)
            if mask.sum() == 0:
                recalls.append(0.0)
                continue
            correct  = (preds[mask] == c).float().sum()
            recalls.append((correct / mask.sum()).item())
    return acc, loss_val.item(), recalls

history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'fbeta': [], 'focal': [], 'bce': [], 'l1': [], 'l2': [], 'ent': [], 'val_recalls': [],}

best_val_acc = 0.0
best_epoch   = 0
best_state   = None

print("\n" + "=" * 75)
print("Iniciando loop de treino com foco em redução de Falsos Negativos")
print("=" * 75)
print(f"{'Época':>6} | {'T-Loss':>8} | {'T-Acc':>7} | {'V-Loss':>8} | {'V-Acc':>7} | {'LR':>8} | Recall A  B  f")
print("-" * 75)

for epoch in range(1, EPOCHS + 1):
    if UPDATE_GRID and epoch % GRID_UPDATE_FREQ == 0:
        model.train()
        with torch.no_grad():
            model.update_grid(X_train_t)

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

    train_acc, train_loss, _        = evaluate(model, X_train_t, y_train_t)
    val_acc,   val_loss,   val_rec  = evaluate(model, X_val_t,   y_val_t)

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_acc)

    if SAVE_BEST and val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    history['val_recalls'].append(val_rec)
    for k_name in ['fbeta', 'focal', 'bce', 'l1', 'l2', 'ent']:
        history[k_name].append(epoch_losses[k_name])

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

if SAVE_BEST and best_state is not None:
    model.load_state_dict(best_state)
    print("Modelo restaurado para o melhor checkpoint.")
    os.makedirs("firmware/generated_headers", exist_ok=True)
    torch.save(best_state, "firmware/generated_headers/kan_best_state.pt")
    print("Salvo em firmware/generated_headers/kan_best_state.pt")

model.eval()
with torch.no_grad():
    y_pred_idx    = torch.argmax(model(X_val_t), dim=1).numpy()

y_pred_labels = encoder.inverse_transform(y_pred_idx)

print("\n" + "=" * 65)
print("RELATÓRIO FINAL — CONJUNTO DE VALIDAÇÃO")
print("=" * 65)
print(classification_report(y_val, y_pred_labels, target_names=class_names, digits=4))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Treino Customizado — KAN ECG (foco em redução de FN)", fontsize=13)

epochs_list = history['epoch']

# (Plots code omitted for brevity) — kept minimal here
plt.tight_layout()
plt.savefig("firmware/generated_headers/treino_customizado.png", dpi=150)
plt.show()
print("Gráfico salvo em firmware/generated_headers/treino_customizado.png")
