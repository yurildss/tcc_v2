# =============================================================================
# src/models/kan_config.py — configuração única do KAN
# =============================================================================
# Usada pelo treino, benchmark e exportação, para que todos construam
# exatamente a mesma arquitetura (antes ela estava repetida em 3 arquivos).
# =============================================================================
from kan import KAN

KAN_HIDDEN = [32, 16]   # camadas ocultas: 46 -> 32 -> 16 -> 8
KAN_GRID   = 5
KAN_K      = 3


def build_kan(n_in, n_out, fast=True):
    """Cria o KAN do TCC.

    fast=True liga model.speed() do pykan: desativa o ramo simbólico (que fica
    zerado e não contribui para a saída) e o salvamento das ativações a cada
    forward (usado só para gráficos). A saída é idêntica, e o treino fica
    ~35x mais rápido. update_grid() continua funcionando: o pykan liga o
    salvamento das ativações temporariamente só para ele.

    auto_save=False evita que o pykan grave checkpoints em ./model a cada
    operação (as mensagens "saving model version" no log).
    """
    model = KAN(width=[n_in] + KAN_HIDDEN + [n_out], grid=KAN_GRID, k=KAN_K, auto_save=False)
    if fast:
        model.speed()
    return model
