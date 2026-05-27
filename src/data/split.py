# =============================================================================
# split_dataset.py (moved to src/data/split.py)
# =============================================================================
import os
import shutil
import random
import numpy as np
from collections import defaultdict

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

INPUT_DIR  = "data/processed/augmented"
OUTPUT_DIR = "data/processed/split"

# Proporções — devem somar 1.0
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "As proporções devem somar 1.0"

SEED = 42

# =============================================================================
# PREPARAÇÃO
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)

for split in ["train", "val", "test"]:
    for cls in os.listdir(INPUT_DIR):
        if os.path.isdir(os.path.join(INPUT_DIR, cls)):
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

print("=" * 65)
print("Split estratificado do dataset")
print(f"  Train : {TRAIN_RATIO*100:.0f}%")
print(f"  Val   : {VAL_RATIO*100:.0f}%  (somente originais)")
print(f"  Test  : {TEST_RATIO*100:.0f}%  (somente originais)")
print("=" * 65)

resumo = {}

for cls in sorted(os.listdir(INPUT_DIR)):
    cls_path = os.path.join(INPUT_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    all_files = [f for f in os.listdir(cls_path) if f.endswith(".csv")]

    # Separa originais de aumentados
    originais  = sorted([f for f in all_files if "_aug_" not in f])
    aumentados = sorted([f for f in all_files if "_aug_" in f])

    n_orig = len(originais)
    n_aug  = len(aumentados)

    if n_orig == 0:
        print(f"\n  {cls}: NENHUM ORIGINAL — pulando")
        continue

    # ---- Divide os ORIGINAIS em val e test ----
    random.shuffle(originais)

    n_val  = max(1, round(n_orig * VAL_RATIO))
    n_test = max(1, round(n_orig * TEST_RATIO))

    # Garante que sobra ao menos 1 para o treino
    n_val  = min(n_val,  n_orig - 2)
    n_test = min(n_test, n_orig - n_val - 1)

    orig_val   = originais[:n_val]
    orig_test  = originais[n_val:n_val + n_test]
    orig_train = originais[n_val + n_test:]

    # ---- Train = originais restantes + TODOS os aumentados ----
    train_files = orig_train + aumentados
    random.shuffle(train_files)

    # ---- Copia os arquivos ----
    for f in train_files:
        shutil.copy2(
            os.path.join(cls_path, f),
            os.path.join(OUTPUT_DIR, "train", cls, f)
        )
    for f in orig_val:
        shutil.copy2(
            os.path.join(cls_path, f),
            os.path.join(OUTPUT_DIR, "val", cls, f)
        )
    for f in orig_test:
        shutil.copy2(
            os.path.join(cls_path, f),
            os.path.join(OUTPUT_DIR, "test", cls, f)
        )

    resumo[cls] = {
        'orig':  n_orig,
        'aug':   n_aug,
        'train': len(train_files),
        'val':   len(orig_val),
        'test':  len(orig_test),
    }

    print(f"\n  Classe {cls}:")
    print(f"    Originais    : {n_orig:>6,}  |  Aumentados: {n_aug:>6,}")
    print(f"    → Train      : {len(train_files):>6,}  "
          f"({len(orig_train)} orig + {n_aug} aug)")
    print(f"    → Val        : {len(orig_val):>6,}  (somente originais)")
    print(f"    → Test       : {len(orig_test):>6,}  (somente originais)")

# =============================================================================
# RELATÓRIO FINAL
# =============================================================================

print("\n\n" + "=" * 65)
print("RELATÓRIO FINAL")
print("=" * 65)
print(f"\n{'Classe':<8} {'Total':>7} {'Train':>7} {'Val':>6} {'Test':>6}")
print("-" * 40)

totais = defaultdict(int)
for cls in sorted(resumo.keys()):
    r = resumo[cls]
    total = r['train'] + r['val'] + r['test']
    print(f"{cls:<8} {total:>7,} {r['train']:>7,} {r['val']:>6,} {r['test']:>6,}")
    totais['train'] += r['train']
    totais['val']   += r['val']
    totais['test']  += r['test']

total_geral = totais['train'] + totais['val'] + totais['test']
print("-" * 40)
print(f"{'TOTAL':<8} {total_geral:>7,} {totais['train']:>7,} "
      f"{totais['val']:>6,} {totais['test']:>6,}")

print(f"\n  Train: {100*totais['train']/total_geral:.1f}%")
print(f"  Val  : {100*totais['val']/total_geral:.1f}%")
print(f"  Test : {100*totais['test']/total_geral:.1f}%")
print(f"\nDataset salvo em: {OUTPUT_DIR}/")
