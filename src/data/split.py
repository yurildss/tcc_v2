# =============================================================================
# src/data/split.py — Split INTER-PACIENTE (DS1/DS2 de de Chazal + marcapassos)
# =============================================================================
# Ordem do pipeline:  prepare (cut_beats) -> SPLIT -> augment (só treino)
#
# - Cada REGISTRO (paciente) vai inteiro para um único conjunto: treino ou teste.
# - Treino = DS1 (22) + 2 marcapassos | Teste = DS2 (22) + 2 marcapassos.
#   Os marcapassos são divididos 2/2 para que as classes B (paced) e f
#   (fusão paced+normal), que só existem nesses registros, apareçam nos dois lados:
#     104 -> f + B no treino | 107 -> B no treino
#     102 -> f + B no teste  | 217 -> f + B no teste
# - Validação = hold-out estratificado de batimentos ORIGINAIS dos registros
#   de treino. É usada só para early stopping / escolha da melhor época.
#   O teste continua 100% inter-paciente e nunca é usado em decisões de treino.
# - O split roda ANTES do augmentation: nenhum batimento sintético pode
#   derivar de um batimento de teste.
#
# Saída:
#   data/processed/train_orig/<classe>/   -> entrada do augment.py
#   data/processed/split/val/<classe>/
#   data/processed/split/test/<classe>/
# =============================================================================

import os
import re
import shutil
import random
from collections import defaultdict

INPUT_DIR      = "data/processed/cut_beats"
TRAIN_ORIG_DIR = "data/processed/train_orig"
SPLIT_DIR      = "data/processed/split"

SEED      = 42
VAL_RATIO = 0.15   # fração dos originais de treino (por classe) reservada p/ validação

# de Chazal et al. (2004), IEEE TBME 51(7) — protocolo inter-paciente padrão
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
PACED_TRAIN = ["104", "107"]
PACED_TEST  = ["102", "217"]

RECORDS_TRAIN = set(DS1 + PACED_TRAIN)
RECORDS_TEST  = set(DS2 + PACED_TEST)

# Nome gerado pelo prepare.py: <classe>_beat_<N>_<record>_labeled.csv
REC_PAT = re.compile(r"_(\d{3})_labeled\.csv$")


def record_of(fname):
    m = REC_PAT.search(fname)
    if m is None:
        raise ValueError(f"Não foi possível extrair o registro de '{fname}'")
    return m.group(1)


def main():
    assert len(RECORDS_TRAIN) == 24 and len(RECORDS_TEST) == 24
    assert not (RECORDS_TRAIN & RECORDS_TEST), "Registro em treino E teste!"

    random.seed(SEED)

    for d in [TRAIN_ORIG_DIR, os.path.join(SPLIT_DIR, "val"), os.path.join(SPLIT_DIR, "test"),
              os.path.join(SPLIT_DIR, "train")]:
        if os.path.exists(d):
            shutil.rmtree(d)   # evita restos do split antigo misturados ao novo

    classes = sorted(c for c in os.listdir(INPUT_DIR)
                     if os.path.isdir(os.path.join(INPUT_DIR, c)))

    recs_vistos = {"train": set(), "val": set(), "test": set()}
    contagem = defaultdict(dict)
    desconhecidos = set()

    for cls in classes:
        files = sorted(f for f in os.listdir(os.path.join(INPUT_DIR, cls)) if f.endswith(".csv"))
        assert not any("_aug_" in f for f in files), \
            f"Aumentados em {INPUT_DIR}/{cls}: o split deve rodar ANTES do augment."

        train_files, test_files = [], []
        for f in files:
            rec = record_of(f)
            if rec in RECORDS_TRAIN:
                train_files.append(f)
            elif rec in RECORDS_TEST:
                test_files.append(f)
            else:
                desconhecidos.add(rec)

        # Hold-out de validação: só originais, só de registros de treino
        random.shuffle(train_files)
        n_val = round(len(train_files) * VAL_RATIO) if len(train_files) >= 2 else 0
        val_files, train_files = train_files[:n_val], train_files[n_val:]

        for split, lista, dst_root in [
            ("train", train_files, TRAIN_ORIG_DIR),
            ("val",   val_files,   os.path.join(SPLIT_DIR, "val")),
            ("test",  test_files,  os.path.join(SPLIT_DIR, "test")),
        ]:
            dst = os.path.join(dst_root, cls)
            os.makedirs(dst, exist_ok=True)
            for f in lista:
                shutil.copy2(os.path.join(INPUT_DIR, cls, f), os.path.join(dst, f))
                recs_vistos[split].add(record_of(f))
            contagem[cls][split] = len(lista)

    # ------------------------------------------------------------------ checagens
    assert not desconhecidos, f"Registros fora de DS1/DS2/marcapasso: {sorted(desconhecidos)}"
    assert not (recs_vistos["train"] & recs_vistos["test"]), "VAZAMENTO treino/teste"
    assert not (recs_vistos["val"] & recs_vistos["test"]),   "VAZAMENTO val/teste"
    assert recs_vistos["val"] <= RECORDS_TRAIN

    faltando = [c for c in classes if contagem[c]["train"] == 0 or contagem[c]["test"] == 0]

    # ------------------------------------------------------------------ relatório
    print("=" * 60)
    print("Split inter-paciente (DS1/DS2 + marcapassos)")
    print("=" * 60)
    print(f"Registros treino ({len(recs_vistos['train'])}): {sorted(recs_vistos['train'])}")
    print(f"Registros teste  ({len(recs_vistos['test'])}): {sorted(recs_vistos['test'])}")
    print(f"\n{'Classe':<7}{'Train':>9}{'Val':>8}{'Test':>9}")
    print("-" * 33)
    tot = defaultdict(int)
    for c in classes:
        r = contagem[c]
        print(f"{c:<7}{r['train']:>9,}{r['val']:>8,}{r['test']:>9,}")
        for k in r:
            tot[k] += r[k]
    print("-" * 33)
    print(f"{'TOTAL':<7}{tot['train']:>9,}{tot['val']:>8,}{tot['test']:>9,}")
    print("\nTreino ainda SEM augmentation — rode src/data/augment.py em seguida.")

    if faltando:
        raise SystemExit(f"\nATENÇÃO: classes ausentes no treino ou no teste: {faltando}")
    print("OK: nenhum registro compartilhado entre treino e teste.")


if __name__ == "__main__":
    main()
