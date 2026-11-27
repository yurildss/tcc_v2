#!/usr/bin/env python3
"""Confere que nenhum paciente (registro) aparece em mais de um split de avaliação."""
import os, re, sys
from itertools import combinations

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/processed/split"
PAT  = re.compile(r"_(\d{3})_labeled\.csv$|_src(\d{3})\.csv$")

recs = {}
for split in ["train", "val", "test"]:
    s, sem_id = set(), 0
    for cls in os.listdir(os.path.join(ROOT, split)):
        for f in os.listdir(os.path.join(ROOT, split, cls)):
            m = PAT.search(f)
            if m: s.add(m.group(1) or m.group(2))
            else: sem_id += 1
    recs[split] = s
    print(f"{split}: {len(s)} registros | {sem_id} arquivos sem ID")

ok = True
for a, b in combinations(recs, 2):
    inter = sorted(recs[a] & recs[b])
    esperado = {a, b} == {"train", "val"}   # val é hold-out dos pacientes de treino
    print(f"{a} ∩ {b}: {inter if inter else 'vazio'}" + ("  (esperado)" if esperado and inter else ""))
    if inter and not esperado:
        ok = False
sys.exit(0 if ok else "VAZAMENTO: pacientes de teste aparecem em train/val")
