# =============================================================================
# augmentation_ecg.py (moved to src/data/augment.py)
# =============================================================================
import os
import random
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

INPUT_DIR  = "data/processed/cut_beats"
OUTPUT_DIR = "data/processed/augmented"
SEED       = 42
FS         = 360

# Classes que recebem augmentation (as demais só copiam/subsamplelam)
CLASSES_AUG = {'A', 'B', 'F', 'R', 'V', 'f'}

# Metas finais por classe
METAS = {
    'A':  6000,
    'B':  8000,
    'F':  4000,
    'L':  8075,
    'N': 10000,
    'R':  8000,
    'V':  8000,
    'f':  4000,
}

# Parâmetros de cada técnica
PARAMS = {
    'timeshift':       {'shift_range':  (-20, 20)},
    'jitter':          {'noise_factor': (0.002, 0.008)},
    'amplitude_scale': {'scale_range':  (0.85, 1.15)},
    'baseline_wander': {'freq_range':   (0.05, 0.5),
                        'amp_range':    (0.01, 0.05)},
    'smooth':          {'window_range': (2, 4)},
}

# =============================================================================
# TÉCNICAS INDIVIDUAIS
# =============================================================================

def t_timeshift(signal, samples, shift_range=(-20, 20)):
    shift = random.randint(*shift_range)
    new_samples = samples + shift
    if new_samples.min() < 0:
        new_samples -= new_samples.min()
    return signal.copy(), new_samples

def t_jitter(signal, samples, noise_factor=(0.002, 0.008)):
    factor = random.uniform(*noise_factor)
    noise  = np.random.normal(0, np.std(signal) * factor,
                              size=len(signal)).astype(np.float32)
    return signal + noise, samples

def t_amplitude_scale(signal, samples, scale_range=(0.85, 1.15)):
    return signal * random.uniform(*scale_range), samples

def t_baseline_wander(signal, samples,
                      freq_range=(0.05, 0.5), amp_range=(0.01, 0.05)):
    amp   = random.uniform(*amp_range) * np.ptp(signal)
    freq  = random.uniform(*freq_range)
    phase = random.uniform(0, 2 * np.pi)
    t     = samples / FS
    wander = (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    return signal + wander, samples

def t_smooth(signal, samples, window_range=(2, 4)):
    w      = random.randint(*window_range)
    kernel = np.ones(w, dtype=np.float32) / w
    padded = np.pad(signal, (w // 2, w - w // 2 - 1), mode='edge')
    result = np.convolve(padded, kernel, mode='valid').astype(np.float32)
    return result, samples

TECNICAS_FN = {
    'timeshift':       t_timeshift,
    'jitter':          t_jitter,
    'amplitude_scale': t_amplitude_scale,
    'baseline_wander': t_baseline_wander,
    'smooth':          t_smooth,
}

# =============================================================================
# APLICADOR COM COMBINAÇÃO ALEATÓRIA
# =============================================================================

def aplicar_combinacao(signal, samples):
    todas = list(TECNICAS_FN.keys())

    n_tec = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
    escolhidas = random.sample(todas, n_tec)

    sig_aug  = signal.copy().astype(np.float32)
    smp_aug  = samples.copy()

    for tec in escolhidas:
        fn     = TECNICAS_FN[tec]
        params = PARAMS[tec]
        sig_aug, smp_aug = fn(sig_aug, smp_aug, **params)

    return sig_aug, smp_aug, escolhidas

# =============================================================================
# CARREGAMENTO DOS ORIGINAIS
# =============================================================================

def carregar_originais(classe):
    path = os.path.join(INPUT_DIR, classe)
    if not os.path.exists(path):
        return []
    result = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith('.csv'):
            continue
        try:
            df = pd.read_csv(os.path.join(path, fname))
            if 'channel_0' not in df.columns or 'sample #' not in df.columns:
                continue
            if 'type' not in df.columns:
                df['type'] = 'NC'
            result.append({'fname': fname, 'df': df})
        except Exception:
            continue
    return result

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)

for cls in METAS:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

print("=" * 65)
print("Data Augmentation — ECG MIT-BIH")
print("=" * 65)

resumo = {}

for cls in sorted(METAS.keys()):
    meta      = METAS[cls]
    originais = carregar_originais(cls)
    n_orig    = len(originais)

    if n_orig == 0:
        print(f"\n  {cls}: NENHUM ARQUIVO ORIGINAL — pulando")
        continue

    print(f"\n  Classe {cls}: {n_orig:,} originais → meta {meta:,}")

    out_path = os.path.join(OUTPUT_DIR, cls)

    # ---- Seleciona quantos originais copiar ----
    if n_orig >= meta:
        selecionados = random.sample(originais, meta)
        print(f"    Subsample: {n_orig:,} → {meta:,} originais")
    else:
        selecionados = originais

    # Copia originais selecionados
    for item in selecionados:
        dst = os.path.join(out_path, item['fname'])
        if not os.path.exists(dst):
            src = os.path.join(INPUT_DIR, cls, item['fname'])
            shutil.copy2(src, dst)

    n_copiados       = len(selecionados)
    n_aug_necessario = max(0, meta - n_copiados)

    print(f"    Originais copiados : {n_copiados:,}")
    print(f"    Aug necessário     : {n_aug_necessario:,}")

    # ---- Augmentation (só para classes deficitárias) ----
    if n_aug_necessario == 0 or cls not in CLASSES_AUG:
        resumo[cls] = {'originais': n_copiados, 'aug': 0, 'total': n_copiados}
        continue

    tec_counter  = defaultdict(int)
    aug_gerado   = 0
    tentativas   = 0
    max_tent     = n_aug_necessario * 15

    while aug_gerado < n_aug_necessario and tentativas < max_tent:
        tentativas += 1

        item    = random.choice(selecionados)
        df_orig = item['df'].copy()
        signal  = df_orig['channel_0'].values.astype(np.float32)
        samples = df_orig['sample #'].values.astype(np.int64)

        try:
            sig_aug, smp_aug, tecnicas_usadas = aplicar_combinacao(signal, samples)
        except Exception:
            continue

        # Validação
        if not np.all(np.isfinite(sig_aug)):
            continue
        if len(sig_aug) != len(signal):
            continue

        # Monta DataFrame
        df_aug              = df_orig.copy()
        df_aug['channel_0'] = sig_aug
        df_aug['sample #']  = smp_aug

        # Nome do arquivo com técnicas usadas no nome para rastreabilidade
        tec_tag  = '_'.join(tecnicas_usadas)
        fname    = f"{cls}_aug_{tec_tag}_{aug_gerado + 1}.csv"
        df_aug.to_csv(os.path.join(out_path, fname), index=False)

        for t in tecnicas_usadas:
            tec_counter[t] += 1
        aug_gerado += 1

    # Log de distribuição de técnicas
    print(f"    Aug gerado         : {aug_gerado:,}")
    if tec_counter:
        print(f"    Uso das técnicas (amostras em que apareceu):")
        for tec in sorted(tec_counter, key=tec_counter.get, reverse=True):
            pct = 100 * tec_counter[tec] / aug_gerado
            print(f"      {tec:<20s}: {tec_counter[tec]:>5}  ({pct:.1f}%)")

    total_cls = n_copiados + aug_gerado
    resumo[cls] = {'originais': n_copiados, 'aug': aug_gerado, 'total': total_cls}

# =============================================================================
# RELATÓRIO FINAL
# =============================================================================

print("\n\n" + "=" * 65)
print("RELATÓRIO FINAL")
print("=" * 65)
print(f"\n{'Classe':<8} {'Meta':>7} {'Orig':>8} {'Aug':>7} {'Total':>8}  Status")
print("-" * 52)

for cls in sorted(METAS.keys()):
    if cls not in resumo:
        print(f"{cls:<8} {METAS[cls]:>7,}  {'—':>7} {'—':>6} {'—':>7}  SEM DADOS")
        continue
    r    = resumo[cls]
    meta = METAS[cls]
    diff = r['total'] - meta
    ok   = "✓" if abs(diff) <= 5 else (f"+{diff}" if diff > 0 else f"{diff}")
    print(f"{cls:<8} {meta:>7,}  {r['originais']:>7,} {r['aug']:>6,} "
          f"{r['total']:>7,}  {ok}")

total = sum(r['total'] for r in resumo.values())
print("-" * 52)
print(f"{'TOTAL':<8} {sum(METAS.values()):>7,}  {'':>7} {'':>6} {total:>7,}")
print(f"\nDataset salvo em: {OUTPUT_DIR}/")
