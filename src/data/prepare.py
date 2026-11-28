# =============================================================================
# cortar_batimentos.py (moved to src/data/prepare.py)
# =============================================================================
# Gera o dataset de batimentos cortados a partir do CSV completo da MIT-BIH.
#
# ESTRATÉGIA:
#   Em vez de usar find_peaks() (que falha para batimentos anômalos),
#   usamos as ANOTAÇÕES já presentes na coluna 'type' como âncoras.
#   Cada linha com type != 'NC' É um pico R confirmado pelo anotador.
#
# ESTRUTURA DO CORTE:
#   Para cada batimento anotado de interesse (A, B, F, L, N, R, V, f):
#     - Pico anterior  = anotação imediatamente anterior no registro
#     - Pico central   = a anotação de interesse
#     - Pico posterior = anotação imediatamente posterior no registro
#   O segmento salvo vai do sample do pico anterior ao do pico posterior,
#   incluindo ambas as extremidades (os picos vizinhos ficam nas bordas).
#
# SAÍDA:
#   data/processed/cut_beats/<classe>/<classe>_beat_<N>_<record>.csv
#   Colunas: channel_0, sample #, type
#   (channel_1 descartado — trabalhamos só com channel_0)
#
# =============================================================================

import os
import pandas as pd
import numpy as np
from collections import defaultdict

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

CSV_FILE     = "data/raw/mitbih_all_records_renumerada.csv"
OUTPUT_DIR   = "data/processed/cut_beats"

# Classes de interesse — apenas essas serão salvas como batimentos centrais
CLASSES_ALVO = {'A', 'B', 'F', 'L', 'N', 'R', 'V', 'f'}

# Para classes com muitos batimentos (ex: N), limita a quantidade salva
# por registro para não criar um dataset gigante e desbalanceado.
# Use None para salvar todos.
MAX_POR_CLASSE_POR_RECORD = {
    'N': 8000,    # Normal é muito abundante — limita
    'L': None,
    'R': None,
    'A': None,
    'V': None,
    'f': None,
    'F': None,
    'B': None,
}

# =============================================================================
# CARREGAMENTO
# =============================================================================

print("Carregando CSV...")
df = pd.read_csv(CSV_FILE)

# Limpa nomes de colunas (remove espaços e aspas extras)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"^['\"]|['\"]$", "", regex=True)
)

# Substitui a anotação "/" por "B" — o caractere "/" quebra caminhos
# de arquivo no Windows e é interpretado como separador de diretório.
# Feito ANTES de qualquer processamento para garantir consistência.
df['type'] = df['type'].replace('/', 'B')

# Garante ordenação por sample dentro de cada record
df = df.sort_values(["record", "sample #"]).reset_index(drop=True)

print(f"Total de linhas  : {len(df):,}")
print(f"Registros únicos : {df['record'].nunique()}")
print(f"Colunas          : {list(df.columns)}")
print()

# Distribuição das anotações
type_counts = df[df['type'] != 'NC']['type'].value_counts()
print("Distribuição das anotações (type != NC):")
print(type_counts.to_string())
print()

# =============================================================================
# SÓ BATIMENTOS VIRAM PICOS
# =============================================================================
# Anotações que NÃO são batimentos (+ mudança de ritmo, ~ ruído, " comentário,
# | artefato, x onda P não conduzida, [ ] início/fim de flutter ...) eram
# tratadas como "pico vizinho", gerando intervalos RR errados. Elas passam a
# ser 'NC' (amostra comum). Símbolos de batimento segundo a PhysioNet
# ('/' já foi trocado por 'B'):
SIMBOLOS_BATIMENTO = set("NLRBAaJSVFejEfQ!")

nao_batimento = (df['type'] != 'NC') & (~df['type'].isin(SIMBOLOS_BATIMENTO))
print(f"Anotações que não são batimentos (viram 'NC'): {int(nao_batimento.sum()):,}")
print(df.loc[nao_batimento, 'type'].value_counts().to_string())
df.loc[nao_batimento, 'type'] = 'NC'
print()

# =============================================================================
# ESTATÍSTICAS POR PACIENTE (normalização inter-paciente)
# =============================================================================
# Features absolutas (RR em amostras, amplitudes em mV) funcionam como uma
# "impressão digital" do paciente e não generalizam para pacientes novos.
# Para cada registro calculamos referências robustas (medianas) que o
# src/features/extract.py usa para normalizar as features:
#   rr_ref  : mediana dos intervalos RR entre batimentos consecutivos (amostras)
#   amp_ref : mediana da amplitude pico-a-pico do QRS (janela ±60 ms) em mV
# Não usam rótulo de classe, só a posição dos batimentos — no ESP32, em modo
# "grava e analisa", essas medianas são calculadas sobre o registro gravado.
STATS_FILE   = "data/processed/record_stats.json"
MEIA_JANELA_QRS = int(round(0.06 * 360))


def estatisticas_registro(df_record):
    samples = df_record['sample #'].to_numpy()
    sig     = df_record['channel_0'].to_numpy(dtype=np.float64)
    # posições (0..len-1) dos batimentos dentro do registro
    pos     = np.flatnonzero(df_record['type'].to_numpy() != 'NC')
    rr = np.diff(samples[pos])
    rr = rr[rr > 0]
    ptp = []
    for p in pos:
        ini, fim = max(0, p - MEIA_JANELA_QRS), min(len(sig), p + MEIA_JANELA_QRS + 1)
        ptp.append(np.ptp(sig[ini:fim]))
    return {
        'rr_ref':    float(np.median(rr)),
        'amp_ref':   float(np.median(ptp)),
        'n_batidas': int(len(pos)),
    }


# =============================================================================
# FUNÇÃO PRINCIPAL DE CORTE
# =============================================================================

def extrair_batimentos(df_record, record_name):
    """
    Extrai todos os batimentos de um único registro.

    Recebe o DataFrame de UM registro já filtrado.
    Retorna lista de dicts: {classe, segment_df}
    """
    resultados = []

    # Todas as linhas anotadas (picos R) — em ordem de sample
    anotados = df_record[df_record['type'] != 'NC'].reset_index(drop=True)

    if len(anotados) < 3:
        return resultados   # impossível formar um corte completo

    # Itera por todas as anotações (exceto a primeira e a última,
    # que não têm pico anterior/posterior no mesmo registro)
    for i in range(1, len(anotados) - 1):
        tipo_central = anotados.loc[i, 'type']

        # Ignora se não é uma classe de interesse
        if tipo_central not in CLASSES_ALVO:
            continue

        # Samples dos três picos
        sample_prev    = int(anotados.loc[i - 1, 'sample #'])
        sample_central = int(anotados.loc[i,     'sample #'])
        sample_next    = int(anotados.loc[i + 1, 'sample #'])

        # Extrai o segmento: do pico anterior ao pico posterior (inclusive)
        mask = (
            (df_record['sample #'] >= sample_prev) &
            (df_record['sample #'] <= sample_next)
        )
        seg = df_record.loc[mask, ['channel_0', 'sample #', 'type']].copy()
        seg = seg.reset_index(drop=True)

        # Verificação de sanidade
        if len(seg) < 10:
            continue   # segmento muito curto — provavelmente erro nos dados

        # Confirma que o pico central está presente no segmento
        n_central = (seg['type'] == tipo_central).sum()
        if n_central == 0:
            continue

        resultados.append({
            'classe':  tipo_central,
            'segment': seg,
            'record':  record_name,
        })

    return resultados


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

# Contadores
contadores   = defaultdict(int)   # total salvo por classe
skipped_max  = defaultdict(int)   # pulados por limite por classe

# Cria diretórios de saída
for cls in CLASSES_ALVO:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

records = df['record'].unique()
print(f"Processando {len(records)} registros...\n")

stats_por_registro = {}

for rec_idx, record_name in enumerate(sorted(records)):
    df_record = df[df['record'] == record_name].copy()
    stats_por_registro[str(record_name).split('_')[0]] = estatisticas_registro(df_record)

    batimentos = extrair_batimentos(df_record, record_name)

    # Conta por classe neste registro (para aplicar o limite)
    count_neste_record = defaultdict(int)

    for bat in batimentos:
        cls = bat['classe']

        # Aplica limite por record
        limite = MAX_POR_CLASSE_POR_RECORD.get(cls, None)
        if limite is not None and count_neste_record[cls] >= limite:
            skipped_max[cls] += 1
            continue

        # Nome do arquivo
        n_global = contadores[cls] + 1
        fname    = f"{cls}_beat_{n_global}_{record_name}.csv"
        fpath    = os.path.join(OUTPUT_DIR, cls, fname)

        bat['segment'].to_csv(fpath, index=False)

        contadores[cls]          += 1
        count_neste_record[cls]  += 1

    if (rec_idx + 1) % 10 == 0 or rec_idx == len(records) - 1:
        total_salvo = sum(contadores.values())
        print(f"  [{rec_idx+1:3d}/{len(records)}] {record_name:<20s} "
              f"| salvo até agora: {total_salvo:,}")

import json
os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
with open(STATS_FILE, "w") as f:
    json.dump(stats_por_registro, f, indent=1, sort_keys=True)
rr_refs  = [v['rr_ref']  for v in stats_por_registro.values()]
amp_refs = [v['amp_ref'] for v in stats_por_registro.values()]
print(f"\nEstatísticas por paciente salvas em {STATS_FILE}")
print(f"  rr_ref : {min(rr_refs):.0f} a {max(rr_refs):.0f} amostras "
      f"({60*360/max(rr_refs):.0f} a {60*360/min(rr_refs):.0f} bpm)")
print(f"  amp_ref: {min(amp_refs):.2f} a {max(amp_refs):.2f} mV")

# =============================================================================
# RELATÓRIO FINAL
# =============================================================================

print("\n" + "=" * 55)
print("RELATÓRIO FINAL")
print("=" * 55)
print(f"\n{'Classe':<8} {'Salvo':>8} {'Pulado (limite)':>16}")
print("-" * 35)

total = 0
for cls in sorted(CLASSES_ALVO):
    salvo  = contadores[cls]
    pulado = skipped_max[cls]
    total += salvo
    print(f"{cls:<8} {salvo:>8,} {pulado:>16,}")

print("-" * 35)
print(f"{'TOTAL':<8} {total:>8,}")
print(f"\nArquivos em: {OUTPUT_DIR}/")

# Verificação rápida de um arquivo de cada classe
print("\nVerificação — primeiro arquivo de cada classe:")
for cls in sorted(CLASSES_ALVO):
    cls_dir = os.path.join(OUTPUT_DIR, cls)
    files   = sorted(os.listdir(cls_dir))
    if not files:
        print(f"  {cls}: NENHUM ARQUIVO GERADO")
        continue

    # Lê o primeiro arquivo e verifica estrutura
    fpath = os.path.join(cls_dir, files[0])
    df_test = pd.read_csv(fpath)
    tipos   = df_test[df_test['type'] != 'NC']['type'].unique()
    n_anot  = len(df_test[df_test['type'] != 'NC'])

    print(f"  {cls}: {files[0]}")
    print(f"       linhas={len(df_test)} | "
          f"picos anotados={n_anot} | "
          f"tipos presentes={sorted(tipos)}")
