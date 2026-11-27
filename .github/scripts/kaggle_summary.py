#!/usr/bin/env python3
"""Monta o resumo do run no GitHub a partir da saída baixada do Kaggle.

Uso: python3 kaggle_summary.py <pasta_saida> <usuario/notebook> <status>
- Converte o log do Kaggle (JSON) em kaggle_log.txt, dentro da própria pasta.
- Escreve no $GITHUB_STEP_SUMMARY: status, link, fim do log e relatórios de métricas.
"""
import glob
import json
import os
import sys

out_dir, kernel_id, status = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)

# ---- Log: o Kaggle entrega uma lista JSON de {stream_name, time, data}
texto = ""
logs = glob.glob(os.path.join(out_dir, "*.log"))
if logs:
    bruto = open(logs[0], encoding="utf-8", errors="replace").read()
    try:
        texto = "".join(e.get("data", "") for e in json.loads(bruto))
    except json.JSONDecodeError:
        texto = bruto
    open(os.path.join(out_dir, "kaggle_log.txt"), "w", encoding="utf-8").write(texto)

linhas = [l for l in texto.splitlines() if "Warning" not in l and "warnings.warn" not in l]
fim_log = "\n".join(linhas[-40:]) or "(log não disponível)"

icone = {"success": "✅", "failure": "❌", "cancelled": "⏹️"}.get(status, "❔")
md = [
    f"## {icone} Pipeline no Kaggle: {status}",
    f"Notebook: https://www.kaggle.com/code/{kernel_id}",
    "",
    "Log completo e arquivos gerados estão no artefato **kaggle-output** deste run.",
    "",
    "### Fim do log",
    "```", fim_log, "```",
]

# ---- Relatórios de classificação gerados pelo benchmark, se existirem
for rel in sorted(glob.glob(os.path.join(out_dir, "**", "benchmark_resultados", "*_report.txt"),
                            recursive=True)):
    nome = os.path.basename(rel).replace("_report.txt", "")
    md += [f"### Relatório — {nome}", "```", open(rel, encoding="utf-8").read().strip(), "```"]

resumo = os.environ.get("GITHUB_STEP_SUMMARY")
if resumo:
    with open(resumo, "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
else:
    print("\n".join(md))
