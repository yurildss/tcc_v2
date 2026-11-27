#!/usr/bin/env python3
"""Acompanha a execução de um notebook do Kaggle até terminar.

Uso: python3 kaggle_wait.py <usuario/notebook> [intervalo_s]
Sai com código 0 se o notebook terminou com sucesso e 1 se falhou ou foi cancelado.
"""
import re
import subprocess
import sys
import time

kernel_id = sys.argv[1]
intervalo = int(sys.argv[2]) if len(sys.argv) > 2 else 60

ATIVOS = {"queued", "running", "new"}
FIM_OK = {"complete"}
FIM_ERRO = {"error", "cancelrequested", "cancelacknowledged"}

# Logo após o push, o Kaggle ainda pode responder com o status da versão
# ANTERIOR (ex.: "complete"). Só aceitamos um fim depois de ver a versão nova
# entrar na fila/rodar, ou depois de 5 min sem vê-la ativa.
GRACE_S = 300

inicio = time.time()
viu_ativo = False
ultimo = None
falhas_cli = 0

while True:
    r = subprocess.run(["kaggle", "kernels", "status", kernel_id],
                       capture_output=True, text=True)
    m = re.search(r'status "([^"]+)"', r.stdout)
    if r.returncode != 0 or m is None:
        falhas_cli += 1
        print(f"::warning::Falha ao consultar status ({falhas_cli}): {r.stdout.strip()} {r.stderr.strip()}")
        if falhas_cli >= 10:
            sys.exit("Não foi possível consultar o Kaggle 10 vezes seguidas.")
        time.sleep(intervalo)
        continue
    falhas_cli = 0

    # Versões novas do CLI respondem "KernelWorkerStatus.RUNNING"; antigas, "running"
    status = m.group(1).split(".")[-1].lower()
    minutos = int((time.time() - inicio) / 60)
    if status != ultimo:
        print(f"[{minutos:>3} min] status: {status}", flush=True)
        ultimo = status

    if status in ATIVOS:
        viu_ativo = True
    elif status in FIM_OK | FIM_ERRO:
        if viu_ativo or time.time() - inicio > GRACE_S:
            break

    time.sleep(intervalo)

if status in FIM_OK:
    print(f"Notebook concluído com sucesso em ~{minutos} min.")
    sys.exit(0)
print(f"::error::Notebook terminou com status '{status}' após ~{minutos} min.")
sys.exit(1)
