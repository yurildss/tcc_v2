# =============================================================================
# src/export/export_all_esp32.py — etapa 7/7 do pipeline
# =============================================================================
# Gera os headers C++ para o ESP32 a partir dos artefatos salvos em disco
# pelas etapas anteriores (não depende de variáveis de notebook):
#
#   firmware/generated_headers/kan_best_state.pt  -> kan_model.h   (+ verificação)
#   firmware/generated_headers/scaler.pkl         -> scaler.h
#   data/processed/split/X_test.npy, y_test.npy   -> dataset.h     (amostras do TESTE)
#   benchmark_resultados/random_forest.pkl        -> rf_model.h    (+ verificação)
#   benchmark_resultados/xgboost.pkl              -> xgboost_model.h (+ verificação)
#
# CNN1D e LSTM: os exportadores originais foram escritos para arquiteturas
# diferentes das definidas hoje em src/models/benchmark.py, então não são
# exportados (ver aviso no log).
#
# Exportadores de RF e XGBoost restaurados da versão original
# (exportar_todos_esp32.py, commit 123a194).
# =============================================================================
import json
import os
import re
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.export.export_kan_cpp import export_kan_to_cpp, verify_export
from src.export.generate_dataset_h import generate_dataset_h
from src.export.generate_scaler_h import generate_scaler_h

HEADERS_DIR    = "firmware/generated_headers"
SPLIT_DIR      = "data/processed/split"
BENCH_DIR      = "benchmark_resultados"
KAN_STATE_PATH = os.path.join(HEADERS_DIR, "kan_best_state.pt")
KAN_WIDTH_OCULTAS = [32, 16]      # igual a src/models/pipeline_kan.py
KAN_GRID, KAN_K   = 5, 3
AMOSTRAS_POR_CLASSE = 50          # dataset.h: 8 classes x 50 = 400 amostras


def titulo(txt):
    print("\n" + "=" * 60 + f"\n{txt}\n" + "=" * 60)


def verificar(nome, y_true, y_python, y_espelho):
    acc_py = accuracy_score(y_true, y_python) * 100
    acc_es = accuracy_score(y_true, y_espelho) * 100
    conc   = float(np.mean(np.asarray(y_python) == np.asarray(y_espelho))) * 100
    ok = conc >= 99.0
    print(f"  [{'OK' if ok else 'AVISO'}] {nome}: Python={acc_py:.2f}%  espelho C++={acc_es:.2f}%"
          f"  concordância={conc:.2f}%")
    return ok


# =============================================================================
# RANDOM FOREST (micromlgen — gera if/else exatos)
# =============================================================================
def exportar_random_forest(rf_model, encoder, output_path, X_amostras, y_amostras):
    titulo("Exportando Random Forest via micromlgen")
    from micromlgen import port
    cpp_code = port(rf_model, classname="RFClassifier",
                    classmap={i: c for i, c in enumerate(encoder.classes_)})
    cpp_code += (
        "\n// ── Wrapper predict_from_array ──────────────────────────\n"
        "int predict_from_array(const float* input) {\n"
        "    return RFClassifier::predict(input);\n"
        "}\n"
    )
    with open(output_path, "w") as f:
        f.write(cpp_code)
    print(f"  Salvo em {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    y_py = rf_model.predict(X_amostras)
    print(f"  Acurácia Python nas {len(y_amostras)} amostras do dataset.h: "
          f"{accuracy_score(y_amostras, y_py) * 100:.2f}%  (micromlgen é exato)")


# =============================================================================
# XGBOOST (árvores convertidas em if/else C++)
# =============================================================================
_LEAF  = re.compile(r"(\d+):leaf=([+-]?\d*\.?\d+(?:e[+-]?\d+)?)")
_SPLIT = re.compile(r"(\d+):\[f(\d+)<([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\]\s+yes=(\d+),no=(\d+)")


def _parse_arvore(tree_str):
    nodes = {}
    for line in tree_str.strip().split("\n"):
        line = line.strip()
        m = _LEAF.match(line)
        if m:
            nodes[int(m.group(1))] = ("leaf", float(m.group(2)))
            continue
        m = _SPLIT.match(line)
        if m:
            nodes[int(m.group(1))] = ("split", int(m.group(2)), float(m.group(3)),
                                      int(m.group(4)), int(m.group(5)))
    return nodes


def _avaliar_arvore(nodes, x):
    nid = 0
    while True:
        n = nodes[nid]
        if n[0] == "leaf":
            return n[1]
        _, feat, thr, yes, no = n
        nid = yes if np.float32(x[feat]) < np.float32(thr) else no


def _base_scores(booster, n_cls):
    """base_score do XGBoost (escalar nas versões antigas, vetor por classe nas novas)."""
    try:
        cfg = json.loads(booster.save_config())
        raw = str(cfg["learner"]["learner_model_param"]["base_score"])
        vals = [float(v) for v in raw.strip("[]").split(",") if v.strip()]
        if len(vals) == 1:
            vals = vals * n_cls
        if len(vals) == n_cls:
            return np.array(vals, dtype=np.float64)
    except Exception as e:
        print(f"  (base_score não lido: {e}; usando 0)")
    return np.zeros(n_cls)


def exportar_xgboost(xgb_model, output_path, X_amostras, y_amostras):
    titulo("Exportando XGBoost (árvores em if/else)")
    booster = xgb_model.get_booster()
    trees   = booster.get_dump(dump_format="text")
    n_cls   = int(xgb_model.n_classes_)
    n_feat  = int(xgb_model.n_features_in_)
    n_rounds = len(trees) // n_cls
    parsed  = [_parse_arvore(t) for t in trees]
    base    = _base_scores(booster, n_cls)
    print(f"  {len(trees)} árvores | {n_cls} classes | {n_rounds} rounds")

    def emitir(nodes, nid=0, ind=4):
        n = nodes.get(nid)
        sp = " " * ind
        if n is None:
            return f"{sp}return 0.0f;\n"
        if n[0] == "leaf":
            return f"{sp}return {n[1]:.9g}f;\n"
        _, feat, thr, yes, no = n
        return (f"{sp}if (input[{feat}] < {thr:.9g}f) {{\n" + emitir(nodes, yes, ind + 2)
                + f"{sp}}} else {{\n" + emitir(nodes, no, ind + 2) + f"{sp}}}\n")

    L = ["// xgboost_model.h — gerado por src/export/export_all_esp32.py",
         "#ifndef XGBOOST_MODEL_H", "#define XGBOOST_MODEL_H", "",
         f"#define XGB_N_CLASSES  {n_cls}", f"#define XGB_N_FEATURES {n_feat}", ""]
    for t, nodes in enumerate(parsed):
        L += [f"static float xgb_tree_{t}(const float* input) {{", emitir(nodes).rstrip("\n"), "}", ""]
    base_c = ", ".join(f"{b:.9g}f" for b in base)
    L += ["int predict_from_array(const float* input) {",
          f"    float scores[{n_cls}] = {{ {base_c} }};"]
    for t in range(len(parsed)):
        L.append(f"    scores[{t % n_cls}] += xgb_tree_{t}(input);")
    L += ["    int best = 0;",
          f"    for (int i = 1; i < {n_cls}; i++) if (scores[i] > scores[best]) best = i;",
          "    return best;   // argmax dos scores = argmax do softmax", "}", "",
          "#endif // XGBOOST_MODEL_H", ""]
    with open(output_path, "w") as f:
        f.write("\n".join(L))
    print(f"  Salvo em {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")

    # Espelho do C++ em Python: mesma soma de árvores + argmax
    y_py = xgb_model.predict(X_amostras)
    y_es = []
    for x in X_amostras:
        s = base.copy()
        for t, nodes in enumerate(parsed):
            s[t % n_cls] += _avaliar_arvore(nodes, x)
        y_es.append(int(np.argmax(s)))
    verificar("XGBoost", y_amostras, y_py, y_es)


# =============================================================================
# EXECUÇÃO
# =============================================================================
def main():
    from kan import KAN

    os.makedirs(HEADERS_DIR, exist_ok=True)
    encoder = joblib.load(os.path.join(HEADERS_DIR, "encoder.pkl"))
    scaler  = joblib.load(os.path.join(HEADERS_DIR, "scaler.pkl"))
    class_names = [str(c) for c in encoder.classes_]
    X_test = np.load(os.path.join(SPLIT_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy")).astype(np.int64)

    titulo("scaler.h e dataset.h")
    generate_scaler_h(scaler, encoder, os.path.join(HEADERS_DIR, "scaler.h"))
    X_ds, y_ds = generate_dataset_h(X_test, y_test, class_names, per_class=AMOSTRAS_POR_CLASSE,
                                    output=os.path.join(HEADERS_DIR, "dataset.h"))

    # ---- KAN (modelo principal do TCC: falha aqui interrompe o pipeline)
    titulo("Exportando KAN (B-splines numéricas, espelho exato do pykan)")
    width = [X_test.shape[1]] + KAN_WIDTH_OCULTAS + [len(class_names)]
    model = KAN(width=width, grid=KAN_GRID, k=KAN_K)
    model.load_state_dict(torch.load(KAN_STATE_PATH, map_location="cpu"))
    model.eval()
    export_kan_to_cpp(model, output_path=os.path.join(HEADERS_DIR, "kan_model.h"),
                      class_names=class_names, use_progmem=True)
    # Verifica nas MESMAS amostras que vão para o ESP32 (dataset.h)
    verify_export(model, torch.tensor(X_ds), y_ds, n_samples=len(y_ds))

    # ---- Modelos de comparação (falha só gera aviso)
    falhas = []
    for nome, arquivo, fn in [
        ("RandomForest", "random_forest.pkl",
         lambda m: exportar_random_forest(m, encoder, os.path.join(HEADERS_DIR, "rf_model.h"), X_ds, y_ds)),
        ("XGBoost", "xgboost.pkl",
         lambda m: exportar_xgboost(m, os.path.join(HEADERS_DIR, "xgboost_model.h"), X_ds, y_ds)),
    ]:
        caminho = os.path.join(BENCH_DIR, arquivo)
        try:
            fn(joblib.load(caminho))
        except Exception as e:
            falhas.append(nome)
            print(f"  AVISO: {nome} não exportado ({type(e).__name__}: {e})")

    print("\n  AVISO: CNN1D e LSTM não exportados — os exportadores originais foram escritos")
    print("  para arquiteturas diferentes das atuais em src/models/benchmark.py.")

    # ---- Resumo
    titulo("Resumo — tamanho dos headers (flash do ESP32)")
    for nome, arq in [("KAN", "kan_model.h"), ("RandomForest", "rf_model.h"),
                      ("XGBoost", "xgboost_model.h"), ("scaler", "scaler.h"), ("dataset", "dataset.h")]:
        p = os.path.join(HEADERS_DIR, arq)
        if os.path.exists(p):
            kb = os.path.getsize(p) / 1024
            print(f"  {nome:<13} {kb:>9.1f} KB (texto do .h)")
        else:
            print(f"  {nome:<13} {'não gerado':>12}")
    print(f"\nHeaders em {HEADERS_DIR}/ — copie para a pasta do sketch do ESP32.")
    if falhas:
        print(f"Modelos com falha na exportação: {falhas}")


if __name__ == "__main__":
    sys.exit(main())
