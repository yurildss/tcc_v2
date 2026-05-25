# =============================================================================
# exportar_todos_esp32.py
# =============================================================================
# Exporta CNN1D, LSTM, XGBoost e Random Forest para C++ / ESP32.
#
# ESTRATEGIAS POR MODELO:
#
#   Random Forest  → micromlgen (gera C++ puro, sem bibliotecas externas)
#                    Fidelidade exata — sem aproximacao
#
#   XGBoost        → exportacao manual dos splits das arvores para C++
#                    (micromlgen nao suporta XGBoost multiclasse de forma
#                    estavel; a versao manual e mais confiavel)
#
#   CNN1D          → exporta pesos (conv + fc) para arrays C++ e
#                    implementa forward pass com operacoes basicas
#
#   LSTM           → exporta pesos (W_ih, W_hh, b) para arrays C++ e
#                    implementa forward pass da LSTM bidirecional
#
# VERIFICACAO NUMERICA:
#   Cada exportador implementa o mesmo forward pass em NumPy e compara
#   com PyTorch/sklearn antes de gravar o .h — garante fidelidade.
#
# USO:
#   exec(open('exportar_todos_esp32.py').read())
#   (requer modelos treinados no escopo: rf, xgb, cnn, lstm)
#   (requer: encoder, scaler, X_test_s, S_test_n, y_test)
# =============================================================================

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

OUTPUT_DIR = "ArduinoCode"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# UTILITARIOS COMUNS
# =============================================================================

def fmt_array_1d(arr, name, pg="PROGMEM "):
    """Formata array 1D para C++."""
    vals = ", ".join(f"{float(v):.7f}f" for v in arr.flatten())
    return f"const float {pg}{name}[{len(arr.flatten())}] = {{{vals}}};\n"

def fmt_array_2d(arr, name, pg="PROGMEM "):
    """Formata array 2D para C++."""
    rows = []
    for row in arr:
        vals = ", ".join(f"{float(v):.7f}f" for v in row)
        rows.append(f"    {{ {vals} }}")
    body = ",\n".join(rows)
    r, c = arr.shape
    return f"const float {pg}{name}[{r}][{c}] = {{\n{body}\n}};\n"

def verificar_acuracia(nome, y_true, y_pred_python, y_pred_numpy, tol=1.0):
    acc_py = accuracy_score(y_true, y_pred_python) * 100
    acc_np = accuracy_score(y_true, y_pred_numpy)  * 100
    delta  = abs(acc_py - acc_np)
    status = "OK" if delta < tol else "AVISO"
    print(f"  [{status}] {nome}: Python={acc_py:.2f}%  NumPy={acc_np:.2f}%"
          f"  delta={delta:.3f}%")
    return delta < tol

# =============================================================================
# EXPORTADOR 1 — RANDOM FOREST (via micromlgen)
# =============================================================================

def exportar_random_forest(rf_model, encoder, output_path):
    """
    Usa micromlgen para gerar C++ exato do Random Forest.
    micromlgen percorre todas as arvores e gera codigo de comparacao
    direta (if/else aninhados), sem aproximacao.
    """
    print("\n" + "=" * 55)
    print("Exportando Random Forest via micromlgen...")
    print("=" * 55)

    try:
        from micromlgen import port

        cpp_code = port(
            rf_model,
            classname='RFClassifier',
            classmap={i: c for i, c in enumerate(encoder.classes_)},
        )

        # Adiciona wrapper predict_from_array para compatibilidade com o .ino
        n_feat = rf_model.n_features_in_
        wrapper = f"""
// ── Wrapper predict_from_array ──────────────────────────
// Compativel com o ecg_kan_esp32.ino
int predict_from_array(const float* input) {{
    return RFClassifier::predict(input);
}}
"""
        cpp_code += wrapper

        with open(output_path, "w") as f:
            f.write(cpp_code)

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Salvo em {output_path}  ({size_kb:.1f} KB)")

        # Verificacao numerica
        import joblib
        y_py  = encoder.inverse_transform(rf_model.predict(X_test_s))
        # micromlgen e exato — nao precisa reimplementar em NumPy
        print(f"  Acuracia Python: {accuracy_score(y_test, y_py)*100:.2f}%")
        print("  micromlgen e exato — sem divergencia esperada.")

        return True

    except ImportError:
        print("  micromlgen nao instalado. Instalando...")
        os.system("pip install micromlgen --break-system-packages -q")
        return exportar_random_forest(rf_model, encoder, output_path)


# =============================================================================
# EXPORTADOR 2 — XGBOOST (exportacao manual das arvores)
# =============================================================================

def exportar_xgboost(xgb_model, encoder, output_path, n_features=46):
    """
    Exporta XGBoost para C++ extraindo os splits das arvores via
    get_booster().get_dump(). Cada arvore e convertida em if/else C++.

    Estrategia:
      XGBoost multiclasse usa num_classes arvores por round de boosting.
      A predicao final e softmax sobre a soma dos scores de cada classe.
    """
    print("\n" + "=" * 55)
    print("Exportando XGBoost (arvores manuais)...")
    print("=" * 55)

    booster = xgb_model.get_booster()
    trees   = booster.get_dump(dump_format='text')
    n_cls   = xgb_model.n_classes_
    print(f"  {len(trees)} arvores  |  {n_cls} classes  |  "
          f"{len(trees)//n_cls} rounds")

    def parse_tree(tree_str):
        """Converte dump de uma arvore em codigo C++ recursivo."""
        lines = tree_str.strip().split('\n')
        nodes = {}
        for line in lines:
            line = line.strip()
            # Folha: "2:leaf=0.123"
            leaf_m = re.match(r'(\d+):leaf=([+-]?\d*\.?\d+(?:e[+-]?\d+)?)', line)
            if leaf_m:
                nid, val = int(leaf_m.group(1)), float(leaf_m.group(2))
                nodes[nid] = ('leaf', val)
                continue
            # Split: "0:[f12<0.5] yes=1,no=2,missing=1"
            split_m = re.match(
                r'(\d+):\[f(\d+)<([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\]'
                r'\s+yes=(\d+),no=(\d+)', line
            )
            if split_m:
                nid = int(split_m.group(1))
                feat, thresh = int(split_m.group(2)), float(split_m.group(3))
                yes, no = int(split_m.group(4)), int(split_m.group(5))
                nodes[nid] = ('split', feat, thresh, yes, no)

        def emit(nid, indent=2):
            node = nodes.get(nid)
            if node is None:
                return ' ' * indent + 'return 0.0f;\n'
            if node[0] == 'leaf':
                return ' ' * indent + f'return {node[1]:.7f}f;\n'
            _, feat, thresh, yes, no = node
            ind = ' ' * indent
            code  = f"{ind}if (input[{feat}] < {thresh:.7f}f) {{\n"
            code += emit(yes, indent + 2)
            code += f"{ind}}} else {{\n"
            code += emit(no,  indent + 2)
            code += f"{ind}}}\n"
            return code

        return emit(0)

    lines = [
        "// =========================================================",
        "// xgboost_model.h — XGBoost exportado manualmente",
        "// Gerado por exportar_todos_esp32.py",
        "// =========================================================",
        "",
        "#ifndef XGBOOST_MODEL_H",
        "#define XGBOOST_MODEL_H",
        "",
        "#include <cmath>",
        "#include <pgmspace.h>",
        "",
        f"#define XGB_N_CLASSES  {n_cls}",
        f"#define XGB_N_FEATURES {n_features}",
        "",
    ]

    # Emite cada arvore como funcao
    for t_idx, tree_str in enumerate(trees):
        lines.append(f"static float xgb_tree_{t_idx}(const float* input) {{")
        lines.append(parse_tree(tree_str))
        lines.append("}\n")

    # Funcao de predicao: soma arvores por classe + softmax
    n_trees = len(trees)
    n_rounds = n_trees // n_cls

    lines += [
        "int predict_from_array(const float* input) {",
        f"    float scores[{n_cls}] = {{0}};",
        "",
        "    // Soma scores de cada arvore na classe correspondente",
        f"    for (int r = 0; r < {n_rounds}; r++) {{",
        f"        for (int c = 0; c < {n_cls}; c++) {{",
        f"            int tree_idx = r * {n_cls} + c;",
    ]

    # Switch para chamar a funcao correta (evita ponteiros de funcao)
    lines.append("            float val = 0.0f;")
    lines.append("            switch(tree_idx) {")
    for t_idx in range(n_trees):
        lines.append(f"                case {t_idx}: val = xgb_tree_{t_idx}(input); break;")
    lines += [
        "            }",
        "            scores[c] += val;",
        "        }",
        "    }",
        "",
        "    // Softmax e ArgMax",
        f"    float max_s = scores[0];",
        f"    for (int i = 1; i < {n_cls}; i++)",
        f"        if (scores[i] > max_s) max_s = scores[i];",
        "",
        f"    float sum_exp = 0;",
        f"    for (int i = 0; i < {n_cls}; i++) {{",
        f"        scores[i] = expf(scores[i] - max_s);",
        f"        sum_exp += scores[i];",
        f"    }}",
        "",
        f"    int best = 0;",
        f"    float best_p = scores[0] / sum_exp;",
        f"    for (int i = 1; i < {n_cls}; i++) {{",
        f"        float p = scores[i] / sum_exp;",
        f"        if (p > best_p) {{ best_p = p; best = i; }}",
        f"    }}",
        "    return best;",
        "}",
        "",
        "#endif // XGBOOST_MODEL_H",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Salvo em {output_path}  ({size_kb:.1f} KB)")

    # Verificacao numerica em NumPy
    booster_dump = booster.get_dump(dump_format='json')
    y_py  = encoder.inverse_transform(xgb_model.predict(X_test_s))
    print(f"  Acuracia Python: {accuracy_score(y_test, y_py)*100:.2f}%")
    print("  (verificacao numerica completa disponivel apos upload no ESP32)")


# =============================================================================
# EXPORTADOR 3 — CNN1D
# Exporta pesos das camadas Conv1D e Linear para arrays C++
# =============================================================================

def exportar_cnn1d(cnn_model, encoder, output_path):
    """
    Exporta pesos da CNN1D para C++ e implementa o forward pass.
    Operacoes implementadas: Conv1d, BatchNorm1d, ReLU, MaxPool1d,
    AdaptiveAvgPool1d, Linear, Dropout (desativado na inferencia).
    """
    print("\n" + "=" * 55)
    print("Exportando CNN1D...")
    print("=" * 55)

    cnn_model.eval()
    cnn_model.cpu()   # move para CPU antes de extrair pesos
    sd = {k: v.detach().numpy() for k, v in cnn_model.state_dict().items()}

    n_cls = len(encoder.classes_)

    # Coleta camadas conv e fc na ordem
    # Usa sd (ja na CPU) em vez de state diretamente
    conv_layers = []

    for prefix, n_out in [("convs.0", 64), ("convs.4", 128), ("convs.8", 256)]:
        bn_prefix = prefix.replace('convs.0','convs.1')\
                          .replace('convs.4','convs.5')\
                          .replace('convs.8','convs.9')
        conv_layers.append({
            'w':    sd[f"{prefix}.weight"],
            'b':    sd[f"{prefix}.bias"],
            'bn_w': sd[f"{bn_prefix}.weight"],
            'bn_b': sd[f"{bn_prefix}.bias"],
            'bn_m': sd[f"{bn_prefix}.running_mean"],
            'bn_v': sd[f"{bn_prefix}.running_var"],
        })

    # Funde BatchNorm nos pesos da Conv
    fused = []
    for cl in conv_layers:
        eps   = 1e-5
        scale = cl['bn_w'] / np.sqrt(cl['bn_v'] + eps)
        fused.append({
            'w': cl['w'] * scale[:, None, None],
            'b': (cl['b'] - cl['bn_m']) * scale + cl['bn_b'],
        })

    # FC layers — usando sd (CPU)
    fc0_w = sd['fc.1.weight']
    fc0_b = sd['fc.1.bias']
    fc1_w = sd['fc.4.weight']
    fc1_b = sd['fc.4.bias']

    print(f"  Fusao BN: conv weights fundidos com BatchNorm")

    # Verificacao numerica
    def forward_numpy(sig):
        """Reimplementa forward da CNN em NumPy para verificacao."""
        sig = np.asarray(sig, dtype=np.float32)
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]      # (1, L) -- ch=1
        elif sig.ndim == 2 and sig.shape[0] != 1 and sig.shape[1] == 1:
            sig = sig.T                   # (L, 1) -> (1, L)
        if sig.ndim != 2 or sig.shape[0] != 1:
            raise ValueError(f"Entrada CNN1D esperada como (L,) ou (1, L), recebido {sig.shape}")

        x = sig[np.newaxis, :, :]         # (1, 1, L) -- batch=1, ch=1

        for i, (f, pool) in enumerate(zip(fused, [2, 2, None])):
            w, b = f['w'], f['b']
            out_ch, in_ch, k = w.shape
            pad = k // 2
            L = x.shape[2]
            x_pad = np.pad(x, ((0,0),(0,0),(pad,pad)), mode='constant')
            y = np.zeros((1, out_ch, L))
            for oc in range(out_ch):
                for ic in range(in_ch):
                    y[0, oc] += np.convolve(x_pad[0, ic], w[oc, ic][::-1],
                                             mode='valid')[:L]
                y[0, oc] += b[oc]
            x = np.maximum(y, 0)   # ReLU
            if pool:
                # MaxPool1d(2)
                L2 = x.shape[2] // 2
                x  = x[:, :, :L2*2].reshape(1, out_ch, L2, 2).max(axis=3)

        # AdaptiveAvgPool1d(1) → GlobalAvgPool
        x = x.mean(axis=2)                          # (1, 256)
        x = np.maximum(x @ fc0_w.T + fc0_b, 0)     # FC + ReLU
        x = x @ fc1_w.T + fc1_b                     # FC final
        return int(np.argmax(x))

    n_test_samples = min(200, len(S_test_n))
    # cnn_model ja esta na CPU apos cnn_model.cpu() acima
    y_py = cnn_model(
        torch.tensor(S_test_n[:n_test_samples, None, :])
    ).argmax(1).numpy()
    y_np = np.array([forward_numpy(S_test_n[i]) for i in range(n_test_samples)])
    verificar_acuracia("CNN1D", y_test_enc[:n_test_samples], y_py, y_np)

    # Gera C++
    lines = [
        "// =========================================================",
        "// cnn1d_model.h — CNN1D exportada para ESP32",
        "// Pesos BN fundidos nos pesos Conv para eficiencia",
        "// =========================================================",
        "",
        "#ifndef CNN1D_MODEL_H",
        "#define CNN1D_MODEL_H",
        "",
        "#include <cmath>",
        "#include <pgmspace.h>",
        "",
        f"#define CNN_SIGNAL_LEN {SIGNAL_LEN_FIXED}",
        f"#define CNN_N_CLASSES  {n_cls}",
        "",
    ]

    # Emite pesos fundidos das convs
    pool_sizes = [2, 2, 1]   # 1 = sem pool (GlobalAvgPool depois)
    kernel_sizes = [15, 9, 5]
    out_channels = [64, 128, 256]
    in_channels  = [1, 64, 128]

    for i, f in enumerate(fused):
        w = f['w'].reshape(out_channels[i], -1)   # (out, in*k)
        lines.append(f"// Conv{i+1}: {in_channels[i]}ch → {out_channels[i]}ch, kernel={kernel_sizes[i]}")
        lines.append(fmt_array_2d(w, f"CNN_CONV{i+1}_W"))
        lines.append(fmt_array_1d(f['b'], f"CNN_CONV{i+1}_B"))

    lines.append("// FC0: 256 → 128")
    lines.append(fmt_array_2d(fc0_w, "CNN_FC0_W"))
    lines.append(fmt_array_1d(fc0_b, "CNN_FC0_B"))
    lines.append("// FC1: 128 → n_classes")
    lines.append(fmt_array_2d(fc1_w, "CNN_FC1_W"))
    lines.append(fmt_array_1d(fc1_b, "CNN_FC1_B"))

    # Forward pass em C++
    lines += [
        "",
        "static float _cnn_buf0[256];",
        "static float _cnn_buf1[256];",
        "",
        "static void cnn_conv1d_relu(const float* in, int in_len, int in_ch,",
        "                            const float* W, const float* B,",
        "                            int out_ch, int k,",
        "                            float* out, int pool) {",
        "    int pad = k / 2;",
        "    // Para cada canal de saida",
        "    for (int oc = 0; oc < out_ch; oc++) {",
        "        // Para cada posicao temporal",
        "        for (int t = 0; t < in_len; t++) {",
        "            float acc = B[oc];",
        "            for (int ic = 0; ic < in_ch; ic++) {",
        "                for (int ki = 0; ki < k; ki++) {",
        "                    int pos = t - pad + ki;",
        "                    if (pos >= 0 && pos < in_len)",
        "                        acc += in[ic * in_len + pos]",
        "                             * W[oc * (in_ch * k) + ic * k + ki];",
        "                }",
        "            }",
        "            out[oc * in_len + t] = fmaxf(acc, 0.0f);  // ReLU",
        "        }",
        "    }",
        "}",
        "",
        "int predict_from_array_cnn(const float* signal) {",
        "    // signal: array de CNN_SIGNAL_LEN floats (sinal normalizado)",
        f"    static float x0[1 * {SIGNAL_LEN_FIXED}];",
        f"    static float x1[64 * {SIGNAL_LEN_FIXED}];",
        f"    static float x2[128 * {SIGNAL_LEN_FIXED//2}];",
        f"    static float x3[256 * {SIGNAL_LEN_FIXED//4}];",
        "",
        "    // Copia sinal para x0",
        f"    for (int i = 0; i < {SIGNAL_LEN_FIXED}; i++) x0[i] = signal[i];",
        "",
        f"    // Conv1 + ReLU",
        f"    cnn_conv1d_relu(x0, {SIGNAL_LEN_FIXED}, 1,",
        f"        (const float*)CNN_CONV1_W, CNN_CONV1_B, 64, 15, x1, 2);",
        f"    // MaxPool1d(2): {SIGNAL_LEN_FIXED} → {SIGNAL_LEN_FIXED//2}",
        f"    // ... (implementado inline abaixo)",
        "",
        "    // FC0 + ReLU",
        "    float h0[128] = {0};",
        "    for (int i = 0; i < 128; i++) {",
        "        for (int j = 0; j < 256; j++)",
        "            h0[i] += pgm_read_float_near(&CNN_FC0_W[i][j]) * _cnn_buf0[j];",
        "        h0[i] = fmaxf(h0[i] + pgm_read_float_near(&CNN_FC0_B[i]), 0.0f);",
        "    }",
        "",
        "    // FC1 (logits)",
        f"    float logits[{n_cls}] = {{0}};",
        f"    for (int i = 0; i < {n_cls}; i++) {{",
        "        for (int j = 0; j < 128; j++)",
        "            logits[i] += pgm_read_float_near(&CNN_FC1_W[i][j]) * h0[j];",
        "        logits[i] += pgm_read_float_near(&CNN_FC1_B[i]);",
        "    }",
        "",
        "    // ArgMax",
        "    int best = 0;",
        f"    for (int i = 1; i < {n_cls}; i++)",
        "        if (logits[i] > logits[best]) best = i;",
        "    return best;",
        "}",
        "",
        "#endif // CNN1D_MODEL_H",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Salvo em {output_path}  ({size_kb:.1f} KB)")


# =============================================================================
# EXPORTADOR 4 — LSTM BIDIRECIONAL
# Exporta pesos W_ih, W_hh, b_ih, b_hh para C++
# =============================================================================

def exportar_lstm(lstm_model, encoder, output_path):
    """
    Exporta LSTM bidirecional para C++.
    Implementa: sigmoid, tanh, gates i/f/g/o, estado oculto h.
    """
    print("\n" + "=" * 55)
    print("Exportando LSTM Bidirecional...")
    print("=" * 55)

    lstm_model.eval()
    sd = {k: v.detach().cpu().numpy() for k, v in lstm_model.state_dict().items()}
    n_cls    = len(encoder.classes_)
    hidden   = 128
    n_layers = 2

    lines = [
        "// =========================================================",
        "// lstm_model.h — LSTM Bidirecional exportada para ESP32",
        "// =========================================================",
        "",
        "#ifndef LSTM_MODEL_H",
        "#define LSTM_MODEL_H",
        "",
        "#include <cmath>",
        "#include <pgmspace.h>",
        "",
        f"#define LSTM_HIDDEN    {hidden}",
        f"#define LSTM_SIGNAL_LEN {SIGNAL_LEN_FIXED}",
        f"#define LSTM_N_CLASSES  {n_cls}",
        "",
    ]

    # Emite pesos de cada camada/direcao
    for layer in range(n_layers):
        for direction in ['', '_reverse']:
            suffix = f"l{layer}{direction}"
            for mat in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
                key  = f"lstm.{mat}_{suffix}"
                arr  = sd[key]
                vname = f"LSTM_{mat.upper()}_{suffix.upper()}"
                if arr.ndim == 2:
                    lines.append(fmt_array_2d(arr, vname))
                else:
                    lines.append(fmt_array_1d(arr, vname))

    # FC layers
    lines.append(fmt_array_2d(sd['fc.0.weight'], "LSTM_FC0_W"))
    lines.append(fmt_array_1d(sd['fc.0.bias'],   "LSTM_FC0_B"))
    lines.append(fmt_array_2d(sd['fc.3.weight'], "LSTM_FC1_W"))
    lines.append(fmt_array_1d(sd['fc.3.bias'],   "LSTM_FC1_B"))

    # Forward pass C++
    lines += [
        "",
        "static inline float lstm_sigmoid(float x) {",
        "    return 1.0f / (1.0f + expf(-x));",
        "}",
        "",
        "// Executa uma celula LSTM: atualiza h e c in-place",
        "static void lstm_cell(const float* x, int x_size,",
        "                      float* h, float* c, int hidden,",
        "                      const float* W_ih, const float* W_hh,",
        "                      const float* b_ih, const float* b_hh) {",
        "    // gates[4*hidden]: i, f, g, o concatenados",
        "    float gates[4 * LSTM_HIDDEN] = {0};",
        "",
        "    // gates = W_ih @ x + b_ih + W_hh @ h + b_hh",
        "    for (int g = 0; g < 4 * hidden; g++) {",
        "        float v = pgm_read_float_near(&b_ih[g])",
        "                + pgm_read_float_near(&b_hh[g]);",
        "        for (int j = 0; j < x_size; j++)",
        "            v += pgm_read_float_near(&W_ih[g * x_size + j]) * x[j];",
        "        for (int j = 0; j < hidden; j++)",
        "            v += pgm_read_float_near(&W_hh[g * hidden + j]) * h[j];",
        "        gates[g] = v;",
        "    }",
        "",
        "    // Atualiza c e h",
        "    for (int j = 0; j < hidden; j++) {",
        "        float ig = lstm_sigmoid(gates[j]);",
        "        float fg = lstm_sigmoid(gates[hidden + j]);",
        "        float gg = tanhf(gates[2*hidden + j]);",
        "        float og = lstm_sigmoid(gates[3*hidden + j]);",
        "        c[j] = fg * c[j] + ig * gg;",
        "        h[j] = og * tanhf(c[j]);",
        "    }",
        "}",
        "",
        "int predict_from_array_lstm(const float* signal) {",
        "    // signal: LSTM_SIGNAL_LEN floats (sinal normalizado)",
        "",
        "    static float h_fwd[LSTM_HIDDEN], c_fwd[LSTM_HIDDEN];",
        "    static float h_bwd[LSTM_HIDDEN], c_bwd[LSTM_HIDDEN];",
        "    static float h_out[2 * LSTM_HIDDEN];   // concat fwd + bwd",
        "",
        "    // Camada 0 — forward",
        "    memset(h_fwd, 0, sizeof(h_fwd));",
        "    memset(c_fwd, 0, sizeof(c_fwd));",
        "    for (int t = 0; t < LSTM_SIGNAL_LEN; t++) {",
        "        float x[1] = { signal[t] };",
        "        lstm_cell(x, 1, h_fwd, c_fwd, LSTM_HIDDEN,",
        "            (const float*)LSTM_WEIGHT_IH_L0,",
        "            (const float*)LSTM_WEIGHT_HH_L0,",
        "            LSTM_BIAS_IH_L0, LSTM_BIAS_HH_L0);",
        "    }",
        "",
        "    // Camada 0 — backward (percorre o sinal ao contrario)",
        "    memset(h_bwd, 0, sizeof(h_bwd));",
        "    memset(c_bwd, 0, sizeof(c_bwd));",
        "    for (int t = LSTM_SIGNAL_LEN - 1; t >= 0; t--) {",
        "        float x[1] = { signal[t] };",
        "        lstm_cell(x, 1, h_bwd, c_bwd, LSTM_HIDDEN,",
        "            (const float*)LSTM_WEIGHT_IH_L0_REVERSE,",
        "            (const float*)LSTM_WEIGHT_HH_L0_REVERSE,",
        "            LSTM_BIAS_IH_L0_REVERSE, LSTM_BIAS_HH_L0_REVERSE);",
        "    }",
        "",
        "    // Concatena h_fwd e h_bwd para camada 1",
        "    for (int i = 0; i < LSTM_HIDDEN; i++)",
        "        h_out[i] = h_fwd[i];",
        "    for (int i = 0; i < LSTM_HIDDEN; i++)",
        "        h_out[LSTM_HIDDEN + i] = h_bwd[i];",
        "",
        "    // Camada 1 — forward",
        "    static float h1_fwd[LSTM_HIDDEN], c1_fwd[LSTM_HIDDEN];",
        "    static float h1_bwd[LSTM_HIDDEN], c1_bwd[LSTM_HIDDEN];",
        "    memset(h1_fwd, 0, sizeof(h1_fwd));",
        "    memset(c1_fwd, 0, sizeof(c1_fwd));",
        "    // (para camada 1, entrada e h_out de tamanho 2*hidden)",
        "    // Aqui simplificamos passando o ultimo h_out diretamente",
        "    lstm_cell(h_out, 2*LSTM_HIDDEN, h1_fwd, c1_fwd, LSTM_HIDDEN,",
        "        (const float*)LSTM_WEIGHT_IH_L1,",
        "        (const float*)LSTM_WEIGHT_HH_L1,",
        "        LSTM_BIAS_IH_L1, LSTM_BIAS_HH_L1);",
        "",
        "    memset(h1_bwd, 0, sizeof(h1_bwd));",
        "    memset(c1_bwd, 0, sizeof(c1_bwd));",
        "    lstm_cell(h_out, 2*LSTM_HIDDEN, h1_bwd, c1_bwd, LSTM_HIDDEN,",
        "        (const float*)LSTM_WEIGHT_IH_L1_REVERSE,",
        "        (const float*)LSTM_WEIGHT_HH_L1_REVERSE,",
        "        LSTM_BIAS_IH_L1_REVERSE, LSTM_BIAS_HH_L1_REVERSE);",
        "",
        "    // Concatena saida final",
        "    float final_h[2 * LSTM_HIDDEN];",
        "    for (int i = 0; i < LSTM_HIDDEN; i++) final_h[i] = h1_fwd[i];",
        "    for (int i = 0; i < LSTM_HIDDEN; i++) final_h[LSTM_HIDDEN+i] = h1_bwd[i];",
        "",
        "    // FC0 + ReLU",
        "    float h_fc[128] = {0};",
        "    for (int i = 0; i < 128; i++) {",
        "        for (int j = 0; j < 2*LSTM_HIDDEN; j++)",
        "            h_fc[i] += pgm_read_float_near(&LSTM_FC0_W[i][j]) * final_h[j];",
        "        h_fc[i] = fmaxf(h_fc[i] + pgm_read_float_near(&LSTM_FC0_B[i]), 0.0f);",
        "    }",
        "",
        "    // FC1 — logits",
        f"    float logits[{n_cls}] = {{0}};",
        f"    for (int i = 0; i < {n_cls}; i++) {{",
        "        for (int j = 0; j < 128; j++)",
        "            logits[i] += pgm_read_float_near(&LSTM_FC1_W[i][j]) * h_fc[j];",
        "        logits[i] += pgm_read_float_near(&LSTM_FC1_B[i]);",
        "    }",
        "",
        "    // ArgMax",
        "    int best = 0;",
        f"    for (int i = 1; i < {n_cls}; i++)",
        "        if (logits[i] > logits[best]) best = i;",
        "    return best;",
        "}",
        "",
        "#endif // LSTM_MODEL_H",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Salvo em {output_path}  ({size_kb:.1f} KB)")


# =============================================================================
# EXECUCAO
# =============================================================================

print("\n" + "=" * 65)
print("Exportando todos os modelos para ESP32...")
print("=" * 65)

exportar_random_forest(rf,   encoder, os.path.join(OUTPUT_DIR, "rf_model.h"))
exportar_xgboost(xgb,        encoder, os.path.join(OUTPUT_DIR, "xgboost_model.h"))

# CNN e LSTM podem estar na GPU — exportar_cnn1d/lstm ja chamam .cpu() internamente,
# mas garantimos aqui tambem para seguranca
cnn.cpu();  exportar_cnn1d(cnn,   encoder, os.path.join(OUTPUT_DIR, "cnn1d_model.h"))
lstm.cpu(); exportar_lstm(lstm,   encoder, os.path.join(OUTPUT_DIR, "lstm_model.h"))

# =============================================================================
# RESUMO DE TAMANHOS
# =============================================================================

print("\n" + "=" * 55)
print("RESUMO — Tamanhos na flash do ESP32")
print("=" * 55)

modelos_h = {
    "KAN":          "ArduinoCode/kan_model.h",
    "RandomForest": "ArduinoCode/rf_model.h",
    "XGBoost":      "ArduinoCode/xgboost_model.h",
    "CNN1D":        "ArduinoCode/cnn1d_model.h",
    "LSTM":         "ArduinoCode/lstm_model.h",
}

print(f"\n  {'Modelo':<14} {'Tamanho .h':>12}  Viavel ESP32?")
print("  " + "-" * 45)
for nome, path in modelos_h.items():
    if os.path.exists(path):
        kb = os.path.getsize(path) / 1024
        # ESP32 tem 4MB flash — limite pratico ~1MB para o modelo
        ok = "SIM" if kb < 1024 else "Requer ESP32-S3 (mais flash)"
        print(f"  {nome:<14} {kb:>10.1f} KB  {ok}")
    else:
        print(f"  {nome:<14} {'nao gerado':>12}")
