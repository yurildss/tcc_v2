# =============================================================================
# src/export/export_kan_cpp.py
# =============================================================================
# Exporta um MultKAN (pykan) treinado para C++ (ESP32) avaliando as B-splines
# numericamente — sem prune() nem auto_symbolic(), que derrubavam a acurácia
# no dispositivo.
#
# As bases B-spline são calculadas pela recursão de Cox–de Boor sobre o grid
# estendido inteiro, exatamente como kan/spline.py (B_batch): sem prender x ao
# domínio do grid; fora dele a parte spline vale 0. Isso dá concordância de
# 100% entre PyTorch e C++ (a versão com clamp + De Boor divergia quando
# alguma entrada caía fora do grid).
#
# Restaurado da versão original (exportar_kan_cpp.py, commit 123a194): a cópia
# em src/ tinha virado um stub sem o forward pass em C++ nem a verificação.
#
# Forward pass por nó j da camada l:
#   out[j] = node_bias[l][j]
#            + sum_i( mask[i,j] * (scale_base[i,j] * silu(x_i)
#                                 + scale_sp[i,j]  * bspline(x_i, grid[i], coef[i,j])) )
#
# Uso: chamado por src/export/export_all_esp32.py
# =============================================================================

import os
import torch
import numpy as np


# =============================================================================
# EXPORTADOR PRINCIPAL
# =============================================================================

def export_kan_to_cpp(model, output_path="firmware/generated_headers/kan_model.h",
                      class_names=None, use_progmem=True):
    """
    Exporta o MultKAN para um header C++ com forward pass numérico completo.

    Parâmetros
    ----------
    model       : MultKAN treinado (pykan)
    output_path : caminho do .h de saída
    class_names : lista de nomes das classes (para comentários)
    use_progmem : armazena arrays na flash do ESP32 (recomendado)
    """
    model.eval()

    # -------------------------------------------------------------------------
    # 1. Metadados da arquitetura
    # -------------------------------------------------------------------------
    # width é lista de listas: [[23,0],[16,0],[8,0],[8,0]]
    # Pegamos apenas o primeiro elemento de cada sublista (neurônios reais)
    raw_width = model.width
    width = []
    for w in raw_width:
        if isinstance(w, (list, tuple)):
            width.append(int(w[0]))
        else:
            width.append(int(w))

    depth = model.depth
    k     = model.k

    print("=" * 60)
    print("Exportando KAN para C++ (forward pass numérico)")
    print("=" * 60)
    print(f"Arquitetura : {' → '.join(map(str, width))}")
    print(f"Profundidade: {depth} camadas")
    print(f"Ordem spline: k = {k}")

    # -------------------------------------------------------------------------
    # 2. Extrai pesos de cada camada
    # -------------------------------------------------------------------------
    layers_data = []

    for l_idx in range(depth):
        layer = model.act_fun[l_idx]

        n_in  = int(layer.in_dim)
        n_out = int(layer.out_dim)

        # grid       : (n_in, n_knots)
        grid       = layer.grid.detach().cpu().float().numpy()
        # coef       : (n_in, n_out, n_coef)
        coef       = layer.coef.detach().cpu().float().numpy()
        # scale_base : (n_in, n_out)
        scale_base = layer.scale_base.detach().cpu().float().numpy()
        # scale_sp   : (n_in, n_out)
        scale_sp   = layer.scale_sp.detach().cpu().float().numpy()
        # mask       : (n_in, n_out)
        mask       = layer.mask.detach().cpu().float().numpy()

        # node_bias  : model.node_bias[l_idx] — shape (n_out,) ou None
        nb = model.node_bias[l_idx]
        if nb is not None:
            node_bias = nb.detach().cpu().float().numpy().flatten()
        else:
            node_bias = np.zeros(n_out, dtype=np.float32)

        n_knots = grid.shape[1]
        n_coef  = coef.shape[2]

        print(f"\n  Camada {l_idx}: {n_in} → {n_out}")
        print(f"    grid       : {grid.shape}")
        print(f"    coef       : {coef.shape}")
        print(f"    scale_base : {scale_base.shape}")
        print(f"    scale_sp   : {scale_sp.shape}")
        print(f"    mask       : {mask.shape}")
        print(f"    node_bias  : {node_bias.shape}")

        layers_data.append({
            'n_in':       n_in,
            'n_out':      n_out,
            'n_knots':    n_knots,
            'n_coef':     n_coef,
            'grid':       grid,        # (n_in, n_knots)
            'coef':       coef,        # (n_in, n_out, n_coef)
            'scale_base': scale_base,  # (n_in, n_out)
            'scale_sp':   scale_sp,    # (n_in, n_out)
            'mask':       mask,        # (n_in, n_out)
            'node_bias':  node_bias,   # (n_out,)
        })

    # -------------------------------------------------------------------------
    # 3. Gera o arquivo C++
    # -------------------------------------------------------------------------
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True
    )

    pg = "PROGMEM " if use_progmem else ""

    def fmt_row(arr):
        """Array 1-D → string C++ de floats."""
        return ", ".join(f"{float(v):.7f}f" for v in arr)

    lines = []

    # --- Cabeçalho ---
    lines += [
        "// =========================================================================",
        "// kan_model.h — KAN forward pass numérico (B-splines sem aproximação)",
        "// Gerado por: exportar_kan_cpp.py",
        "//",
        "// Sem prune() nem auto_symbolic() — acurácia idêntica ao Python.",
        "//",
        f"// Arquitetura : {' → '.join(map(str, width))}",
        f"// Ordem spline: k = {k}",
        "// =========================================================================",
        "",
        "#ifndef KAN_MODEL_H",
        "#define KAN_MODEL_H",
        "",
        "#include <cmath>",
        "#include <pgmspace.h>",
        "",
        f"#define KAN_DEPTH     {depth}",
        f"#define KAN_K         {k}",
        f"#define KAN_N_IN      {width[0]}",
        f"#define KAN_N_CLASSES {width[-1]}",
        "",
    ]

    if class_names:
        lines.append("// Classes: " + ", ".join(
            f"{i}={c}" for i, c in enumerate(class_names)))
        lines.append("")

    # --- Arrays por camada ---
    for l_idx, ld in enumerate(layers_data):
        n_in    = ld['n_in']
        n_out   = ld['n_out']
        n_knots = ld['n_knots']
        n_coef  = ld['n_coef']

        lines.append(f"// ── Camada {l_idx}  ({n_in} entradas → {n_out} saídas) ──")

        # grid: (n_in, n_knots) — um por entrada
        lines.append(f"// grid[i][*] = knots da entrada i")
        lines.append(
            f"const float {pg}L{l_idx}_GRID[{n_in}][{n_knots}] = {{")
        for i in range(n_in):
            lines.append(f"    /* in={i:2d} */ {{ {fmt_row(ld['grid'][i])} }},")
        lines.append("};\n")

        # coef: (n_in, n_out, n_coef)
        lines.append(f"// coef[i][j][*] = coeficientes da aresta (in=i, out=j)")
        lines.append(
            f"const float {pg}L{l_idx}_COEF[{n_in}][{n_out}][{n_coef}] = {{")
        for i in range(n_in):
            lines.append(f"    /* in={i:2d} */ {{")
            for j in range(n_out):
                lines.append(f"        /* out={j:2d} */ {{ {fmt_row(ld['coef'][i, j])} }},")
            lines.append("    },")
        lines.append("};\n")

        # scale_base: (n_in, n_out)
        lines.append(f"// scale_base[i][j] = peso residual SiLU da aresta (i→j)")
        lines.append(
            f"const float {pg}L{l_idx}_SCALE_BASE[{n_in}][{n_out}] = {{")
        for i in range(n_in):
            lines.append(f"    /* in={i:2d} */ {{ {fmt_row(ld['scale_base'][i])} }},")
        lines.append("};\n")

        # scale_sp: (n_in, n_out)
        lines.append(f"// scale_sp[i][j] = peso da spline da aresta (i→j)")
        lines.append(
            f"const float {pg}L{l_idx}_SCALE_SP[{n_in}][{n_out}] = {{")
        for i in range(n_in):
            lines.append(f"    /* in={i:2d} */ {{ {fmt_row(ld['scale_sp'][i])} }},")
        lines.append("};\n")

        # mask: (n_in, n_out)
        lines.append(f"// mask[i][j] = 1 se a aresta (i→j) está ativa, 0 se podada")
        lines.append(
            f"const float {pg}L{l_idx}_MASK[{n_in}][{n_out}] = {{")
        for i in range(n_in):
            lines.append(f"    /* in={i:2d} */ {{ {fmt_row(ld['mask'][i])} }},")
        lines.append("};\n")

        # node_bias: (n_out,)
        lines.append(f"// node_bias[j] = bias do nó de saída j")
        lines.append(
            f"const float {pg}L{l_idx}_BIAS[{n_out}] = {{ {fmt_row(ld['node_bias'])} }};\n")

    # --- Bases B-spline exatamente como o pykan (kan/spline.py: B_batch) ---
    # Cox–de Boor sobre o grid ESTENDIDO inteiro, sem prender x ao domínio:
    # fora de [knots[0], knots[n-1]) todas as bases valem 0 e só sobra a
    # parte SiLU — igual ao PyTorch. (A versão anterior prendia x ao domínio
    # útil do grid, o que divergia do PyTorch para entradas fora dele.)
    max_coef  = max(ld['n_coef'] for ld in layers_data)
    max_knots = max(ld['n_knots'] for ld in layers_data)
    lines += [
        "// =========================================================================",
        "// kan_bases — bases B-spline de ordem k (Cox–de Boor), espelho do pykan",
        "// Preenche B[0..n_knots-k-2]. Fora do grid estendido, todas valem 0.",
        "// =========================================================================",
        f"#define KAN_MAX_KNOTS {max_knots}",
        f"#define KAN_MAX_COEF  {max_coef}",
        "static void kan_bases(float x, const float* knots, int n_knots, int k, float* B) {",
        "    float g[KAN_MAX_KNOTS];",
        "    for (int i = 0; i < n_knots; i++) g[i] = pgm_read_float_near(&knots[i]);",
        "    float N[KAN_MAX_KNOTS];",
        "    // ordem 0: indicadora do intervalo [g[i], g[i+1])",
        "    for (int i = 0; i < n_knots - 1; i++)",
        "        N[i] = (x >= g[i] && x < g[i + 1]) ? 1.0f : 0.0f;",
        "    // ordens 1..k (mesma fórmula e mesma ordem de cálculo do pykan)",
        "    for (int p = 1; p <= k; p++) {",
        "        for (int i = 0; i < n_knots - 1 - p; i++) {",
        "            float d1 = g[i + p] - g[i];",
        "            float d2 = g[i + p + 1] - g[i + 1];",
        "            float t1 = (d1 != 0.0f) ? (x - g[i]) / d1 * N[i] : 0.0f;",
        "            float t2 = (d2 != 0.0f) ? (g[i + p + 1] - x) / d2 * N[i + 1] : 0.0f;",
        "            N[i] = t1 + t2;",
        "        }",
        "    }",
        "    for (int i = 0; i < n_knots - 1 - k; i++) B[i] = N[i];",
        "}",
        "",
        "// SiLU: x * sigmoid(x) = x / (1 + exp(-x))",
        "static inline float silu(float x) {",
        "    return x / (1.0f + expf(-x));",
        "}",
        "",
    ]

    max_dim = max(width)
    lines += [
        f"static float _buf_a[{max_dim}];  // activações de entrada da camada atual",
        f"static float _buf_b[{max_dim}];  // activações de saída  da camada atual",
        "",
        "// =========================================================================",
        "// predict_from_array",
        "// Recebe ponteiro para array de features já escalonadas.",
        "// Retorna o índice da classe com maior logit.",
        "// As bases dependem só de (x_i, grid_i): são calculadas uma vez por",
        "// entrada e reaproveitadas para todas as saídas j.",
        "// =========================================================================",
        "int predict_from_array(const float* input) {",
        "    float B[KAN_MAX_COEF];",
        f"    for (int i = 0; i < {width[0]}; i++) _buf_a[i] = input[i];",
        "",
    ]
    for l_idx, ld in enumerate(layers_data):
        n_in, n_out = ld['n_in'], ld['n_out']
        n_knots, n_coef = ld['n_knots'], ld['n_coef']
        lines += [
            f"    // ── Camada {l_idx} ({n_in} → {n_out}) ──────────────────────────",
            f"    for (int j = 0; j < {n_out}; j++) _buf_b[j] = pgm_read_float_near(&L{l_idx}_BIAS[j]);",
            f"    for (int i = 0; i < {n_in}; i++) {{",
            f"        float x = _buf_a[i];",
            f"        float base_val = silu(x);",
            f"        kan_bases(x, L{l_idx}_GRID[i], {n_knots}, KAN_K, B);",
            f"        for (int j = 0; j < {n_out}; j++) {{",
            f"            float m = pgm_read_float_near(&L{l_idx}_MASK[i][j]);",
            f"            if (m == 0.0f) continue;  // aresta inativa — pula",
            f"            float sp_val = 0.0f;",
            f"            for (int c = 0; c < {n_coef}; c++)",
            f"                sp_val += pgm_read_float_near(&L{l_idx}_COEF[i][j][c]) * B[c];",
            f"            float sb = pgm_read_float_near(&L{l_idx}_SCALE_BASE[i][j]);",
            f"            float ss = pgm_read_float_near(&L{l_idx}_SCALE_SP[i][j]);",
            f"            _buf_b[j] += m * (sb * base_val + ss * sp_val);",
            f"        }}",
            f"    }}",
            f"    for (int i = 0; i < {n_out}; i++) _buf_a[i] = _buf_b[i];",
            f"",
        ]

    n_classes = width[-1]
    lines += [
        "    // ArgMax",
        "    int   best     = 0;",
        "    float best_val = _buf_a[0];",
        f"    for (int i = 1; i < {n_classes}; i++) {{",
        "        if (_buf_a[i] > best_val) { best_val = _buf_a[i]; best = i; }",
        "    }",
        "    return best;",
        "}",
        "",
    ]

    # --- Wrapper predict(f0, f1, ..., fN) para compatibilidade com .ino ---
    params   = ", ".join(f"float x_{i}" for i in range(width[0]))
    arr_fill = "\n".join(f"    arr[{i}] = x_{i};" for i in range(width[0]))
    lines += [
        "// Wrapper: recebe features como argumentos individuais",
        f"int predict({params}) {{",
        f"    float arr[{width[0]}];",
        arr_fill,
        "    return predict_from_array(arr);",
        "}",
        "",
        "#endif // KAN_MODEL_H",
    ]

    # --- Grava ---
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    total_floats = sum(
        ld['n_in'] * ld['n_knots']           # grid
        + ld['n_in'] * ld['n_out'] * ld['n_coef']  # coef
        + ld['n_in'] * ld['n_out'] * 3       # scale_base + scale_sp + mask
        + ld['n_out']                         # node_bias
        for ld in layers_data
    )
    size_kb = total_floats * 4 / 1024

    print(f"\nArquivo gerado : {output_path}")
    print(f"Floats totais  : {total_floats:,}")
    print(f"Tamanho flash  : ~{size_kb:.1f} KB")
    return output_path


# =============================================================================
# VERIFICAÇÃO NUMÉRICA
# Reimplementa o forward pass em NumPy e compara com PyTorch
# =============================================================================

def verify_export(model, X_val_t, y_val_enc, n_samples=200):
    """
    Compara acurácia do PyTorch com a implementação NumPy espelho do C++.
    Se coincidirem, o C++ gerado terá a mesma acurácia.
    """
    print("\nVerificando exportação numericamente...")
    model.eval()

    raw_width = model.width
    width = [int(w[0]) if isinstance(w, (list, tuple)) else int(w) for w in raw_width]
    depth = model.depth
    k     = model.k

    def bases_np(x, knots, k):
        """Bases B-spline como o pykan (Cox–de Boor no grid estendido, sem clamp)."""
        g = knots
        n = len(g)
        N = np.array([1.0 if (g[i] <= x < g[i + 1]) else 0.0 for i in range(n - 1)])
        for p in range(1, k + 1):
            novo = np.zeros(n - 1 - p)
            for i in range(n - 1 - p):
                d1, d2 = g[i + p] - g[i], g[i + p + 1] - g[i + 1]
                t1 = (x - g[i]) / d1 * N[i] if d1 != 0 else 0.0
                t2 = (g[i + p + 1] - x) / d2 * N[i + 1] if d2 != 0 else 0.0
                novo[i] = t1 + t2
            N = novo
        return N   # n - k - 1 bases = n_coef


    def silu_np(x):
        return x / (1.0 + np.exp(-x))

    def forward_np(model, x_np):
        a = x_np.astype(np.float64).copy()
        for l_idx in range(depth):
            layer      = model.act_fun[l_idx]
            grid       = layer.grid.detach().cpu().numpy().astype(np.float64)
            coef       = layer.coef.detach().cpu().numpy().astype(np.float64)
            scale_base = layer.scale_base.detach().cpu().numpy().astype(np.float64)
            scale_sp   = layer.scale_sp.detach().cpu().numpy().astype(np.float64)
            mask       = layer.mask.detach().cpu().numpy().astype(np.float64)
            nb = model.node_bias[l_idx]
            b = (nb.detach().cpu().numpy().flatten().astype(np.float64)
                 if nb is not None else np.zeros(layer.out_dim))
            b = b.copy()
            for i in range(int(layer.in_dim)):
                x = a[i]
                B = bases_np(x, grid[i], k)
                sp = coef[i] @ B            # (n_out,)
                b += mask[i] * (scale_base[i] * silu_np(x) + scale_sp[i] * sp)
            a = b

        return int(np.argmax(a))

    X_np = X_val_t.numpy()[:n_samples]
    y_np = y_val_enc[:n_samples]

    with torch.no_grad():
        y_torch = torch.argmax(model(X_val_t[:n_samples]), dim=1).numpy()

    y_mirror = np.array([forward_np(model, X_np[i]) for i in range(n_samples)])
    correct_torch = int(np.sum(y_torch == y_np))
    correct_np    = int(np.sum(y_mirror == y_np))
    concordancia  = 100.0 * float(np.mean(y_mirror == y_torch))

    acc_torch = 100.0 * correct_torch / n_samples
    acc_np    = 100.0 * correct_np    / n_samples

    print(f"  PyTorch (referência) : {acc_torch:.2f}%")
    print(f"  NumPy espelho do C++ : {acc_np:.2f}%")
    print(f"  Concordância amostra a amostra (PyTorch x espelho C++): {concordancia:.2f}%")

    delta = abs(acc_torch - acc_np)
    if delta < 1.0:
        print(f"  OK — diferença {delta:.2f}% < 1%. Exportação consistente.")
    else:
        print(f"  AVISO — diferença {delta:.2f}% >= 1%. Verifique os pesos.")

    return acc_np
