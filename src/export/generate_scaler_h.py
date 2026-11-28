# =============================================================================
# src/export/generate_scaler_h.py
# =============================================================================
# Gera scaler.h (média e desvio do StandardScaler do treino) para o ESP32.
# Restaurado da versão original (gerar_scaler_h.py, commit 123a194).
# =============================================================================
import os

# Nomes das 46 features na ordem exata de src/features/extract.py
FEATURE_NAMES = [
    # Bloco 1 — RR Intervals [0-4]
    "RR_pre (amostras)",
    "RR_post (amostras)",
    "Razao RR post/pre",
    "RR_pre normalizado",
    "RR_post normalizado",
    # Bloco 2 — Amplitudes [5-10]
    "Amp pico anterior",
    "Amp pico central",
    "Amp pico posterior",
    "Amp central relativa (baseline)",
    "Razao amp central/anterior",
    "Razao amp central/posterior",
    # Bloco 3 — Morfologia QRS [11-14]
    "Largura QRS (s)",
    "Upstroke QRS (s)",
    "Downstroke QRS (s)",
    "Assimetria QRS",
    # Bloco 4 — Segmento pre-QRS [15-19]
    "Pre-QRS: mean - baseline",
    "Pre-QRS: std",
    "Pre-QRS: max (onda P?)",
    "Pre-QRS: skewness",
    "Pre-QRS: kurtosis",
    # Bloco 5 — Segmento pos-QRS [20-24]
    "Pos-QRS: mean - baseline",
    "Pos-QRS: std",
    "Pos-QRS: max (onda T?)",
    "Pos-QRS: skewness",
    "Pos-QRS: kurtosis",
    # Bloco 6 — Areas [25-27]
    "Area segmento pre",
    "Area segmento QRS",
    "Area segmento pos",
    # Bloco 7 — Variancia [28-30]
    "Variancia isoeletrica pre",
    "Variancia QRS",
    "Variancia isoeletrica pos",
    # Bloco 8 — FFT do QRS [31-34]
    "FFT <5 Hz",
    "FFT 5-15 Hz",
    "FFT 15-40 Hz",
    "FFT >40 Hz",
    # Bloco 9 — Wavelet do QRS [35-39]
    "Wavelet energia lv1 (detalhe)",
    "Wavelet energia lv2 (detalhe)",
    "Wavelet energia lv3 (detalhe)",
    "Wavelet energia lv4 (detalhe)",
    "Wavelet energia approx",
    # Bloco 10 — Onda P [40-42]
    "Onda P: presente (0 ou 1)",
    "Onda P: amplitude",
    "Onda P: posicao relativa",
    # Bloco 11 — Skew/Kurt por segmento [43-45]
    "Skewness segmento pre",
    "Skewness segmento pos",
    "Kurtosis QRS central",
]

assert len(FEATURE_NAMES) == 46


def generate_scaler_h(scaler, encoder, output_path="firmware/generated_headers/scaler.h"):
    means, stds = scaler.mean_, scaler.scale_
    num_features = len(means)
    class_names  = list(encoder.classes_)
    num_classes  = len(class_names)
    assert num_features == len(FEATURE_NAMES), (
        f"scaler tem {num_features} features, esperado {len(FEATURE_NAMES)}")

    class_map = ", ".join(f"{i}={c}" for i, c in enumerate(class_names))
    lines = [
        "// scaler.h — gerado automaticamente por src/export/generate_scaler_h.py",
        "// NAO edite manualmente.",
        f"// Features: {num_features}  |  Classes: {num_classes}",
        "",
        "#ifndef SCALER_H",
        "#define SCALER_H",
        "",
        f"#define NUM_FEATURES {num_features}",
        f"#define NUM_CLASSES  {num_classes}",
        "",
        f"// Classes: {class_map}",
        "",
        "const float SCALER_MEAN[] = {",
    ]
    lines += [f"    {v:.8f}f,  // [{i:2d}] {n}" for i, (v, n) in enumerate(zip(means, FEATURE_NAMES))]
    lines += ["};", "", "const float SCALER_STD[] = {"]
    lines += [f"    {v:.8f}f,  // [{i:2d}] {n}" for i, (v, n) in enumerate(zip(stds, FEATURE_NAMES))]
    lines += ["};", "", "#endif // SCALER_H", ""]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  scaler.h gerado em {output_path} ({num_features} features, classes: {class_map})")
    return output_path
