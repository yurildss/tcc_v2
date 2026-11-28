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
    "RR_pre / RR mediano do paciente",
    "RR_post / RR mediano do paciente",
    "Razao RR post/pre",
    "RR_pre (s)",
    "RR_post (s)",
    # Bloco 2 — Amplitudes [5-10]
    "Amp pico anterior (rel. baseline) / amp_ref",
    "Amp pico central (rel. baseline) / amp_ref",
    "Amp pico posterior (rel. baseline) / amp_ref",
    "QRS pico-a-pico / amp_ref",
    "(Amp central - anterior) / amp_ref",
    "(Amp central - posterior) / amp_ref",
    # Bloco 3 — Morfologia QRS [11-14]
    "Largura QRS medida (s)",
    "QRS: inicio ate pico (s)",
    "QRS: pico ate fim (s)",
    "QRS: fracao antes do pico",
    # Bloco 4 — Segmento pre-QRS [15-19]
    "Pre-QRS: mean - baseline / amp_ref",
    "Pre-QRS: std / amp_ref",
    "Pre-QRS: max (onda P?) / amp_ref",
    "Pre-QRS: skewness",
    "Pre-QRS: kurtosis",
    # Bloco 5 — Segmento pos-QRS [20-24]
    "Pos-QRS: mean - baseline / amp_ref",
    "Pos-QRS: std / amp_ref",
    "Pos-QRS: max (onda T?) / amp_ref",
    "Pos-QRS: skewness",
    "Pos-QRS: kurtosis",
    # Bloco 6 — Areas [25-27]
    "Area segmento pre / amp_ref",
    "Area segmento QRS / amp_ref",
    "Area segmento pos / amp_ref",
    # Bloco 7 — Variancia [28-30]
    "Variancia isoeletrica pre / amp_ref^2",
    "Variancia QRS / amp_ref^2",
    "Variancia isoeletrica pos / amp_ref^2",
    # Bloco 8 — FFT do QRS [31-34]
    "FFT <5 Hz / amp_ref",
    "FFT 5-15 Hz / amp_ref",
    "FFT 15-40 Hz / amp_ref",
    "FFT >40 Hz / amp_ref",
    # Bloco 9 — Wavelet do QRS [35-39]
    "Wavelet energia lv1 (detalhe) / amp_ref^2",
    "Wavelet energia lv2 (detalhe) / amp_ref^2",
    "Wavelet energia lv3 (detalhe) / amp_ref^2",
    "Wavelet energia lv4 (detalhe) / amp_ref^2",
    "Wavelet energia approx / amp_ref^2",
    # Bloco 10 — Onda P [40-42]
    "Onda P: presente (0 ou 1)",
    "Onda P: amplitude / amp_ref",
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
