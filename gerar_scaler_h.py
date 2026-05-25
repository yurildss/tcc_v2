# =============================================================================
# gerar_scaler_h.py
# Execute apos o treino para gerar o scaler.h para o ESP32.
# Le o scaler.pkl salvo pelo kolmo_corrigido.py e exporta os valores
# de media e desvio padrao para C++.
# =============================================================================

import joblib
import numpy as np

scaler  = joblib.load("ArduinoCode/scaler.pkl")
encoder = joblib.load("ArduinoCode/encoder.pkl")

means        = scaler.mean_
stds         = scaler.scale_
num_features = len(means)
num_classes  = len(encoder.classes_)
class_names  = list(encoder.classes_)

# Nomes das 46 features na ordem exata do features_estruturais.py
feature_names = [
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

assert len(feature_names) == 46, \
    f"Lista de nomes tem {len(feature_names)} entradas, esperado 46."

assert num_features == 46, (
    f"scaler.pkl tem {num_features} features, esperado 46. "
    f"Regere o scaler com kolmo_corrigido.py atualizado."
)

lines = []
lines.append("// scaler.h — gerado automaticamente por gerar_scaler_h.py")
lines.append("// NAO edite manualmente.")
lines.append(f"// Features: {num_features}  |  Classes: {num_classes}")
lines.append("")
lines.append("#ifndef SCALER_H")
lines.append("#define SCALER_H")
lines.append("")
lines.append(f"#define NUM_FEATURES {num_features}")
lines.append(f"#define NUM_CLASSES  {num_classes}")
lines.append("")

class_map = ", ".join([f"{i}={c}" for i, c in enumerate(class_names)])
lines.append(f"// Classes: {class_map}")
lines.append("")

lines.append("const float SCALER_MEAN[] = {")
for i, (v, name) in enumerate(zip(means, feature_names)):
    lines.append(f"    {v:.8f}f,  // [{i:2d}] {name}")
lines.append("};")
lines.append("")

lines.append("const float SCALER_STD[] = {")
for i, (v, name) in enumerate(zip(stds, feature_names)):
    lines.append(f"    {v:.8f}f,  // [{i:2d}] {name}")
lines.append("};")
lines.append("")
lines.append("#endif // SCALER_H")

output = "ArduinoCode/scaler.h"
with open(output, "w") as f:
    f.write("\n".join(lines))

print(f"scaler.h gerado em {output}")
print(f"  Features : {num_features}")
print(f"  Classes  : {num_classes} -> {class_names}")
