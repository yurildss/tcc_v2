# gerar_scaler_h.py moved to src/export/generate_scaler_h.py
import joblib
import numpy as np
def generate_scaler_h(scaler_path="firmware/generated_headers/scaler.pkl", output_path="firmware/generated_headers/scaler.h"):
    scaler  = joblib.load(scaler_path)
    encoder = joblib.load(scaler_path.replace('scaler.pkl','encoder.pkl')) if False else None
    means        = scaler.mean_
    stds         = scaler.scale_
    num_features = len(means)
    lines = []
    lines.append("// scaler.h — gerado automaticamente")
    lines.append("#ifndef SCALER_H")
    lines.append("#define SCALER_H")
    lines.append(f"#define NUM_FEATURES {num_features}")
    lines.append("")
    lines.append("const float SCALER_MEAN[] = {")
    for v in means:
        lines.append(f"    {v:.8f}f,")
    lines.append("};")
    lines.append("")
    lines.append("const float SCALER_STD[] = {")
    for v in stds:
        lines.append(f"    {v:.8f}f,")
    lines.append("};")
    lines.append("")
    lines.append("#endif // SCALER_H")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return output_path
