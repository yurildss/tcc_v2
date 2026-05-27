# exportar_todos_esp32.py moved to src/export/export_all_esp32.py
import os
import numpy as np

def fmt_array_1d(arr, name, pg="PROGMEM "):
    vals = ", ".join(f"{float(v):.7f}f" for v in arr.flatten())
    return f"const float {pg}{name}[{len(arr.flatten())}] = {{{vals}}};\n"

def fmt_array_2d(arr, name, pg="PROGMEM "):
    rows = []
    for row in arr:
        vals = ", ".join(f"{float(v):.7f}f" for v in row)
        rows.append(f"    {{ {vals} }}")
    body = ",\n".join(rows)
    r, c = arr.shape
    return f"const float {pg}{name}[{r}][{c}] = {{\n{body}\n}};\n"

def exportar_random_forest(rf_model, encoder, output_path):
    try:
        from micromlgen import port
        cpp_code = port(rf_model, classname='RFClassifier', classmap={i: c for i, c in enumerate(encoder.classes_)})
        wrapper = "\nint predict_from_array(const float* input) { return RFClassifier::predict(input); }\n"
        cpp_code += wrapper
        with open(output_path, "w") as f:
            f.write(cpp_code)
        return True
    except ImportError:
        print("micromlgen nao instalado — instale antes de exportar RF")
        return False

def exportar_xgboost(xgb_model, encoder, output_path, n_features=46):
    # Implementação completa mantida no original — this is a placeholder
    print("exportar_xgboost placeholder — use original script if needed")
    return False
