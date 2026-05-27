# gerar_dataset_h.py moved to src/export/generate_dataset_h.py
import os
import random
import numpy as np
import joblib

def generate_dataset_h(dataset_path, scaler, encoder, samples_per_class=50, output="firmware/generated_headers/dataset.h", fs=360):
    random.seed(42)
    np.random.seed(42)
    classes = sorted([c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))])
    X_raw, y_raw = [], []
    for label_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith(".csv")]
        random.shuffle(files)
        loaded = 0
        for fname in files:
            if loaded >= samples_per_class:
                break
            # Skipping actual feature extraction here — assume precomputed externally
            loaded += 1
            # placeholder zeros
            X_raw.append(np.zeros(46, dtype=np.float32))
            y_raw.append(label_idx)
    X_scaled = scaler.transform(np.array(X_raw)).astype(np.float32)
    n_samples = len(X_scaled)
    with open(output, 'w') as f:
        f.write("// dataset.h generated placeholder\n")
    return output
