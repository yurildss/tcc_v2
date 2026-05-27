# =============================================================================
# features_estruturais.py (moved to src/features/extract.py)
# =============================================================================
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

FS = 360  # frequência de amostragem em Hz


def load_signal_with_structure(file_path):
    df = pd.read_csv(file_path)
    signal  = df['channel_0'].values.astype(np.float32)
    samples = df['sample #'].values.astype(np.int64)
    types   = df['type'].values

    peak_indices = np.where(types != 'NC')[0]

    if len(peak_indices) != 3:
        return None

    p_prev    = peak_indices[0]
    p_central = peak_indices[1]
    p_next    = peak_indices[2]

    return {
        'signal':    signal,
        'samples':   samples,
        'p_prev':    p_prev,
        'p_central': p_central,
        'p_next':    p_next,
    }


def extract_features(signal_data, fs=FS):
    sig       = signal_data['signal']
    samples   = signal_data['samples']
    p_prev    = signal_data['p_prev']
    p_central = signal_data['p_central']
    p_next    = signal_data['p_next']

    n = len(sig)
    features = []

    rr_pre  = float(samples[p_central] - samples[p_prev])
    rr_post = float(samples[p_next]    - samples[p_central])
    rr_mean = (rr_pre + rr_post) / 2.0

    features.append(rr_pre)
    features.append(rr_post)
    features.append(rr_post / (rr_pre + 1e-6))
    features.append(rr_pre  / (rr_mean + 1e-6))
    features.append(rr_post / (rr_mean + 1e-6))

    amp_prev    = float(sig[p_prev])
    amp_central = float(sig[p_central])
    amp_next    = float(sig[p_next])

    margin = int(0.05 * fs)
    seg_iso_pre  = sig[p_prev  + margin : p_central - margin]
    seg_iso_post = sig[p_central + margin : p_next   - margin]

    baseline_pre  = float(np.median(seg_iso_pre))  if len(seg_iso_pre)  > 5 else 0.0
    baseline_post = float(np.median(seg_iso_post)) if len(seg_iso_post) > 5 else 0.0
    baseline      = (baseline_pre + baseline_post) / 2.0

    amp_central_rel = amp_central - baseline

    features.append(amp_prev)
    features.append(amp_central)
    features.append(amp_next)
    features.append(amp_central_rel)
    features.append(amp_central / (amp_prev   + 1e-6))
    features.append(amp_central / (amp_next   + 1e-6))

    half_qrs = int(0.06 * fs)
    qrs_start = max(0,   p_central - half_qrs)
    qrs_end   = min(n-1, p_central + half_qrs)
    qrs_seg   = sig[qrs_start : qrs_end + 1]

    threshold = baseline + 0.5 * amp_central_rel
    above     = np.where(qrs_seg > threshold)[0]
    qrs_width = float(len(above)) / fs if len(above) > 0 else 0.0

    local_peak = p_central - qrs_start
    upstroke   = float(local_peak) / fs if local_peak > 0 else 0.0
    downstroke = float(len(qrs_seg) - local_peak - 1) / fs
    qrs_asymmetry = upstroke / (downstroke + 1e-6)

    features.append(qrs_width)
    features.append(upstroke)
    features.append(downstroke)
    features.append(qrs_asymmetry)

    pre_seg = sig[p_prev + margin : p_central - half_qrs]

    if len(pre_seg) > 5:
        pre_mean   = float(np.mean(pre_seg))
        pre_std    = float(np.std(pre_seg))
        pre_max    = float(np.max(pre_seg))
        pre_skew   = float(skew(pre_seg))
        pre_kurt   = float(kurtosis(pre_seg))
    else:
        pre_mean = pre_std = pre_max = pre_skew = pre_kurt = 0.0

    features.append(pre_mean  - baseline)
    features.append(pre_std)
    features.append(pre_max   - baseline)
    features.append(pre_skew)
    features.append(pre_kurt)

    post_seg = sig[p_central + half_qrs : p_next - margin]

    if len(post_seg) > 5:
        post_mean  = float(np.mean(post_seg))
        post_std   = float(np.std(post_seg))
        post_max   = float(np.max(post_seg))
        post_skew  = float(skew(post_seg))
        post_kurt  = float(kurtosis(post_seg))
    else:
        post_mean = post_std = post_max = post_skew = post_kurt = 0.0

    features.append(post_mean  - baseline)
    features.append(post_std)
    features.append(post_max   - baseline)
    features.append(post_skew)
    features.append(post_kurt)

    def area_above_baseline(seg, base):
        if len(seg) == 0:
            return 0.0
        return float(np.sum(seg - base)) / fs

    features.append(area_above_baseline(pre_seg,  baseline))
    features.append(area_above_baseline(qrs_seg,  baseline))
    features.append(area_above_baseline(post_seg, baseline))

    features.append(float(np.var(seg_iso_pre))  if len(seg_iso_pre)  > 5 else 0.0)
    features.append(float(np.var(qrs_seg)))
    features.append(float(np.var(seg_iso_post)) if len(seg_iso_post) > 5 else 0.0)

    win_size = int(0.25 * fs)
    cen_start = max(0,   p_central - win_size)
    cen_end   = min(n,   p_central + win_size)
    cen_seg   = sig[cen_start : cen_end]

    yf = np.abs(rfft(cen_seg.astype(np.float64)))
    xf = rfftfreq(len(cen_seg), 1 / fs)

    features.append(float(yf[(xf >= 0)  & (xf < 5)].sum()))
    features.append(float(yf[(xf >= 5)  & (xf < 15)].sum()))
    features.append(float(yf[(xf >= 15) & (xf < 40)].sum()))
    features.append(float(yf[(xf >= 40)].sum()))

    coeffs = pywt.wavedec(cen_seg.astype(np.float64), 'db4', level=4)
    for c in coeffs:
        features.append(float(np.sum(c ** 2)))

    p_wave_window = sig[p_prev + margin : p_central - int(0.04 * fs)]

    if len(p_wave_window) > 10:
        p_wave_max     = float(np.max(p_wave_window) - baseline)
        p_wave_present = 1.0 if p_wave_max > 0.05 else 0.0
        local_max_idx  = int(np.argmax(p_wave_window))
        p_wave_pos_rel = float(local_max_idx) / (len(p_wave_window) + 1e-6)
    else:
        p_wave_max = p_wave_present = p_wave_pos_rel = 0.0

    features.append(p_wave_present)
    features.append(p_wave_max)
    features.append(p_wave_pos_rel)

    features.append(float(skew(pre_seg))     if len(pre_seg)  > 5 else 0.0)
    features.append(float(skew(post_seg))    if len(post_seg) > 5 else 0.0)
    features.append(float(kurtosis(cen_seg)) if len(cen_seg)  > 5 else 0.0)

    assert len(features) == 46, f"Esperado 46 features, obtido {len(features)}"
    return features


def build_dataset(root_dir, fs=FS):
    import os
    X, y = [], []
    skipped = 0

    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        loaded = 0
        for file in os.listdir(class_path):
            if not file.endswith('.csv'):
                continue

            data = load_signal_with_structure(os.path.join(class_path, file))
            if data is None:
                skipped += 1
                continue

            try:
                feat = extract_features(data, fs)
                X.append(feat)
                y.append(class_name)
                loaded += 1
            except Exception as e:
                skipped += 1

        print(f"  {class_name}: {loaded} amostras carregadas")

    if skipped > 0:
        print(f"  (ignorados: {skipped} arquivos sem 3 picos ou com erro)")

    return np.array(X, dtype=np.float32), np.array(y)
