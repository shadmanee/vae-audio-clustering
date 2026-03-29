import numpy as np, librosa
from config import config

def check_global_min_max(specs, n_sub=200):
    mins, maxs = [], []
    for spec in specs[:n_sub]:
        x = np.load(spec)
        mins.append(x.min())
        maxs.append(x.max())

    # print("min of mins:", min(mins))
    # print("max of maxs:", max(maxs))
    # print("avg min:", sum(mins)/len(mins))
    # print("avg max:", sum(maxs)/len(maxs))
    
    return {
        "min_min": min(mins),
        "max_max": max(maxs),
        "avg_min": sum(mins)/n_sub,
        "avg_max": sum(maxs)/n_sub
    }
    
def minmax_normalize(x, min_value=0.0, max_value=1.0):
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min: return np.full_like(x, min_value, dtype=np.float32)
    x = (x - x_min) / (x_max - x_min)
    x = x * (max_value - min_value) + min_value
    return x.astype(np.float32)

def extract_mel_spectrogram(audio_path):
    audio, _ = librosa.load(path=audio_path, sr=config.SAMPLE_RATE)
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db