import librosa, random
import numpy as np
# from config import RAW_AUDIO_DIR_EN, RAW_AUDIO_DIR_BN, FEATURES_DIR, N_SUBSET, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from config import * # type: ignore

random.seed(42)

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def minmax_normalize(x, min_value=0.0, max_value=1.0):
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min: return np.full_like(x, min_value, dtype=np.float32)
    x = (x - x_min) / (x_max - x_min)
    x = x * (max_value - min_value) + min_value
    return x.astype(np.float32)

def extract_mel_spectrogram(audio_path):
    audio, _ = librosa.load(path=audio_path, sr=SAMPLE_RATE)
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    # return minmax_normalize(spec_db)
    return spec_db

def main():
    en_audio_list = [p for p in RAW_AUDIO_DIR_EN.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
    audio_list = en_audio_list
    audio_list = random.sample(audio_list, N_SUBSET) if N_SUBSET is not None else audio_list

    for audio_path in audio_list:
        try:
            spec = extract_mel_spectrogram(audio_path)
            out_path = FEATURES_DIR / f"{audio_path.stem}.npy"
            np.save(out_path, spec)
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            
if __name__ == "__main__":
    main()