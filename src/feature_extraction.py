import random, numpy as np, config
from utils.audio_data import extract_mel_spectrogram

random.seed(42)

config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    en_audio_list = [p for p in config.RAW_AUDIO_DIR_EN.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
    bn_audio_list = [p for p in config.RAW_AUDIO_DIR_BN.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
    audio_list = en_audio_list + bn_audio_list
    audio_list = random.sample(audio_list, config.N_SUBSET) if config.N_SUBSET is not None else audio_list

    for audio_path in audio_list:
        try:
            spec = extract_mel_spectrogram(audio_path)
            out_path = config.FEATURES_DIR / f"{audio_path.stem}.npy"
            np.save(out_path, spec)
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            
if __name__ == "__main__":
    main()