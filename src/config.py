from pathlib import Path

RAW_AUDIO_DIR_EN = Path("data/audio/en_clips")
RAW_AUDIO_DIR_BN = Path("data/audio/bn_clips")
FEATURES_DIR = Path("data/features")

N_SUBSET = 500 # an integer or None

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 735
N_MELS = 64