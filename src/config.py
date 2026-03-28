from pathlib import Path

RAW_AUDIO_DIR_EN = Path("data/audio/en_clips")
RAW_AUDIO_DIR_BN = Path("data/audio/bn_clips")
FEATURES_DIR = Path("data/features")

N_SUBSET = 2500 # an integer or None

# spectrogram parameters
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 735
N_MELS = 64

# dataset related parameters
BATCH_SIZE = [32, 64, 128]
SHUFFLE = True

# vae user input parameters
INPUT_HEIGHT = 64
INPUT_WIDTH = 91
INPUT_DIM = INPUT_HEIGHT * INPUT_WIDTH

# vae tunable parameters
LATENT_DIM = [8, 16, 32, 64][2]
HIDDEN_DIM_1 = [256, 512, 1024, 2048][1]
HIDDEN_DIM_2 = [64, 128, 256, 512][1]
DROPOUT = [0.0, 0.1, 0.2, 0.3][0]
BETA = [0.1, 0.5, 1.0, 2.0, 4.0][4]
LR = [1e-4, 3e-4, 1e-3, 3e-3][0]
EPOCHS = 30