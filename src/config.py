from pathlib import Path

class BaseConfig:
    # general user defined (fixed) parameters
    RAW_AUDIO_DIR_EN = Path("data/audio/en_clips")
    RAW_AUDIO_DIR_BN = Path("data/audio/bn_clips")
    FEATURES_DIR = Path("data/features")
    RESULT_DIR = Path("results")

    DEBUG = False

    N_SUBSET = 1000 # an integer or None
    
    MODEL_TYPE = "basic"

    # spectrogram parameters
    SAMPLE_RATE = 22050
    N_FFT = 2048
    HOP_LENGTH = 735
    N_MELS = 64
    
    # vae user defined (fixed) parameters
    INPUT_HEIGHT = 64
    INPUT_WIDTH = 91
    INPUT_DIM = INPUT_HEIGHT * INPUT_WIDTH
    
    # general user defined (fixed) parameters
    EPOCHS = 2
    TRIALS = 2
    SHUFFLE = True

    # dataset related shared tunable parameters
    BATCH_SIZE: int = 16

    # vae related tunable parameters
    # LATENT_DIM = [8, 16, 32, 64][2]
    # HIDDEN_DIM_1 = [256, 512, 1024, 2048][1]
    # HIDDEN_DIM_2 = [64, 128, 256, 512][1]
    # BETA = [0.1, 0.5, 1.0, 2.0, 4.0]
    # LR = [1e-4, 3e-4, 1e-3, 3e-3][0]
    LATENT_DIM: int = 16
    HIDDEN_DIM_1: int = 512
    HIDDEN_DIM_2: int = 256
    BETA: float = 1.0
    BETA_TYPE: str = "fixed" # TODO: idt this should be a hyperparameter
    
    # general shared tunable parameters
    LR: float = 1e-4 # suggest_float returns float
    
config = BaseConfig()

"""
Basic VAE trial run best parameters:
{'latent_dim': 16,
 'hidden_dim_1': 512,
 'hidden_dim_2': 256,
 'lr': 0.00047128915005434553,
 'batch_size': 16}
"""
