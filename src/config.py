from pathlib import Path
import os

class BaseConfig:
    # general user defined (fixed) parameters
    RAW_AUDIO_DIR_EN = Path("data/audio/en_clips") # or "en_clips"
    RAW_AUDIO_DIR_BN = Path("data/audio/bn_clips")
    FEATURES_DIR = Path("data/features/en") # "en_bn" (for en + bn audio features) or "en_only"
    LYRICS_DIR_EN = Path("data/lyrics/en") # "test_en" or "en"
    EMBEDDINGS_DIR = Path("data/embeddings/en") # "test_en" or "en"
    RESULT_DIR = Path("results")
    METADATA_DIR = Path("data/metadata") # TODO: update to metadata file?
    TSNE_DIR = RESULT_DIR / "clustering" / "tsne"

    N_SUBSET = 2500 # an integer or None

    # spectrogram parameters
    SAMPLE_RATE = 22050
    N_FFT = 2048
    HOP_LENGTH = 735
    N_MELS = 64
    
    # vae user defined (fixed) parameters
    INPUT_HEIGHT = 64
    INPUT_WIDTH = 91
    INPUT_DIM = INPUT_HEIGHT * INPUT_WIDTH
    MODEL_TYPE = "cvae" # options: basic, conv, cvae, beta
    VAE_TYPE = "conv" # ONLY CHANGE WHEN WORKING WITH BETA/CVAE: basic/conv
    
    # general user defined (fixed) parameters
    EPOCHS = 5 # TODO: update 
    TRIALS = 5 # TODO: update
    SHUFFLE = True
    # anneal over first 20% of total epochs
    ANNEALING_EPOCHS = int(0.2 * EPOCHS)

    # dataset related shared tunable parameters
    BATCH_SIZE: int = 16

    # vae related tunable parameters
    LATENT_DIM: int = 32
    HIDDEN_DIM_1: int = 512 # conv channel layer-3
    HIDDEN_DIM_2: int = 256 # conv channel layer-2
    HIDDEN_DIM_3: int = 128 # conv channel layer-1
    BETA: float = 1.0
    
    # NOT TUNABLE - beta parameters
    # cases: 1. constant beta, fixed, 2. constant beta, annealing, 3. tunable beta, fixed, 4. tunable beta, annealing
    BETA_TYPE: str = "annealing" # or "annealing"
    BETA_CHOICE: str = "tunable" # or "tunable"
    
    # general shared tunable parameters
    LR: float = 1e-4 # suggest_float returns float
    
    # HuggingFace
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    
config = BaseConfig()

"""
Basic VAE trial run best parameters:
{'latent_dim': 16,
 'hidden_dim_1': 512,
 'hidden_dim_2': 256,
 'lr': 0.00047128915005434553,
 'batch_size': 16}
"""
