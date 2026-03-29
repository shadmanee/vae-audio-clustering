from pathlib import Path

from sentence_transformers import SentenceTransformer
from config import config
import random, numpy as np, pandas as pd, os

from dotenv import load_dotenv

load_dotenv() 

token = os.environ.get("HF_TOKEN", None)

random.seed(42)

config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
model = SentenceTransformer('all-mpnet-base-v2', token=token)

if __name__ == "__main__":
    en_lyric_list = [p.stem for p in config.LYRICS_DIR_EN.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    
    # all_feature_files = list(config.LYRICS_DIR_EN.glob("*.npy"))    
    # parent_ids = [f.stem.split('_')[0] for f in all_feature_files]
    # unique_song_ids = sorted(list(set(parent_ids)))
    
    df_meta = pd.read_csv("data/metadata/metadata_en.csv")
    relevant_meta = df_meta[df_meta['lyric_file_stem'].isin(en_lyric_list)]
    
    for lyric_path in en_lyric_list:
        song_id = relevant_meta[relevant_meta["lyric_file_stem"]==lyric_path]["audio_file_stem"].item()
        song_id = str(song_id)
        
        try:
            main_file_path = Path(config.LYRICS_DIR_EN / Path(f"{lyric_path}.txt"))
            if main_file_path.exists():
                with open(main_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Generate vector (384 dimensions)
                z_text = model.encode(text)
                
                # Save as .npy using the song_id as name
                np.save(config.EMBEDDINGS_DIR / f"{song_id}.npy", z_text)
            else:
                print(f"Warning: File not found at {main_file_path}")
                
        except Exception as e:
            print(f"Error processing {song_id}: {e}")
