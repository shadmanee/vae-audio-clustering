from pathlib import Path
import numpy as np, pandas as pd
from config import config
from utils.common import split_data, save_result_to_csv, combine_audio_and_lyrics, combine_audio_lyrics_and_genre
from datasets import AudioSpectrogramDataset
from run_models import run_KMeans

def extract_raw_features(loader):
    features, names, labels = [], [], []
    for batch in loader:
        x, paths, y = batch
        features.append(x.numpy().reshape(len(x), -1))
        if paths is not None or paths != -1: names.extend(paths)
        if y is not None or y != -1: labels.append(np.array(y))
    return (
        np.concatenate(features, axis=0),
        names,
        np.concatenate(labels, axis=0)
    )

def run_direct_clustering(
    modality: str = "audio",
    save_dir = "direct",
    root: Path = Path("."),
    features_dir = config.FEATURES_DIR,
):
    """
    Direct spectral feature clustering — no model, raw features -> KMeans.
    """
    npy_dir = root / features_dir
    
    dataset = AudioSpectrogramDataset(dataset_dir=npy_dir)

    train_loader, test_loader = split_data(
        dataset=dataset, ratio=0.8,
        batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE
    )

    train_feats, train_names, train_labels = extract_raw_features(train_loader)
    test_feats,  test_names,  test_labels  = extract_raw_features(test_loader)

    if modality == "audio_lyrics":
        train_feats = combine_audio_and_lyrics(train_feats, train_names, root=root)
        test_feats  = combine_audio_and_lyrics(test_feats,  test_names,  root=root)
        
    elif modality == "audio_lyrics_genre":
        labeled_df = pd.read_csv(root / config.METADATA_DIR / "metadata_en.csv")
        train_feats = combine_audio_lyrics_and_genre(train_feats, train_names, labeled_df=labeled_df, root=root)
        test_feats  = combine_audio_lyrics_and_genre(test_feats,  test_names,  labeled_df=labeled_df, root=root)

    all_feats  = np.concatenate([train_feats,  test_feats],  axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    print(f"Raw feature shape: {all_feats.shape}")

    kmeans, metrics = run_KMeans(
        latent_vecs=all_feats,
        model_type=save_dir,
        root=root
    )
    
    # print("I AM HERE")

    # save_result_to_csv(history=metrics, model_name=f"direct_{modality}", root=root)

    return kmeans, metrics, all_feats, all_labels