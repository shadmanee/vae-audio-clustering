from utils.common import extract_latents_with_names, load_model, split_data, combine_audio_lyrics_and_genre
from datasets import AudioSpectogramDatasetwithLabels
from run_models import run_KMeans, run_DBSCAN, run_Agglomerative
from visualizations import plot_latent_space_by_cluster
from config import config

import torch, numpy as np, pandas as pd
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path(".")

cvae, cfg = load_model(model_type="cvae", weights_path=root/config.RESULT_DIR/r"models\hard_1_cvae_audio_only_20260421_194729.pt", cfg_path=root/config.RESULT_DIR/r"models\hard_1_cvae_audio_only_20260421_194729_cfg.json")

npy_dir = root / cfg.FEATURES_DIR
labeled_df = pd.read_csv(root/cfg.METADATA_DIR/"metadata_en_bn.csv")
dataset = AudioSpectogramDatasetwithLabels(dataset_dir=npy_dir, labeled_df=labeled_df)
train_loader, test_loader = split_data(dataset=dataset, ratio=0.8, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE)

train_latents, train_labels, train_audio = extract_latents_with_names(model=cvae, loader=train_loader, device=device)
test_latents,  test_labels, test_audio  = extract_latents_with_names(model=cvae, loader=test_loader,  device=device)

vae_latents = np.concatenate([train_latents, test_latents], axis=0)
vae_labels = np.concatenate([train_labels, test_labels], axis=0)
vae_audio_names = np.concatenate([train_audio, test_audio], axis=0)

latent_vecs_hybrid = combine_audio_lyrics_and_genre(latent_vecs=vae_latents, audio_names=vae_audio_names, labeled_df=labeled_df, root=root, cfg=cfg)

import umap
reducer = umap.UMAP(n_components=50, metric='cosine', random_state=42) # latent dim = 64
z_reduced = reducer.fit_transform(latent_vecs_hybrid)
# z_reduced = latent_vecs_hybrid

vae_kmeans, vae_kmeans_metrics, vae_kmeans_df = run_KMeans(z_reduced, model_type="hard_3_cvae_audio_lyrics_genre", root=root, k_range=range(2, 20), true_labels=vae_labels)
vae_dbscan, vae_dbscan_metrics, vae_dbscan_df  = run_DBSCAN(z_reduced, model_type="hard_3_cvae_audio_lyrics_genre", root=root, true_labels=vae_labels)
vae_agglomerative, vae_agglomerative_metrics, vae_agglomerative_df = run_Agglomerative(z_reduced, model_type="hard_3_cvae_audio_lyrics_genre", root=root, k_range=range(2, 20), true_labels=vae_labels)

TSNE_DIR = root / config.TSNE_DIR
TSNE_DIR.mkdir(exist_ok=True, parents=True)

method = "tsne"

plot_latent_space_by_cluster(
    latent_vecs=z_reduced,
    cluster_labels=vae_kmeans.labels_,
    method=method,
    save_path=TSNE_DIR/f"hard_3_cvae_audio_lyrics_genre_kmeans_{method}",
    plot_title=f"{method.upper()} Visualization of Conditional VAE Clusters using K_Means (Audio + Lyrics + Genre)"
)

plot_latent_space_by_cluster(
    latent_vecs=z_reduced,
    cluster_labels=vae_dbscan.labels_,
    method=method,
    save_path=TSNE_DIR/f"hard_3_cvae_audio_lyrics_genre_dbscan_{method}",
    plot_title=f"{method.upper()} Visualization of Conditional VAE Clusters using DB-SCAN (Audio + Lyrics + Genre)"
)

plot_latent_space_by_cluster(
    latent_vecs=z_reduced,
    cluster_labels=vae_agglomerative.labels_,
    method=method,
    save_path=TSNE_DIR/f"hard_3_cvae_audio_lyrics_genre_agglomerative_{method}",
    plot_title=f"{method.upper()} Visualization of Conditional VAE Clusters using Agglomerative Clustering (Audio + Lyrics + Genre)"
)