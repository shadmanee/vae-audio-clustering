from utils.common import extract_latents_with_names, load_model, split_data, combine_audio_and_lyrics
from datasets import AudioSpectrogramDataset
from run_models import run_KMeans, run_DBSCAN, run_Agglomerative
from visualizations import plot_latent_space_by_cluster
from config import config

import torch, numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path(".")

conv_vae, cfg = load_model(model_type="conv", weights_path=root/config.RESULT_DIR/r"models\medium_1_conv_vae_audio_only_20260421_192034.pt", cfg_path=root/config.RESULT_DIR/r"models\medium_1_conv_vae_audio_only_20260421_192034_cfg.json")

npy_dir = root / cfg.FEATURES_DIR
dataset = AudioSpectrogramDataset(dataset_dir=npy_dir)
train_loader, test_loader = split_data(dataset=dataset, ratio=0.8, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE)

train_latents, _, train_audio = extract_latents_with_names(model=conv_vae, loader=train_loader, device=device)
test_latents,  _, test_audio  = extract_latents_with_names(model=conv_vae, loader=test_loader,  device=device)

vae_latents = np.concatenate([train_latents, test_latents], axis=0)
vae_audio_names = np.concatenate([train_audio, test_audio], axis=0)

latent_vecs_hybrid = combine_audio_and_lyrics(latent_vecs=vae_latents, audio_names=vae_audio_names, root=root, cfg=cfg)

import umap
reducer = umap.UMAP(n_components=20, metric='cosine', random_state=42) #latent dim = 32
z_reduced = reducer.fit_transform(latent_vecs_hybrid)

vae_kmeans, vae_kmeans_metrics, vae_kmeans_df = run_KMeans(z_reduced, model_type="medium_2_conv_vae_audio_lyrics", root=root, k_range=range(2, 20))
vae_dbscan, vae_dbscan_metrics, vae_dbscan_df  = run_DBSCAN(z_reduced, model_type="medium_2_conv_vae_audio_lyrics", root=root)
vae_agglomerative, vae_agglomerative_metrics, vae_agglomerative_df = run_Agglomerative(z_reduced, model_type="medium_2_conv_vae_audio_lyrics", root=root, k_range=range(2, 20))

TSNE_DIR = root / config.TSNE_DIR
TSNE_DIR.mkdir(exist_ok=True, parents=True)

method = "tsne"

plot_latent_space_by_cluster(
    latent_vecs=z_reduced,
    cluster_labels=vae_kmeans.labels_,
    method=method,
    save_path=TSNE_DIR/f"medium_2_conv_vae_audio_lyrics_kmeans_{method}",
    plot_title=f"{method.upper()} Visualization of Convolutional VAE Clusters using K_Means (Audio + Lyrics)"
)

plot_latent_space_by_cluster(
    latent_vecs=z_reduced,
    cluster_labels=vae_dbscan.labels_,
    method=method,
    save_path=TSNE_DIR/f"medium_2_conv_vae_audio_lyrics_dbscan_{method}",
    plot_title=f"{method.upper()} Visualization of Convolutional VAE Clusters using DB-SCAN (Audio + Lyrics)"
)

plot_latent_space_by_cluster(
    latent_vecs=z_reduced,
    cluster_labels=vae_agglomerative.labels_,
    method=method,
    save_path=TSNE_DIR/f"medium_2_conv_vae_audio_lyrics_agglomerative_{method}",
    plot_title=f"{method.upper()} Visualization of Convolutional VAE Clusters using Agglomerative Clustering (Audio + Lyrics)"
)