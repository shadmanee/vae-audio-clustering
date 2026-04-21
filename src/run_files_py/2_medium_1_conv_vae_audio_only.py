from utils.common import extract_latents, save_model, load_model
from run_models import run_VAE, run_KMeans, run_DBSCAN, run_Agglomerative
from visualizations import plot_latent_space_by_cluster
from config import config

import torch, numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path(".")

conv_vae, latent_dim, train_loader, test_loader, vae_history, study = run_VAE(model_type="conv", root=root, plot_dir_name="medium_1_conv_vae_audio_only")

weight_path, cfg_path = save_model(model=conv_vae, model_name="medium_1_conv_vae_audio_only", cfg=conv_vae.config, root=root, num_classes=0)

train_latents, _ = extract_latents(model=conv_vae, loader=train_loader, device=device)
test_latents,  _  = extract_latents(model=conv_vae, loader=test_loader,  device=device)

vae_latents = np.concatenate([train_latents, test_latents], axis=0)

vae_kmeans, vae_kmeans_metrics, vae_kmeans_df = run_KMeans(vae_latents, model_type="medium_1_conv_vae_audio_only", root=root, k_range=range(2, 20))
vae_dbscan, vae_dbscan_metrics, vae_dbscan_df  = run_DBSCAN(vae_latents, model_type="medium_1_conv_vae_audio_only", root=root)
vae_agglomerative, vae_agglomerative_metrics, vae_agglomerative_df = run_Agglomerative(vae_latents, model_type="medium_1_conv_vae_audio_only", root=root, k_range=range(2, 20))

TSNE_DIR = root / config.TSNE_DIR
TSNE_DIR.mkdir(exist_ok=True, parents=True)

method = "tsne"

plot_latent_space_by_cluster(
    latent_vecs=vae_latents,
    cluster_labels=vae_kmeans.labels_,
    method=method,
    save_path=TSNE_DIR/f"medium_1_conv_vae_audio_only_kmeans_{method}",
    plot_title=f"{method.upper()} Visualization of Convolutional VAE Clusters using K_Means (Audio Only)"
)

plot_latent_space_by_cluster(
    latent_vecs=vae_latents,
    cluster_labels=vae_dbscan.labels_,
    method=method,
    save_path=TSNE_DIR/f"medium_1_conv_vae_audio_only_dbscan_{method}",
    plot_title=f"{method.upper()} Visualization of Convolutional VAE Clusters using DB-SCAN (Audio Only)"
)

plot_latent_space_by_cluster(
    latent_vecs=vae_latents,
    cluster_labels=vae_agglomerative.labels_,
    method=method,
    save_path=TSNE_DIR/f"medium_1_conv_vae_audio_only_agglomerative_{method}",
    plot_title=f"{method.upper()} Visualization of Convolutional VAE Clusters using Agglomerative Clustering (Audio Only)"
)