from utils.common import extract_latents, save_model, load_model
from run_models import run_VAE, run_KMeans, run_DBSCAN, run_Agglomerative
from visualizations import plot_latent_space_by_cluster
from config import config

import torch, numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path(".")

cvae, latent_dim, train_loader, test_loader, vae_history, study = run_VAE(model_type="cvae", root=root, plot_dir_name="hard_1_cvae_audio_only")

weight_path, cfg_path = save_model(model=cvae, model_name="hard_1_cvae_audio_only", cfg=cvae.config, root=root, num_classes=2)

train_latents, train_labels = extract_latents(model=cvae, loader=train_loader, device=device)
test_latents,  test_labels  = extract_latents(model=cvae, loader=test_loader,  device=device)

vae_latents = np.concatenate([train_latents, test_latents], axis=0)
vae_labels = np.concatenate([train_labels, test_labels], axis=0)

vae_kmeans, vae_kmeans_metrics, vae_kmeans_df = run_KMeans(vae_latents, model_type="hard_1_cvae_audio_only", root=root, true_labels=vae_labels, k_range=range(2, 20))
vae_dbscan, vae_dbscan_metrics, vae_dbscan_df  = run_DBSCAN(vae_latents, model_type="hard_1_cvae_audio_only", root=root, true_labels=vae_labels)
vae_agglomerative, vae_agglomerative_metrics, vae_agglomerative_df = run_Agglomerative(vae_latents, model_type="hard_1_cvae_audio_only", root=root, true_labels=vae_labels, k_range=range(2, 20))

TSNE_DIR = root / config.TSNE_DIR
TSNE_DIR.mkdir(exist_ok=True, parents=True)

method = "tsne"

plot_latent_space_by_cluster(
    latent_vecs=vae_latents,
    cluster_labels=vae_kmeans.labels_,
    method=method,
    save_path=TSNE_DIR/f"hard_1_cvae_audio_only_kmeans_{method}",
    plot_title=f"{method.upper()} Visualization of Conditional VAE Clusters using K_Means (Audio Only)"
)

plot_latent_space_by_cluster(
    latent_vecs=vae_latents,
    cluster_labels=vae_dbscan.labels_,
    method=method,
    save_path=TSNE_DIR/f"hard_1_cvae_audio_only_dbscan_{method}",
    plot_title=f"{method.upper()} Visualization of Conditional VAE Clusters using DB-SCAN (Audio Only)"
)

plot_latent_space_by_cluster(
    latent_vecs=vae_latents,
    cluster_labels=vae_agglomerative.labels_,
    method=method,
    save_path=TSNE_DIR/f"hard_1_cvae_audio_only_agglomerative_{method}",
    plot_title=f"{method.upper()} Visualization of Conditional VAE Clusters using Agglomerative Clustering (Audio Only)"
)
