from utils.common import extract_latents, save_model
from run_models import run_VAE, run_KMeans, run_DBSCAN, run_Agglomerative
from visualizations import plot_latent_space_by_cluster
from config import config

import torch, numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path(".")

ae, latent_dim, train_loader, test_loader, vae_history, study = run_VAE(model_type="ae", root=root, plot_dir_name="0_ae_baseline_audio_only")

weight_path, cfg_path = save_model(model=ae, model_name="0_ae_baseline_audio_only", cfg=ae.config, root=root, num_classes=0)

train_latents, _ = extract_latents(model=ae, loader=train_loader, device=device)
test_latents,  _  = extract_latents(model=ae, loader=test_loader,  device=device)

ae_latents = np.concatenate([train_latents, test_latents], axis=0)

ae_kmeans, ae_kmeans_metrics, ae_kmeans_df = run_KMeans(ae_latents, model_type="0_ae_baseline_audio_only", root=root, k_range=range(2, 20))
ae_dbscan, ae_dbscan_metrics, ae_dbscan_df  = run_DBSCAN(ae_latents, model_type="0_ae_baseline_audio_only", root=root)
ae_agglomerative, ae_agglomerative_metrics, ae_agglomerative_df = run_Agglomerative(ae_latents, model_type="0_ae_baseline_audio_only", root=root, k_range=range(2, 20))

TSNE_DIR = root / config.TSNE_DIR
TSNE_DIR.mkdir(exist_ok=True, parents=True)

method = "tsne"

plot_latent_space_by_cluster(
    latent_vecs=ae_latents,
    cluster_labels=ae_kmeans.labels_,
    method=method,
    save_path=TSNE_DIR/f"0_ae_baseline_audio_only_kmeans_{method}",
    plot_title=f"{method.upper()} Visualization of AE Baseline Clusters using K_Means (Audio Only)"
)

plot_latent_space_by_cluster(
    latent_vecs=ae_latents,
    cluster_labels=ae_dbscan.labels_,
    method=method,
    save_path=TSNE_DIR/f"0_ae_baseline_audio_only_dbscan_{method}",
    plot_title=f"{method.upper()} Visualization of AE Baseline Clusters using DB-SCAN (Audio Only)"
)

plot_latent_space_by_cluster(
    latent_vecs=ae_latents,
    cluster_labels=ae_agglomerative.labels_,
    method=method,
    save_path=TSNE_DIR/f"0_ae_baseline_audio_only_agglomerative_{method}",
    plot_title=f"{method.upper()} Visualization of AE Baseline Clusters using Agglomerative Clustering (Audio Only)"
)