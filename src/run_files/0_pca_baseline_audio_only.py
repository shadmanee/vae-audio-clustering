import torch, numpy as np, joblib
from pathlib import Path
from datetime import datetime

from config import config
from run_models import run_PCA, run_KMeans, run_DBSCAN, run_Agglomerative
from visualizations import plot_latent_space_by_cluster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path(".")

pca_baseline, train_loader, test_loader, pca_history = run_PCA(root=root, model_name="0_pca_baseline_audio_only", variance_threshold=0.8)

# Save PCA model
pca_save_dir = root / config.RESULT_DIR / "models"
pca_save_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pca_save_path = pca_save_dir / f"0_pca_baseline_audio_only_{timestamp}.joblib"
joblib.dump(pca_baseline, pca_save_path)
print(f"PCA model saved to: {pca_save_path}")

pca_latents = np.concatenate([
    pca_baseline.transform(train_loader),
    pca_baseline.transform(test_loader)
])

pca_kmeans, pca_metrics, pca_kmeans_df = run_KMeans(latent_vecs=pca_latents, model_type="0_pca_baseline_audio_only", k_range=range(2, 20), scale=False)
pca_dbscan, pca_metrics, pca_dbscan_df = run_DBSCAN(latent_vecs=pca_latents, model_type="0_pca_baseline_audio_only", scale=False)
pca_agglomerative, pca_metrics, pca_agglomerative_df = run_Agglomerative(latent_vecs=pca_latents, model_type="0_pca_baseline_audio_only", k_range=range(2, 20), scale=False)

TSNE_DIR = root / config.TSNE_DIR
TSNE_DIR.mkdir(exist_ok=True, parents=True)

method = "tsne"

plot_latent_space_by_cluster(
    latent_vecs=pca_latents,
    cluster_labels=pca_kmeans.labels_,
    method=method,
    save_path=TSNE_DIR/f"0_pca_baseline_audio_only_kmeans_{method}",
    plot_title=f"{method.upper()} Visualization of PCA Baseline Clusters using K_Means (Audio Only)"
)

plot_latent_space_by_cluster(
    latent_vecs=pca_latents,
    cluster_labels=pca_dbscan.labels_,
    method=method,
    save_path=TSNE_DIR/f"0_pca_baseline_audio_only_dbscan_{method}",
    plot_title=f"{method.upper()} Visualization of PCA Baseline Clusters using DB-SCAN (Audio Only)"
)

plot_latent_space_by_cluster(
    latent_vecs=pca_latents,
    cluster_labels=pca_agglomerative.labels_,
    method=method,
    save_path=TSNE_DIR/f"0_pca_baseline_audio_only_agglomerative_{method}",
    plot_title=f"{method.upper()} Visualization of PCA Baseline Clusters using Agglomerative Clustering (Audio Only)"
)