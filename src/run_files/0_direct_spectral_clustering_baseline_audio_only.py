from direct_spectral_feature_clustering import run_direct_clustering
from run_models import run_KMeans, run_DBSCAN, run_Agglomerative
from visualizations import plot_latent_space_by_cluster
from config import config

from pathlib import Path

root = Path(".")

all_feats, _ = run_direct_clustering(modality="audio", root=root)

direct_kmeans, direct_metrics, direct_kmeans_df = run_KMeans(latent_vecs=all_feats, model_type="0_direct_spectral_clustering_baseline_audio_only", k_range=range(2, 20), scale=False)
# direct_dbscan, direct_metrics, direct_dbscan_df = run_DBSCAN(latent_vecs=all_feats, model_type="0_direct_spectral_clustering_baseline_audio_only", scale=False)
# direct_agglomerative, direct_metrics, direct_agglomerative_df = run_Agglomerative(latent_vecs=all_feats, model_type="0_direct_spectral_clustering_baseline_audio_only", k_range=range(2, 20), scale=False)

TSNE_DIR = root / config.TSNE_DIR
TSNE_DIR.mkdir(exist_ok=True, parents=True)

method = "tsne"

plot_latent_space_by_cluster(
    latent_vecs=all_feats,
    cluster_labels=direct_kmeans.labels_,
    method=method,
    save_path=TSNE_DIR/f"0_direct_spectral_clustering_baseline_audio_only_{method}",
    plot_title=f"{method.upper()} Visualization of Direct (Baseline) Spectral Clusters using K_Means (Audio Only)"
)

# plot_latent_space_by_cluster(
#     latent_vecs=all_feats,
#     cluster_labels=direct_dbscan.labels_,
#     method=method,
#     save_path=TSNE_DIR/f"0_direct_spectral_clustering_baseline_audio_only_{method}",
#     plot_title=f"{method.upper()} Visualization of Direct (Baseline) Spectral Clusters using DB-SCAN (Audio Only)"
# )

# plot_latent_space_by_cluster(
#     latent_vecs=all_feats,
#     cluster_labels=direct_agglomerative.labels_,
#     method=method,
#     save_path=TSNE_DIR/f"0_direct_spectral_clustering_baseline_audio_only_{method}",
#     plot_title=f"{method.upper()} Visualization of Direct (Baseline) Spectral Clusters using Agglomerative Clustering (Audio Only)"
# )