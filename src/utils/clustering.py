import os, numpy as np, matplotlib.pyplot as plt, pandas as pd

from sklearn.metrics import silhouette_score, silhouette_samples

def _save_elbow_plot(visualizer, save_dir):
    path = os.path.join(save_dir, "elbow_plot.png")
    visualizer.show(outpath=path, clear_figure=True)
    print(f"Elbow plot saved to {path}")
    
def _save_silhouette_plot(latent_vecs, kmeans, optimal_k, save_dir):
    sample_silhouette_values = silhouette_samples(latent_vecs, kmeans.labels_)
    avg_silhouette = silhouette_score(latent_vecs, kmeans.labels_)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i in range(optimal_k):
        cluster_values = np.sort(sample_silhouette_values[kmeans.labels_ == i]) # type: ignore
        size = cluster_values.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=avg_silhouette, color="red", linestyle="--", label=f"Avg silhouette = {avg_silhouette:.4f}") # type: ignore
    ax.set_title(f"Silhouette Plot (k={optimal_k})")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.legend(loc="best")
    plt.tight_layout()

    path = os.path.join(save_dir, "silhouette_plot.png")
    plt.savefig(path)
    plt.close()
    print(f"Silhouette plot saved to {path}")
    
def _append_metrics_to_csv(metrics, model_type, save_dir):
    csv_path = os.path.join(save_dir, "clustering_results.csv")
    row = {"model_type": model_type, **metrics}
    df_new = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # Overwrite the row for this model_type if it already exists
        df_existing = df_existing[df_existing["model_type"] != model_type]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")