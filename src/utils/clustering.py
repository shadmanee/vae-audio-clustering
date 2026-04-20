import os, numpy as np, matplotlib.pyplot as plt, pandas as pd

from pathlib import Path

from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram

def _save_elbow_plot(visualizer, save_dir):
    path = os.path.join(save_dir, "elbow_plot.png")
    visualizer.show(outpath=path, clear_figure=True)
    # print(f"Elbow plot saved to {path}")
    
def _save_silhouette_plot(latent_vecs, kmeans, optimal_k, save_dir):
    sample_silhouette_values = silhouette_samples(latent_vecs, kmeans.labels_)
    avg_silhouette = silhouette_score(latent_vecs, kmeans.labels_)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i in range(optimal_k):
        cluster_values = np.sort(sample_silhouette_values[kmeans.labels_ == i])
        size = cluster_values.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=avg_silhouette, color="red", linestyle="--", label=f"Avg silhouette = {avg_silhouette:.4f}")
    ax.set_title(f"Silhouette Plot (k={optimal_k})")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.legend(loc="best")
    plt.tight_layout()

    path = os.path.join(save_dir, "kmeans_silhouette_plot.png")
    plt.savefig(path)
    plt.close()
    # print(f"Silhouette plot saved to {path}")
    
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
    # print(f"Metrics saved to {csv_path}")

def compute_supervised_metrics(true_labels, pred_labels):
    """
    Compute ARI, NMI, and Cluster Purity.
    Only called when true labels are meaningful (not all -1).
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Cluster Purity
    total, n = 0, len(true_labels)
    for cluster_id in np.unique(pred_labels):
        mask = pred_labels == cluster_id
        if mask.sum() == 0:
            continue
        most_common = np.bincount(true_labels[mask]).max()
        total += most_common
    purity = total / n

    return {
        "ari":    adjusted_rand_score(true_labels, pred_labels),
        "nmi":    normalized_mutual_info_score(true_labels, pred_labels),
        "purity": purity
    }


def _has_true_labels(true_labels):
    """Check if true labels are meaningful (not placeholder -1 values)."""
    return true_labels is not None and np.any(np.array(true_labels) != -1)

def _preprocess_latents(
    latent_vecs,
    scale: bool = True,
    l2_normalize: bool = False,
    pca_var: float | None = None
):
    X = np.asarray(latent_vecs, dtype=np.float32)

    if scale:
        X = StandardScaler().fit_transform(X)

    if l2_normalize:
        X = Normalizer(norm="l2").fit_transform(X)

    if pca_var is not None and 0 < pca_var < 1:
        X = PCA(n_components=pca_var, svd_solver="full", random_state=0).fit_transform(X)

    return X


def _safe_internal_metrics(X, labels, ignore_noise: bool = False):
    labels = np.asarray(labels)

    if ignore_noise:
        mask = labels != -1
        X_eval = X[mask]
        y_eval = labels[mask]
    else:
        X_eval = X
        y_eval = labels

    unique = np.unique(y_eval)
    if len(unique) < 2:
        return {
            "silhouette": None,
            "ch_index": None,
            "db_index": None,
            "n_eval_samples": int(len(y_eval))
        }

    # CH/DB/silhouette need >= 2 clusters and enough samples
    try:
        sil = silhouette_score(X_eval, y_eval)
    except Exception:
        sil = None

    try:
        ch = calinski_harabasz_score(X_eval, y_eval)
    except Exception:
        ch = None

    try:
        db = davies_bouldin_score(X_eval, y_eval)
    except Exception:
        db = None

    return {
        "silhouette": sil,
        "ch_index": ch,
        "db_index": db,
        "n_eval_samples": int(len(y_eval))
    }


def _cluster_size_summary(labels):
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def _score_for_model_selection(metrics, noise_ratio: float = 0.0):
    """
    Higher is better.
    Silhouette drives the score.
    Small penalty for high DB index.
    Stronger penalty for too much noise in DBSCAN.
    """
    sil = metrics.get("silhouette")
    ch = metrics.get("ch_index")
    db = metrics.get("db_index")

    if sil is None:
        return -np.inf

    score = sil
    if db is not None:
        score -= 0.05 * db
    if ch is not None and ch > 0:
        score += 0.0001 * np.log1p(ch)

    score -= 0.35 * noise_ratio
    return score

def _save_k_vs_silhouette_plot(search_df: pd.DataFrame, save_dir: Path):
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    plot_df = search_df.dropna(subset=["k", "silhouette"]).copy()
    if plot_df.empty:
        print("Warning: no valid silhouette values available for k-vs-silhouette plot.")
        return

    plot_df = plot_df.sort_values("k")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["k"], plot_df["silhouette"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average silhouette score")
    plt.title("Silhouette score vs k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "kmeans_k_vs_silhouette.png", dpi=300, bbox_inches="tight")
    plt.close()
    
def _project_to_2d(X):
    if X.shape[1] <= 2:
        return X[:, :2]
    return PCA(n_components=2, random_state=0).fit_transform(X)

def _save_dbscan_k_distance_plot(X, min_samples: int, save_dir: Path):
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    kth_dist = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(kth_dist)), kth_dist)
    plt.xlabel("Sorted sample index")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
    plt.title(f"DBSCAN k-distance plot (min_samples={min_samples})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"dbscan_k_distance_min_samples_{min_samples}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_dbscan_heatmap(search_df: pd.DataFrame, save_dir: Path):
    if search_df.empty:
        return

    pivot = search_df.pivot_table(
        index="min_samples",
        columns="eps",
        values="silhouette",
        aggfunc="max"
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Silhouette")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns], rotation=45)
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.title("DBSCAN silhouette heatmap")
    plt.tight_layout()
    plt.savefig(save_dir / "dbscan_silhouette_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_dbscan_coverage_plot(search_df: pd.DataFrame, save_dir: Path):
    if search_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for ms in sorted(search_df["min_samples"].unique()):
        df_ms = search_df[search_df["min_samples"] == ms].sort_values("eps")
        plt.plot(df_ms["eps"], df_ms["coverage"], marker="o", label=f"min_samples={ms}")

    plt.xlabel("eps")
    plt.ylabel("Coverage")
    plt.title("DBSCAN coverage vs eps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "dbscan_coverage_vs_eps.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_cluster_scatter_2d(X, labels, save_path: Path, title: str):
    X2 = _project_to_2d(X)

    plt.figure(figsize=(7, 6))
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        mask = labels == lab
        label_name = "noise" if lab == -1 else f"cluster {lab}"
        plt.scatter(X2[mask, 0], X2[mask, 1], s=18, alpha=0.8, label=label_name)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_agglomerative_k_vs_silhouette(search_df: pd.DataFrame, save_dir: Path):
    if search_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for (linkage_name, metric_name), df_sub in search_df.groupby(["linkage", "metric"]):
        df_sub = df_sub.sort_values("optimal_k")
        plt.plot(
            df_sub["optimal_k"],
            df_sub["silhouette"],
            marker="o",
            label=f"{linkage_name}/{metric_name}"
        )

    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette")
    plt.title("Agglomerative silhouette vs k")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / "agglomerative_k_vs_silhouette.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_agglomerative_heatmap(search_df: pd.DataFrame, save_dir: Path):
    if search_df.empty:
        return

    df = search_df.copy()
    df["cfg"] = df["linkage"] + "/" + df["metric"]

    pivot = df.pivot_table(
        index="cfg",
        columns="optimal_k",
        values="silhouette",
        aggfunc="max"
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Silhouette")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.xlabel("k")
    plt.ylabel("linkage / metric")
    plt.title("Agglomerative silhouette heatmap")
    plt.tight_layout()
    plt.savefig(save_dir / "agglomerative_silhouette_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_truncated_dendrogram(X, linkage_method: str, metric: str, save_dir: Path, max_points: int = 300):
    n = len(X)
    if n == 0:
        return

    rng = np.random.default_rng(0)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    if linkage_method == "ward" and metric != "euclidean":
        return

    Z = scipy_linkage(X_sub, method=linkage_method, metric=metric)

    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=20, no_labels=True)
    plt.title(f"Truncated dendrogram ({linkage_method}/{metric})")
    plt.xlabel("Merged groups")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(save_dir / f"dendrogram_{linkage_method}_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()