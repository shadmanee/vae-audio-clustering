"""
visualization.py
================
Dimensionality-reduction plots for clustering results.

Two reduction methods are supported:
    - t-SNE  (sklearn.manifold.TSNE)
    - UMAP   (umap-learn)   →  `pip install umap-learn`

Public API
----------
    # t-SNE
    plot_tsne_pca_vs_vae(...)          – side-by-side: PCA+KMeans vs VAE+KMeans
    plot_tsne_clustering_comparison(…) – one subplot per algorithm (DBSCAN, Agglomerative, …)

    # UMAP
    plot_umap_pca_vs_vae(...)          – same layout as t-SNE counterpart
    plot_umap_clustering_comparison(…) – same layout as t-SNE counterpart

    # Save helpers
    save_fig(fig, path)                – save any figure returned by the functions above
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

_UMAP_AVAILABLE = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_umap() -> None:
    if not _UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is required for UMAP plots.\n"
            "Install it with:  pip install umap-learn"
        )


def _scatter_with_noise(
    ax: plt.Axes,
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    silhouette: float | None = None,
) -> plt.cm.ScalarMappable:
    """
    Scatter plot that colours noise points (label == -1) in gray and
    assigns a distinct Spectral colour to every real cluster.

    Returns the ScalarMappable for a potential colorbar on the caller side.
    """
    unique_labels = np.unique(labels)
    real_labels   = unique_labels[unique_labels != -1]
    cmap_colors   = plt.cm.get_cmap("Spectral")(
        np.linspace(0, 1, max(len(real_labels), 1))
    )
    color_map = {lbl: cmap_colors[i] for i, lbl in enumerate(real_labels)}

    for label in unique_labels:
        mask       = labels == label
        color      = "gray" if label == -1 else color_map[label]
        label_text = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            label=label_text,
            alpha=0.6,
            edgecolors="w",
            s=30,
        )

    full_title = title
    if silhouette is not None:
        full_title += f"\n(Silhouette: {silhouette:.4f})"
    ax.set_title(full_title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    if len(unique_labels) < 15:
        ax.legend(loc="best", markerscale=0.8, fontsize="small")

    # Return a dummy mappable so callers can attach a colorbar if they want
    sm = plt.cm.ScalarMappable(
        cmap="Spectral",
        norm=plt.Normalize(vmin=real_labels.min() if len(real_labels) else 0,
                           vmax=real_labels.max() if len(real_labels) else 1),
    )
    sm.set_array([])
    return sm


# ---------------------------------------------------------------------------
# t-SNE  —  ported directly from the notebook
# ---------------------------------------------------------------------------

def plot_tsne_pca_vs_vae(
    pca_latents:  np.ndarray,
    vae_latents:  np.ndarray,
    pca_clusters: np.ndarray,
    vae_clusters: np.ndarray,
    pca_sil:      float,
    vae_sil:      float,
    save_path:    Path | str | None = None,
) -> plt.Figure:
    """
    Side-by-side t-SNE comparison of PCA+KMeans vs ConvVAE+KMeans.

    Mirrors Cell 68 of the notebook.

    Parameters
    ----------
    pca_latents  : PCA-reduced feature matrix  (n_samples, n_pca_dims)
    vae_latents  : VAE latent / hybrid matrix  (n_samples, n_latent_dims)
    pca_clusters : cluster labels from PCA+KMeans
    vae_clusters : cluster labels from VAE+KMeans
    pca_sil      : silhouette score for PCA+KMeans
    vae_sil      : silhouette score for VAE+KMeans
    save_path    : optional file path to save the figure (PNG/PDF/…)
    """
    print("Computing t-SNE for PCA features...")
    tsne_pca = TSNE(n_components=2, random_state=42).fit_transform(pca_latents)

    print("Computing t-SNE for VAE features...")
    tsne_vae = TSNE(n_components=2, random_state=42).fit_transform(vae_latents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    scatter1 = ax1.scatter(
        tsne_pca[:, 0], tsne_pca[:, 1],
        c=pca_clusters, cmap="viridis", alpha=0.7,
    )
    ax1.set_title(
        f"t-SNE  –  PCA + KMeans  (audio + lyrics)\n"
        f"(Silhouette: {round(pca_sil, 4)})"
    )
    fig.colorbar(scatter1, ax=ax1)

    scatter2 = ax2.scatter(
        tsne_vae[:, 0], tsne_vae[:, 1],
        c=vae_clusters, cmap="plasma", alpha=0.7,
    )
    ax2.set_title(
        f"t-SNE  –  ConvVAE + KMeans  (audio + lyrics)\n"
        f"(Silhouette: {round(vae_sil, 4)})"
    )
    fig.colorbar(scatter2, ax=ax2)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
    return fig


def plot_tsne_clustering_comparison(
    X_data:       np.ndarray,
    cluster_list: Sequence[np.ndarray],
    titles:       Sequence[str],
    scores:       Sequence[float],
    save_path:    Path | str | None = None,
) -> plt.Figure:
    """
    One t-SNE subplot per clustering algorithm.

    t-SNE is computed **once** on X_data and reused for every subplot so
    that all plots share the same spatial layout — matching Cell 70 of
    the notebook.

    Parameters
    ----------
    X_data       : raw feature matrix used for clustering  (n_samples, n_features)
    cluster_list : list of label arrays, one per algorithm
    titles       : subplot titles, e.g. ["DBSCAN", "Agglomerative"]
    scores       : silhouette score for each algorithm
    save_path    : optional file path to save the figure
    """
    print("Computing t-SNE embeddings (this may take a minute)...")
    tsne_results = TSNE(
        n_components=2, random_state=42, init="pca", learning_rate="auto"
    ).fit_transform(X_data)

    n_plots = len(cluster_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, labels, title, score in zip(axes, cluster_list, titles, scores):
        _scatter_with_noise(
            ax=ax,
            embedding=tsne_results,
            labels=np.asarray(labels),
            title=f"t-SNE  –  {title}",
            silhouette=score,
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# UMAP  —  mirrors the t-SNE functions above
# ---------------------------------------------------------------------------

def plot_umap_pca_vs_vae(
    pca_latents:  np.ndarray,
    vae_latents:  np.ndarray,
    pca_clusters: np.ndarray,
    vae_clusters: np.ndarray,
    pca_sil:      float,
    vae_sil:      float,
    n_neighbors:  int   = 15,
    min_dist:     float = 0.1,
    save_path:    Path | str | None = None,
) -> plt.Figure:
    """
    Side-by-side UMAP comparison of PCA+KMeans vs ConvVAE+KMeans.

    UMAP counterpart of plot_tsne_pca_vs_vae.

    Parameters
    ----------
    pca_latents  : PCA-reduced feature matrix  (n_samples, n_pca_dims)
    vae_latents  : VAE latent / hybrid matrix  (n_samples, n_latent_dims)
    pca_clusters : cluster labels from PCA+KMeans
    vae_clusters : cluster labels from VAE+KMeans
    pca_sil      : silhouette score for PCA+KMeans
    vae_sil      : silhouette score for VAE+KMeans
    n_neighbors  : UMAP neighbours parameter (controls local vs global structure)
    min_dist     : UMAP minimum distance (controls cluster compactness)
    save_path    : optional file path to save the figure
    """
    _check_umap()

    print("Computing UMAP for PCA features...")
    umap_pca = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
    ).fit_transform(pca_latents)

    print("Computing UMAP for VAE features...")
    umap_vae = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
    ).fit_transform(vae_latents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    scatter1 = ax1.scatter(
        umap_pca[:, 0], umap_pca[:, 1],
        c=pca_clusters, cmap="viridis", alpha=0.7,
    )
    ax1.set_title(
        f"UMAP  –  PCA + KMeans  (audio + lyrics)\n"
        f"(Silhouette: {round(pca_sil, 4)})"
    )
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    fig.colorbar(scatter1, ax=ax1)

    scatter2 = ax2.scatter(
        umap_vae[:, 0], umap_vae[:, 1],
        c=vae_clusters, cmap="plasma", alpha=0.7,
    )
    ax2.set_title(
        f"UMAP  –  ConvVAE + KMeans  (audio + lyrics)\n"
        f"(Silhouette: {round(vae_sil, 4)})"
    )
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    fig.colorbar(scatter2, ax=ax2)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
    return fig


def plot_umap_clustering_comparison(
    X_data:       np.ndarray,
    cluster_list: Sequence[np.ndarray],
    titles:       Sequence[str],
    scores:       Sequence[float],
    n_neighbors:  int   = 15,
    min_dist:     float = 0.1,
    save_path:    Path | str | None = None,
) -> plt.Figure:
    """
    One UMAP subplot per clustering algorithm.

    UMAP is computed **once** on X_data and shared across all subplots —
    matching the single-embedding approach used in
    plot_tsne_clustering_comparison.

    Parameters
    ----------
    X_data       : raw feature matrix used for clustering  (n_samples, n_features)
    cluster_list : list of label arrays, one per algorithm
    titles       : subplot titles, e.g. ["DBSCAN", "Agglomerative"]
    scores       : silhouette score for each algorithm
    n_neighbors  : UMAP neighbours parameter
    min_dist     : UMAP minimum distance
    save_path    : optional file path to save the figure
    """
    _check_umap()

    print("Computing UMAP embeddings...")
    umap_results = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=42,
    ).fit_transform(X_data)

    n_plots = len(cluster_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, labels, title, score in zip(axes, cluster_list, titles, scores):
        _scatter_with_noise(
            ax=ax,
            embedding=umap_results,
            labels=np.asarray(labels),
            title=f"UMAP  –  {title}  (audio + lyrics)",
            silhouette=score,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, path: Path | str, dpi: int = 150) -> None:
    """Save a figure returned by any plot function in this module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {path}")