from pathlib import Path

import torch
import torch.optim as optim

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from config import config
from datasets import AudioSpectrogramDataset, AudioSpectogramGenreDataset
from utils.common import split_data, train_vae, create_new_config, save_result_to_csv
from utils.clustering import _save_elbow_plot, _save_silhouette_plot, _append_metrics_to_csv
from models.vae import VAE
from models.pca_baseline import PCABaseline
from tuning import run_tuning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_VAE(model_type: str, root: Path=Path("."), features_dir=config.FEATURES_DIR):
    npy_dir = root / features_dir
    num_labels = 0
    if model_type == "cvae":
        df_meta_path = root / config.METADATA_DIR / "metadata_en.csv" # "metadata_popular_en.csv" if considering only genres that appear > 10x
        labeled_df = pd.read_csv(df_meta_path)
        num_labels = len(labeled_df["label"].unique())
        dataset = AudioSpectogramGenreDataset(dataset_dir=npy_dir, labeled_df=labeled_df)
    else:
        dataset = AudioSpectrogramDataset(dataset_dir=npy_dir)
    
    # hyperparameter tuning
    tuning_study = run_tuning(model_type=model_type, dataset=dataset, num_classes=num_labels, device=device, epochs=config.EPOCHS, trials=config.TRIALS, root=root)

    print(f"Best trial for `{model_type}`:\nScore: {tuning_study.best_trial.value:.4f}")
    for k, v in tuning_study.best_trial.params.items():
        print(f"    {k:<25} {v}")
    
    # converting the best hyperparameters into BaseConfig object
    # best_params -> dict -> BaseConfig
    best_trial_cfg = create_new_config(best_params=tuning_study.best_params)
    
    train_loader, test_loader = split_data(dataset=dataset, ratio=0.8, batch_size=best_trial_cfg.BATCH_SIZE, shuffle=config.SHUFFLE)

    vae = VAE(cfg=best_trial_cfg, model_type=model_type, num_classes=num_labels).to(device)
    print("\n\nFINAL MODEL:\n", "="*20, "\n", vae, "\n", "="*20, "\n\n\n")
    
    optimizer = optim.Adam(vae.parameters(), lr=best_trial_cfg.LR)
    
    history = train_vae(model=vae, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=config.EPOCHS, annealing_epochs=config.ANNEALING_EPOCHS, beta=config.BETA, beta_type=config.BETA_TYPE, device=device, root=root)
    
    save_result_to_csv(study=tuning_study, history=history, model_name=model_type, root=root)
    
    return vae, best_trial_cfg.LATENT_DIM, train_loader, test_loader, history, tuning_study

def run_PCA(n_components, train_loader, test_loader=None, root=Path(".")):
    # Replaces your original code entirely
    pca_baseline = PCABaseline(variance_threshold=0.90, n_components=n_components)
    pca_baseline.fit(train_loader, test_loader)

    # Compare reconstruction error against your VAE recon loss
    train_error = pca_baseline.reconstruction_error(train_loader)
    test_error  = pca_baseline.reconstruction_error(test_loader)
    
    history = {"train_recon": [], "test_recon": []}
    history["train_recon"].append(train_error)
    history["test_recon"].append(test_error)
    
    save_result_to_csv(history=history, model_name="pca", root=root)
    
    return pca_baseline, history

def run_KMeans(latent_vecs, model_type, root: Path=Path(".")):
    save_dir = root / config.RESULT_DIR / Path(f"clustering/{model_type}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(save_dir.exists())
    
    elbow_model = KMeans(init='k-means++', random_state=0, n_init=10)
    visualizer = KElbowVisualizer(elbow_model, k=(2, 12), timings=True, force_model=True) # type: ignore
    visualizer.fit(latent_vecs)
    
    _save_elbow_plot(visualizer, save_dir=save_dir)
    
    optimal_k = visualizer.elbow_value_
    if optimal_k is None:
        print("\n" * 3, "=" * 20)
        print("Warning: elbow method could not find an optimal k. Defaulting to k=4.")
        print("=" * 20, "\n" * 3)
        optimal_k = 4
    
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(latent_vecs)  
    
    _save_silhouette_plot(latent_vecs, kmeans, optimal_k, save_dir)
    
    metrics = {
        "optimal_k":  optimal_k,
        "inertia":    kmeans.inertia_,
        "silhouette": silhouette_score(latent_vecs, kmeans.labels_),
        "ch_index":   calinski_harabasz_score(latent_vecs, kmeans.labels_)
    }
    
    save_dir = root / config.RESULT_DIR / Path(f"clustering/")
    _append_metrics_to_csv(metrics, f"{model_type}_kmeans", save_dir)

    print("\n" * 3, "=" * 20)
    print(f"Optimal k        : {optimal_k}")
    print(f"Inertia (WCSS)   : {metrics['inertia']:.4f}")
    print(f"Silhouette Score : {metrics['silhouette']:.4f}")
    print(f"CH Index         : {metrics['ch_index']:.4f}")
    print("=" * 20, "\n" * 3)

    return kmeans, metrics

def run_DBSCAN(latent_vecs, model_type, eps: float=2.5, min_samples: int=5, root: Path=Path(".")):
    """
    Run DBSCAN clustering on latent vectors.
 
    Parameters
    ----------
    latent_vecs  : array-like of shape (n_samples, n_features)
    model_type   : str  - used for labelling saved results
    eps          : float - maximum neighbourhood distance (DBSCAN hyperparameter)
    min_samples  : int   - minimum points to form a core point (DBSCAN hyperparameter)
    root         : Path  - project root for result saving
 
    Returns
    -------
    dbscan  : fitted DBSCAN object
    metrics : dict with clustering quality metrics (or empty dict if < 2 clusters found)
    """
    save_dir = root / config.RESULT_DIR / Path("clustering/")
    save_dir.mkdir(parents=True, exist_ok=True)
 
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(latent_vecs)
 
    # DBSCAN marks noise as -1; silhouette/CH/DB require at least 2 proper clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
 
    print("\n" * 3, "=" * 20)
    print(f"DBSCAN  eps={eps}  min_samples={min_samples}")
    print(f"Clusters found   : {n_clusters}")
    print(f"Noise points     : {n_noise}")
 
    if n_clusters > 1:
        metrics = {
            "optimal_k":  n_clusters,
            "silhouette": silhouette_score(latent_vecs, labels),
            "ch_index":   calinski_harabasz_score(latent_vecs, labels),
            "db_index":   davies_bouldin_score(latent_vecs, labels),
            "n_noise":    int(n_noise),
        }
        print(f"Silhouette Score : {metrics['silhouette']:.4f}")
        print(f"CH Index         : {metrics['ch_index']:.4f}")
        print(f"Davies-Bouldin   : {metrics['db_index']:.4f}")
        _append_metrics_to_csv(metrics, f"{model_type}_dbscan", save_dir)
    else:
        metrics = {}
        print("Warning: DBSCAN found < 2 clusters. Try adjusting 'eps' or 'min_samples'.")
        print("Metrics not computed.")
 
    print("=" * 20, "\n" * 3)
 
    return dbscan, metrics
 
 
def run_Agglomerative(latent_vecs, model_type, n_clusters=None, linkage='ward', root=Path(".")):
    """
    Run Agglomerative (hierarchical) clustering on latent vectors.
 
    Parameters
    ----------
    latent_vecs : array-like of shape (n_samples, n_features)
    model_type  : str  - used for labelling saved results
    n_clusters  : int  - number of clusters; if None, falls back to run_KMeans elbow value
    linkage     : str  - linkage criterion: 'ward', 'complete', 'average', or 'single'
    root        : Path - project root for result saving
 
    Returns
    -------
    agg     : fitted AgglomerativeClustering object
    metrics : dict with clustering quality metrics
    """
    save_dir = root / config.RESULT_DIR / Path("clustering/")
    save_dir.mkdir(parents=True, exist_ok=True)
 
    if n_clusters is None:
        # Fall back to the elbow method to pick k automatically
        print("n_clusters not provided for Agglomerative — running elbow method to determine k...")
        elbow_model = KMeans(init='k-means++', random_state=0, n_init=10)
        visualizer  = KElbowVisualizer(elbow_model, k=(2, 12), timings=True, force_model=True) # type: ignore
        visualizer.fit(latent_vecs)
        n_clusters = visualizer.elbow_value_ or 4   # default to 4 if elbow fails
 
    agg    = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage) # type: ignore
    labels = agg.fit_predict(latent_vecs)
 
    metrics = {
        "optimal_k":  n_clusters,
        "silhouette": silhouette_score(latent_vecs, labels),
        "ch_index":   calinski_harabasz_score(latent_vecs, labels),
        "db_index":   davies_bouldin_score(latent_vecs, labels),
    }
 
    _append_metrics_to_csv(metrics, f"{model_type}_agglomerative", save_dir)
 
    print("\n" * 3, "=" * 20)
    print(f"Agglomerative  n_clusters={n_clusters}  linkage='{linkage}'")
    print(f"Silhouette Score : {metrics['silhouette']:.4f}")
    print(f"CH Index         : {metrics['ch_index']:.4f}")
    print(f"Davies-Bouldin   : {metrics['db_index']:.4f}")
    print("=" * 20, "\n" * 3)
 
    return agg, metrics
    

    


