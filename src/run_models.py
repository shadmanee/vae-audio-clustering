from pathlib import Path

import torch
import torch.optim as optim

import pandas as pd, numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from config import config
from datasets import AudioSpectrogramDataset, AudioSpectogramDatasetwithLabels
from utils.common import split_data, train_vae, create_new_config, save_result_to_csv
from utils.clustering import _save_elbow_plot, _save_silhouette_plot, _append_metrics_to_csv, _has_true_labels, compute_supervised_metrics
from models.vae import VAE
from models.pca_baseline import PCABaseline
from tuning import run_tuning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_VAE(model_type: str, plot_dir_name=config.MODEL_TYPE, root: Path=Path("."), features_dir=config.FEATURES_DIR):
    npy_dir = root / features_dir
    num_labels = 0
    if model_type == "cvae":
        df_meta_path = root / config.METADATA_DIR / "metadata_en.csv"
        labeled_df = pd.read_csv(df_meta_path)
        num_labels = len(labeled_df["label"].unique())
        # print("\n"*3, "="*20, f"\n{num_labels}\n", "="*20, "\n"*3)
        dataset = AudioSpectogramDatasetwithLabels(dataset_dir=npy_dir, labeled_df=labeled_df)
    else:
        dataset = AudioSpectrogramDataset(dataset_dir=npy_dir)
    
    # hyperparameter tuning
    tuning_study = run_tuning(model_type=model_type, dataset=dataset, plot_dir_name=plot_dir_name, num_classes=num_labels, device=device, epochs=config.EPOCHS, trials=config.TRIALS, root=root)

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
    
    history = train_vae(model=vae, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=config.EPOCHS, annealing_epochs=config.ANNEALING_EPOCHS, beta=config.BETA, beta_type=config.BETA_TYPE, device=device, root=root, plot_model_dir_name=plot_dir_name)
    
    save_result_to_csv(study=tuning_study, history=history, model_name=plot_dir_name, root=root)
    
    return vae, best_trial_cfg.LATENT_DIM, train_loader, test_loader, history, tuning_study

def run_PCA(model_name, variance_threshold: float = 0.90, root: Path = Path("."), features_dir = config.FEATURES_DIR):
    npy_dir = root / features_dir

    dataset = AudioSpectrogramDataset(dataset_dir=npy_dir)

    train_loader, test_loader = split_data(
        dataset=dataset, ratio=0.8,
        batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE
    )

    pca_baseline = PCABaseline(variance_threshold=variance_threshold)
    pca_baseline.fit(train_loader, test_loader)
    pca_baseline.plot()
    pca_baseline.summary()

    train_error = pca_baseline.reconstruction_error(train_loader)
    test_error  = pca_baseline.reconstruction_error(test_loader)
    print(f"Train Recon MSE : {train_error:.4f}")
    print(f"Test  Recon MSE : {test_error:.4f}")

    history = {
        "train_recon": [train_error],
        "test_recon":  [test_error],
        "optimal_n":   [pca_baseline.optimal_n]
    }

    save_result_to_csv(history=history, model_name=model_name, root=root)

    return pca_baseline, train_loader, test_loader, history

def run_AE(
    plot_dir_name = "ae",
    root: Path = Path("."),
    features_dir = config.FEATURES_DIR
):
    npy_dir    = root / features_dir
    num_labels = 0

    dataset = AudioSpectrogramDataset(dataset_dir=npy_dir)

    tuning_study = run_tuning(
        model_type="ae",  # separate search space for AE
        dataset=dataset,
        plot_dir_name=plot_dir_name,
        num_classes=num_labels,
        device=device,
        epochs=config.EPOCHS,
        trials=config.TRIALS,
        root=root
    )

    print(f"Best trial:\nScore: {tuning_study.best_trial.value:.4f}")
    for k, v in tuning_study.best_trial.params.items():
        print(f"    {k:<25} {v}")

    best_trial_cfg = create_new_config(best_params=tuning_study.best_params)

    train_loader, test_loader = split_data(
        dataset=dataset, ratio=0.8,
        batch_size=best_trial_cfg.BATCH_SIZE, shuffle=config.SHUFFLE
    )
    
    ae = VAE(cfg=best_trial_cfg, model_type="ae", num_classes=num_labels).to(device)
    print("\n\nFINAL AE MODEL:\n", "="*20, "\n", ae, "\n", "="*20, "\n\n")

    optimizer = optim.Adam(ae.parameters(), lr=best_trial_cfg.LR)

    history = train_vae(
        model=ae,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=config.EPOCHS,
        annealing_epochs=config.ANNEALING_EPOCHS,
        beta=config.BETA,
        beta_type=config.BETA_TYPE,
        device=device,
        root=root
    )

    save_result_to_csv(
        study=tuning_study, history=history,
        model_name=plot_dir_name, root=root
    )

    return ae, best_trial_cfg.LATENT_DIM, train_loader, test_loader, history, tuning_study

# def run_PCA(n_components, train_loader, test_loader=None, root=Path(".")):
#     # Replaces your original code entirely
#     pca_baseline = PCABaseline(variance_threshold=0.90, n_components=n_components)
#     pca_baseline.fit(train_loader, test_loader)

#     # Compare reconstruction error against your VAE recon loss
#     train_error = pca_baseline.reconstruction_error(train_loader)
#     test_error  = pca_baseline.reconstruction_error(test_loader)
    
#     history = {"train_recon": [], "test_recon": []}
#     history["train_recon"].append(train_error)
#     history["test_recon"].append(test_error)
    
#     save_result_to_csv(history=history, model_name="pca", root=root)
    
#     return pca_baseline, history

def run_KMeans(latent_vecs, model_type, true_labels=None, root: Path = Path(".")):
    save_dir = root / config.RESULT_DIR / Path(f"clustering/{model_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    elbow_model = KMeans(init='k-means++', random_state=0, n_init=10)
    visualizer  = KElbowVisualizer(elbow_model, k=(2, 12), timings=True, force_model=True) # type: ignore
    visualizer.fit(latent_vecs)
    _save_elbow_plot(visualizer, save_dir=save_dir)

    optimal_k = visualizer.elbow_value_
    if optimal_k is None:
        print("Warning: elbow method could not find an optimal k. Defaulting to k=4.")
        optimal_k = 4

    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(latent_vecs)
    _save_silhouette_plot(latent_vecs, kmeans, optimal_k, save_dir)

    metrics = {
        "optimal_k":  optimal_k,
        "inertia":    kmeans.inertia_,
        "silhouette": silhouette_score(latent_vecs, kmeans.labels_),
        "ch_index":   calinski_harabasz_score(latent_vecs, kmeans.labels_),
        "db_index":   davies_bouldin_score(latent_vecs, kmeans.labels_),  # ← added
        "ari":        None,
        "nmi":        None,
        "purity":     None
    }

    if _has_true_labels(true_labels):
        supervised = compute_supervised_metrics(true_labels, kmeans.labels_)
        metrics.update(supervised)

    save_dir_csv = root / config.RESULT_DIR / Path("clustering/")
    _append_metrics_to_csv(metrics, f"{model_type}_kmeans", save_dir_csv)

    print("\n" * 3, "=" * 20)
    print(f"Optimal k        : {optimal_k}")
    print(f"Inertia (WCSS)   : {metrics['inertia']:.4f}")
    print(f"Silhouette Score : {metrics['silhouette']:.4f}")
    print(f"CH Index         : {metrics['ch_index']:.4f}")
    print(f"DB Index         : {metrics['db_index']:.4f}")
    if _has_true_labels(true_labels):
        print(f"ARI              : {metrics['ari']:.4f}")
        print(f"NMI              : {metrics['nmi']:.4f}")
        print(f"Purity           : {metrics['purity']:.4f}")
    print("=" * 20, "\n" * 3)

    return kmeans, metrics

def run_DBSCAN(latent_vecs, model_type, true_labels=None, eps: float=2.5, min_samples: int=5, root: Path=Path(".")):
    """
    Run DBSCAN clustering on latent vectors. If fewer than 2 clusters are found,
    automatically tunes eps and min_samples to find valid clustering.

    Parameters
    ----------
    latent_vecs  : array-like of shape (n_samples, n_features)
    model_type   : str  - used for labelling saved results
    eps          : float - initial maximum neighbourhood distance
    min_samples  : int   - initial minimum points to form a core point
    root         : Path  - project root for result saving

    Returns
    -------
    dbscan  : fitted DBSCAN object
    metrics : dict with clustering quality metrics (or empty dict if tuning also fails)
    """
    save_dir = root / config.RESULT_DIR / Path("clustering/")
    save_dir.mkdir(parents=True, exist_ok=True)

    def _fit_and_count(e, ms):
        m = DBSCAN(eps=e, min_samples=ms).fit(latent_vecs)
        lbs = m.labels_
        n_cls = len(set(lbs)) - (1 if -1 in lbs else 0)
        return m, lbs, n_cls

    dbscan, labels, n_clusters = _fit_and_count(eps, min_samples)

    if n_clusters < 2:
        print("Initial DBSCAN found < 2 clusters. Tuning eps and min_samples...")

        eps_values        = np.linspace(0.1, 10.0, 30)
        min_samples_values = [3, 5, 7, 10, 15]
        best_score        = -np.inf
        best_params       = None
        best_model        = None
        best_labels       = None

        for e in eps_values:
            for ms in min_samples_values:
                m, lbs, n_cls = _fit_and_count(e, ms)
                if n_cls < 2:
                    continue
                score = silhouette_score(latent_vecs, lbs)
                if score > best_score:
                    best_score  = score
                    best_params = (e, ms)
                    best_model  = m
                    best_labels = lbs

        if best_params is None:
            print("Warning: tuning failed to find valid clustering. Returning empty metrics.")
            return dbscan, {}

        eps, min_samples = best_params
        dbscan, labels   = best_model, best_labels
        n_clusters       = len(set(labels)) - (1 if -1 in labels else 0) # type: ignore
        print(f"Best params found — eps={eps:.3f}  min_samples={min_samples}")

    n_noise = (labels == -1).sum()

    print("\n" * 3, "=" * 20)
    print(f"DBSCAN  eps={eps}  min_samples={min_samples}")
    print(f"Clusters found   : {n_clusters}")
    print(f"Noise points     : {n_noise}")

    metrics = {
        "optimal_k":  n_clusters,
        "silhouette": silhouette_score(latent_vecs, labels), # type: ignore
        "ch_index":   calinski_harabasz_score(latent_vecs, labels), # type: ignore
        "db_index":   davies_bouldin_score(latent_vecs, labels), # type: ignore
        "n_noise":    int(n_noise),
        "ari":        None,
        "nmi":        None,
        "purity":     None
    }

    if _has_true_labels(true_labels):
        supervised = compute_supervised_metrics(true_labels[labels != -1], labels[labels != -1])  # type: ignore
        metrics.update(supervised)
    
    print("\n" * 3, "=" * 20)
    print(f"DB-SCAN\nSilhouette Score : {metrics['silhouette']:.4f}")
    print(f"CH Index         : {metrics['ch_index']:.4f}")
    print(f"DB Index         : {metrics['db_index']:.4f}")
    if _has_true_labels(true_labels):
        print(f"ARI              : {metrics['ari']:.4f}")
        print(f"NMI              : {metrics['nmi']:.4f}")
        print(f"Purity           : {metrics['purity']:.4f}")
    print("=" * 20, "\n" * 3)

    _append_metrics_to_csv(metrics, f"{model_type}_dbscan", save_dir)

    return dbscan, metrics
  
def run_Agglomerative(latent_vecs, model_type, true_labels=None, n_clusters=None, linkage='ward', root=Path(".")):
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
        "ari":        None,
        "nmi":        None,
        "purity":     None
    }

    if _has_true_labels(true_labels):
        supervised = compute_supervised_metrics(true_labels, labels)
        metrics.update(supervised)
 
    _append_metrics_to_csv(metrics, f"{model_type}_agglomerative", save_dir)
 
    print("\n" * 3, "=" * 20)
    print(f"Agglomerative  n_clusters={n_clusters}  linkage='{linkage}'")
    print(f"Silhouette Score : {metrics['silhouette']:.4f}")
    print(f"CH Index         : {metrics['ch_index']:.4f}")
    print(f"DB Index         : {metrics['db_index']:.4f}")
    if _has_true_labels(true_labels):
        print(f"ARI              : {metrics['ari']:.4f}")
        print(f"NMI              : {metrics['nmi']:.4f}")
        print(f"Purity           : {metrics['purity']:.4f}")
    print("=" * 20, "\n" * 3)
 
    return agg, metrics
    

    


