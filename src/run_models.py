from pathlib import Path

import torch
import torch.optim as optim

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from config import config
from datasets import AudioSpectrogramDataset
from utils.common import split_data, train_vae, create_new_config, save_result_to_csv
from utils.clustering import _save_elbow_plot, _save_silhouette_plot, _append_metrics_to_csv
from models.vae import VAE
from models.pca_baseline import PCABaseline
from tuning import run_tuning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

npy_dir = config.FEATURES_DIR
dataset = AudioSpectrogramDataset(dataset_dir=npy_dir)

def run_VAE(model_type: str):
    # if model_type == "basic":
    # hyperparameter tuning
    tuning_study = run_tuning(model_type=model_type, dataset=dataset, device=device, epochs=config.EPOCHS, trials=config.TRIALS)

    print(f"Best trial for `{model_type}`:\nScore: {tuning_study.best_trial.value:.4f}")
    for k, v in tuning_study.best_trial.params.items():
        print(f"    {k:<25} {v}")
    
    # converting the best hyperparameters into BaseConfig object
    # best_params -> dict -> BaseConfig
    best_trial_cfg = create_new_config(best_params=tuning_study.best_params)
    # print("="*20, "\n")
    # print(best_trial_cfg.__dict__)
    # print("\n", "="*20)
    
    train_loader, test_loader = split_data(dataset=dataset, ratio=0.8, batch_size=best_trial_cfg.BATCH_SIZE, shuffle=config.SHUFFLE)

    vae = VAE(cfg=best_trial_cfg, model_type=model_type).to(device)
    print("\n\n\n", "="*20, "\n", vae, "\n", "="*20)
    
    optimizer = optim.Adam(vae.parameters(), lr=best_trial_cfg.LR)
    
    history = train_vae(model=vae, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=config.EPOCHS, beta=config.BETA, beta_type=config.BETA_TYPE, device=device)
    
    save_result_to_csv(study=tuning_study, history=history, model_name=model_type)
    
    return vae, best_trial_cfg.LATENT_DIM, train_loader, test_loader, history
    
    # else:
    #     pass

def run_PCA(n_components, train_loader, test_loader=None):
    # Replaces your original code entirely
    pca_baseline = PCABaseline(variance_threshold=0.90, n_components=n_components)
    pca_baseline.fit(train_loader, test_loader)

    # Compare reconstruction error against your VAE recon loss
    train_error = pca_baseline.reconstruction_error(train_loader)
    test_error  = pca_baseline.reconstruction_error(test_loader)
    
    history = {"train_recon": [], "test_recon": []}
    history["train_recon"].append(train_error)
    history["test_recon"].append(test_error)
    
    save_result_to_csv(history=history, model_name="pca")
    
    return pca_baseline, history

def run_KMeans(latent_vecs, model_type):
    save_dir = config.RESULT_DIR / Path(f"clustering/{model_type}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    elbow_model = KMeans(init='k-means++', random_state=0, n_init=10)
    visualizer = KElbowVisualizer(elbow_model, k=(2, 12), timings=True, force_model=True) # type: ignore
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
        "ch_index":   calinski_harabasz_score(latent_vecs, kmeans.labels_)
    }
    
    _append_metrics_to_csv(metrics, model_type, save_dir)

    print(f"Optimal k        : {optimal_k}")
    print(f"Inertia (WCSS)   : {metrics['inertia']:.4f}")
    print(f"Silhouette Score : {metrics['silhouette']:.4f}")
    print(f"CH Index         : {metrics['ch_index']:.4f}")

    return kmeans, metrics
    

    


