import os

import torch, numpy as np

from run_models import run_PCA, run_VAE, run_KMeans
from utils.common import extract_latents, extract_latents_with_names, combine_audio_and_lyrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def solve_easy():
    basic_vae, latent_dim, train_loader, test_loader, vae_history, _ = run_VAE(model_type="basic") #type: ignore
    pca_baseline, pca_history = run_PCA(n_components=latent_dim, train_loader=train_loader, test_loader=test_loader)
    
    vae_latents = np.concatenate([
        extract_latents(model=basic_vae, loader=train_loader, device=device),
        extract_latents(model=basic_vae, loader=test_loader,  device=device)
    ])
    
    pca_latents = np.concatenate([
        pca_baseline.transform(train_loader),
        pca_baseline.transform(test_loader)
    ])
    
    vae_kmeans, vae_metrics = run_KMeans(vae_latents, model_type="basic")
    pca_kmeans, pca_metrics = run_KMeans(pca_latents, model_type="pca")
    
    return vae_history, pca_history, vae_metrics, pca_metrics

def solve_medium():
    conv_vae, latent_dim, train_loader, test_loader, vae_history, _ = run_VAE(model_type="conv")
    pca_baseline, pca_history = run_PCA(n_components=latent_dim, train_loader=train_loader, test_loader=test_loader)
    
    train_vae_latents, _, audio_names1 = extract_latents_with_names(model=conv_vae, loader=train_loader)
    test_vae_latents, _, audio_names2 = extract_latents_with_names(model=conv_vae, loader=test_loader)
    
    latent_vecs = np.concatenate([train_vae_latents, test_vae_latents], axis=0)
    audio_names = np.concatenate([audio_names1, audio_names2], axis=0)
    
    latent_vecs_hybrid = combine_audio_and_lyrics(latent_vecs=latent_vecs, audio_names=audio_names)
    
    vae_kmeans, vae_metrics = run_KMeans(latent_vecs=latent_vecs_hybrid, model_type="conv")
    
    pca_latents = np.concatenate([
        pca_baseline.transform(train_loader),
        pca_baseline.transform(test_loader)
    ])
    
    pca_latent_vecs_hybrid = combine_audio_and_lyrics(latent_vecs=pca_latents, audio_names=audio_names)
    
    pca_kmeans, pca_metrics = run_KMeans(pca_latents, model_type="pca")
    
    return vae_history, pca_history, vae_metrics, pca_metrics
    
if __name__ == "__main__":
    a, c, b, d = solve_medium()
    print("VAE history: \n", a)
    print("VAE + K-Means history: \n", b)
    print("-" * 20)
    print("PCA history: \n", c)
    print("PCA + K-Means history: \n", d)