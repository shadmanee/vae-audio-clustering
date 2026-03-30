from pathlib import Path

from models.vae import VAE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from typing import Tuple
from config import config, BaseConfig

import matplotlib.pyplot as plt, numpy as np, pandas as pd, os

def split_data(dataset, ratio=0.8, batch_size=32, shuffle=True) -> Tuple[DataLoader, DataLoader]:
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def loader_to_numpy(loader):
    xs = []
    for x,_ in loader:
        xs.append(x.view(x.size(0), -1).numpy())
    return np.concatenate(xs)
    
# TODO: more complex annealing: https://github.com/hubertrybka/vae-annealing
# annealing starts at 0.1 and is capped at beta
def vae_loss(x_hat, x, mu, logvar, epoch, beta_params):
    beta = beta_params.get("beta", 1.0)
    beta_type = beta_params.get("beta_type", "fixed")
    
    if beta_type == "annealing":
        annealing_epochs = beta_params.get("annealing_epochs", None)
        epoch = epoch if annealing_epochs is None else annealing_epochs
        beta = min(beta, (epoch + 1) / 10)
    
    recon = F.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total = recon + beta * kl
    
    return total, recon, kl

def train_one_epoch(model, loader, optimizer, epoch, beta_params, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.DEBUG: print(f"Training epoch {epoch + 1}...")
    model.train()
    total_sum = recon_sum = kl_sum = n = 0
    
    for x, _ in loader:              
        x = x.to(device)
        if len(x.shape) == 3: x = x.unsqueeze(1) # for convVAE, channel has to be included?         
        optimizer.zero_grad() 
        x_hat, mu, logvar = model(x)
        
        loss, recon, kl = vae_loss(x_hat=x_hat, x=x, mu=mu, logvar=logvar, epoch=epoch, beta_params=beta_params)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_sum += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
        n += x.size(0)
              
    train_total = total_sum / max(n, 1)
    train_recon = recon_sum / max(n, 1)
    train_kl = kl_sum / max(n, 1)
    
    return {
        "loss": train_total,
        "recon": train_recon,
        "kl": train_kl
    }
    
def evaluate_one_epoch(model, loader, epoch, beta_params, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_sum = recon_sum = kl_sum = n = 0

    with torch.no_grad():
        for x, _ in loader:              
            x = x.to(device)
            if len(x.shape) == 3: x = x.unsqueeze(1) # for convVAE, channel has to be included?          
            x_hat, mu, logvar = model(x)
            
            loss, recon, kl = vae_loss(x_hat=x_hat, x=x, mu=mu, logvar=logvar, epoch=epoch, beta_params=beta_params)
            
            total_sum += loss.item()
            recon_sum += recon.item()
            kl_sum += kl.item()
            n += x.size(0)

    test_total = total_sum / max(n, 1)
    test_recon = recon_sum / max(n, 1)
    test_kl = kl_sum / max(n, 1)
    
    return {
        "loss": test_total,
        "recon": test_recon,
        "kl": test_kl
    }
    
def train_vae(model: VAE, train_loader, test_loader, optimizer, epochs, annealing_epochs, beta=1.0, beta_type="fixed", device=None, trial_i=None, root: Path=Path(".")):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    beta_params = {
        "beta": beta,
        "beta_type": beta_type,
        "annealing_epochs": annealing_epochs
    }

    history = {"train_total": [], "test_total": [],
               "train_recon": [], "test_recon": [],
               "train_kl": [], "test_kl": []}
    for epoch in range(epochs):
        train_stats = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, epoch=epoch, beta_params=beta_params, device=device)
        test_stats = evaluate_one_epoch(model=model, loader=test_loader, epoch=epoch, beta_params=beta_params, device=device)
        train_total = train_stats["loss"]
        test_total = test_stats["loss"]
        train_recon = train_stats["recon"]
        test_recon = test_stats["recon"]
        train_kl = train_stats["kl"]
        test_kl = test_stats["kl"]
        history["train_total"].append(train_total)
        history["test_total"].append(test_total)
        history["train_recon"].append(train_recon)
        history["test_recon"].append(test_recon)
        history["train_kl"].append(train_kl)
        history["test_kl"].append(test_kl)
        
        # huge print
        print("-" * 50)
        print(f"Epoch {epoch + 1} / {epochs}")
        print(f"{'Metric':<12} | {'Train':<12} | {'Test':<12}")
        print("-" * 50)
        print(f"{'Total Loss':<12} | {train_total:<12.4f} | {test_total:<12.4f}")
        print(f"{'Recon':<12} | {train_recon:<12.4f} | {test_recon:<12.4f}")
        print(f"{'KL Div':<12} | {train_kl:<12.4f} | {test_kl:<12.4f}")
        print("-" * 50 + "\n")
    
    plot_title = model.model_type
    
    if trial_i is not None:
        plot_title = f"{plot_title}_trial_{trial_i + 1}" 
        plots_dir = root / config.RESULT_DIR / Path("trials/plots/")
    
    else: plots_dir = root / config.RESULT_DIR / Path("final/plots/")
    
    
        
    plots_dir.mkdir(parents=True, exist_ok=True)
    file_name = plots_dir / f"{plot_title}.png"
    
    plot_history(history=history, title=plot_title, file_path=file_name)
        
    return history

def plot_history(history, title, file_path):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_total"], label="train total")
    plt.plot(history["test_total"], label="test total")
    plt.plot(history["train_recon"], label="train recon")
    plt.plot(history["test_recon"], label="test recon")
    plt.plot(history["train_kl"], label="train kl")
    plt.plot(history["test_kl"], label="test kl")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    
def create_new_config(best_params: dict):
    """
    Create a new config object from best_params dict returned by study.best_params.
    Only overrides attributes that exist in best_params — everything else is
    inherited from base_cfg (or BaseConfig defaults if base_cfg is None).
    """
    cfg = BaseConfig()
    conv_special_params = {"CHANNEL_1", "CHANNEL_2_MULTIPLIER", "CHANNEL_3_MULTIPLIER"}
    if conv_special_params.issubset(best_params.keys()):
        cfg.HIDDEN_DIM_3 = best_params["CHANNEL_1"]
        cfg.HIDDEN_DIM_2 = best_params["CHANNEL_1"] * best_params["CHANNEL_2_MULTIPLIER"]
        cfg.HIDDEN_DIM_1 = best_params["CHANNEL_1"] * best_params["CHANNEL_2_MULTIPLIER"] * best_params["CHANNEL_3_MULTIPLIER"]

    for key, value in best_params.items():
        if key in conv_special_params: continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"Warning: '{key}' from best_params not found in config — skipping.")
    
    return cfg

def save_result_to_csv(study=None, history=None, model_name=None, save_dir=config.RESULT_DIR, root: Path=Path(".")):
    trial_dir = root / save_dir / Path("trials")
    final_dir = root / save_dir / Path("final")
    trial_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    
    if study is not None:
        df: pd.DataFrame = study.trials_dataframe()
        filepath = os.path.join(trial_dir, f"{model_name}_tuning_results.csv")
        df.to_csv(filepath, index=False)
    
    if history is not None:
        df_history = pd.DataFrame(history)
        history_path = final_dir / f"{model_name}_training_history.csv"
        df_history.to_csv(history_path, index_label="epoch")
        
def extract_latents(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    latents = []
    names = []
    
    with torch.no_grad():
        for x, filenames in loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            latents.append(mu.cpu().numpy())
            names.extend(filenames)
            
    return np.concatenate(latents)

# DIFFERENT FROM THE PREVIOUS EASY TASK FUNCTION
def extract_latents_with_names(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    latents = []
    names = []
    with torch.no_grad():
        for x, filenames in loader:
            x = x.to(device)
            x = x.unsqueeze(1)
            mu, _ = model.encoder(x)
            latents.append(mu.cpu().numpy())
            names.extend(filenames)
            
    return np.concatenate(latents,axis=0), names

def combine_audio_and_lyrics(latent_vecs, audio_names, root: Path=Path(".")):
    hybrid = []
    for i, full_name in enumerate(audio_names):
        parent_stem = str(Path(full_name).stem).split("_")[0]
        z_audio = latent_vecs[i]
        embeddings_path = root / config.EMBEDDINGS_DIR / f"{parent_stem}.npy"
                
        if embeddings_path.exists():
            z_text = np.load(embeddings_path)
            z_combined = np.concatenate([z_audio, z_text])
            hybrid.append(z_combined)
        else:
            print(f"Warning: No lyrics for {parent_stem}")
            
    return np.array(hybrid)
        