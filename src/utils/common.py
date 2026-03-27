import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from typing import Tuple
import config, matplotlib.pyplot as plt

def split_data(dataset, ratio=0.8, batch_size=32, shuffle=True) -> Tuple[DataLoader, DataLoader]:
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
    
# TODO: more complex annealing: https://github.com/hubertrybka/vae-annealing
# annealing starts at 0.1 and is capped at beta
def vae_loss(x_hat, x, mu, logvar, epoch, beta=1.0, beta_type="fixed"):
    recon = F.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta = min(beta, (epoch + 1) / 10) if beta_type == "annealing" else beta
    total = recon + beta * kl
    
    return total, recon, kl

def train_one_epoch(model, loader, optimizer, epoch, beta=1.0, beta_type="fixed", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.DEBUG: print(f"Training epoch {epoch + 1}...")
    model.train()
    total_sum = recon_sum = kl_sum = n = 0
    
    for batch_i, (x, _) in enumerate(loader):
        if config.DEBUG: print(f"Loading training data - batch {batch_i + 1}...")
        x = x.to(device)
        optimizer.zero_grad()
        if config.DEBUG: print(f"Training model on batch {batch_i + 1}")
        x_hat, mu, logvar = model(x)
        if config.DEBUG: print(f"Calculating training stats on {batch_i + 1}...")
        loss, recon, kl = vae_loss(x_hat=x_hat, x=x, mu=mu, logvar=logvar, epoch=epoch, beta=beta, beta_type=beta_type)
        loss.backward()
        optimizer.step()

        total_sum += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
        n += x.size(0) # TODO: = why (incresaing by the exact no. of samples in each batch?)
              
    train_total = total_sum / max(n, 1)
    train_recon = recon_sum / max(n, 1)
    train_kl = kl_sum / max(n, 1)
    
    return {
        "loss": train_total,
        "recon": train_recon,
        "kl": train_kl
    }
    
def evaluate_one_epoch(model, loader, epoch, beta=1.0, beta_type="fixed", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.DEBUG: print(f"Evaluating epoch {epoch + 1}...")
    model.eval()
    total_sum = recon_sum = kl_sum = n = 0

    with torch.no_grad():
        for batch_i, (x, _) in enumerate(loader):
            if config.DEBUG: print(f"Loading test/validation data - batch {batch_i + 1}")
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            if config.DEBUG: print(f"Calculating evaluation stats on batch {batch_i + 1}")
            loss, recon, kl = vae_loss(x_hat=x_hat, x=x, mu=mu, logvar=logvar, epoch=epoch, beta=beta, beta_type=beta_type)
            
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
    
def train_vae(model, train_loader, test_loader, optimizer, epochs, beta=1.0, beta_type="fixed", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {"train_total": [], "test_total": [],
               "train_recon": [], "test_recon": [],
               "train_kl": [], "test_kl": []}
    for epoch in range(epochs):
        train_stats = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, epoch=epoch, beta=beta, beta_type=beta_type, device=device)
        test_stats = evaluate_one_epoch(model=model, loader=test_loader, epoch=epoch, beta=beta, beta_type=beta_type, device=device)
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
        
    return history

def plot_history(history, title):
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
    plt.show()