import torch
import torch.optim as optim

import config
from datasets import AudioSpectrogramDataset
from utils.common import split_data, train_vae
from models.vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

npy_dir = config.FEATURES_DIR
dataset = AudioSpectrogramDataset(dataset_dir=npy_dir)
train_loader, test_loader = split_data(dataset=dataset, ratio=0.8, batch_size=config.BATCH_SIZE[2], shuffle=config.SHUFFLE)

basic_vae = VAE(cfg=config, model_type="basic").to(device)
print(basic_vae)
optimizer = optim.Adam(basic_vae.parameters(), lr=config.LR)
history = train_vae(model=basic_vae, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=10, beta=config.BETA, beta_type="annealing", device=device)