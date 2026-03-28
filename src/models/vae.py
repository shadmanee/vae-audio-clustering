import torch
import torch.nn as nn
from config import BaseConfig

class VAE(nn.Module):
    def __init__(self, cfg: BaseConfig, model_type="basic"):
        super().__init__()
        self.config = cfg
        self.model_type = model_type # "basic", "conv" or "beta"/"cvae"
        
        if self.model_type == "basic":
            from .encoders.vanilla_encoder import Encoder 
            from .decoders.vanilla_decoder import Decoder
        elif self.model_type == "conv":
            pass
        elif self.model_type == "beta":
            pass
        elif self.model_type == "cvae":
            pass
        else:
            raise ValueError(f"{self.model_type} is not a valid model type.")
        
        self.encoder = Encoder(
            input_height=self.config.INPUT_HEIGHT,
            input_width=self.config.INPUT_WIDTH,
            hidden_dim_1=self.config.HIDDEN_DIM_1,
            hidden_dim_2=self.config.HIDDEN_DIM_2,
            latent_dim=self.config.LATENT_DIM
        )
        self.decoder = Decoder(
            output_height=self.config.INPUT_HEIGHT,
            output_width=self.config.INPUT_WIDTH,
            hidden_dim_1=self.config.HIDDEN_DIM_1,
            hidden_dim_2=self.config.HIDDEN_DIM_2,
            latent_dim=self.config.LATENT_DIM
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, self.config.INPUT_HEIGHT, self.config.INPUT_WIDTH)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar