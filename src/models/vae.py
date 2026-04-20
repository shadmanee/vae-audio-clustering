import torch
import torch.nn as nn
from config import BaseConfig

class VAE(nn.Module):
    def __init__(self, cfg: BaseConfig, num_classes=0, model_type="basic"):
        super().__init__()
        self.config = cfg
        self.model_type = model_type
        self.is_conditional = model_type == "cvae"
        self.num_classes = num_classes

        intermediate_dims = [self.config.HIDDEN_DIM_1, self.config.HIDDEN_DIM_2]

        if self.model_type == "basic":
            from .encoders.vanilla_encoder import Encoder
            from .decoders.vanilla_decoder import Decoder

        elif self.model_type in ["conv", "beta"]:
            from .encoders.conv_encoder import Encoder
            from .decoders.conv_decoder import Decoder
            intermediate_dims.append(self.config.HIDDEN_DIM_3)

        elif self.model_type == "cvae":
            from .encoders.conditional_encoder import Encoder
            from .decoders.conditional_decoder import Decoder
            intermediate_dims.append(self.config.HIDDEN_DIM_3)

        elif self.model_type == "ae":
            from .encoders.conv_encoder import Encoder
            from .decoders.conv_decoder import Decoder
            intermediate_dims.append(self.config.HIDDEN_DIM_3)

        else:
            raise ValueError(
                f"'{self.model_type}' is not a valid model type. "
                f"Choose from: 'basic', 'conv', 'beta', 'cvae', or 'ae'."
            )

        layer_params = {
            "input_height": self.config.INPUT_HEIGHT,
            "input_width": self.config.INPUT_WIDTH,
            "intermediate_dims": intermediate_dims,
            "latent_dim": self.config.LATENT_DIM,
            "num_classes": self.num_classes if self.is_conditional else 0,
        }

        self.encoder = Encoder(layer_params=layer_params)
        self.decoder = Decoder(layer_params=layer_params)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, y=None):
        if self.is_conditional:
            return self.encoder(x, y)
        return self.encoder(x)

    def decode(self, z, y=None):
        if self.is_conditional:
            x_recon = self.decoder(z, y)
        else:
            x_recon = self.decoder(z)

        return x_recon.view(-1, 1, self.config.INPUT_HEIGHT, self.config.INPUT_WIDTH)

    def forward(self, x, y=None):
        # Encode
        if self.is_conditional:
            mu, logvar = self.encoder(x, y)
        else:
            mu, logvar = self.encoder(x)

        # Latent sample/code
        if self.model_type == "ae":
            z = mu                 # deterministic latent
            logvar = torch.zeros_like(mu)  # optional placeholder for compatibility
        else:
            z = self.reparameterize(mu, logvar)

        # Decode
        if self.is_conditional:
            x_recon = self.decoder(z, y)
        else:
            x_recon = self.decoder(z)

        x_recon = x_recon.view(-1, 1, self.config.INPUT_HEIGHT, self.config.INPUT_WIDTH)

        return x_recon, mu, logvar


if __name__ == "__main__":
    from config import BaseConfig

    # Unconditional
    vae = VAE(cfg=BaseConfig(), model_type="basic")
    print(vae)

    # Conditional
    import torch.nn.functional as F
    cfg = BaseConfig()
    cvae = VAE(cfg=cfg, num_classes=19, model_type="cvae")
    x = torch.zeros(4, 64, 91)
    y = F.one_hot(torch.randint(0, 9, (4,)), num_classes=9).float()
    out, mu, logvar = cvae(x, y)
    print(f"CVAE output: {out.shape}")  # (4, 64, 91)