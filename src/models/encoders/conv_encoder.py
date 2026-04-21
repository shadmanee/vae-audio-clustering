import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()

        self.input_height = layer_params["input_height"]
        self.input_width = layer_params["input_width"]
        self.intermediate_dims = list(layer_params["intermediate_dims"])[::-1]
        self.latent_dim = layer_params["latent_dim"]

        dims = [1] + self.intermediate_dims
        layers = []

        for in_ch, out_ch in zip(dims, dims[1:]):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            ])

        layers.append(nn.Flatten())
        self.feature_extractor = nn.Sequential(*layers)

        self.flattened_size = self._get_flattened_size()

        self.mu_layer = nn.Linear(self.flattened_size, self.latent_dim)
        self.logvar_layer = nn.Linear(self.flattened_size, self.latent_dim)

    def _get_flattened_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_height, self.input_width)
            out = self.feature_extractor(dummy)
            return int(np.prod(out.shape[1:]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_extractor(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


if __name__ == "__main__":
    # Unconditional
    enc = Encoder({
        "input_height": 64, "input_width": 91,
        "intermediate_dims": [128, 64, 32], "latent_dim": 16
    })
    print(enc)

    # Conditional
    enc_cvae = Encoder({
        "input_height": 64, "input_width": 91,
        "intermediate_dims": [128, 64, 32], "latent_dim": 16,
        "num_classes": 9
    })
    print(enc_cvae)