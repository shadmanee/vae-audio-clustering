import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()
        self.input_height  = layer_params["input_height"]
        self.input_width   = layer_params["input_width"]
        intermediate_dims  = layer_params["intermediate_dims"][::-1]  # descending -> ascending for conv
        latent_dim         = layer_params["latent_dim"]
        self.num_classes   = layer_params.get("num_classes", 0)  # 0 = unconditional

        dims = [1] + list(intermediate_dims)
        layers = []
        for in_ch, out_ch in zip(dims, dims[1:]):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Flatten())

        self.net = nn.Sequential(*layers)

        self.flattened_size = self._get_flattened_size()

        # If conditional, label is concatenated after flattening
        self.mu_layer     = nn.Linear(self.flattened_size + self.num_classes, latent_dim)
        self.logvar_layer = nn.Linear(self.flattened_size + self.num_classes, latent_dim)

    def _get_flattened_size(self):
        """Pass a dummy tensor through conv layers to get exact flattened size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_height, self.input_width)
            return int(np.prod(self.net(dummy).shape[1:]))

    def forward(self, x, y=None):
        h = self.net(x)  # (batch, flattened_size)
        if y is not None:
            h = torch.cat([h, y], dim=1)  # (batch, flattened_size + num_classes)
        return self.mu_layer(h), self.logvar_layer(h)


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