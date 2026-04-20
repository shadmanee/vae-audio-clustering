import torch.nn as nn, torch, numpy as np

class Encoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()

        self.input_height = layer_params["input_height"]
        self.input_width = layer_params["input_width"]
        self.intermediate_dims = list(layer_params["intermediate_dims"])[::-1]
        self.latent_dim = layer_params["latent_dim"]
        self.num_classes = layer_params["num_classes"]
        self.cond_dim = layer_params.get("cond_dim", 32)
        self.fusion_dim = layer_params.get("fusion_dim", max(128, self.intermediate_dims[0]))

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

        self.condition_proj = nn.Sequential(
            nn.Linear(self.num_classes, self.cond_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.flattened_size + self.cond_dim, self.fusion_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.mu_layer = nn.Linear(self.fusion_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(self.fusion_dim, self.latent_dim)

    def _get_flattened_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_height, self.input_width)
            out = self.feature_extractor(dummy)
            return int(np.prod(out.shape[1:]))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_extractor(x)
        y_feat = self.condition_proj(y)
        fused = torch.cat([h, y_feat], dim=1)
        fused = self.fusion(fused)

        mu = self.mu_layer(fused)
        logvar = self.logvar_layer(fused)
        return mu, logvar