import torch.nn as nn, torch, numpy as np

class Decoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()

        self.input_height = layer_params["input_height"]
        self.input_width = layer_params["input_width"]
        self.intermediate_dims = list(layer_params["intermediate_dims"])
        self.latent_dim = layer_params["latent_dim"]

        dims = self.intermediate_dims + [1]
        spatial_dims = self._compute_spatial_dims(n_layers=len(self.intermediate_dims))

        self.start_h, self.start_w = spatial_dims[0]
        self.flattened_size = self.intermediate_dims[0] * self.start_h * self.start_w

        self.fc = nn.Linear(self.latent_dim, self.flattened_size)

        layers = [
            nn.Unflatten(1, (self.intermediate_dims[0], self.start_h, self.start_w))
        ]

        for i, (in_ch, out_ch) in enumerate(zip(dims, dims[1:])):
            current_h, current_w = spatial_dims[i]
            target_h, target_w = spatial_dims[i + 1]

            out_pad_h = target_h - ((current_h - 1) * 2 - 2 + 3)
            out_pad_w = target_w - ((current_w - 1) * 2 - 2 + 3)

            out_pad_h = max(0, min(1, out_pad_h))
            out_pad_w = max(0, min(1, out_pad_w))

            layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=(out_pad_h, out_pad_w),
                )
            )

            if out_ch != 1:
                layers.extend([
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(inplace=True),
                ])

        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def _compute_spatial_dims(self, n_layers: int) -> list[tuple[int, int]]:
        dims = [(self.input_height, self.input_width)]
        h, w = self.input_height, self.input_width

        for _ in range(n_layers):
            h = (h + 1) // 2
            w = (w + 1) // 2
            dims.append((h, w))

        dims.reverse()
        return dims

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        x_hat = self.decoder(h)
        return x_hat


if __name__ == "__main__":
    import torch.nn.functional as F

    # Unconditional
    dec = Decoder({
        "input_height": 64, "input_width": 91,
        "intermediate_dims": [128, 64, 32], "latent_dim": 32
    })
    print(dec)

    # Conditional
    dec_cvae = Decoder({
        "input_height": 64, "input_width": 91,
        "intermediate_dims": [128, 64, 32], "latent_dim": 32,
        "num_classes": 9
    })
    print(dec_cvae)