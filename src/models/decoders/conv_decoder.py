import torch.nn as nn, torch, numpy as np

class Decoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()
        self.input_height = layer_params["input_height"]
        self.input_width = layer_params["input_width"]
        intermediate_dims = layer_params["intermediate_dims"]
        latent_dim = layer_params["latent_dim"]
        self.num_classes = layer_params.get("num_classes", 0)  # 0 = unconditional

        dims = intermediate_dims + [1]
        spatial_dims = self._compute_spatial_dims(n_layers=len(intermediate_dims))
        flattened_size = intermediate_dims[0] * spatial_dims[0][0] * spatial_dims[0][1]

        # If conditional, fc input is z concatenated with one-hot label y
        self.fc = nn.Linear(latent_dim + self.num_classes, flattened_size)

        layers = [nn.Unflatten(1, (intermediate_dims[0], spatial_dims[0][0], spatial_dims[0][1]))]

        for i, (in_ch, out_ch) in enumerate(zip(dims, dims[1:])):
            target_h, target_w = spatial_dims[i + 1]
            current_h, current_w = spatial_dims[i]

            out_pad_h = target_h - (current_h * 2 - 1) - 1 + 1
            out_pad_w = target_w - (current_w * 2 - 1) - 1 + 1
            out_pad_h = max(0, min(1, out_pad_h))
            out_pad_w = max(0, min(1, out_pad_w))

            layers.append(nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=3, stride=2, padding=1,
                output_padding=(out_pad_h, out_pad_w)
            ))  # type: ignore

            if out_ch != 1:
                layers.append(nn.BatchNorm2d(out_ch))  # type: ignore
                layers.append(nn.LeakyReLU(inplace=True))  # type: ignore

        layers.append(nn.Sigmoid())  # type: ignore
        self.net = nn.Sequential(*layers)

    def _compute_spatial_dims(self, n_layers):
        dims = [(self.input_height, self.input_width)]
        h, w = self.input_height, self.input_width
        for _ in range(n_layers):
            h = (h + 1) // 2
            w = (w + 1) // 2
            dims.append((h, w))
        dims.reverse()
        return dims

    def forward(self, z, y=None):
        if y is not None:
            z = torch.cat([z, y], dim=1)  # (batch, latent_dim + num_classes)
        h = self.fc(z)
        return self.net(h)


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