import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()
        output_height     = layer_params["input_height"]
        output_width      = layer_params["input_width"]
        intermediate_dims = layer_params["intermediate_dims"][::-1]
        latent_dim        = layer_params["latent_dim"]
        self.num_classes  = layer_params.get("num_classes", 0)  # 0 = unconditional

        output_dim = output_height * output_width

        # If conditional, label is concatenated to latent vector
        fc_input_dim = latent_dim + self.num_classes

        dims = [fc_input_dim] + list(intermediate_dims)
        last_dim = dims[-1]

        layers = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Linear(last_dim, output_dim))
        layers.append(nn.Unflatten(1, (output_height, output_width)))

        self.net = nn.Sequential(*layers)

    def forward(self, z, y=None):
        if y is not None:
            z = torch.cat([z, y], dim=1)
        return self.net(z)


if __name__ == "__main__":
    import torch.nn.functional as F

    # Unconditional
    dec = Decoder({
        "input_height": 64, "input_width": 91,
        "intermediate_dims": [256, 128], "latent_dim": 32
    })
    z = torch.zeros(4, 32)
    out = dec(z)
    print(dec)

    # Conditional
    dec_cvae = Decoder({
        "input_height": 64, "input_width": 91,
        "intermediate_dims": [256, 128], "latent_dim": 32,
        "num_classes": 19
    })
    y = F.one_hot(torch.randint(0, 19, (4,)), num_classes=19).float()
    out = dec_cvae(z, y)
    print(dec_cvae)