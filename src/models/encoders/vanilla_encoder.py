import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()
        input_height      = layer_params["input_height"]
        input_width       = layer_params["input_width"]
        intermediate_dims = layer_params["intermediate_dims"]
        latent_dim        = layer_params["latent_dim"]
        self.num_classes  = layer_params.get("num_classes", 0)  # 0 = unconditional

        # If conditional, label is concatenated to flattened input
        self.input_dim = input_height * input_width + self.num_classes

        dims = [self.input_dim] + intermediate_dims
        layers = [nn.Flatten()]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(inplace=True))

        self.net = nn.Sequential(*layers)

        self.mu_layer     = nn.Linear(intermediate_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(intermediate_dims[-1], latent_dim)

    def forward(self, x, y=None):
        h = self.net(x) if y is None else self.net(torch.cat([x.flatten(1), y], dim=1))
        return self.mu_layer(h), self.logvar_layer(h)


if __name__ == "__main__":
    # Unconditional
    enc = Encoder({
        "input_height": 64, "input_width": 91,
        "intermediate_dims": [256, 128], "latent_dim": 32
    })
    x = torch.zeros(4, 64, 91)
    mu, logvar = enc(x)
    print(enc)
    
    # no. of unique genres in the original metadata = 19

    # Conditional
    enc_cvae = Encoder({
        "input_height": 64, "input_width": 91,
        "intermediate_dims": [256, 128], "latent_dim": 32,
        "num_classes": 19
    })
    import torch.nn.functional as F
    y = F.one_hot(torch.randint(0, 19, (4,)), num_classes=19).float()
    mu, logvar = enc_cvae(x, y)
    print(enc_cvae)