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
            layers.append(nn.Linear(in_dim, out_dim)) # type: ignore
            layers.append(nn.LeakyReLU(inplace=True)) # type: ignore

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




    
    
    
    
# vanilla VAE encoder: input (flatten) -> hidden layer 1 (linear + leaky relu) -> hidden layer 2 (linear + leaky relu) -> mu and logvar layers
# class Encoder(nn.Module):
#     def __init__(self, input_height, input_width, hidden_dim_1, hidden_dim_2, latent_dim):
#         super().__init__()
#         self.input_dim = input_height * input_width
#         self.net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.input_dim, hidden_dim_1),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(hidden_dim_1, hidden_dim_2),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.mu_layer = nn.Linear(hidden_dim_2, latent_dim)
#         self.logvar_layer = nn.Linear(hidden_dim_2, latent_dim)
        
#     def forward(self, x):
#         h = self.net(x) # TODO: i dont understand the syntax here
#         mu = self.mu_layer(h)
#         logvar = self.logvar_layer(h)
#         return mu, logvar