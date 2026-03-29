import torch.nn as nn

# ================================
# DYNAMIC: for more hidden layers
# ================================
class Encoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()
        input_height = layer_params["input_height"]
        input_width = layer_params["input_width"]
        intermediate_dims = layer_params["intermediate_dims"]  # [256, 128]
        latent_dim = layer_params["latent_dim"]
        
        self.input_dim = input_height * input_width
        
        dims = [self.input_dim] + intermediate_dims  # e.g. [5824, 256, 128]
        layers = [nn.Flatten()]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim)) # type: ignore
            layers.append(nn.LeakyReLU(inplace=True)) # type: ignore

        self.net = nn.Sequential(*layers)
        
        self.mu_layer     = nn.Linear(intermediate_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(intermediate_dims[-1], latent_dim)
        
    def forward(self, x):
        h = self.net(x) # TODO: i dont understand the syntax here
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar
    
    
if __name__ == "__main__":
    enc = Encoder({"input_height": 64, "input_width": 91, "intermediate_dims": [256, 128], "latent_dim": 32})
    
    print(enc)




    
    
    
    
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