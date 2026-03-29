import torch.nn as nn

# vanilla VAE decoder: input -> hidden layer 1 (linear + leaky relu) -> hidden layer 2 (linear + leaky relu) -> output (unflatten) 

# =====================
# DYNAMIC HIDDEN LAYERS
# =====================
class Decoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()
        output_height = layer_params["input_height"]
        output_width = layer_params["input_width"]
        intermediate_dims = layer_params["intermediate_dims"][::-1]
        latent_dim = layer_params["latent_dim"]
        
        output_dim = output_height * output_width
        
        dims = [latent_dim] + intermediate_dims
        layers = []
        
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(inplace=True))
            
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Unflatten(1, (output_height, output_width)))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.net(z)
    
if __name__ == "__main__":
    dec = Decoder({"input_height": 64, "input_width": 91, "intermediate_dims": [256, 128], "latent_dim": 32})
    print(dec)
    
    
    
    
# class Decoder(nn.Module):
#     def __init__(self, output_height, output_width, hidden_dim_1, hidden_dim_2, latent_dim):
#         super().__init__()
#         self.output_dim = output_height * output_width
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim_2),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(hidden_dim_2, hidden_dim_1),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(hidden_dim_1, self.output_dim),
#             nn.Unflatten(1, (output_height, output_width))
#         )
        
#     def forward(self, z):
#         return self.net(z)