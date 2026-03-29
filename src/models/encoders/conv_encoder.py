import torch.nn as nn, torch, numpy as np

class Encoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()
        self.input_height = layer_params["input_height"]
        self.input_width = layer_params["input_width"]
        intermediate_dims = layer_params["intermediate_dims"][::-1]  # [256, 128] -> [128, 256]
        latent_dim = layer_params["latent_dim"]
        
        dims = [1] + intermediate_dims  # e.g. [1, 128, 256]
        layers = []
        
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)) # type: ignore
            layers.append(nn.BatchNorm2d(out_dim)) # type: ignore
            layers.append(nn.LeakyReLU(inplace=True)) # type: ignore

        layers.append(nn.Flatten())
        
        self.net = nn.Sequential(*layers)
        
        self.flattened_size = self._get_flattened_size()
        
        self.mu_layer     = nn.Linear(self.flattened_size, latent_dim)
        self.logvar_layer = nn.Linear(self.flattened_size, latent_dim)
        
    def _get_flattened_size(self):
        """
        Pass a dummy tensor through the conv layers to get the exact flattened size.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_height, self.input_width)
            out = self.net(dummy)
            
            return int(np.prod(out.shape[1:]))
        
    def forward(self, x):
        h = self.net(x) # TODO: i dont understand the syntax here
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar       
    
if __name__ == "__main__":
    enc = Encoder({"input_height": 64, "input_width": 91, "intermediate_dims": [128, 64, 32], "latent_dim": 16})
    
    print(enc)