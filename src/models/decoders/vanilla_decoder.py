import torch.nn as nn

# vanilla VAE decoder: input -> hidden layer 1 (linear + leaky relu) -> hidden layer 2 (linear + leaky relu) -> output (unflatten) 
class Decoder(nn.Module):
    def __init__(self, output_height, output_width, hidden_dim_1, hidden_dim_2, latent_dim):
        super().__init__()
        self.output_dim = output_height * output_width
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim_1, self.output_dim),
            nn.Unflatten(1, (output_height, output_width))
        )
        
    def forward(self, z):
        return self.net(z)