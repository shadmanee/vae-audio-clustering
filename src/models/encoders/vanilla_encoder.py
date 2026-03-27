import torch.nn as nn

# vanilla VAE encoder: input (flatten) -> hidden layer 1 (linear + leaky relu) -> hidden layer 2 (linear + leaky relu) -> mu and logvar layers
class Encoder(nn.Module):
    def __init__(self, input_height, input_width, hidden_dim_1, hidden_dim_2, latent_dim):
        super().__init__()
        self.input_dim = input_height * input_width
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim_1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(inplace=True),
        )
        self.mu_layer = nn.Linear(hidden_dim_2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim_2, latent_dim)
        
    def forward(self, x):
        h = self.net(x) # TODO: i dont understand the syntax here
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar       