import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layer):
        super().__init__()
        self.layers = nn.ModuleList( [nn.Linear(dim_in, dim_hidden)] + [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layer)] + [nn.Linear(dim_hidden, dim_out)] )
        self.gelu = nn.GELU()

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if idx !=0 and idx != len(self.layers) - 1:
                x0 = x
                x = layer(x)
                x = x0 + self.gelu(x)
            elif idx == 0:
                x = self.gelu(layer(x))
            elif idx == len(self.layers) - 1:
                x = layer(x)
        return x


# variational autoencoder model
class VAE(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layer):
        super().__init__()
        self.encoder = MLP(dim_in, dim_hidden, dim_hidden, num_layer)
        self.mu = MLP(dim_hidden, dim_hidden, dim_in, 2)
        self.logvar = MLP(dim_hidden, dim_hidden, dim_in, 2)
        self.decoder = MLP(dim_in, dim_hidden, dim_hidden, num_layer)
        self.recon = MLP(dim_hidden, dim_hidden, dim_in, 2)

        self.mseloss = nn.MSELoss()

        self.dim_in = dim_in
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def decode(self, z):
        z = self.decoder(z)
        recon = self.recon(z)
        return recon
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_fn(self, x):
        recon, mu, logvar = self.forward(x)
        recon_loss = self.mseloss(recon, x)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld
    
    # sampling from the model
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.dim_in).to(device) * 10
        return self.decode(z)
