import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

import typing as ty
import math
from src.ldm.vae.encoder import Encoder
from src.ldm.vae.decoder import Decoder
from src.ldm.vae.core import Transformer


class VAE(nn.Module):
    def __init__(self, d_numerical, categories, num_layers, d_token, n_head = 1, factor = 4, bias = True):
        super(VAE, self).__init__()
        self.num_layers = num_layers
        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        self.n_head = n_head
        self.factor = factor
        self.bias = bias

        self.encoder = Encoder(num_layers, d_numerical, categories, d_token, n_head, factor, bias)
        self.decoder = Decoder(num_layers, d_numerical, categories, d_token, n_head, factor, bias)

    def get_embedding(self, x_num, x_cat):
        mu_z, _ = self.encoder(x_num, x_cat)  # Only take mu, ignore logvar
        return mu_z.detach()  # Return just the mu values

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat):
        mu_z, logvar_z = self.encoder(x_num, x_cat)
        z = self.reparameterize(mu_z, logvar_z)
        x_num_rec, x_cat_rec = self.decoder(z[:, 1:])
        return x_num_rec, x_cat_rec, mu_z, logvar_z

    def sample(self, num_samples, current_device):
        # Generate noise with CLS token
        z = torch.randn(num_samples, self.d_numerical + len(self.categories) + 1, self.d_token).to(current_device)
        # Remove CLS token before decoding
        return self.decoder(z[:, 1:])

