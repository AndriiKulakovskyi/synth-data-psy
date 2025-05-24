import math
import torch
import torch.nn as nn
from torch import Tensor
from src.ldm.vae.core import Transformer


class Decoder(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias=True):
        super(Decoder, self).__init__()
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)
        
    def forward(self, z):
        h = self.decoder(z)
        x_hat_num, x_hat_cat = self.detokenizer(h)
        return x_hat_num, x_hat_cat


class Reconstructor(nn.Module):
    """
    Reconstruction head - inverse of tokenizer.
    """
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))  
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num  = h[:, :self.d_numerical]
        h_cat  = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
      
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat