import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scanpy as sc
import numpy as np
import umap
import torch.autograd as autograd
import scipy.sparse
import random
from sklearn.decomposition import PCA
import anndata
import pandas as pd
from typing import List
import time
from itertools import combinations
from torch.distributions import Normal, kl_divergence as kl
from torch.utils.data import DataLoader, TensorDataset,Dataset


torch.backends.cudnn.benchmark = True

from typing import Optional, Union
import collections
from typing import Iterable, List

from torch.distributions import Normal
from torch.nn import ModuleList
import jax.numpy as jnp






# Net + Loss function

def log_nb_positive(
    x: Union[torch.Tensor, jnp.ndarray],
    mu: Union[torch.Tensor, jnp.ndarray],
    theta: Union[torch.Tensor, jnp.ndarray],
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
        lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

class Encoder(nn.Module):
    def __init__(self, p_dim, latent_dim=128):
        super(Encoder, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(p_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mean = nn.Linear(512, latent_dim)  # Output layer for mean
        self.fc_var = nn.Linear(512, latent_dim)   # Output layer for variance
        #self.fc_library = nn.Linear(512, 1)        # Output layer for library size
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        # Forward pass through the network
        x = self.fc1(x)
        x = self.relu(self.bn1(x))

        x = self.fc2(x)
        x = self.relu(self.bn2(x))

        # Separate paths for mean, variance, and library size
        q_m = self.fc_mean(x)
        q_v = torch.exp(self.fc_var(x)) + 1e-4
        #library = self.fc_library(x)  # Predicted log library size

        z = reparameterize_gaussian(q_m, q_v)
        
        return q_m, q_v, z



class Decoder(nn.Module):
    def __init__(self, p_dim, v_dim, latent_dim=256):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        
        # Main decoder pathway
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + v_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, p_dim),
        )
        
        # Additional pathway for the batch effect (ec)
        self.decoder_ec = nn.Sequential(
            nn.Linear(v_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, p_dim),
        )

        # Parameters for ZINB distribution
        self.px_scale_decoder = nn.Linear(p_dim, p_dim)  # mean (rate) of ZINB
        self.px_r_decoder = nn.Linear(p_dim, p_dim)  # dispersion

    def forward(self, z, ec):
        # Main decoding
        z_ec = torch.cat((z, ec), dim=-1)
        decoded = self.decoder(z_ec)
        decoded_ec = self.decoder_ec(ec)

        # Combining outputs
        combined = self.relu(decoded + decoded_ec)

        # NB parameters with safe exponential

        px_scale = torch.exp(self.px_scale_decoder(combined))
        px_r = torch.exp(self.px_r_decoder(combined))

        # Scale the mean (px_scale) with the predicted library size
        px_rate = px_scale
        
        return px_rate, px_r

   

class VAE(nn.Module):
    def __init__(self, p_dim, v_dim, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = Encoder(p_dim, latent_dim)
        self.decoder = Decoder(p_dim, v_dim, latent_dim)


    def forward(self, x, ec):
        # Encoding
        q_m, q_v, z = self.encoder(x)

        # Decoding
        px_scale, px_r = self.decoder(z, ec)

        # Reconstruction Loss
        #reconst_loss = F.mse_loss(px_scale, x)
        reconst_loss = -log_nb_positive(x, px_scale, px_r)
        # KL Divergence
        mean = torch.zeros_like(q_m)
        scale = torch.ones_like(q_v)
        kl_divergence = kl(Normal(q_m, torch.sqrt(q_v)), Normal(mean, scale)).sum(dim=1)

        return reconst_loss, kl_divergence, z, px_scale


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        # Apply log softmax to the output
        log_preds = F.log_softmax(output, dim=-1)
        
        # Compute the negative log likelihood loss
        loss = F.nll_loss(log_preds, target, reduction=self.reduction)
        
        return loss



class discriminator(nn.Module):
    def __init__(self, n_input, domain_number):
        super(discriminator, self).__init__()
        n_hidden = 128

        # Define layers
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, domain_number)
        self.loss = CrossEntropy()

    def forward(self, x, batch_ids, generator=False):
        # Forward pass through layers
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        output = self.fc3(h)
        softmax_probs = F.softmax(output, dim=1)
        
        D_loss = self.loss(output, batch_ids)
        if self.loss.reduction == 'mean':
             D_loss = D_loss.mean()
        elif self.loss.reduction == 'sum':
             D_loss = D_loss.sum()

        return D_loss


