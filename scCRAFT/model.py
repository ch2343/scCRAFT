import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scanpy as sc
import numpy as np
import umap
import torch.autograd as autograd
from tqdm.notebook import tqdm
from tqdm import tqdm
import scipy.sparse
import random
from sklearn.decomposition import PCA
import anndata
import pandas as pd
from typing import List
import time
from scCRAFT.networks import *
from scCRAFT.utils import *
    
    
# Main training class
class SCIntegrationModel(nn.Module):
    def __init__(self, adata, batch_key, z_dim):
        super(SCIntegrationModel, self).__init__()
        self.p_dim = adata.shape[1]
        self.z_dim = z_dim
        self.v_dim = np.unique(adata.obs[batch_key]).shape[0]
        
        # Correctly initialize VAE with p_dim, v_dim, and latent_dim
        self.VAE = VAE(p_dim=self.p_dim, v_dim=self.v_dim, latent_dim=self.z_dim)
        self.D_Z = discriminator(self.z_dim, self.v_dim)
        self.mse_loss = torch.nn.MSELoss()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move models to CUDA if available
        self.VAE.to(self.device)
        self.D_Z.to(self.device)

        # Initialize weights
        self.VAE.apply(weights_init_normal)
        self.D_Z.apply(weights_init_normal)

    def train_model(self, adata, batch_key, epochs, d_coef, kl_coef, warmup_epoch):
        # Optimizer for VAE (Encoder + Decoder)
        optimizer_G = optim.Adam(self.VAE.parameters(), lr=0.001, weight_decay=0.)
        # Optimizer for Discriminator
        optimizer_D_Z = optim.Adam(self.D_Z.parameters(), lr=0.001, weight_decay=0.)

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor

        progress_bar = tqdm(total=epochs, desc="Overall Progress", leave=False)
        for epoch in range(epochs):
                set_seed(epoch)
                data_loader = generate_balanced_dataloader(adata, batch_size = 512, batch_key= batch_key)
                self.VAE.train()
                self.D_Z.train()
                all_losses = 0
                D_loss = 0
                T_loss = 0
                V_loss = 0
                for i, (x, v, labels_low, labels_high) in enumerate(data_loader):
                    x = x.to(self.device)
                    v = v.to(self.device)
                    labels_low = labels_low.to(self.device)
                    labels_high = labels_high.to(self.device)
                    batch_size = x.size(0)
                    v_true = v
                    v_one_hot = torch.zeros(batch_size, self.v_dim).to(x.device)
                    # Use scatter_ to put 1s in the indices indicated by v
                    v = v.unsqueeze(1)  # Ensure v is of shape [batch_size, 1] if it's not already
                    
                    v_one_hot.scatter_(1, v, 1).to(v.device)

                    reconst_loss, kl_divergence, z, x_tilde = self.VAE(x, v_one_hot)
                    reconst_loss = torch.clamp(reconst_loss, max = 1e5)

                    loss_cos = (1 - torch.sum(F.normalize(x_tilde, p=2) * F.normalize(x, p=2), 1)).mean()
                    loss_VAE = torch.mean(reconst_loss.mean() + kl_coef * kl_divergence.mean())
                    
                    for disc_iter in range(10):
                        optimizer_D_Z.zero_grad()
                        loss_D_Z = self.D_Z(z, v_true)
                        loss_D_Z.backward(retain_graph=True)
                        optimizer_D_Z.step()
                    
                    optimizer_G.zero_grad()
                    loss_DA = self.D_Z(z, v_true, generator = False)
                    
                    triplet_loss = create_triplets(z, labels_low, labels_high, v_true, margin = 5)
                    if epoch < warmup_epoch:
                        all_loss = - 0 * loss_DA + 1 * loss_VAE + 1 * triplet_loss + 20 * loss_cos
                    else:
                        all_loss = - d_coef * loss_DA + 1 * loss_VAE + 1 * triplet_loss + 20 * loss_cos
                        
                    all_loss.backward()
                    optimizer_G.step()
                    all_losses += all_loss
                    D_loss += loss_DA
                    T_loss += triplet_loss
                    V_loss += loss_VAE
                progress_bar.update(1)  # Increment the progress bar by one for each batch processed
                progress_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", all_loss=all_losses.item(), disc_loss=D_loss.item())
        progress_bar.close()

def train_integration_model(adata, batch_key='batch', z_dim=256, epochs = 150, d_coef = 0.2, kl_coef = 0.005, warmup_epoch = 50):
    number_of_cells = adata.n_obs
    number_of_batches = np.unique(adata.obs[batch_key]).shape[0]
    
    # Default number of epochs
    if epochs == 150:
        # Check if the number of cells goes above 100000
        if number_of_cells > 100000:
            calculated_epochs = int(1.5 * number_of_cells / (number_of_batches * 512))
            # If the calculated value is larger than the default, use it instead
            if calculated_epochs > epochs:
                epochs = calculated_epochs
    else:
        epochs = epochs
    model = SCIntegrationModel(adata, batch_key, z_dim)
    print(epochs)
    start_time = time.time() 
    model.train_model(adata, batch_key=batch_key, epochs=epochs, d_coef = d_coef, kl_coef = kl_coef, warmup_epoch = warmup_epoch)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    model.VAE.eval()
    return model.VAE


    
def obtain_embeddings(adata, VAE, dim = 50):
    VAE.eval()
    data_loader = generate_adata_to_dataloader(adata)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_z = []
    all_indices = [] 
    #all_x_tilde =[]
    for i, (x, indices) in enumerate(data_loader):
        x = x.to(device)
        batch_size = x.size(0)
        _,_,z = VAE.encoder(x)
        all_z.append(z)
        all_indices.extend(indices.tolist())
        
        
    all_z_combined = torch.cat(all_z, dim=0)
    all_indices_tensor = torch.tensor(all_indices)
    all_z_reordered = all_z_combined[all_indices_tensor.argsort()]
    all_z_np = all_z_reordered.cpu().detach().numpy()

    # Create anndata object with reordered embeddings
    adata.obsm['X_scCRAFT'] = all_z_np
    pca = PCA(n_components= dim)
    # Fit and transform the data
    X_scCRAFT_pca = pca.fit_transform(adata.obsm['X_scCRAFT'])

    # Store the PCA-reduced data back into adata.obsm
    adata.obsm['X_scCRAFT'] = X_scCRAFT_pca

    return adata
