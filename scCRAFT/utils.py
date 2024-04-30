import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scanpy as sc
import numpy as np

import scipy.sparse
import random
import anndata
import pandas as pd
from itertools import combinations
from torch.distributions import Normal, kl_divergence as kl
from torch.utils.data import DataLoader, TensorDataset,Dataset





def weights_init_normal(m):
    classname = m.__class__.__name__
    # Skip if it's an instance of _DomainSpecificBatchNorm  
    if classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def generate_balanced_dataloader(adata, batch_size, batch_key='batch'):
    if not adata.obs_names.is_unique:
        print("Error: Indices are not unique!")
        raise AssertionError("Indices are not unique. Please ensure the indices are unique before proceeding.")
    # Map unique batch keys to integers
    unique_batches = adata.obs[batch_key].unique()
    batch_to_int = {batch: i for i, batch in enumerate(unique_batches)}
    unsupervised_labels1 = adata.obs['leiden1'].cat.codes.values
    unsupervised_labels2 = adata.obs['leiden2'].cat.codes.values
    # Separate the dataset by batches and sample indices
    batch_indices = []
    batch_labels_list = []
    for batch in unique_batches:
        # Find the indices for the current batch
        batch_indices_in_adata = adata.obs[adata.obs[batch_key] == batch].index
        
        # Sample indices from the current batch
        if len(batch_indices_in_adata) >= batch_size:
            sampled_indices = np.random.choice(batch_indices_in_adata, batch_size, replace=False)
        else:
            # If not enough cells, sample with replacement
            sampled_indices = np.random.choice(batch_indices_in_adata, batch_size, replace=True)
        
        # Get the integer positions of the sampled indices
        sampled_indices_pos = [adata.obs_names.get_loc(idx) for idx in sampled_indices]
        batch_indices.extend(sampled_indices_pos)
        
        # Map the batch keys to integers and add to the label list
        batch_labels_list.extend([batch_to_int[batch]] * batch_size)
    
    # Extract the feature data
    X_sampled = adata.X[batch_indices, :]

    # Convert features to tensor
    if isinstance(X_sampled, np.ndarray):
        X_tensor = torch.tensor(X_sampled, dtype=torch.float32)
    else:  # if it's a sparse matrix
        X_tensor = torch.tensor(X_sampled.toarray(), dtype=torch.float32)
    
    # Convert batch labels to tensor
    v_tensor = torch.tensor(batch_labels_list, dtype=torch.int64)
    label_tensor1 = torch.tensor(unsupervised_labels1[batch_indices], dtype=torch.int64)
    label_tensor2 = torch.tensor(unsupervised_labels2[batch_indices], dtype=torch.int64)
    
    # Create a TensorDataset and DataLoader
    combined_dataset = TensorDataset(X_tensor, v_tensor, label_tensor1, label_tensor2)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size * 2, shuffle=True)
    
    return dataloader

# Example usage (assuming `adata` is your AnnData object):
# data_loader = generate_balanced_dataloader(adata, batch_size=256, batch_key='batch')

def count_labels_per_batch(labels, batch_ids):
    unique_batches = batch_ids.unique()
    label_counts_per_batch = {batch: (labels[batch_ids == batch].unique(), 
                                      torch.stack([(labels[batch_ids == batch] == l).sum() for l in labels[batch_ids == batch].unique()]))
                              for batch in unique_batches}
    return label_counts_per_batch



def create_triplets(embeddings, labels, labels_high, batch_ids, margin=1.0, num_triplets_per_label=15):
    label_counts_per_batch = count_labels_per_batch(labels, batch_ids)
    triplets = []

    for batch_id, (unique_labels, counts) in label_counts_per_batch.items():
        for label in unique_labels:
            indices_with_label = (labels == label) & (batch_ids == batch_id)
            indices_without_label = (labels != label) & (batch_ids == batch_id)
            
            positive_pairs = list(combinations(torch.where(indices_with_label)[0], 2))
            negative_indices = torch.where(indices_without_label)[0]

            # Randomly sample triplets
            sampled_positive_pairs = random.sample(positive_pairs, min(num_triplets_per_label, len(positive_pairs)))
            # Ensure the number of negative samples doesn't exceed the available negatives
            num_negative_samples = min(len(sampled_positive_pairs), len(negative_indices))
            sampled_negative_indices = random.sample(list(negative_indices), num_negative_samples)

            for (anchor_idx, positive_idx), negative_idx in zip(sampled_positive_pairs, sampled_negative_indices):
                anchor, positive, negative = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
                if labels_high[anchor_idx] != labels_high[positive_idx]:
                    # Only consider anchor-negative pair for loss calculation
                    #triplet_loss = torch.relu(-torch.norm(anchor - negative) + margin)
                    continue
                else:
                    # Standard triplet loss calculation for anchor-positive and anchor-negative
                    triplet_loss = torch.relu(torch.norm(anchor - positive) - torch.norm(anchor - negative) + margin)
                    triplets.append(triplet_loss)

    if triplets:
        return torch.mean(torch.stack(triplets))
    else:
        return torch.tensor(0.0)


def set_seed(seed):
    """
    Set the seed for different packages to ensure reproducibility.

    Parameters:
    seed (int): The seed number.
    """
    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

def multi_resolution_cluster(adata, resolution1=0.5, resolution2=7, method="Louvain"):
    """
    Performs PCA, neighbors calculation, and clustering with specified resolutions and method.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - resolution1: float, the resolution parameter for the first clustering.
    - resolution2: float, the resolution parameter for the second clustering.
    - method: str, clustering method to use ("Louvain" or "Leiden").

    The function updates `adata` in place, adding two new columns to `adata.obs`:
    - 'leiden1': contains cluster labels from the first clustering.
    - 'leiden2': contains cluster labels from the second clustering.
    """
    # Perform PCA
    sc.tl.pca(adata, n_comps=50)
    # Compute neighbors using the PCA representation
    sc.pp.neighbors(adata, use_rep="X_pca")
    
    # Determine the clustering function based on the method
    if method.lower() == "louvain":
        clustering_function = sc.tl.louvain
    elif method.lower() == "leiden":
        clustering_function = sc.tl.leiden
    else:
        raise ValueError("Method should be 'Louvain' or 'Leiden'")
    
    # Perform the first round of clustering
    clustering_function(adata, resolution=resolution1)
    adata.obs['leiden1'] = adata.obs[method.lower()]
    
    # Perform the second round of clustering with a different resolution
    clustering_function(adata, resolution=resolution2)
    adata.obs['leiden2'] = adata.obs[method.lower()]
    

# Loader only for Visualization
def generate_adata_to_dataloader(adata, batch_size= 2048):

    if isinstance(adata.X, scipy.sparse.spmatrix):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    
    # Create a DataLoader for batch-wise processing
    dataset = torch.utils.data.TensorDataset(X_tensor, torch.arange(len(X_tensor))) # include indices
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader
    
