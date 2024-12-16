# scCRAFT
 *scCRAFT (**sc**RNA-seq batch **C**orrection and **R**eliable **A**nchor-**F**ree integration with partial **T**opology)*
<img src="model.png" alt="Model Architecture"  width="600" height="470"/>

## Installation
* Requested enviroment creation, packages including Pytorch and jax of corresponding version can be found in scVI install page: https://docs.scvi-tools.org/en/stable/installation.html
* scCRAFT can also be downloaded from GitHub:
```bash
git clone https://github.com/ch2343/scCRAFT.git
cd scCRAFT
pip install .
```

Normally the installation time is less than 5 minutes.

## Quick Start
### Basic Usage
Starting with raw count matrices formatted as AnnData objects, scCRAFT uses a standard pipline adopted by Seurat and Scanpy to preprocess data.
```python
import scCRAFT
from scCRAFT.model import *

# read AnnData (all batches merged into one adata with aligned genes and metadata contains the batch indicator 'batch' for example in adata.obs)
adata = sc.read_h5ad("adata.h5ad")
adata.raw = adata
sc.pp.filter_genes(adata, min_cells=5)
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
adata = adata[:, adata.var['highly_variable']]
```
Then scCRAFT performed the clustering, training and embedding obtaining.
```python
multi_resolution_cluster(adata, resolution1 = 0.5, method = 'Leiden')
VAE = train_integration_model(adata, batch_key = 'batch', epochs = 150, d_coef = 0.2, kl_coef = 0.005, warmup_epoch = 50)
obtain_embeddings(adata, VAE)

#Visualization
sc.pp.neighbors(adata, use_rep="X_scCRAFT")
sc.tl.umap(adata, min_dist=0.5)
sc.pl.umap(adata, color=['batch', 'celltype'], frameon=False, ncols=1)
```
The evaluating procedure `obtain_embeddings()` saves the integrated latent representation of cells in `adata.obsm['X_scCRAFT']` directly, which can be used for downstream integrative analysis.

#### Parameters in `scCRAFT training`:
* `method`: Method of the clustering. 'Leiden' will take more time but might provide more accurate result. *Default*: `Louvain`.
* `resolution1`: Coefficient of the low resolution clustering. A higher low resolution might separate same cell type. *Default*: `0.5`.
* `epochs`: Number of steps for training. *Default*: `150`. Use `epochs=50` for datasets with batch number over 80. (Drop this manually if even more batches)
* `warmup_epoch`: Number of steps for warm-up training. *Default*: `50`. warmup_epoch should be about 1/3 of the epochs (adjust according to the epochs choice)
* `d_coef`: The coefficient of discriminator loss in the overall loss. Higher value means stronger mixing. *Default*: `0.2`.
* `kl_coef`: kl divergence proportion in the VAE loss. *Default*: `0.005`. Sometimes drop it to 0.0005 or 0.0001 to achieve better cell conservation.


The default setting of the parameter works in general. For the real setting with little batch effects, we recommend to use a lower value such as `kl_coef = 0.0005`.

## Evaluation Metrics
### Basic Usage
Since we adopt a few calculation with R, we have to use the robject to import the necessary calculation with R code:
```python
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

#Perhaps require the library relocation (to the library contains kBET and lisi library)
robjects.r('.libPaths(c("current path", "library location"))')

rscript = '''
library(kBET)
library(lisi)
'''
robjects.r(rscript)
kbet = robjects.r('kBET')
lisi = robjects.r['compute_lisi']
```

```python
calculate_metrics(adata, batch_key='batch', celltype_key='celltype', all = False,
                       savepath=None, subsample=None, seed=0,
                      org='human', tp_thr=3., is_raw=False, use_raw=False, n_neighbors=5, is_embed=False, embed='X_pca'
                     )
```
#### Parameters in `evaluation metrics`:
* `all`: The metrics we use. All means all the metrics we have adopted in the evaluation. The 'False' option will remove the calculation of LISI and positive cells (which is computationally slow for large datasets). *Default*: `False`.
* `n_neighbors`: Coefficient of the KNN. *Default*: `5`. Here we can also adopt the default value of 15 in scanpy pipeline.
* `embed`: The embedding we want to evaluate. *Default*: `X_pca`. For scCRAFT, we use 'X_scCRAFT'.

The default setting of the parameter is the one we use to evaluate our model in the paper.

