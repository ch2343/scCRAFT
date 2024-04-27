import torch.utils.data
import scanpy as sc
from sklearn.cluster import KMeans
from scipy.spatial import distance
import scipy.sparse
import scib 
import numpy as np
import random
import pandas as pd
from torch.distributions import  kl_divergence as kl
torch.backends.cudnn.benchmark = True
from sklearn.neighbors import KDTree
from sklearn import preprocessing
import scipy


# LISI
def calculate_LISI(adata, labels=None, total_cells=None, batch_key='batch', celltype_key='celltype', embed = 'X_pca'):
    if celltype_key is None:
        # Calculate bLISI
        lisi_b = 0
        lisi_c = lisi_f1 = np.nan
        if adata.shape[0] < 90:
            perplexity = int(adata.shape[0]/6)
        else:
            perplexity = 30
        lisi_res = lisi(adata.obsm[embed], adata.obs, batch_key, perplexity=perplexity)
        lisi_b = (np.mean(lisi_res)[batch_key]-1.)/(len(set(adata.obs[batch_key]))-1.)
    else:
        # Calculate 1-cLISI
        lisi_res = lisi(adata.obsm[embed], adata.obs, celltype_key)
        lisi_c = 1 - (np.mean(lisi_res)-1.)/(len(set(adata.obs[celltype_key]))-1.)

        # Calculate bLISI
        lisi_b = 0
        for label in labels:
            adata_sub = adata[adata.obs[celltype_key] == label]
            if adata_sub.shape[0] < 90:
                perplexity = int(adata_sub.shape[0]/6)
            else:
                perplexity = 30
            lisi_res = lisi(adata_sub.obsm[embed], adata_sub.obs, batch_key, perplexity=perplexity)
            lisi_batch = (np.mean(lisi_res)-1.)/(len(set(adata_sub.obs[batch_key]))-1.)
            lisi_b += lisi_batch*adata_sub.shape[0]
        lisi_b /= total_cells
        lisi_c = lisi_c.item()
        lisi_b = lisi_b.item()  # This will print only the value without the data type

        # Calcualte F1 score
        lisi_f1 = (2*lisi_c*lisi_b)/(lisi_c + lisi_b)
    
    return lisi_c, lisi_b, lisi_f1


# Positive and true positive cells defined in iMAP
def positive_true_positive(adata, batch_key='batch', celltype_key='celltype', use_raw=False,k1=20, k2=100, tp_thr=3., distance='cosine', embed='X_pca'):
    celltype_list = adata.obs[celltype_key]
    batch_list = adata.obs[batch_key]

    temp_c = adata.obs[celltype_key].value_counts()
    temp_b = pd.crosstab(adata.obs[celltype_key], adata.obs[batch_key])
    temp_b_prob = temp_b.divide(temp_b.sum(1), axis=0)
    
    if use_raw:
        if isinstance(adata.X, scipy.sparse.csr.csr_matrix):
            X = adata.X.todense()
        else:
            X = adata.X
    else:
        X = adata.obsm[embed]
    if distance == 'cosine':
        X = preprocessing.normalize(X, axis=1)

    t1 = KDTree(X)

    p_list = []
    tp_list = []

    for cell in range(len(X)):

        # Discriminate positive cells
        neig1 = min(k1, temp_c[celltype_list[cell]])
        NNs = t1.query(X[cell].reshape(1,-1), neig1+1, return_distance=False)[0, 1:]
        c_NN = celltype_list[NNs]
        true_rate = sum(c_NN == celltype_list[cell])/neig1
        if true_rate > 0.5:
            p_list.append(True)
        else:
            p_list.append(False)

        # Discriminate true positive cells
        if p_list[cell] == True:
            neig2 = min(k2, temp_c[celltype_list[cell]])
            NNs = t1.query(X[cell].reshape(1,-1), neig2, return_distance=False)[0]
            NNs_c = celltype_list[NNs]
            NNs_i = NNs_c == celltype_list[cell]
            NNs = NNs[NNs_i] # get local neighbors that are from the same cell type
            neig2 = len(NNs)
            NNs_b = batch_list[NNs]

            max_b = 0
            b_prob = temp_b_prob.loc[celltype_list[cell]]
            for b in set(batch_list):
                if b_prob[b] > 0 and b_prob[b] < 1:
                    p_b = sum(NNs_b == b)
                    stat_b = abs(p_b - neig2*b_prob[b]) / np.sqrt(neig2*b_prob[b]*(1-b_prob[b]))
                    max_b = max(max_b, stat_b)
            if max_b <= tp_thr:
                tp_list.append(True)
            else:
                tp_list.append(False)
        else:
            tp_list.append(False)

    pos_rate = sum(p_list)/len(p_list)
    truepos_rate = sum(tp_list)/len(tp_list)
    return pos_rate, truepos_rate


    
def calculate_metrics(adata, batch_key='batch', celltype_key='celltype', all = False,
                       savepath=None, subsample=None, seed=0,
                      org='human', tp_thr=3., is_raw=False, use_raw=False, n_neighbors=5, is_embed=False, embed='X_pca'
                     ):
#     if not is_embed:
#         use_raw = True
    if subsample:
        random.seed(seed)
        np.random.seed(seed)
        sample_idx = np.random.choice(adata.shape[0], int(subsample*adata.shape[0]), replace=False)
        adata = adata[sample_idx].copy()
        if not is_raw:
            adata_raw = adata_raw[sample_idx].copy()
        del sample_idx
        print('Data size:', adata.shape)
    labels = set(adata.obs[celltype_key])
    labels_ = labels.copy()
    total_cells = adata.shape[0]
    for label in labels_:
        adata_sub = adata[adata.obs[celltype_key] == label]
        if len(set(adata_sub.obs[batch_key])) == 1 or adata_sub.shape[0] < 10:
            print('Cell cluster %s contains only one batch or has less than 10 cells. Skip.' % label)
            total_cells -= adata_sub.shape[0]
            labels.remove(label)
    
    if all:
        print('LISI---')
        cLISI, bLISI, LISI_F1 = calculate_LISI(adata, batch_key=batch_key, celltype_key=celltype_key, 
                                               labels=labels, total_cells=total_cells, embed = embed)
    pcr_score = scib.metrics.pcr_comparison(
            adata, adata, embed=embed, covariate=batch_key, verbose=False
        )
    print('ASW---')
    asw_label = scib.metrics.silhouette(
            adata, label_key=celltype_key, embed=embed, metric='euclidean'
        )
        # silhouette coefficient per batch

    asw_batch = scib.metrics.silhouette_batch(
            adata,
            batch_key=batch_key,
            label_key=celltype_key,
            embed=embed,
            metric='euclidean',
            return_all=False,
            verbose=False,
        )
    print('kBET---')
    kbet_score = scib.metrics.kBET(
            adata,
            batch_key=batch_key,
            label_key=celltype_key,
            type_=None,
            embed=embed,
            scaled=True,
            verbose=False,
        )

    adata.obs[celltype_key] = adata.obs[celltype_key].astype('category')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=embed)
    graph_conn = scib.metrics.graph_connectivity(adata, celltype_key)

    scib.cl.opt_louvain(
            adata,
            label_key=celltype_key,
            cluster_key='cluster',
            plot=False,
            inplace=True,
            force=True,
            verbose=False
        )
    print('NMI, ARI ---')
    NMI = scib.me.nmi(adata, group1='cluster', group2=celltype_key)

    ARI = scib.me.ari(adata, group1='cluster', group2=celltype_key)
    if all:
        print('positive and true positive rate---')
        pos_rate, truepos_rate = positive_true_positive(adata, batch_key=batch_key, celltype_key=celltype_key, 
                                                        k1=20, k2=100, tp_thr=tp_thr, embed=embed)
        df = pd.DataFrame({'ASW_label': [asw_label],
                        'ARI': [ARI],
                        'NMI': [NMI],
                        '1-cLISI': [cLISI],
                        'bLISI': [bLISI],
                        'ASW_batch': [asw_batch],
                        'kBET Accept Rate': [kbet_score],
                        'graph connectivity': [graph_conn],
                        'PCR_batch': [pcr_score],
                        'pos rate': [pos_rate],
                        'true pos rate': [truepos_rate],
                        'F1 LISI': [LISI_F1]
                      }, index=[embed])
    else:
        df = pd.DataFrame({'ASW_label': [asw_label],
                        'ARI': [ARI],
                        'NMI': [NMI],
                        'ASW_batch': [asw_batch],
                        'kBET Accept Rate': [kbet_score],
                        'graph connectivity': [graph_conn],
                        'PCR_batch': [pcr_score]
                      }, index=[embed])
    if savepath:
        df.to_csv(savepath)
    return df

