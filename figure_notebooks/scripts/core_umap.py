import numpy as np
import scanpy as sc
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import scarches
from scarches.dataset import remove_sparsity
from lataq.models import EMBEDCVAE, TRANVAE

adata_raw = sc.read(
    '../../data/hlca_counts_commonvars.h5ad'
)
adata_raw.obs['ann_finest_level'] = [
        adata_raw.obs.loc[cell, f"ann_level_{highest_lev}"]
        for cell, highest_lev in zip(
            adata_raw.obs.index, adata_raw.obs.ann_highest_res
        )
]

condition_key = 'sample'
cell_type_key = ['ann_finest_level']

lataq_model = EMBEDCVAE.load('../../notebooks/hlca_core_sample', adata=adata_raw)
lataq_model.model.cuda()


data_latent = lataq_model.get_latent(
    adata_raw.X.A.astype('float32'), 
    adata_raw.obs[condition_key].values,
    mean=False,
)

adata_latent = sc.AnnData(data_latent)
adata_latent.obs = adata_raw.obs.copy()
sc.pp.pca(adata_latent)
sc.pp.neighbors(adata_latent, n_neighbors=15)
sc.tl.umap(adata_latent)

adata_latent.write('../hlca_core_sample_integrated.h5ad')