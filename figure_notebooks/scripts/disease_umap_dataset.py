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

ct_map = {
        "Alveolar epithelium": ["AT1", "AT2", "AT2 proliferating"],
        "B cell lineage": ["B cells", "Plasma cells"],
        "Cancer": ["New"],
        "Endothelial": [
            "EC arterial",
            "EC aerocyte capillary",
            "EC general capillary",
            "EC venous systemic",
            "EC venous pulmonary",
            "Lymphatic EC mature",
            "Lymphatic EC proliferating",
            "Lymphatic EC differentiating",
        ],
        "Epithelial": [
            "Basal resting",
            "Suprabasal",
            "Deuterosomal",
            "Multiciliated (nasal)",
            "Multiciliated (non-nasal)",
            "Club (non-nasal)",
            "Club (nasal)",
            "Goblet (nasal)",
            "Goblet (bronchial)",
            "Goblet (subsegmental)",
            "Transitional Club-AT2",
            "Ionocyte",
            "Tuft",
            "Neuroendocrine",
            "SMG serous (nasal)",
            "SMG serous (bronchial)",
            "SMG mucous",
            "SMG duct",
            "AT1",
            "AT2",
            "AT2 proliferating",
        ],
        "Erythroblast": ["New"],
        "Fibroblast lineage": [
            "Peribronchial fibroblasts",
            "Adventitial fibroblasts",
            "Alveolar fibroblasts",
            "Pericytes",
            "Subpleural fibroblasts",
            "Myofibroblasts",
            "Smooth muscle",
            "Fibromyocytes",
            "SM activated stress response",
        ],  # they don't distinguish between fibroblasts and smooth muscle
        "Mast cells": ["Mast cells"],
        "Myeloid": [
            "DC1",
            "DC2",
            "Migratory DCs",
            "Plasmacytoid DCs",
            "Alveolar macrophages",
            "Alveolar Mφ CCL3+",
            "Alveolar Mφ MT-positive",
            "Alveolar Mφ proliferating",
            "Monocyte-derived Mφ",
            "Interstitial Mφ perivascular",
            "Classical monocytes",
            "Non-classical monocytes",
            "Mast cells",
        ],
        "T cell lineage": [
            "CD4 T cells",
            "CD8 T cells",
            "T cells proliferating",
            "NK cells",
        ],  # these are also NK cells
    }

adata_raw = sc.read(
    '../../data/lambrechts_sub.h5ad'
)

adata_ref = sc.read(
    '../../data/hlca_counts_commonvars.h5ad'
)
adata_raw.obs['ann_finest_level'] = [
        adata_raw.obs.loc[cell, f"ann_level_{highest_lev}"]
        for cell, highest_lev in zip(
            adata_raw.obs.index, adata_raw.obs.ann_highest_res
        )
]

adata_raw = adata_raw[:, adata_ref.var_names]

condition_key = 'dataset'
cell_type_key = ['ann_finest_level']

lataq_model = EMBEDCVAE.load('../../notebooks/hlca_core_dataset_disease', adata=adata_raw)
lataq_model.model.cuda()

results_dict = lataq_model.classify(
    adata_raw.X.A, 
    adata_raw.obs[condition_key], 
    get_prob=False,
    metric='dist',
    threshold=-np.inf
)

data_latent = lataq_model.get_latent(
    adata_ref.X.A.astype('float32'), 
    adata_ref.obs[condition_key].values,
    mean=False
)

adata_latent_ref = sc.AnnData(data_latent)
adata_latent_ref.obs = adata_ref.obs.copy()

data_latent = lataq_model.get_latent(
    adata_raw.X.A, 
    adata_raw.obs[condition_key].values,
    mean=False,
)

adata_latent = sc.AnnData(data_latent)
adata_latent.obs = adata_raw.obs.copy()
adata_latent.obs['probs'] = results_dict[cell_type_key[0]]['probs']
adata_latent.obs['pred'] = results_dict['ann_finest_level']['preds']
adata_latent.obs['outcome'] = [
    adata_latent.obs['pred'].iloc[i] in ct_map[adata_latent.obs['ann_finest_level'].iloc[i]] for i in range(len(adata_latent))
]
adata_latent.obs['log_probs'] = np.log1p(results_dict['ann_finest_level']['probs'])
#adata_latent_full = adata_latent_ref.concatenate(adata_latent, batch_key='query')

labeled_set = lataq_model.get_landmarks_info()
labeled_set.obs['condition'] = 'labeled landmark'
unlabeled_set = lataq_model.get_landmarks_info(landmark_set='unlabeled')
unlabeled_set.obs['condition'] = 'unlabeled landmark'

adata_latent_full = adata_latent_ref.concatenate([adata_latent, labeled_set, unlabeled_set], batch_key='query')
sc.pp.pca(adata_latent_full)
sc.pp.neighbors(adata_latent_full, n_neighbors=15)
sc.tl.umap(adata_latent_full)

adata_latent_full.write('../hlca_core_dataset_disease.h5ad')