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

ct_map = {'NK cells': ['NK cells'],
 'Neuroendocrine': ['Neuroendocrine'],
 'CD8 T cells': ['CD8 T cells'],
 'Suprabasal': ['Suprabasal'],
 'Alveolar fibroblasts': ['Alveolar fibroblasts'],
 'Plasma cells': ['Plasma cells'],
 'Mast cells': ['Mast cells'],
 'Alveolar macrophages': ['Alveolar macrophages'],
 'SMG mucous': ['SMG mucous'],
 'AT1': ['AT1'],
 'Classical monocytes': ['Classical monocytes'],
 'Mesothelium': ['Mesothelium'],
 'DC2': ['DC2'],
 'Non-classical monocytes': ['Non-classical monocytes'],
 'DC1': ['DC1'],
 'Deuterosomal': ['Deuterosomal'],
 'AT2 proliferating': ['AT2 proliferating'],
 'Pericytes': ['Pericytes'],
 'EC venous pulmonary': ['EC venous pulmonary'],
 'AT2': ['AT2'],
 'EC venous systemic': ['EC venous systemic'],
 'Plasmacytoid DCs': ['Plasmacytoid DCs'],
 'CD4 T cells': ['CD4 T cells'],
 'SMG serous': ['SMG serous (nasal)', 'SMG serous (bronchial)'],
 'Secretory': ['Club (non-nasal)',
  'Club (nasal)',
  'Goblet (nasal)',
  'Goblet (bronchial)',
  'Goblet (subsegmental)',
  'Transitional Club-AT2'],
 'Basal': ['Basal resting', 'Suprabasal'],
 'Lymphatic EC': ['Lymphatic EC mature',
  'Lymphatic EC proliferating',
  'Lymphatic EC differentiating'],
 'Goblet': ['Goblet (nasal)', 'Goblet (bronchial)', 'Goblet (subsegmental)'],
 'Fibroblasts': ['Peribronchial fibroblasts',
  'Adventitial fibroblasts',
  'Alveolar fibroblasts',
  'Pericytes',
  'Subpleural fibroblasts'],
 'Club': ['Club (non-nasal)', 'Club (nasal)', 'Transitional Club-AT2'],
 'Rare': ['Ionocyte', 'Tuft', 'Neuroendocrine'],
 'Interstitial macrophages': ['Monocyte-derived Mφ',
  'Interstitial Mφ perivascular'],
 'Multiciliated': ['Multiciliated (nasal)', 'Multiciliated (non-nasal)'],
 'EC capillary': ['EC aerocyte capillary', 'EC general capillary'],
 'Mature B cells': ['B cells'],
 'NK_ITGAD+': ['NK cells'],
 'CD4T cells TRM': ['CD4 T cells'],
 'Arterial Pulmonary': ['EC arterial'],
 'Arterial Systemic': ['EC arterial'],
 'CD4 T cells MAIT': ['CD4 T cells'],
 'CD8 T cells TRM': ['CD8 T cells'],
 'CD4 T cells naive': ['CD4 T cells'],
 'Airway smooth muscle': ['Smooth muscle'],
 'NK_XCL1+': ['NK cells'],
 'CD8 T cells GZMK+': ['CD8 T cells'],
 'Plasmablasts': ['Plasma cells'],
 'Vascular smooth muscle': ['Smooth muscle'],
 'Naive B cells': ['B cells'],
 'CD8 T cells ME': ['CD8 T cells'],
 'Erythrocytes': ['New'],
 'Basal proliferating': ['Basal resting', 'Suprabasal'],
 'Schwann myelinating': ['New'],
 'Gamma-delta T cells': ['New'],
 'DC activated': ['Migratory DCs'],
 'Megakaryocytes': ['New'],
 'NKT cells': ['New'],
 'Chondrocytes': ['New'],
 'Regulatory T cells': ['New'],
 'ILCs': ['New'],
 'Mφ proliferating': ['Alveolar Mφ proliferating',
  'Interstitial Mφ perivascular',
  'Monocyte-derived Mφ',
  'Alveolar macrophages'],
 'Schwann nonmyelinating': ['New']}

adata_raw = sc.read(
    '../../data/Meyer_2021_raw.h5ad'
)

adata_meyer = sc.read(
    '../../data/HLCA_meyer_adata_full_tcnorm_log1p.h5ad'
)

adata_ref = sc.read(
    '../../data/hlca_counts_commonvars.h5ad'
)

adata_meyer = adata_meyer[adata_meyer.obs['study'] == 'Meyer_2021']
adata_raw.var = adata_raw.var.set_index('gene_symbols')
adata_raw = adata_raw[:, ~adata_raw.var.index.duplicated(keep='first')]
adata_raw.obs = adata_meyer.obs.reindex(adata_raw.obs.index).copy()
adata_raw = adata_raw[:, adata_ref.var_names]

adata_raw.obs['ann_finest_level'] = [
        adata_raw.obs.loc[cell, f"original_ann_level_{highest_lev}"]
        for cell, highest_lev in zip(
            adata_raw.obs.index, adata_raw.obs.original_ann_highest_res
        )
]

condition_key = 'dataset'
cell_type_key = ['ann_finest_level']

lataq_model = EMBEDCVAE.load('../../notebooks/hlca_core_dataset_healthy', adata=adata_raw)
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
    mean=False,
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

adata_latent_full.write('../hlca_core_dataset_healthy.h5ad')