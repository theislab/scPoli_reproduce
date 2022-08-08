import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scIB.metrics import metrics
from lataq_reproduce.exp_dict import EXPERIMENT_INFO
from lataq_reproduce.utils import label_encoder

adata = sc.read('../../data/hlca_counts_commonvars.h5ad')
adata_latent = sc.read('../../notebooks/hlca_core_integrated_sample.h5ad')
condition_key = 'sample'
cell_type_key = ['ann_finest_level']

conditions, _ = label_encoder(adata, condition_key=condition_key)
labels, _ = label_encoder(adata, condition_key=cell_type_key[0])
adata.obs["batch"] = conditions.squeeze(axis=1)
adata.obs["celltype"] = labels.squeeze(axis=1)
adata.obs["batch"] = adata.obs["batch"].astype("category")
adata.obs["celltype"] = adata.obs["celltype"].astype("category")
conditions, _ = label_encoder(adata_latent, condition_key=condition_key)
labels, _ = label_encoder(adata_latent, condition_key=cell_type_key[0])
adata_latent.obs["batch"] = conditions.squeeze(axis=1)
adata_latent.obs["celltype"] = labels.squeeze(axis=1)
adata_latent.obs["batch"] = adata_latent.obs["batch"].astype("category")
adata_latent.obs["celltype"] = adata_latent.obs["celltype"].astype("category")
sc.pp.pca(adata)
sc.pp.pca(adata_latent)

scores = metrics(
    adata,
    adata_latent,
    "batch",
    "celltype",
    isolated_labels_asw_=True,
    silhouette_=True,
    graph_conn_=True,
    pcr_=True,
    isolated_labels_f1_=True,
    nmi_=True,
    ari_=True,
)

scores = scores.T
scores = scores[
    [
        "NMI_cluster/label",
        "ARI_cluster/label",
        "ASW_label",
        "ASW_label/batch",
        "PCR_batch",
        "isolated_label_F1",
        "isolated_label_silhouette",
        "graph_conn",
    ]
]
scores.to_pickle('../hlca_core_sample_lataq_scib.pickle')
