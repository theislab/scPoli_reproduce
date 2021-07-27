import numpy as np
import scanpy as sc
import torch
import os


adata= sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_tumor_shrinked.h5ad'))
print(adata)

condition_key = "study"
cell_type_key = "cell_type"

studies = adata.obs[condition_key].unique().tolist()
cell_types = adata.obs[cell_type_key].unique().tolist()
print(studies)

for study in studies:
    study_ad = adata[adata.obs[condition_key].isin([study])]
    study_cts = study_ad.obs[cell_type_key].unique().tolist()
    size = len(study_ad)
    missing_ct = []
    for ct in cell_types:
        if ct not in study_cts:
            missing_ct.append(ct)
    print(study, missing_ct, size)

