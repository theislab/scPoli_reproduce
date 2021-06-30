import numpy as np
import scanpy as sc
import torch
import os

adata = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/tumor_hvg.h5ad'))
adata.raw = None
print(adata)
batches = adata.obs['study'].copy()
#conditions = adata.obs['donor'].copy()
cts = adata.obs['cell_type'].copy()
#adata.X = adata.layers['counts']
del(adata.var)
del(adata.uns)
del(adata.layers)
del(adata.obs)
del(adata.obsm)
#del(adata.obs['tech'])
#del(adata.obs['celltype'])
#del(adata.obs['size_factors'])
adata.obs['study'] = batches
#adata.obs['condition'] = conditions
adata.obs['cell_type'] = cts
print(adata)
print(adata.obs['study'].unique().tolist())
print(adata.obs['cell_type'].unique().tolist())
adata.write_h5ad(filename=os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_tumor_shrinked.h5ad'))
