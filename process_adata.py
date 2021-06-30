import scanpy as sc
import os


adata = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/azimuth.h5ad'))
adata.X = adata.raw.X.copy()
#print(adata)
#print(adata.obs["tech"].unique().tolist())
#print(adata.obs["celltype"].unique().tolist())

#adata.X = adata.layers['counts']
adata.raw = adata.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=4000,
    batch_key='dataset_origin',
    subset=True)
adata.X = adata.raw[:, adata.var_names].X
adata.obs['study'] = adata.obs['dataset_origin'].tolist()
adata.obs['cell_type'] = adata.obs['cell_type'].tolist()
adata.write_h5ad(filename=os.path.expanduser(f'~/Documents/benchmarking_datasets/azimuth_hvg.h5ad'))