import scanpy as sc
import numpy as np
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


data = 'pancreas'
#adata = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_scvelo_shrinked.h5ad'))
adata = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_pancreas_shrinked.h5ad'))
reference = ["inDrop1", "inDrop2", "inDrop3", "inDrop4", "fluidigmc1", "smartseq2", "smarter"]
query = ["celseq", "celseq2"]
condition_key = 'study'

ref = sc.read(os.path.expanduser(f'~/Documents/seurat_benchmarks/{data}/ref.h5ad'))
q1 = sc.read(os.path.expanduser(f'~/Documents/seurat_benchmarks/{data}/q1.h5ad'))
q2 = sc.read(os.path.expanduser(f'~/Documents/seurat_benchmarks/{data}/q2.h5ad'))

print(ref)
print(ref.obsm['X_spca'].shape)
ref_data = ref.obsm['X_spca']
ref_batches = ref.obs[condition_key].tolist()
ref_cts = ref.obs['cell_type'].tolist()
ref_preds = ref.obs['cell_type'].tolist()


print(q1)
print(q1.obsm['X_qrmapping'].shape)
q1_data = q1.obsm['X_qrmapping']
q1_batches = adata.obs[condition_key][adata.obs[condition_key].isin([query[0]])].tolist()
q1_cts = adata.obs['cell_type'][adata.obs[condition_key].isin([query[0]])].tolist()
q1_preds = q1.obs["predicted.id"].tolist()

print(q2)
print(q2.obsm['X_qrmapping'].shape)
q2_data = q2.obsm['X_qrmapping']
q2_batches = adata.obs[condition_key][adata.obs[condition_key].isin([query[1]])].tolist()
q2_cts = adata.obs['cell_type'][adata.obs[condition_key].isin([query[1]])].tolist()
q2_preds = q2.obs["predicted.id"].tolist()

query_data = np.concatenate((q1_data, q2_data), axis=0)
query_batches = q1_batches + q2_batches
query_cts = q1_cts + q2_cts
query_preds = q1_preds + q2_preds

new_data = np.concatenate((ref_data, query_data), axis=0)
new_batches = ref_batches + query_batches
new_cts = ref_cts + query_cts
new_preds = ref_preds + query_preds

adata_latent = sc.AnnData(new_data)
adata_latent.obs['celltype'] = new_cts
adata_latent.obs['batch'] = new_batches
adata_latent.obs['predictions'] = new_preds
adata_latent.write_h5ad(filename=os.path.expanduser(f'~/Documents/seurat_benchmarks/{data}/full_adata.h5ad'))

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(
    os.path.expanduser(
        f'~/Documents/seurat_benchmarks/{data}/full_umap_batch.png'),
    bbox_inches='tight')
plt.close()
sc.pl.umap(adata_latent,
           color=['celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(
    os.path.expanduser(
        f'~/Documents/seurat_benchmarks/{data}/full_umap_ct.png'),
    bbox_inches='tight')
plt.close()
sc.pl.umap(adata_latent,
           color=['predictions'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(
    os.path.expanduser(
        f'~/Documents/seurat_benchmarks/{data}/full_umap_pred.png'),
    bbox_inches='tight')
plt.close()

text_file_q = open(os.path.expanduser(f'~/Documents/seurat_benchmarks/{data}/query_acc_report.txt'),"w")
n = text_file_q.write(classification_report(
    y_true=np.array(query_cts),
    y_pred=np.array(query_preds),
    labels=np.array(list(set((query_cts))))
))
text_file_q.close()