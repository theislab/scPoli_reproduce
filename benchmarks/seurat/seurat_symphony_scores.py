import scanpy as sc
import pandas as pd
from lataq.metrics.metrics import metrics
from lataq_reproduce.exp_dict import EXPERIMENT_INFO

scores_list = []
for d in ['pancreas', 'pbmc', 'scvelo', 'lung', 'tumor', 'brain']:
    print(d)
    condition_key = EXPERIMENT_INFO[d]['condition_key']
    cell_type_key = EXPERIMENT_INFO[d]['cell_type_key'][0]
    adata = sc.read(f'../../data/{d}.h5ad')
    adata_symphony = sc.AnnData(
        X=adata.obsm['X_symphony'],
        obs=adata.obs,
        #var=adata.var
    )
    scores = metrics(
        adata, 
        adata_symphony, 
        condition_key, 
        cell_type_key,
        nmi_=False,
        ari_=False,
        silhouette_=False,
        pcr_=True,
        graph_conn_=True,
        isolated_labels_=False,
        hvg_score_=False,
        ebm_=True,
        knn_=True,
    )
    scores = scores.assign(data=d)
    scores = scores.assign(method='symphony')
    scores_list.append(scores)
    scores_df = pd.concat(scores_list)
    scores_df.to_pickle('scores.pickle')
    if adata.obsm['X_seurat'] is not None:
        adata_seurat = sc.AnnData(
            X=adata.obsm['X_seurat'],
            obs=adata.obs,
            #var=adata.var
        )
        scores = metrics(
            adata, 
            adata_seurat, 
            condition_key, 
            cell_type_key,
            nmi_=False,
            ari_=False,
            silhouette_=False,
            pcr_=True,
            graph_conn_=True,
            isolated_labels_=False,
            hvg_score_=False,
            ebm_=True,
            knn_=True,
        )
        scores = scores.assign(data=d)
        scores = scores.assign(method='seurat')
        scores_list.append(scores)
        scores_df = pd.concat(scores_list)
        scores_df.to_pickle('scores.pickle')