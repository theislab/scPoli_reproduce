import logging
from sacred import Experiment
import seml
import scanpy as sc
import numpy as np
import pandas as pd
import scvi
import matplotlib.pyplot as plt
from scarches.dataset.trvae.data_handling import remove_sparsity
from lataq_reproduce.exp_dict import EXPERIMENT_INFO
from lataq.metrics.metrics import metrics
import time

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite=None
    db_collection=None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(
        data: str,
        overwrite: int,
    ):
    logging.info('Received the following configuration:')
    logging.info(
        f'Dataset: {data}'
    )

    DATA_DIR = '/storage/groups/ml01/workspace/carlo.dedonno/LATAQ/data'
    RES_PATH = (
        f'/storage/groups/ml01/workspace/carlo.dedonno/'
        f'lataq_reproduce/results/scvi/{data}'
    )
    EXP_PARAMS = EXPERIMENT_INFO[data]
    FILE_NAME = EXP_PARAMS['file_name']

    adata = sc.read(f'{DATA_DIR}/{FILE_NAME}')
    condition_key = EXP_PARAMS['condition_key']
    cell_type_key = EXP_PARAMS['cell_type_key']
    reference = EXP_PARAMS['reference']
    query = EXP_PARAMS['query']

    adata = remove_sparsity(adata)
    source_adata = adata[adata.obs.study.isin(reference)].copy()
    target_adata = adata[adata.obs.study.isin(query)].copy()
    logging.info('Data loaded succesfully')

    scvi.data.setup_anndata(source_adata, batch_key=condition_key)

    vae_ref = scvi.model.SCVI(source_adata, **arches_params)
    ref_time = time.time()
    vae_ref.train()
    ref_time = time.time() - ref_time
    vae_ref.save(f'{RES_PATH}/scvi_model', overwrite=True)
    #save ref time

    vae_q = scvi.model.SCVI.load_query_data(
        target_adata,
        f'{RES_PATH}/scvi_model',
    )
    query_time = time.time()
    vae_q.train(
        max_epochs=200,
        plan_kwargs=dict(weight_decay=0.0),
    )
    query_time = time.time() - query_time
    vae_q.save(
        f'{RES_PATH}/scvi_query_model', 
        overwrite=True
    )
    #save query time
    adata_latent = sc.AnnData(vae_q.get_latent_representation())
    adata_latent.obs[cell_type_key[0]] = target_adata.obs[cell_type_key[0]].tolist()
    adata_latent.obs[condition_key] = target_adata.obs[condition_key].tolist()
    adata_latent.write_h5ad(
        f'{RES_PATH}/adata_latent.h5ad'
    )

    sc.pp.neighbors(adata_latent)
    sc.tl.leiden(adata_latent)
    sc.tl.umap(adata_latent)
    sc.pl.umap(
        adata_latent,
        color=[condition_key],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/adata_latent_batch.png',
        bbox_inches='tight'
    )
    plt.close()
    sc.pl.umap(
        adata_latent,
        color=[cell_type_key[0]],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/adata_latent_celltype.png',
        bbox_inches='tight'
    )
    plt.close()

    adata_full = target_adata.concatenate(source_adata)
    adata_full.obs[condition_key].cat.rename_categories(
        ["Query", "Reference"], 
        inplace=True
    )

    adata_latent = sc.AnnData(vae_q.get_latent_representation(adata_full))
    adata_latent.obs[cell_type_key[0]] = adata_full.obs[cell_type_key[0]].tolist()
    adata_latent.obs[condition_key] = adata_full.obs[condition_key].tolist()
    adata_latent.write_h5ad(
        f'{RES_PATH}/adata_latent_full.h5ad'
    )

    sc.pp.neighbors(adata_latent)
    sc.tl.leiden(adata_latent)
    sc.tl.umap(adata_latent)
    sc.pl.umap(
        adata_latent,
        color=[condition_key],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/adata_full_latent_batch.png',
        bbox_inches='tight'
    )
    plt.close()
    sc.pl.umap(
        adata_latent,
        color=[cell_type_key[0]],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/adata_full_latent_celltype.png',
        bbox_inches='tight'
    )
    plt.close()

    # adata_latent = sc.AnnData(adata_latent)
    # adata_latent.obs[condition_key] = adata.obs[condition_key].tolist()
    # adata_latent.obs[cell_type_key[0]] = adata.obs[cell_type_key[0]].tolist()
    conditions, _ = label_encoder(adata_latent, condition_key=condition_key)
    labels, _ = label_encoder(adata_latent, condition_key=cell_type_key[0])
    adata_latent.obs[condition_key] = conditions.squeeze(axis=1)
    adata_latent.obs[cell_type_key[0]] = labels.squeeze(axis=1)

    scores = metrics(
        adata, 
        adata_latent, 
        condition_key, 
        cell_type_key[0],
        nmi_=True,
        ari_=False,
        silhouette_=False,
        pcr_=True,
        graph_conn_=True,
        isolated_labels_=True,
        hvg_score_=False,
        knn_=True,
        ebm_=True,
    )
    
    scores = scores.T
    scores = scores[['NMI_cluster/label', 
                     #'ARI_cluster/label',
                     #'ASW_label',
                     #'ASW_label/batch',
                     'PCR_batch', 
                     'isolated_label_F1',
                     'isolated_label_silhouette',
                     'graph_conn',
                     'ebm',
                     'knn',
                    ]]

    results = {
        'classification_report': np.nan,
        'classification_report_query': np.nan,
        'integration_scores': scores
    }
    return results