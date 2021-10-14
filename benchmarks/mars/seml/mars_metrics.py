# Maybe needs some tweaks. I ran it directly in their repo, since it was not installable. Also may update the datasets to match the other scripts

import logging
from sacred import Experiment
import seml

import scanpy as sc
from scIB.metrics import metrics

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

    RES_PATH = (
        f'/storage/groups/ml01/workspace/carlo.dedonno/'
        f'lataq_reproduce/results/mars/{data}'
    )

    adata = sc.read(f'{RES_PATH}/adata_original.h5ad')
    adata_latent = sc.read(f'{RES_PATH}/adata_latent.h5ad')

    scores = metrics(
        adata, 
        adata_latent, 
        'batch', 
        'celltype',
        isolated_labels_asw_=True,
        silhouette_=True,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=True,
        nmi_=True,
        ari_=True
    )
    
    scores = scores.T
    scores = scores[[
        'NMI_cluster/label', 
        'ARI_cluster/label',
        'ASW_label',
        'ASW_label/batch',
        'PCR_batch', 
        'isolated_label_F1',
        'isolated_label_silhouette',
        'graph_conn',
    ]]
    results = {
        'integration_scores': scores
    }
    return results