# Maybe needs some tweaks. I ran it directly in their repo, since it was not installable. Also may update the datasets to match the other scripts

import logging
from sacred import Experiment
import seml

import scanpy as sc
from scIB.metrics import metrics_fast

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

    scores = metrics_fast(
        adata,
        adata_latent,
        'batch',
        'celltype'
    )

    results = {
        'integration_scores': scores.T
    }
    return results