import logging
from sacred import Experiment
import seml
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.metrics import classification_report
from scarches.dataset.trvae.data_handling import remove_sparsity
from lataq.models import EMBEDCVAE
from utils import entropy_batch_mixing, knn_purity, label_encoder
from exp_dict import EXPERIMENT_INFO
from shutil import rmtree
np.random.seed(420)

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
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(
        data: str,
        latent_dim: int,
        loss_metric: str,
        clustering_res: float,
        hidden_layers: int,
        n_epochs: int,
        n_pre_epochs: int,
        alpha_epoch_anneal: int,
        eta: float,
        overwrite: int,
    ):
    logging.info('Received the following configuration:')
    logging.info(
        f'Dataset: {data}, latent_dim: {latent_dim}, loss metric: {loss_metric},'
        f'clustering res: {clustering_res}, hidden layers: {hidden_layers}'
    )

    DATA_DIR = '/storage/groups/ml01/workspace/carlo.dedonno/LATAQ/data'
    REF_PATH = f'/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/tmp/ref_model_embedcvae_{overwrite}'
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

    early_stopping_kwargs = {
        "early_stopping_metric": "val_classifier_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    EPOCHS = n_epochs
    PRE_EPOCHS = n_pre_epochs
    tranvae = EMBEDCVAE(
        adata=source_adata,
        condition_key=condition_key,
        cell_type_keys=cell_type_key,
        hidden_layer_sizes=[128]*int(hidden_layers),
        latent_dim=int(latent_dim),
        use_mmd=False,
    )
    tranvae.train(
        n_epochs=EPOCHS,
        early_stopping_kwargs=early_stopping_kwargs,
        alpha_epoch_anneal=alpha_epoch_anneal,
        pretraining_epochs=PRE_EPOCHS,
        clustering_res=clustering_res,
        labeled_loss_metric=loss_metric,
        unlabeled_loss_metric=loss_metric,
        eta=eta,
    )
    tranvae.save(REF_PATH, overwrite=True)
    tranvae_query = EMBEDCVAE.load_query_data(
        adata=target_adata,
        reference_model=REF_PATH,
        labeled_indices=[],
    )
    tranvae_query.train(
            n_epochs=EPOCHS,
            early_stopping_kwargs=early_stopping_kwargs,
            alpha_epoch_anneal=alpha_epoch_anneal,
            pretraining_epochs=PRE_EPOCHS,
            clustering_res=clustering_res,
            eta=eta,
            labeled_loss_metric=loss_metric,
            unlabeled_loss_metric=loss_metric
        )
    results_dict = tranvae_query.classify(
            adata.X, 
            adata.obs[condition_key], 
            metric=loss_metric
        )
    for i in range(len(cell_type_key)):
        preds = results_dict[cell_type_key[i]]['preds']
        probs = results_dict[cell_type_key[i]]['probs']
        classification_df = pd.DataFrame(
            classification_report(
                y_true=adata.obs[cell_type_key[i]], 
                y_pred=preds,
                output_dict=True
            )
        ).transpose()

    latent_adata = tranvae_query.get_latent(
        x = adata.X,
        c = adata.obs[condition_key]
    )
    latent_adata = sc.AnnData(latent_adata)
    latent_adata.obs[condition_key] = adata.obs[condition_key].tolist()
    latent_adata.obs[cell_type_key[0]] = adata.obs[cell_type_key[0]].tolist()
    conditions, _ = label_encoder(latent_adata, condition_key=condition_key)
    labels, _ = label_encoder(latent_adata, condition_key=cell_type_key[0])
    latent_adata.obs[condition_key] = conditions.squeeze(axis=1)
    latent_adata.obs[cell_type_key[0]] = labels.squeeze(axis=1)
    ebm = entropy_batch_mixing(latent_adata, condition_key, n_neighbors=15)
    knn = knn_purity(latent_adata, cell_type_key[0], n_neighbors=15)

    results = {
        'classification_report': classification_df,
        'ebm': ebm,
        'knn': knn,
    }

    rmtree(REF_PATH)
    return results
