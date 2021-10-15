import logging
from sacred import Experiment
import seml
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.metrics import classification_report
from lataq.metrics.metrics import metrics
from scarches.dataset.trvae.data_handling import remove_sparsity
from lataq.models import EMBEDCVAE, TRANVAE
from lataq_reproduce.utils import label_encoder
from lataq_reproduce.exp_dict import EXPERIMENT_INFO
from shutil import rmtree

np.random.seed(420)

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(ann_levels, overwrite, runs):
    logging.info("Received the following configuration:")
    model = "tranvae"
    latent_dim = 25
    loss_metric = "dist"
    clustering_res = 2.0
    hidden_layers = 3
    n_epochs = 100
    n_pre_epochs = 80
    alpha_epoch_anneal = 1e6
    eta = 1
    runs = runs + 1
    DATA_DIR = "/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/data"
    REF_PATH = f"/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/tmp/ref_model_embedcvae_{overwrite}"
    FILE_NAME = "adata_lung.h5ad"
    adata = sc.read(f"{DATA_DIR}/{FILE_NAME}")
    condition_key = "dataset"
    cell_type_key = [f"ann_level_{i}" for i in ann_levels]
    query = ["Northwestern_Misharin_2018Reyfman"]
    # reference = [
    #    'Vanderbilt_Kropski_bioRxivHabermann_vand',
    #    'Stanford_Krasnow_bioRxivTravaglini',
    #    'Sanger_Teichmann_2019VieiraBraga'
    # ]

    adata = remove_sparsity(adata)
    source_adata = adata[~adata.obs[condition_key].isin(query)].copy()
    target_adata = adata[adata.obs[condition_key].isin(query)].copy()
    logging.info("Data loaded succesfully")

    early_stopping_kwargs = {
        "early_stopping_metric": "val_landmark_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    EPOCHS = n_epochs
    PRE_EPOCHS = n_pre_epochs
    hyperbolic_log1p = False

    if loss_metric == "hyperbolic_log1p":
        loss_metric = "hyperbolic"
        hyperbolic_log1p = True

    if model == "embedcvae":
        tranvae = EMBEDCVAE(
            adata=source_adata,
            condition_key=condition_key,
            cell_type_keys=cell_type_key,
            hidden_layer_sizes=[128] * int(hidden_layers),
            latent_dim=int(latent_dim),
            use_mmd=False,
        )
    elif model == "tranvae":
        tranvae = TRANVAE(
            adata=source_adata,
            condition_key=condition_key,
            cell_type_keys=cell_type_key,
            hidden_layer_sizes=[128] * int(hidden_layers),
            latent_dim=int(latent_dim),
            use_mmd=False,
        )
    logging.info("Model instantiated")
    tranvae.train(
        n_epochs=EPOCHS,
        early_stopping_kwargs=early_stopping_kwargs,
        alpha_epoch_anneal=alpha_epoch_anneal,
        pretraining_epochs=PRE_EPOCHS,
        clustering_res=clustering_res,
        labeled_loss_metric=loss_metric,
        unlabeled_loss_metric=loss_metric,
        hyperbolic_log1p=hyperbolic_log1p,
        eta=eta,
    )
    tranvae.save(REF_PATH, overwrite=True)
    logging.info("Model trained and saved, initiate surgery")
    if model == "embedcvae":
        tranvae_query = EMBEDCVAE.load_query_data(
            adata=target_adata,
            reference_model=REF_PATH,
            labeled_indices=[],
        )
    elif model == "tranvae":
        tranvae_query = TRANVAE.load_query_data(
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
        unlabeled_loss_metric=loss_metric,
    )
    logging.info("Computing metrics")

    results_dict = tranvae_query.classify(
        adata.X, adata.obs[condition_key], metric=loss_metric
    )
    classification_dfs = {}
    for i in range(len(cell_type_key)):
        preds = results_dict[cell_type_key[i]]["preds"]
        probs = results_dict[cell_type_key[i]]["probs"]
        classification_dfs[f"level_{ann_levels[i]}"] = pd.DataFrame(
            classification_report(
                y_true=adata.obs[cell_type_key[i]], y_pred=preds, output_dict=True
            )
        ).transpose()

    results_dict_query = tranvae_query.classify(
        target_adata.X, target_adata.obs[condition_key], metric=loss_metric
    )
    classification_query_dfs = {}
    for i in range(len(cell_type_key)):
        preds = results_dict_query[cell_type_key[i]]["preds"]
        probs = results_dict_query[cell_type_key[i]]["probs"]
        classification_query_dfs[f"level_{ann_levels[i]}"] = pd.DataFrame(
            classification_report(
                y_true=target_adata.obs[cell_type_key[i]],
                y_pred=preds,
                output_dict=True,
            )
        ).transpose()

    # logging.info('Compute integration metrics')
    # latent_adata = tranvae_query.get_latent(
    #     x = adata.X,
    #     c = adata.obs[condition_key]
    # )
    # latent_adata = sc.AnnData(latent_adata)
    # latent_adata.obs[condition_key] = adata.obs[condition_key].tolist()
    # latent_adata.obs[cell_type_key[0]] = adata.obs[cell_type_key[0]].tolist()
    # conditions, _ = label_encoder(latent_adata, condition_key=condition_key)
    # labels, _ = label_encoder(latent_adata, condition_key=cell_type_key[0])
    # latent_adata.obs[condition_key] = conditions.squeeze(axis=1)
    # latent_adata.obs[cell_type_key[0]] = labels.squeeze(axis=1)

    # scores = metrics(
    #     adata,
    #     latent_adata,
    #     condition_key,
    #     cell_type_key[0],
    #     nmi_=False,
    #     ari_=False,
    #     silhouette_=False,
    #     pcr_=True,
    #     graph_conn_=True,
    #     isolated_labels_=False,
    #     hvg_score_=False,
    #     ebm_=True,
    #     knn_=True,
    # )
    # logging.info('Completed integration metrics')
    # scores = scores.T
    # scores = scores[[#'NMI_cluster/label',
    #                     #'ARI_cluster/label',
    #                     #'ASW_label',
    #                     #'ASW_label/batch',
    #                     #'PCR_batch',
    #                     #'isolated_label_F1',
    #                     #'isolated_label_silhouette',
    #                     #'graph_conn',
    #                     'ebm',
    #                     'knn',
    #                 ]]

    results = {
        "classification_report": classification_dfs,
        "classification_report_query": classification_query_dfs,
        #'integration_scores': scores
    }

    rmtree(REF_PATH)
    return results
