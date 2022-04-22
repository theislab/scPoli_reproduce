import logging
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seml
# from lataq.metrics.metrics import metrics
from lataq.models import EMBEDCVAE, TRANVAE
from sacred import Experiment
from scarches.dataset.trvae.data_handling import remove_sparsity
from scIB.metrics import metrics
from sklearn.metrics import classification_report

from lataq_reproduce.exp_dict import EXPERIMENT_INFO
from lataq_reproduce.utils import label_encoder

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
def run(
    data: str,
    model: str,
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
    logging.info("Received the following configuration:")
    logging.info(
        f"Dataset: {data}, latent_dim: {latent_dim}, loss metric: {loss_metric},"
        f"clustering res: {clustering_res}, hidden layers: {hidden_layers}"
    )

    DATA_DIR = "/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/data"
    REF_PATH = f"/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/tmp/ref_model_embedcvae_{overwrite}"
    RES_PATH = (
        f"/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/results/hyperopt"
    )
    EXP_PARAMS = EXPERIMENT_INFO[data]
    FILE_NAME = EXP_PARAMS["file_name"]
    adata = sc.read(f"{DATA_DIR}/{FILE_NAME}")
    condition_key = EXP_PARAMS["condition_key"]
    cell_type_key = EXP_PARAMS["cell_type_key"]
    reference = EXP_PARAMS["reference"]
    query = EXP_PARAMS["query"]

    adata = remove_sparsity(adata)
    source_adata = adata[adata.obs[condition_key].isin(reference)].copy()
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
    # hyperbolic_log1p = False

    # if loss_metric == 'hyperbolic_log1p':
    #    loss_metric = 'hyperbolic'
    #    hyperbolic_log1p=True

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
        # labeled_loss_metric=loss_metric,
        # unlabeled_loss_metric=loss_metric,
        # hyperbolic_log1p=hyperbolic_log1p,
        eta=eta,
        weight_decay=0,
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
        weight_decay=0,
        # labeled_loss_metric=loss_metric,
        # unlabeled_loss_metric=loss_metric
    )

    logging.info("Computing metrics")
    results_dict = tranvae_query.classify(
        adata.X, adata.obs[condition_key], metric=loss_metric
    )
    for i in range(len(cell_type_key)):
        preds = results_dict[cell_type_key[i]]["preds"]
        results_dict[cell_type_key[i]]["probs"]
        classification_df = pd.DataFrame(
            classification_report(
                y_true=adata.obs[cell_type_key[i]], y_pred=preds, output_dict=True
            )
        ).transpose()

    results_dict_query = tranvae_query.classify(
        target_adata.X, target_adata.obs[condition_key], metric=loss_metric
    )
    for i in range(len(cell_type_key)):
        preds = results_dict_query[cell_type_key[i]]["preds"]
        results_dict_query[cell_type_key[i]]["probs"]
        classification_df_query = pd.DataFrame(
            classification_report(
                y_true=target_adata.obs[cell_type_key[i]],
                y_pred=preds,
                output_dict=True,
            )
        ).transpose()

    logging.info("Compute integration metrics")
    conditions, _ = label_encoder(adata, condition_key=condition_key)
    labels, _ = label_encoder(adata, condition_key=cell_type_key[0])
    adata.obs["batch"] = conditions.squeeze(axis=1)
    adata.obs["celltype"] = labels.squeeze(axis=1)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    adata_latent = tranvae_query.get_latent(x=adata.X, c=adata.obs[condition_key])
    adata_latent = sc.AnnData(adata_latent)
    adata_latent.obs[condition_key] = adata.obs[condition_key].tolist()
    adata_latent.obs[cell_type_key[0]] = adata.obs[cell_type_key[0]].tolist()
    conditions, _ = label_encoder(adata_latent, condition_key=condition_key)
    labels, _ = label_encoder(adata_latent, condition_key=cell_type_key[0])
    adata_latent.obs["batch"] = conditions.squeeze(axis=1)
    adata_latent.obs["celltype"] = labels.squeeze(axis=1)
    adata_latent.obs["batch"] = adata_latent.obs["batch"].astype("category")
    adata_latent.obs["celltype"] = adata_latent.obs["celltype"].astype("category")
    sc.pp.pca(adata)
    sc.pp.pca(adata_latent)

    sc.pp.neighbors(adata_latent)
    sc.tl.umap(adata_latent)
    sc.pl.umap(adata_latent, color=condition_key, show=False, frameon=False)
    plt.savefig(
        f"{RES_PATH}/condition_umap_{model}_{data}_{loss_metric}_{latent_dim}_{hidden_layers}.png",
        bbox_inches="tight",
    )
    plt.close()
    sc.pl.umap(adata_latent, color=cell_type_key[0], show=False, frameon=False)
    plt.savefig(
        f"{RES_PATH}/ct_umap_{model}_{data}_{loss_metric}_{latent_dim}_{hidden_layers}.png",
        bbox_inches="tight",
    )

    # adata_latent.write(f"{RES_PATH}/adata_latent.h5ad")
    # adata.write(f"{RES_PATH}/adata_original.h5ad")
    scores = metrics(
        adata,
        adata_latent,
        "batch",
        "celltype",
        isolated_labels_asw_=True,
        silhouette_=True,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=True,
        nmi_=True,
        ari_=True,
    )

    scores = scores.T
    scores = scores[
        [
            "NMI_cluster/label",
            "ARI_cluster/label",
            "ASW_label",
            "ASW_label/batch",
            "PCR_batch",
            "isolated_label_F1",
            "isolated_label_silhouette",
            "graph_conn",
        ]
    ]
    results = {
        "classification_report": classification_df,
        "classification_report_query": classification_df_query,
        "integration_scores": scores,
    }

    rmtree(REF_PATH)
    return results
