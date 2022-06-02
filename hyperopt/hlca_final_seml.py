import logging
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seml
# from lataq.metrics.metrics import metrics
from lataq.models import EMBEDCVAE
from sacred import Experiment
from scarches.dataset.trvae.data_handling import remove_sparsity
from scIB.metrics import metrics

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
    embedding_dim: int,
    latent_dim: int,
    loss_metric: str,
    inject_condition_info: str,
    clustering_res: float,
    hidden_layers: int,
    n_epochs: int,
    n_pre_epochs: int,
    alpha_epoch_anneal: int,
    eta: float,
    overwrite: int,
):

    DATA_DIR = "/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/data"
    REF_PATH = f"/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hlca_models/ref_{eta}/"
    RES_PATH = (
        f"/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/results/hlca"
    )
    adata = sc.read(f"{DATA_DIR}/hlca_counts.h5ad")
    condition_key = "subject_ID"
    cell_type_key = ["ann_finest_level"]

    adata = remove_sparsity(adata)
    logging.info("Data loaded succesfully")

    early_stopping_kwargs = {
        "early_stopping_metric": "val_landmark_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 100,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    EPOCHS = n_epochs
    PRE_EPOCHS = n_pre_epochs
    if inject_condition_info == "decoder":
        inject_condition = ["decoder"]
    elif inject_condition_info == "both":
        inject_condition = ["encoder", "decoder"]

    tranvae = EMBEDCVAE(
        adata=adata,
        condition_key=condition_key,
        cell_type_keys=cell_type_key,
        hidden_layer_sizes=[128] * int(hidden_layers),
        latent_dim=int(latent_dim),
        use_mmd=False,
        embedding_dim=int(embedding_dim),
        inject_condition=inject_condition,
    )

    logging.info("Model instantiated")
    tranvae.train(
        n_epochs=EPOCHS,
        #early_stopping_kwargs=early_stopping_kwargs,
        use_early_stopping=False,
        pretraining_epochs=PRE_EPOCHS,
        clustering_res=clustering_res,
        use_stratified_sampling=False,
        eta=eta,
    )
    tranvae.save(REF_PATH, overwrite=True)
    logging.info("Computing metrics")

    logging.info("Compute integration metrics")
    conditions, _ = label_encoder(adata, condition_key=condition_key)
    labels, _ = label_encoder(adata, condition_key=cell_type_key[0])
    adata.obs["batch"] = conditions.squeeze(axis=1)
    adata.obs["celltype"] = labels.squeeze(axis=1)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    adata_latent = tranvae.get_latent(x=adata.X, c=adata.obs[condition_key])
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
        f"{RES_PATH}/condition_umap_{embedding_dim}_{latent_dim}_{hidden_layers}_{eta}_{inject_condition_info}.png",
        bbox_inches="tight",
    )
    plt.close()
    sc.pl.umap(adata_latent, color=cell_type_key[0], show=False, frameon=False)
    plt.savefig(
        f"{RES_PATH}/ct_umap_{embedding_dim}_{latent_dim}_{hidden_layers}_{eta}_{inject_condition_info}.png",
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
        "integration_scores": scores,
    }

    rmtree(REF_PATH)
    return results
