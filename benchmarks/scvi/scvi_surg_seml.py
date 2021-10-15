import logging
from sacred import Experiment
import seml

import time
import scanpy as sc
import numpy as np
import scarches as sca
import matplotlib.pyplot as plt
from scarches.dataset.trvae.data_handling import remove_sparsity
from scIB.metrics import metrics

from lataq_reproduce.exp_dict import EXPERIMENT_INFO
from lataq_reproduce.utils import label_encoder

# from lataq.metrics.metrics import metrics

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
    overwrite: int,
):
    logging.info("Received the following configuration:")
    logging.info(f"Dataset: {data}")

    DATA_DIR = "/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/data"
    RES_PATH = (
        f"/storage/groups/ml01/workspace/carlo.dedonno/"
        f"lataq_reproduce/results/scvi/{data}"
    )
    EXP_PARAMS = EXPERIMENT_INFO[data]
    FILE_NAME = EXP_PARAMS["file_name"]

    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
        n_layers=2,
        deeply_inject_covariates=False,
    )

    adata = sc.read(f"{DATA_DIR}/{FILE_NAME}")
    condition_key = EXP_PARAMS["condition_key"]
    cell_type_key = EXP_PARAMS["cell_type_key"]
    reference = EXP_PARAMS["reference"]
    query = EXP_PARAMS["query"]

    adata = remove_sparsity(adata)
    source_adata = adata[adata.obs.study.isin(reference)].copy()
    target_adata = adata[adata.obs.study.isin(query)].copy()
    logging.info("Data loaded succesfully")

    sca.dataset.setup_anndata(
        source_adata, batch_key=condition_key, labels_key=cell_type_key[0]
    )

    vae_ref = sca.models.SCVI(
        source_adata,  # use default params
    )
    ref_time = time.time()
    vae_ref.train()
    ref_time = time.time() - ref_time
    vae_ref.save(f"{RES_PATH}/scvi_model", overwrite=True)
    # save ref time

    vae_q = sca.models.SCVI.load_query_data(
        target_adata,
        f"{RES_PATH}/scvi_model",
    )
    query_time = time.time()
    vae_q.train(
        max_epochs=200,
        plan_kwargs=dict(weight_decay=0.0),
    )
    query_time = time.time() - query_time
    vae_q.save(f"{RES_PATH}/scvi_query_model", overwrite=True)
    # save query time
    adata_latent = sc.AnnData(vae_q.get_latent_representation())
    adata_latent.obs["celltype"] = target_adata.obs[cell_type_key[0]].tolist()
    adata_latent.obs["batch"] = target_adata.obs[condition_key].tolist()
    adata_latent.write_h5ad(f"{RES_PATH}/adata_latent.h5ad")

    sc.pp.neighbors(adata_latent)
    sc.tl.leiden(adata_latent)
    sc.tl.umap(adata_latent)
    sc.pl.umap(adata_latent, color=["batch"], frameon=False, wspace=0.6, show=False)
    plt.savefig(f"{RES_PATH}/adata_latent_batch.png", bbox_inches="tight")
    plt.close()
    sc.pl.umap(adata_latent, color=["celltype"], frameon=False, wspace=0.6, show=False)
    plt.savefig(f"{RES_PATH}/adata_latent_celltype.png", bbox_inches="tight")
    plt.close()

    adata_full = target_adata.concatenate(source_adata, batch_key="query")
    adata_full.obs["query"] = adata_full.obs["query"].astype("category")
    adata_full.obs["query"].cat.rename_categories(["Query", "Reference"], inplace=True)

    adata_latent = sc.AnnData(vae_q.get_latent_representation(adata_full))
    adata_latent.obs["celltype"] = adata_full.obs[cell_type_key[0]].tolist()
    adata_latent.obs["batch"] = adata_full.obs[condition_key].tolist()
    adata_latent.obs["query"] = adata_full.obs["query"].tolist()
    adata_latent.write_h5ad(f"{RES_PATH}/adata_latent_full.h5ad")

    sc.pp.neighbors(adata_latent)
    sc.tl.leiden(adata_latent)
    sc.tl.umap(adata_latent)
    sc.pl.umap(adata_latent, color=["batch"], frameon=False, wspace=0.6, show=False)
    plt.savefig(f"{RES_PATH}/adata_full_latent_batch.png", bbox_inches="tight")
    plt.close()
    sc.pl.umap(adata_latent, color=["query"], frameon=False, wspace=0.6, show=False)
    plt.savefig(f"{RES_PATH}/adata_full_latent_query.png", bbox_inches="tight")
    plt.close()
    sc.pl.umap(adata_latent, color=["celltype"], frameon=False, wspace=0.6, show=False)
    plt.savefig(f"{RES_PATH}/adata_full_latent_celltype.png", bbox_inches="tight")
    plt.close()

    conditions, _ = label_encoder(adata, condition_key=condition_key)
    labels, _ = label_encoder(adata, condition_key=cell_type_key[0])
    adata.obs["batch"] = conditions.squeeze(axis=1)
    adata.obs["celltype"] = labels.squeeze(axis=1)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    conditions, _ = label_encoder(adata_latent, condition_key="batch")
    labels, _ = label_encoder(adata_latent, condition_key="celltype")
    adata_latent.obs["batch"] = conditions.squeeze(axis=1)
    adata_latent.obs["celltype"] = labels.squeeze(axis=1)
    adata_latent.obs["batch"] = adata_latent.obs["batch"].astype("category")
    adata_latent.obs["celltype"] = adata_latent.obs["celltype"].astype("category")
    sc.pp.pca(adata)
    sc.pp.pca(adata_latent)

    adata_latent.write(f"{RES_PATH}/adata_latent.h5ad")
    adata.write(f"{RES_PATH}/adata_original.h5ad")
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
        "reference_time": ref_time,
        "query_time": query_time,
        "classification_report": np.nan,
        "classification_report_query": np.nan,
        "integration_scores": scores,
    }
    return results
