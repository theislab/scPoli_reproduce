import logging
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seml
# from lataq.metrics.metrics import metrics
from lataq.models import EMBEDCVAE
from sacred import Experiment
from scarches.dataset.trvae.data_handling import remove_sparsity
from scIB.metrics import metrics
from sklearn.decomposition import KernelPCA
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_distances

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
    holdout: str,
    n_epochs: int,
    n_pre_epochs: int,
    overwrite: int,
    runs: int,
):
    runs = runs + 1
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

    adata = remove_sparsity(adata)
    source_adata = adata[~adata.obs[condition_key].isin([holdout])].copy()
    target_adata = adata[adata.obs[condition_key].isin([holdout])].copy()
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
    HIDDEN_LAYERS = 3
    LATENT_DIM = 25
    ALPHA_EPOCH_ANNEAL = 1e6
    ETA = 1
    CLUSTERING_RES = 1.0
    LOSS_METRIC = "dist"

    lataq = EMBEDCVAE(
        adata=source_adata,
        condition_key=condition_key,
        cell_type_keys=cell_type_key,
        hidden_layer_sizes=[128] * int(HIDDEN_LAYERS),
        latent_dim=int(LATENT_DIM),
    )
    logging.info("Model instantiated")
    lataq.train(
        n_epochs=EPOCHS,
        early_stopping_kwargs=early_stopping_kwargs,
        alpha_epoch_anneal=ALPHA_EPOCH_ANNEAL,
        pretraining_epochs=PRE_EPOCHS,
        clustering_res=CLUSTERING_RES,
        eta=ETA,
    )
    lataq.save(REF_PATH, overwrite=True)
    logging.info("Model trained and saved, initiate surgery")
    lataq_query = EMBEDCVAE.load_query_data(
        adata=target_adata,
        reference_model=REF_PATH,
        labeled_indices=[],
    )
    lataq_query.train(
        n_epochs=EPOCHS,
        early_stopping_kwargs=early_stopping_kwargs,
        alpha_epoch_anneal=ALPHA_EPOCH_ANNEAL,
        pretraining_epochs=PRE_EPOCHS,
        clustering_res=CLUSTERING_RES,
        eta=ETA,
    )

    # generate embedding plot
    embedding = lataq_query.model.embedding.weight.detach().cpu().numpy()
    pca = KernelPCA(n_components=2, kernel="cosine")
    emb_pca = pca.fit_transform(embedding)
    emb_pca = pca.transform(embedding)
    conditions = lataq_query.conditions_
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    import seaborn as sns

    sns.scatterplot(emb_pca[:, 0], emb_pca[:, 1], conditions, ax=ax)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    for i, c in enumerate(conditions):
        ax.plot([0, emb_pca[i, 0]], [0, emb_pca[i, 1]])
        ax.text(emb_pca[i, 0], emb_pca[i, 1], c)
    sns.despine()
    plt.savefig(
        f"{RES_PATH}/pca_embeddings.png",
        bbox_inches="tight",
    )
    # generate cosine distances and find min
    cos_dist = cosine_distances(embedding)
    np.fill_diagonal(cos_dist, np.inf)
    min_cos_dist = np.min(cos_dist, axis=1)

    eucl_dist = cosine_distances(embedding)
    np.fill_diagonal(eucl_dist, np.inf)
    min_eucl_dist = np.min(eucl_dist, axis=1)
    distances = pd.DataFrame(
        {
            "condition": conditions,
            "cos_dist": min_cos_dist,
            "eucl_dist": min_eucl_dist,
        }
    )

    logging.info("Computing metrics")
    results_dict = lataq_query.classify(
        adata.X, adata.obs[condition_key], metric=LOSS_METRIC
    )
    for i in range(len(cell_type_key)):
        preds = results_dict[cell_type_key[i]]["preds"]
        results_dict[cell_type_key[i]]["probs"]
        classification_df = pd.DataFrame(
            classification_report(
                y_true=adata.obs[cell_type_key[i]], y_pred=preds, output_dict=True
            )
        ).transpose()

    results_dict_query = lataq_query.classify(
        target_adata.X, target_adata.obs[condition_key], metric=LOSS_METRIC
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

    adata_latent = lataq_query.get_latent(x=adata.X, c=adata.obs[condition_key])
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
        f"{RES_PATH}/condition_umap_{data}_{holdout}.png",
        bbox_inches="tight",
    )
    plt.close()
    sc.pl.umap(adata_latent, color=cell_type_key[0], show=False, frameon=False)
    plt.savefig(
        f"{RES_PATH}/ct_umap__{data}_{holdout}.png",
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
        "distances": distances,
        "classification_report": classification_df,
        "classification_report_query": classification_df_query,
        "integration_scores": scores,
    }

    rmtree(REF_PATH)
    return results
