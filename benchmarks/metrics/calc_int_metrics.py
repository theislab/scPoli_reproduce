from metrics.utils import entropy_batch_mixing, knn_purity
import scIB as scib
import scanpy as sc
import os
import pandas as pd
import numpy as np


def label_encoder(adata, encoder=None, condition_key="condition"):
    """Encode labels of Annotated `adata` matrix.
    Parameters
    ----------
    adata: : `~anndata.AnnData`
         Annotated data matrix.
    encoder: Dict or None
         dictionary of encoded labels. if `None`, will create one.
    condition_key: String
         column name of conditions in `adata.obs` data frame.

    Returns
    -------
    labels: `~numpy.ndarray`
         Array of encoded labels
    label_encoder: Dict
         dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    if encoder is None:
        encoder = {
            k: v
            for k, v in zip(
                sorted(unique_conditions), np.arange(len(unique_conditions))
            )
        }

    labels = np.zeros(adata.shape[0])
    if not set(unique_conditions).issubset(set(encoder.keys())):
        print("Warning: Labels in adata is not a subset of label-encoder!")
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), encoder


def compute_metrics(
    latent_adata, adata, model, dataset, batch_key="study", label_key="cell_type"
):
    latent_adata.obsm["X_pca"] = latent_adata.X

    scores = scib.metrics.metrics(
        adata,
        latent_adata,
        batch_key,
        label_key,
        nmi_=True,
        ari_=True,
        silhouette_=True,
        pcr_=True,
        graph_conn_=True,
        isolated_labels_=True,
        hvg_score_=False,
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

    ebm = entropy_batch_mixing(latent_adata, batch_key, n_neighbors=15)
    knn = knn_purity(latent_adata, label_key, n_neighbors=15)

    scores["EBM"] = ebm
    scores["KNN"] = knn
    scores["method"] = model
    scores["data"] = dataset
    scores["rqr"] = None
    scores["reference_time"] = 0.0
    scores["query_time"] = 0.0

    # calculate mean batch correction
    batch_cor = 0.0
    batch_metrics = ["ASW_label/batch", "PCR_batch", "graph_conn", "EBM"]
    for metric in batch_metrics:
        batch_cor += scores[metric]
    batch_cor = batch_cor / 4

    # calculate mean bio con
    bio_con = 0.0
    bio_metrics = [
        "NMI_cluster/label",
        "ARI_cluster/label",
        "ASW_label",
        "isolated_label_F1",
        "isolated_label_silhouette",
        "KNN",
    ]
    for metric in bio_metrics:
        bio_con += scores[metric]
    bio_con = bio_con / 6

    overall = (batch_cor + bio_con) / 2
    scores["Batch Cor"] = batch_cor
    scores["Bio Con"] = bio_con
    scores["Overall"] = overall

    """
    scores = scib.metrics.metrics(adata, latent_adata, batch_key, label_key, pcr_=True, hvg_score_=False)

    scores = scores.T
    scores = scores[['PCR_batch']]
    scores['method'] = model
    scores['data'] = dataset
    """
    return scores


models = ["tranvae", "scanvi", "scvi"]
experiments = ["semi", "surg"]
datasets = ["pancreas", "lung", "scvelo", "brain", "pbmc", "tumor"]
# datasets = ["brain", "tumor"]

for model in models:
    for experiment in experiments:
        scores = None
        label_key = "cell_type"
        number = 10
        for dataset in datasets:
            if dataset == "pbmc":
                adata = sc.read(
                    os.path.expanduser(
                        f"~/Documents/benchmarking_datasets/benchmark_pbmc_shrinked.h5ad"
                    )
                )
                batch_key = "condition"
            elif dataset == "brain":
                adata = sc.read(
                    os.path.expanduser(
                        f"~/Documents/benchmarking_datasets/benchmark_brain_shrinked.h5ad"
                    )
                )
                batch_key = "study"
            elif dataset == "pancreas":
                adata = sc.read(
                    os.path.expanduser(
                        f"~/Documents/benchmarking_datasets/benchmark_pancreas_shrinked.h5ad"
                    )
                )
                batch_key = "study"
            elif dataset == "scvelo":
                adata = sc.read(
                    os.path.expanduser(
                        f"~/Documents/benchmarking_datasets/benchmark_scvelo_shrinked.h5ad"
                    )
                )
                batch_key = "study"
            elif dataset == "lung":
                adata = sc.read(
                    os.path.expanduser(
                        f"~/Documents/benchmarking_datasets/benchmark_lung_shrinked.h5ad"
                    )
                )
                batch_key = "condition"
            elif dataset == "tumor":
                adata = sc.read(
                    os.path.expanduser(
                        f"~/Documents/benchmarking_datasets/benchmark_tumor_shrinked.h5ad"
                    )
                )
                batch_key = "study"

            print(dataset)
            latent_adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/{model}_benchmarks/batchwise/{experiment}/{dataset}/{number}_full_adata.h5ad"
                )
            )

            sc.pp.normalize_total(adata)

            conditions, _ = label_encoder(adata, condition_key=batch_key)
            celltypes, _ = label_encoder(adata, condition_key=label_key)
            adata.obs[batch_key] = conditions.squeeze(axis=1)
            adata.obs[label_key] = celltypes.squeeze(axis=1)

            adata.X = adata.X.astype(dtype="float64")
            adata.obs[batch_key] = adata.obs[batch_key].astype(dtype="float64")
            adata.obs[label_key] = adata.obs[label_key].astype(dtype="float64")

            conditions, _ = label_encoder(latent_adata, condition_key="batch")
            celltypes, _ = label_encoder(latent_adata, condition_key="celltype")
            latent_adata.obs[batch_key] = conditions.squeeze(axis=1)
            latent_adata.obs[label_key] = celltypes.squeeze(axis=1)

            latent_adata.X = latent_adata.X.astype(dtype="float64")
            latent_adata.obs[batch_key] = latent_adata.obs[batch_key].astype(
                dtype="float64"
            )
            latent_adata.obs[label_key] = latent_adata.obs[label_key].astype(
                dtype="float64"
            )

            df = compute_metrics(
                latent_adata, adata, model, dataset, batch_key, label_key
            )
            scores = pd.concat([scores, df], axis=0) if scores is not None else df
        scores.to_csv(
            os.path.expanduser(
                f"~/Documents/{model}_benchmarks/batchwise/{experiment}/{model}_results.csv"
            ),
            index=False,
        )
