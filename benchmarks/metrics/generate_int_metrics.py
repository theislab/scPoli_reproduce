import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from .utils import entropy_batch_mixing, knn_purity
import scanpy as sc
import seaborn as sns
import scIB as scib
import pandas as pd
from matplotlib import pyplot as plt

models = ["scanvi"]
datasets = ["pbmc", "pancreas", "brain"]
versions = ["first", "deep"]
ratios = [5, 4, 3, 2, 1]


def compute_metrics(
    latent_adata,
    adata,
    model,
    dataset,
    rqr=None,
    batch_key="study",
    label_key="cell_type",
):
    latent_adata.obsm["X_pca"] = latent_adata.X
    print(adata.shape, latent_adata.shape)
    n_batches = len(adata.obs[batch_key].unique().tolist())

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
    scores["rqr"] = rqr / n_batches if rqr is not None else None
    scores.rqr = scores.rqr.round(2)
    scores["reference_time"] = 0.0
    scores["query_time"] = 0.0

    return scores


for dataset in datasets:
    if dataset == "pbmc":
        adata = sc.read(
            os.path.expanduser(
                f"~/Documents/benchmarking_datasets/Immune_ALL_human_wo_villani_rqr_normalized_hvg.h5ad"
            )
        )
        batch_key = "condition"
        label_key = "final_annotation"
        number = 4
    elif dataset == "brain":
        adata = sc.read(
            os.path.expanduser(
                f"~/Documents/benchmarking_datasets/mouse_brain_subsampled_normalized_hvg.h5ad"
            )
        )
        batch_key = "study"
        label_key = "cell_type"
        number = 4
    elif dataset == "pancreas":
        adata = sc.read(
            os.path.expanduser(
                f"~/Documents/benchmarking_datasets/pancreas_normalized.h5ad"
            )
        )
        batch_key = "study"
        label_key = "cell_type"
        number = 5
    # adata = adata_all.raw.to_adata()
    for version in versions:
        scores = None
        for model in models:
            for ratio in ratios:
                if ratio == 5 and dataset in ["pbmc", "brain"]:
                    continue
                """
                elif ratio == 5 and dataset == 'pancreas' and model == 'scanvi':
                    continue
                elif ratio == 4 and dataset in ['pbmc','brain'] and model == 'scanvi':
                    continue
                """
                test_num = ratio
                latent_adata = sc.read(
                    os.path.expanduser(
                        f"~/Documents/benchmarking_results/rqr/{model}/{dataset}/test_{number}_{version}_cond/label_ratio_{ratio}/full_data.h5ad"
                    )
                )
                latent_adata.obs[batch_key] = latent_adata.obs["batch"].values
                latent_adata.obs[label_key] = latent_adata.obs["celltype"].values
                df = compute_metrics(
                    latent_adata, adata, model, dataset, ratio, batch_key, label_key
                )
                scores = pd.concat([scores, df], axis=0) if scores is not None else df
        scores.to_csv(
            os.path.expanduser(
                f"~/Documents/benchmarking_results/rqr/{dataset}scanvi_full_{version}.csv"
            ),
            index=False,
        )
