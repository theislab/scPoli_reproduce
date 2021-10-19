import numpy as np
import pandas as pd
import scanpy as sc
from scIB.metrics import metrics
from sklearn.metrics import classification_report

from lataq_reproduce.exp_dict import EXPERIMENT_INFO

scores_list = []
results_dict_list = []
for d in ["pancreas", "pbmc", "scvelo", "lung", "tumor", "brain"]:
    print(d)
    condition_key = EXPERIMENT_INFO[d]["condition_key"]
    cell_type_key = EXPERIMENT_INFO[d]["cell_type_key"][0]
    reference = EXPERIMENT_INFO[d]["reference"]
    query = EXPERIMENT_INFO[d]["query"]
    adata = sc.read(f"../../data/{d}.h5ad")
    sc.pp.pca(adata)
    adata_symphony = sc.AnnData(
        X=adata.obsm["X_symphony"],
        obs=adata.obs,
        # var=adata.var
    )
    sc.pp.pca(adata_symphony)
    scores = metrics(
        adata,
        adata_symphony,
        condition_key,
        cell_type_key,
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
    results_dict = {
        "data": d,
        "method": "symphony",
        "integration_scores": scores,
        "classification_report": np.nan,
        "classification_report_query": np.nan,
    }
    results_dict_list.append(results_dict)

    if adata.obsm["X_seurat"] is not None:
        adata_seurat = sc.AnnData(
            X=adata.obsm["X_seurat"],
            obs=adata.obs,
            # var=adata.var
        )
        sc.pp.pca(adata_seurat)
        report = pd.DataFrame(
            classification_report(
                y_true=adata[adata.obs[condition_key].isin(query)].obs[cell_type_key],
                y_pred=adata[adata.obs[condition_key].isin(query)].obs["pred_label"],
                output_dict=True,
            )
        ).transpose()

        report_full = pd.DataFrame(
            classification_report(
                y_true=adata.obs[cell_type_key],
                y_pred=adata.obs["pred_label"],
                output_dict=True,
            )
        ).transpose()
        scores = metrics(
            adata,
            adata_seurat,
            condition_key,
            cell_type_key,
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
        results_dict = {
            "data": d,
            "method": "seurat",
            "integration_scores": scores,
            "classification_report": report_full,
            "classification_report_query": report,
        }
        results_dict_list.append(results_dict)

results_df = pd.DataFrame(results_dict_list)
results_df.to_pickle("result_seurat_symphony.pickle")
