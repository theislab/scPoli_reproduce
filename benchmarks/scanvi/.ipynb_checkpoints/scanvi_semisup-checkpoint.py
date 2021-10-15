import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import torch
from sklearn.metrics import classification_report

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction="out")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel("Sample name")


# Experiment Params
# experiments = ["pancreas","pbmc","lung","scvelo","brain"]
experiments = ["tumor"]
test_nrs = [10]
cell_type_key = "cell_type"
for experiment in experiments:
    for test_nr in test_nrs:
        if experiment == "pancreas":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_pancreas_shrinked.h5ad"
                )
            )
            condition_key = "study"
            if test_nr == 1:
                reference = ["Pancreas inDrop"]
                query = [
                    "Pancreas SS2",
                    "Pancreas CelSeq2",
                    "Pancreas CelSeq",
                    "Pancreas Fluidigm C1",
                ]
            elif test_nr == 2:
                reference = ["Pancreas inDrop", "Pancreas SS2"]
                query = ["Pancreas CelSeq2", "Pancreas CelSeq", "Pancreas Fluidigm C1"]
            elif test_nr == 3:
                reference = ["Pancreas inDrop", "Pancreas SS2", "Pancreas CelSeq2"]
                query = ["Pancreas CelSeq", "Pancreas Fluidigm C1"]
            elif test_nr == 4:
                reference = [
                    "Pancreas inDrop",
                    "Pancreas SS2",
                    "Pancreas CelSeq2",
                    "Pancreas CelSeq",
                ]
                query = ["Pancreas Fluidigm C1"]
            elif test_nr == 5:
                reference = [
                    "Pancreas inDrop",
                    "Pancreas SS2",
                    "Pancreas CelSeq2",
                    "Pancreas CelSeq",
                    "Pancreas Fluidigm C1",
                ]
                query = []
            elif test_nr == 10:
                reference = [
                    "inDrop1",
                    "inDrop2",
                    "inDrop3",
                    "inDrop4",
                    "fluidigmc1",
                    "smartseq2",
                    "smarter",
                ]
                query = ["celseq", "celseq2"]
        if experiment == "pbmc":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_pbmc_shrinked.h5ad"
                )
            )
            condition_key = "condition"
            if test_nr == 1:
                reference = ["Oetjen"]
                query = ["10X", "Sun", "Freytag"]
            elif test_nr == 2:
                reference = ["Oetjen", "10X"]
                query = ["Sun", "Freytag"]
            elif test_nr == 3:
                reference = ["Oetjen", "10X", "Sun"]
                query = ["Freytag"]
            elif test_nr == 4:
                reference = ["Oetjen", "10X", "Sun", "Freytag"]
                query = []
            elif test_nr == 10:
                reference = ["Oetjen", "10X", "Sun"]
                query = ["Freytag"]
        if experiment == "brain":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_brain_shrinked.h5ad"
                )
            )
            condition_key = "study"
            if test_nr == 1:
                reference = ["Rosenberg"]
                query = ["Saunders", "Zeisel", "Tabula_muris"]
            elif test_nr == 2:
                reference = ["Rosenberg", "Saunders"]
                query = ["Zeisel", "Tabula_muris"]
            elif test_nr == 3:
                reference = ["Rosenberg", "Saunders", "Zeisel"]
                query = ["Tabula_muris"]
            elif test_nr == 4:
                reference = ["Rosenberg", "Saunders", "Zeisel", "Tabula_muris"]
                query = []
            elif test_nr == 10:
                reference = ["Rosenberg", "Saunders"]
                query = ["Zeisel", "Tabula_muris"]
        if experiment == "scvelo":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_scvelo_shrinked.h5ad"
                )
            )
            condition_key = "study"
            if test_nr == 1:
                reference = ["12.5"]
                query = ["13.5", "14.5", "15.5"]
            elif test_nr == 2:
                reference = ["12.5", "13.5"]
                query = ["14.5", "15.5"]
            elif test_nr == 3:
                reference = ["12.5", "13.5", "14.5"]
                query = ["15.5"]
            elif test_nr == 4:
                reference = ["12.5", "13.5", "14.5", "15.5"]
                query = []
            elif test_nr == 10:
                reference = ["12.5", "13.5"]
                query = ["14.5", "15.5"]
        if experiment == "lung":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_lung_shrinked.h5ad"
                )
            )
            condition_key = "condition"
            if test_nr == 1:
                reference = ["Dropseq_transplant", "10x_Biopsy"]
                query = ["10x_Transplant"]
            elif test_nr == 10:
                reference = ["Dropseq_transplant", "10x_Biopsy"]
                query = ["10x_Transplant"]
        if experiment == "tumor":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_tumor_shrinked.h5ad"
                )
            )
            condition_key = "study"
            if test_nr == 10:
                reference = [
                    "breast",
                    "colorectal",
                    "liver2",
                    "liver1",
                    "lung1",
                    "lung2",
                    "multiple",
                    "ovary",
                    "pancreas",
                    "skin",
                ]
                query = ["melanoma1", "melanoma2", "uveal melanoma"]

        celltypes = adata.obs[cell_type_key].tolist()

        indices = np.arange(len(adata))
        labeled_ind = indices[adata.obs.study.isin(reference)].tolist()
        unlabeled_ind = np.delete(indices, labeled_ind).tolist()

        for index in unlabeled_ind:
            celltypes[index] = "Unknown"
        adata.obs["custom_ct"] = celltypes
        labeled_adata = adata[labeled_ind].copy()
        unlabeled_adata = adata[unlabeled_ind].copy()

        scvi.data.setup_anndata(adata, batch_key=condition_key, labels_key="custom_ct")
        scvi_model = scvi.model.SCVI(adata, n_layers=2)
        ref_time = time.time()
        scvi_model.train()
        scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, "Unknown")
        scanvi_model.train()
        ref_time = time.time() - ref_time
        text_file_t = open(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_runtime.txt"
            ),
            "w",
        )
        m = text_file_t.write(str(ref_time))
        text_file_t.close()
        dir_path_scan = os.path.expanduser(
            f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_model/"
        )
        scanvi_model.save(dir_path_scan, overwrite=True)

        # EVAL UNLABELED
        preds = scanvi_model.predict(unlabeled_adata)
        full_probs = scanvi_model.predict(unlabeled_adata, soft=True)
        probs = []
        for cell_prob in full_probs:
            probs.append(max(cell_prob))
        probs = np.array(probs)
        checks = np.array(len(unlabeled_adata) * ["incorrect"])
        checks[preds == unlabeled_adata.obs[cell_type_key]] = "correct"
        text_file_q = open(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_query_acc_report.txt"
            ),
            "w",
        )
        n = text_file_q.write(
            classification_report(
                y_true=unlabeled_adata.obs[cell_type_key],
                y_pred=preds,
                labels=np.array(unlabeled_adata.obs[cell_type_key].unique().tolist()),
            )
        )
        text_file_q.close()
        correct_probs = probs[preds == unlabeled_adata.obs[cell_type_key]]
        incorrect_probs = probs[preds != unlabeled_adata.obs[cell_type_key]]
        data = [correct_probs, incorrect_probs]
        fig, ax = plt.subplots()
        ax.set_title("Default violin plot")
        ax.set_ylabel("Observed values")
        ax.violinplot(data)
        labels = ["Correct", "Incorrect"]
        set_axis_style(ax, labels)
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_query_uncertainty.png"
            ),
            bbox_inches="tight",
        )

        adata_latent = sc.AnnData(
            scanvi_model.get_latent_representation(unlabeled_adata)
        )
        adata_latent.obs["celltype"] = unlabeled_adata.obs[cell_type_key].tolist()
        adata_latent.obs["batch"] = unlabeled_adata.obs[condition_key].tolist()
        adata_latent.obs["predictions"] = preds.tolist()
        adata_latent.obs["checking"] = checks.tolist()
        adata_latent.write_h5ad(
            filename=os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_query_adata.h5ad"
            )
        )
        sc.pp.neighbors(adata_latent, n_neighbors=8)
        sc.tl.leiden(adata_latent)
        sc.tl.umap(adata_latent)
        sc.pl.umap(adata_latent, color=["batch"], frameon=False, wspace=0.6, show=False)
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_query_umap_batch.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
        sc.pl.umap(
            adata_latent, color=["celltype"], frameon=False, wspace=0.6, show=False
        )
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_query_umap_ct.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
        sc.pl.umap(
            adata_latent, color=["predictions"], frameon=False, wspace=0.6, show=False
        )
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_query_umap_pred.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
        sc.pl.umap(
            adata_latent, color=["checking"], frameon=False, wspace=0.6, show=False
        )
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_query_umap_checks.png"
            ),
            bbox_inches="tight",
        )
        plt.close()

        # EVAL FULL
        preds = scanvi_model.predict()
        full_probs = scanvi_model.predict(soft=True)
        probs = []
        for cell_prob in full_probs:
            probs.append(max(cell_prob))
        probs = np.array(probs)
        text_file_f = open(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_acc_report.txt"
            ),
            "w",
        )
        n = text_file_f.write(
            classification_report(y_true=adata.obs[cell_type_key], y_pred=preds)
        )
        text_file_f.close()
        correct_probs = probs[preds == adata.obs[cell_type_key]]
        incorrect_probs = probs[preds != adata.obs[cell_type_key]]
        data = [correct_probs, incorrect_probs]
        fig, ax = plt.subplots()
        ax.set_title("Default violin plot")
        ax.set_ylabel("Observed values")
        ax.violinplot(data)
        labels = ["Correct", "Incorrect"]
        set_axis_style(ax, labels)
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_uncertainty.png"
            ),
            bbox_inches="tight",
        )

        checks = np.array(len(adata) * ["incorrect"])
        checks[preds == adata.obs[cell_type_key]] = "correct"
        adata_latent = sc.AnnData(scanvi_model.get_latent_representation())
        adata_latent.obs["celltype"] = adata.obs[cell_type_key].tolist()
        adata_latent.obs["batch"] = adata.obs[condition_key].tolist()
        adata_latent.obs["predictions"] = preds.tolist()
        adata_latent.obs["checking"] = checks.tolist()
        adata_latent.write_h5ad(
            filename=os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_adata.h5ad"
            )
        )
        sc.pp.neighbors(adata_latent, n_neighbors=8)
        sc.tl.leiden(adata_latent)
        sc.tl.umap(adata_latent)
        sc.pl.umap(adata_latent, color=["batch"], frameon=False, wspace=0.6, show=False)
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_umap_batch.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
        sc.pl.umap(
            adata_latent, color=["celltype"], frameon=False, wspace=0.6, show=False
        )
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_umap_ct.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
        sc.pl.umap(
            adata_latent, color=["predictions"], frameon=False, wspace=0.6, show=False
        )
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_umap_pred.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
        sc.pl.umap(
            adata_latent, color=["checking"], frameon=False, wspace=0.6, show=False
        )
        plt.savefig(
            os.path.expanduser(
                f"~/Documents/scanvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_umap_checks.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
