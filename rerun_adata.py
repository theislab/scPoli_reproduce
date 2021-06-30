import numpy as np
import scanpy as sc
import torch
import os
import scvi
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


# Experiment Params
experiments = ["scvelo"]
test_nrs = [1,2,3,4]

for experiment in experiments:
    for test_nr in test_nrs:
        if experiment != "pancreas" and test_nr == 4:
            continue
        if experiment == "lung" and test_nr != 1:
            continue
        if experiment == "pancreas":
            adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
            adata = adata_all.raw.to_adata()
            condition_key = "study"
            cell_type_key = "cell_type"
            if test_nr == 1:
                reference = ['Pancreas inDrop']
                query = ['Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq', 'Pancreas Fluidigm C1']
            elif test_nr == 2:
                reference = ['Pancreas inDrop', 'Pancreas SS2']
                query = ['Pancreas CelSeq2', 'Pancreas CelSeq', 'Pancreas Fluidigm C1']
            elif test_nr == 3:
                reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2']
                query = ['Pancreas CelSeq', 'Pancreas Fluidigm C1']
            elif test_nr == 4:
                reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq']
                query = ['Pancreas Fluidigm C1']
            elif test_nr == 5:
                reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq', 'Pancreas Fluidigm C1']
                query = []
        if experiment == "pbmc":
            adata_all = sc.read(os.path.expanduser(
                f'~/Documents/benchmarking_datasets/Immune_ALL_human_wo_villani_normalized_hvg.h5ad'))
            adata = adata_all.raw.to_adata()
            condition_key = 'condition'
            cell_type_key = 'final_annotation'
            if test_nr == 1:
                reference = ['Oetjen']
                query = ['10X', 'Sun', 'Freytag']
            elif test_nr == 2:
                reference = ['Oetjen', '10X']
                query = ['Sun', 'Freytag']
            elif test_nr == 3:
                reference = ['Oetjen', '10X', 'Sun']
                query = ['Freytag']
            elif test_nr == 4:
                reference = ['Oetjen', '10X', 'Sun', 'Freytag']
                query = []
        if experiment == "brain":
            adata_all = sc.read(
                os.path.expanduser(f'~/Documents/benchmarking_datasets/mouse_brain_subsampled_normalized_hvg.h5ad'))
            adata = adata_all.raw.to_adata()
            condition_key = "study"
            cell_type_key = "cell_type"
            if test_nr == 1:
                reference = ['Rosenberg']
                query = ['Saunders', 'Zeisel', 'Tabula_muris']
            elif test_nr == 2:
                reference = ['Rosenberg', 'Saunders']
                query = ['Zeisel', 'Tabula_muris']
            elif test_nr == 3:
                reference = ['Rosenberg', 'Saunders', 'Zeisel']
                query = ['Tabula_muris']
            elif test_nr == 4:
                reference = ['Rosenberg', 'Saunders', 'Zeisel', 'Tabula_muris']
                query = []
        if experiment == "scvelo":
            adata = sc.read(
                os.path.expanduser(f'~/Documents/benchmarking_datasets/endocrinogenesis_hvg.h5ad'))
            condition_key = "study"
            cell_type_key = "cell_type"
            if test_nr == 1:
                reference = ['12.5']
                query = ['13.5', '14.5', '15.5']
            elif test_nr == 2:
                reference = ['12.5', '13.5']
                query = ['14.5', '15.5']
            elif test_nr == 3:
                reference = ['12.5', '13.5', '14.5']
                query = ['15.5']
            elif test_nr == 4:
                reference = ['12.5', '13.5', '14.5', '15.5']
                query = []
        if experiment == "lung":
            adata = sc.read(
                os.path.expanduser(f'~/Documents/benchmarking_datasets/Lung_atlas_public_hvg.h5ad'))
            condition_key = "donor"
            cell_type_key = "cell_type"
            if test_nr == 1:
                reference = ['Dropseq_transplant', '10x_Biopsy']
                query = ['10x_Transplant']

        source_adata = adata[adata.obs.study.isin(reference)].copy()
        target_adata = adata[adata.obs.study.isin(query)].copy()
        scanvi = scvi.model.SCANVI.load(
            dir_path=os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_surg_model'),
            adata=adata,
        )
        preds = scanvi.predict()
        checks = np.array(len(adata) * ['incorrect'])
        checks[preds == adata.obs[cell_type_key]] = 'correct'
        data_latent = scanvi.get_latent_representation()
        adata_latent = sc.AnnData(data_latent)
        adata_latent.obs['celltype'] = adata.obs[cell_type_key].tolist()
        adata_latent.obs['batch'] = adata.obs[condition_key].tolist()
        adata_latent.obs['predictions'] = preds.tolist()
        adata_latent.obs['checking'] = checks.tolist()
        adata_latent.write_h5ad(filename=os.path.expanduser(
            f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_full_adata.h5ad'))
        sc.pp.neighbors(adata_latent, n_neighbors=8)
        sc.tl.leiden(adata_latent)
        sc.tl.umap(adata_latent)
        sc.pl.umap(adata_latent,
                   color=['batch'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_basic_full_umap_batch.png'),
            bbox_inches='tight')
        plt.close()
        sc.pl.umap(adata_latent,
                   color=['celltype'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_basic_full_umap_ct.png'),
            bbox_inches='tight')
        plt.close()
        sc.pl.umap(adata_latent,
                   color=['predictions'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_basic_full_umap_pred.png'),
            bbox_inches='tight')
        plt.close()
        sc.pl.umap(adata_latent,
                   color=['checking'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_basic_full_umap_checks.png'),
            bbox_inches='tight')
        plt.close()

        scanvi = scvi.model.SCANVI.load(
            dir_path=os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_surg_model'),
            adata=target_adata,
        )
        preds = scanvi.predict()
        checks = np.array(len(target_adata)*['incorrect'])
        checks[preds == target_adata.obs[cell_type_key]] = 'correct'

        text_file_q = open(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_surg_acc_report_updated.txt'),
            "w")
        n = text_file_q.write(classification_report(
            y_true=target_adata.obs[cell_type_key],
            y_pred=preds,
            labels=np.array(target_adata.obs[cell_type_key].unique().tolist())
        ))
        text_file_q.close()
        data_latent = scanvi.get_latent_representation()
        adata_latent = sc.AnnData(data_latent)
        adata_latent.obs['celltype'] = target_adata.obs[cell_type_key].tolist()
        adata_latent.obs['batch'] = target_adata.obs[condition_key].tolist()
        adata_latent.obs['predictions'] = preds.tolist()
        adata_latent.obs['checking'] = checks.tolist()
        adata_latent.write_h5ad(filename=os.path.expanduser(
            f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_unlabeled_adata.h5ad'))
        sc.pp.neighbors(adata_latent, n_neighbors=8)
        sc.tl.leiden(adata_latent)
        sc.tl.umap(adata_latent)
        sc.pl.umap(adata_latent,
                   color=['batch'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_basic_query_umap_batch.png'),
            bbox_inches='tight')
        plt.close()
        sc.pl.umap(adata_latent,
                   color=['celltype'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_basic_query_umap_ct.png'),
            bbox_inches='tight')
        plt.close()
        sc.pl.umap(adata_latent,
                   color=['predictions'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_basic_query_umap_pred.png'),
            bbox_inches='tight')
        plt.close()
        sc.pl.umap(adata_latent,
                   color=['checking'],
                   frameon=False,
                   wspace=0.6,
                   show=False
                   )
        plt.savefig(
            os.path.expanduser(
                f'~/Documents/scanvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_basic_query_umap_checks.png'),
            bbox_inches='tight')
        plt.close()