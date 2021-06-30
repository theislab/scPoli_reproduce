import scvi
import numpy as np
import scanpy as sc
import torch
import os
import matplotlib.pyplot as plt
import time

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


# Experiment Params
#experiments = ["pancreas","pbmc","lung","scvelo","brain"]
experiments = ["pbmc", "tumor"]
test_nrs = [10]
cell_type_key = "cell_type"
for experiment in experiments:
    for test_nr in test_nrs:
        if experiment == "pancreas":
            adata = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_pancreas_shrinked.h5ad'))
            condition_key = "study"
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
                reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq',
                             'Pancreas Fluidigm C1']
                query = []
            elif test_nr == 10:
                reference = ["inDrop1", "inDrop2", "inDrop3", "inDrop4", "fluidigmc1", "smartseq2", "smarter"]
                query = ["celseq", "celseq2"]
        if experiment == "pbmc":
            adata = sc.read(os.path.expanduser(
                f'~/Documents/benchmarking_datasets/benchmark_pbmc_shrinked.h5ad'))
            condition_key = 'condition'
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
            elif test_nr == 10:
                reference = ['Oetjen', '10X', 'Sun']
                query = ['Freytag']
        if experiment == "brain":
            adata = sc.read(
                os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_brain_shrinked.h5ad'))
            condition_key = "study"
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
            elif test_nr == 10:
                reference = ['Rosenberg', 'Saunders']
                query = ['Zeisel', 'Tabula_muris']
        if experiment == "scvelo":
            adata = sc.read(
                os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_scvelo_shrinked.h5ad'))
            condition_key = "study"
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
            elif test_nr == 10:
                reference = ['12.5', '13.5']
                query = ['14.5', '15.5']
        if experiment == "lung":
            adata = sc.read(
                os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_lung_shrinked.h5ad'))
            condition_key = "condition"
            if test_nr == 1:
                reference = ['Dropseq_transplant', '10x_Biopsy']
                query = ['10x_Transplant']
            elif test_nr == 10:
                reference = ['Dropseq_transplant', '10x_Biopsy']
                query = ['10x_Transplant']
        if experiment == "tumor":
            adata = sc.read(
                os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_tumor_shrinked.h5ad'))
            condition_key = "study"
            if test_nr == 10:
                reference = ['breast', 'colorectal', 'liver2', 'liver1', 'lung1', 'lung2', 'multiple', 'ovary',
                             'pancreas', 'skin']
                query = ['melanoma1', 'melanoma2', 'uveal melanoma']


        scvi.data.setup_anndata(adata, batch_key=condition_key)
        scvi_model = scvi.model.SCVI(adata, n_layers=2)
        ref_time = time.time()
        scvi_model.train()
        ref_time = time.time() - ref_time

        dir_path_scan = os.path.expanduser(
            f"~/Documents/scvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_model/")
        scvi_model.save(dir_path_scan, overwrite=True)
        text_file_t = open(
            os.path.expanduser(f"~/Documents/scvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_runtime.txt"), "w")
        m = text_file_t.write(str(ref_time))
        text_file_t.close()

        adata_latent = sc.AnnData(scvi_model.get_latent_representation(adata))
        adata_latent.obs['celltype'] = adata.obs[cell_type_key].tolist()
        adata_latent.obs['batch'] = adata.obs[condition_key].tolist()
        adata_latent.write_h5ad(filename=os.path.expanduser(
            f'~/Documents/scvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_adata.h5ad'))

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
                f'~/Documents/scvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_umap_batch.png'),
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
                f'~/Documents/scvi_benchmarks/batchwise/semi/{experiment}/{test_nr}_full_umap_ct.png'),
            bbox_inches='tight')
        plt.close()
