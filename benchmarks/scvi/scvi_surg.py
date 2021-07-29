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

arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    encode_covariates=True,
    dropout_rate=0.2,
    n_layers=2,
    deeply_inject_covariates=False
)


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


        adata_ref = adata[adata.obs.study.isin(reference)].copy()
        adata_query = adata[adata.obs.study.isin(query)].copy()

        scvi.data.setup_anndata(adata_ref, batch_key=condition_key)

        vae_ref = scvi.model.SCVI(adata_ref, **arches_params)
        ref_time = time.time()
        vae_ref.train()
        ref_time = time.time() - ref_time

        dir_path_scan = os.path.expanduser(f"~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_ref_model/")
        vae_ref.save(dir_path_scan, overwrite=True)
        text_file_r = open(
            os.path.expanduser(f"~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_ref_runtime.txt"),
            "w")
        m = text_file_r.write(str(ref_time))
        text_file_r.close()

        vae_q = scvi.model.SCVI.load_query_data(
            adata_query,
            dir_path_scan,
        )
        query_time = time.time()
        vae_q.train(
            max_epochs=200,
            plan_kwargs=dict(weight_decay=0.0),
        )
        query_time = time.time() - query_time
        dir_path_scan = os.path.expanduser(f"~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_model/")
        vae_q.save(dir_path_scan, overwrite=True)

        text_file_q = open(
            os.path.expanduser(f"~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_query_runtime.txt"),
            "w")
        m = text_file_q.write(str(query_time))
        text_file_q.close()

        adata_latent = sc.AnnData(vae_q.get_latent_representation())
        adata_latent.obs['celltype'] = adata_query.obs[cell_type_key].tolist()
        adata_latent.obs['batch'] = adata_query.obs[condition_key].tolist()
        adata_latent.write_h5ad(filename=os.path.expanduser(
            f'~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_query_adata.h5ad'))

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
                f'~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_query_umap_batch.png'),
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
                f'~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_query_umap_ct.png'),
            bbox_inches='tight')
        plt.close()

        adata_full = adata_query.concatenate(adata_ref)
        adata_full.obs.batch.cat.rename_categories(["Query", "Reference"], inplace=True)

        adata_latent = sc.AnnData(vae_q.get_latent_representation(adata_full))
        adata_latent.obs['celltype'] = adata_full.obs[cell_type_key].tolist()
        adata_latent.obs['batch'] = adata_full.obs[condition_key].tolist()
        adata_latent.write_h5ad(filename=os.path.expanduser(
            f'~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_full_adata.h5ad'))

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
                f'~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_full_umap_batch.png'),
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
                f'~/Documents/scvi_benchmarks/batchwise/surg/{experiment}/{test_nr}_full_umap_ct.png'),
            bbox_inches='tight')
        plt.close()