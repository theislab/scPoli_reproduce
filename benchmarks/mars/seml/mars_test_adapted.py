# Maybe needs some tweaks. I ran it directly in their repo, since it was not installable. Also may update the datasets to match the other scripts

import logging
from sacred import Experiment
import seml

#MARS imports
import torch
import os
from anndata import AnnData
from benchmarks.mars.args_parser import get_parser
from benchmarks.mars.model.mars import MARS
from benchmarks.mars.model.experiment_dataset import ExperimentDataset
import warnings
warnings.filterwarnings('ignore')

import time
import scanpy as sc
import numpy as np
import pandas as pd
import scarches as sca
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scarches.dataset.trvae.data_handling import remove_sparsity
from scIB.metrics import metrics_fast

from lataq_reproduce.exp_dict import EXPERIMENT_INFO
from lataq_reproduce.utils import label_encoder
#from lataq.metrics.metrics import metrics

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite=None
    db_collection=None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(
        data: str,
        overwrite: int,
):
    logging.info('Received the following configuration:')
    logging.info(
        f'Dataset: {data}'
    )

    DATA_DIR = '/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/data'
    RES_PATH = (
        f'/storage/groups/ml01/workspace/carlo.dedonno/'
        f'lataq_reproduce/results/mars/{data}'
    )
    EXP_PARAMS = EXPERIMENT_INFO[data]
    FILE_NAME = EXP_PARAMS['file_name']

    def celltype_to_numeric(adata, obs_key):
        """Adds ground truth clusters data."""
        annotations = list(adata.obs[obs_key])
        annotations_set = sorted(set(annotations))

        mapping = {a: idx for idx, a in enumerate(annotations_set)}

        truth_labels = [mapping[a] for a in annotations]
        adata.obs['truth_labels'] = pd.Categorical(values=truth_labels)

        return adata, mapping

    params, unknown = get_parser().parse_known_args()
    params.cuda = True
    params.pretrain_batch = 128
    print('PARAMS:', params)
    if torch.cuda.is_available() and not params.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    params.device = device

    # LOADING DATA
    adata = sc.read(f'{DATA_DIR}/{FILE_NAME}')
    condition_key = EXP_PARAMS['condition_key']
    cell_type_key = EXP_PARAMS['cell_type_key']
    reference = EXP_PARAMS['reference']
    query = EXP_PARAMS['query']
    adata = remove_sparsity(adata)
    #adata_old = adata.copy()
    # Create Int Mapping for celltypes
    adata, celltype_id_map = celltype_to_numeric(adata, cell_type_key[0])
    cell_type_name_map = {v: k for k, v in celltype_id_map.items()}
    
    # Preprocess data
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10, zero_center=True)

    # Make labeled Datasets for Mars
    annotated = []
    labels = []
    batches = []
    labeled_adatas = []
    for batch in reference:
        labeled_adata = adata[adata.obs[condition_key].isin([batch])].copy()
        labeled_adatas.append(labeled_adata)
        y_labeled = np.array(labeled_adata.obs['truth_labels'], dtype=np.int64)
        annotated.append(ExperimentDataset(labeled_adata.X,
                                           labeled_adata.obs_names,
                                           labeled_adata.var_names,
                                           batch,
                                           y_labeled
                                           ))
        labels += labeled_adata.obs[cell_type_key[0]].tolist()
        batches += labeled_adata.obs[condition_key].tolist()
    labeled_adata_full = labeled_adatas[0].concatenate(labeled_adatas[1:])
    
    # Make Unlabeled Datasets for Mars

    unlabeled_adata = adata[adata.obs[condition_key].isin(query)].copy()
    y_unlabeled = np.array(unlabeled_adata.obs['truth_labels'], dtype=np.int64)
    unannotated = ExperimentDataset(
        unlabeled_adata.X,
        unlabeled_adata.obs_names,
        unlabeled_adata.var_names,
        'query',
        y_unlabeled
    )
    labels += unlabeled_adata.obs[cell_type_key[0]].tolist()
    batches += unlabeled_adata.obs[condition_key].tolist()
    n_clusters = len(np.unique(unannotated.y))
    adata_full = labeled_adata_full.concatenate(unlabeled_adata)
    # Make pretrain Dataset
    pretrain = ExperimentDataset(
        adata.X,
        adata.obs_names,
        adata.var_names,
        'Pretrain'
    )
    logging.info('Data loaded succesfully')

    # TRAINING REFERENCE MODEL
    mars = MARS(
        n_clusters,
        params,
        annotated,
        unannotated,
        pretrain,
        hid_dim_1=1000,
        hid_dim_2=100
    )
    ref_time = time.time()
    adata, landmarks, _ = mars.train(evaluation_mode=True)
    ref_time = time.time() - ref_time
    # save ref time
    adata.obs[condition_key] = adata_full.obs[condition_key].values
    adata.obs[cell_type_key[0]] = adata_full.obs[cell_type_key[0]].values

    # TODO: CHECK FROM HERE....
    names = mars.name_cell_types(adata, landmarks, cell_type_name_map)
    print(names)
    unproc_labels = adata.obs['truth_labels'].tolist()
    unproc_pred = adata.obs['MARS_labels'].tolist()

    predictions = []
    for count, label in enumerate(unproc_pred):
        if not isinstance(label, int):
            predictions.append(cell_type_name_map[unproc_labels[count]])
        elif len(names[label]) == 1:
            predictions.append(names[label][-1])
        else:
            predictions.append(names[label][-1][0])

    labels_after = []
    for count, label in enumerate(unproc_labels):
        labels_after.append(cell_type_name_map[label])

    report = pd.DataFrame(
        classification_report(
            y_true=np.array(labels_after)[adata.obs['experiment'] == 'query'],
            y_pred=np.array(predictions)[adata.obs['experiment'] == 'query'],
            labels=np.array(unlabeled_adata.obs[cell_type_key[0]].unique().tolist()),
            output_dict=True,
        )
    ).transpose()

    report_full = pd.DataFrame(
        classification_report(
            y_true=np.array(labels_after),
            y_pred=np.array(predictions),
            output_dict=True
        )
    ).transpose().add_prefix('full_')

    adata_latent = AnnData(adata.obsm['MARS_embedding'])
    adata_latent.obs['celltype'] = labels_after
    adata_latent.obs['predictions'] = predictions
    adata_latent.obs['batch'] = batches
    adata_latent.write_h5ad(f'{RES_PATH}/adata_latent_full.h5ad')

    sc.pp.neighbors(adata_latent)
    sc.tl.leiden(adata_latent)
    sc.tl.umap(adata_latent)
    sc.pl.umap(
        adata_latent,
        color=['batch'],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/full_umap_batch.png',
        bbox_inches='tight'
    )
    plt.close()
    sc.pl.umap(
        adata_latent,
        color=['celltype'],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/full_umap_ct.png',
        bbox_inches='tight'
    )
    plt.close()
    sc.pl.umap(
        adata_latent,
        color=['predictions'],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/full_umap_pred.png',
        bbox_inches='tight'
    )
    plt.close()

    conditions, _ = label_encoder(adata, condition_key=condition_key)
    labels, _ = label_encoder(adata, condition_key=cell_type_key[0])
    adata.obs['batch'] = conditions.squeeze(axis=1)
    adata.obs['celltype'] = labels.squeeze(axis=1)
    conditions, _ = label_encoder(adata_latent, condition_key='batch')
    labels, _ = label_encoder(adata_latent, condition_key='celltype')
    adata_latent.obs['batch'] = conditions.squeeze(axis=1)
    adata_latent.obs['celltype'] = labels.squeeze(axis=1)

    scores = metrics_fast(
        adata,
        adata_latent,
        'batch',
        'celltype',
    )

    scores = scores.T
    # scores = scores[[  # 'NMI_cluster/label',
    #     # 'ARI_cluster/label',
    #     # 'ASW_label',
    #     # 'ASW_label/batch',
    #     'PCR_batch',
    #     # 'isolated_label_F1',
    #     # 'isolated_label_silhouette',
    #     'graph_conn',
    #     'ebm',
    #     'knn',
    # ]]

    results = {
        'reference_time': ref_time,
        'classification_report': report_full,
        'classification_report_query': report,
        'integration_scores': scores
    }
    return results