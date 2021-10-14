import logging
from sacred import Experiment
import seml

import time
import scanpy as sc
import numpy as np
import pandas as pd
import scarches as sca
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scarches.dataset.trvae.data_handling import remove_sparsity
from scIB.metrics import metrics

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
        f'lataq_reproduce/results/scanvi/{data}'
    )
    EXP_PARAMS = EXPERIMENT_INFO[data]
    FILE_NAME = EXP_PARAMS['file_name']

    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
        n_layers=2,
        deeply_inject_covariates=False
    )

    #LOADING DATA
    adata = sc.read(f'{DATA_DIR}/{FILE_NAME}')
    condition_key = EXP_PARAMS['condition_key']
    cell_type_key = EXP_PARAMS['cell_type_key']
    reference = EXP_PARAMS['reference']
    query = EXP_PARAMS['query']

    adata = remove_sparsity(adata)
    source_adata = adata[adata.obs.study.isin(reference)].copy()
    target_adata = adata[adata.obs.study.isin(query)].copy()
    logging.info('Data loaded succesfully')

    sca.dataset.setup_anndata(source_adata, batch_key=condition_key, labels_key=cell_type_key[0])

    #TRAINING REFERENCE MODEL
    vae_ref = sca.models.SCVI(source_adata)
    ref_time = time.time()
    vae_ref.train()
    vae_ref_scan = sca.models.SCANVI.from_scvi_model(
        vae_ref,
        unlabeled_category="Unknown",
    )
    vae_ref_scan.train(max_epochs=20)
    ref_time = time.time() - ref_time
    vae_ref_scan.save(f'{RES_PATH}/scanvi_model', overwrite=True)
    #save ref time

    #TRAINING QUERY MODEL
    vae_q = sca.models.SCANVI.load_query_data(
        target_adata,
        f'{RES_PATH}/scanvi_model',
    )
    vae_q._unlabeled_indices = np.arange(target_adata.n_obs)
    vae_q._labeled_indices = []
    query_time = time.time()
    vae_q.train(
        max_epochs=100,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10,
    )
    query_time = time.time() - query_time
    vae_q.save(f'{RES_PATH}/scanvi_query_model', overwrite=True)

    # EVAL UNLABELED
    preds = vae_q.predict()
    full_probs = vae_q.predict(soft=True)
    probs = full_probs.max(axis=1)
    probs = np.array(probs)
    checks = np.array(len(target_adata) * ['incorrect'])
    checks[preds == target_adata.obs[cell_type_key[0]]] = 'correct'
    
    report = pd.DataFrame(
        classification_report(
            y_true=target_adata.obs[cell_type_key[0]],
            y_pred=preds,
            labels=np.array(target_adata.obs[cell_type_key[0]].unique().tolist()),
            output_dict=True,
        )
    ).transpose()

    correct_probs = probs[preds == target_adata.obs[cell_type_key[0]]]
    incorrect_probs = probs[preds != target_adata.obs[cell_type_key[0]]]
    data = [correct_probs, incorrect_probs]

    #UNCERTAINTY FIGURE
    fig, ax = plt.subplots()
    ax.set_title('Default violin plot')
    ax.set_ylabel('Observed values')
    ax.violinplot(data)
    labels = ['Correct', 'Incorrect']
    plt.savefig(
        f'{RES_PATH}/query_uncertainty.png',
        bbox_inches='tight'
    )

    adata_latent = sc.AnnData(vae_q.get_latent_representation())
    adata_latent.obs['celltype'] = target_adata.obs[cell_type_key[0]].tolist()
    adata_latent.obs['batch'] = target_adata.obs[condition_key].tolist()
    adata_latent.obs['predictions'] = preds.tolist()
    adata_latent.obs['checking'] = checks.tolist()
    adata_latent.write_h5ad(
        f'{RES_PATH}/adata_latent_query.h5ad'
    )

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
        f'{RES_PATH}/query_umap_batch.png',
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
        f'{RES_PATH}_query_umap_ct.png',
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
        f'{RES_PATH}/query_umap_pred.png',
        bbox_inches='tight'
    )
    plt.close()
    sc.pl.umap(
        adata_latent,
        color=['checking'],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/query_umap_checks.png',
        bbox_inches='tight'
    )
    plt.close()

    # EVAL FULL
    adata_full = target_adata.concatenate(source_adata, batch_key='query')
    adata_full.obs['query'] = adata_full.obs['query'].astype('category')
    adata_full.obs['query'].cat.rename_categories(
        ["Query", "Reference"], 
        inplace=True
    )
    preds = vae_q.predict(adata_full)
    full_probs = vae_q.predict(adata_full, soft=True)
    probs = full_probs.max(axis=1)
    probs = np.array(probs)
    checks = np.array(len(adata_full) * ['incorrect'])
    checks[preds == adata_full.obs[cell_type_key[0]]] = 'correct'

    report_full = pd.DataFrame(
        classification_report(
            y_true=adata_full.obs[cell_type_key[0]],
            y_pred=preds,
            output_dict=True
        )
    ).transpose()

    correct_probs = probs[preds == adata_full.obs[cell_type_key[0]]]
    incorrect_probs = probs[preds != adata_full.obs[cell_type_key[0]]]
    data = [correct_probs, incorrect_probs]

    fig, ax = plt.subplots()
    ax.set_title('Default violin plot')
    ax.set_ylabel('Observed values')
    ax.violinplot(data)
    labels = ['Correct', 'Incorrect']
    plt.savefig(
        f'{RES_PATH}/full_uncertainty.png',
        bbox_inches='tight'
    )

    adata_latent = sc.AnnData(vae_q.get_latent_representation(adata_full))
    adata_latent.obs['celltype'] = adata_full.obs[cell_type_key[0]].tolist()
    adata_latent.obs['batch'] = adata_full.obs[condition_key].tolist()
    adata_latent.obs['predictions'] = preds.tolist()
    adata_latent.obs['checking'] = checks.tolist()
    adata_latent.obs['query'] = adata_full.obs['query'].tolist()
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
        color=['query'],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/full_umap_query.png',
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
    sc.pl.umap(
        adata_latent,
        color=['checking'],
        frameon=False,
        wspace=0.6,
        show=False
    )
    plt.savefig(
        f'{RES_PATH}/full_umap_checks.png',
        bbox_inches='tight'
    )
    plt.close()

    conditions, _ = label_encoder(adata, condition_key=condition_key)
    labels, _ = label_encoder(adata, condition_key=cell_type_key[0])
    adata.obs['batch'] = conditions.squeeze(axis=1)
    adata.obs['celltype'] = labels.squeeze(axis=1)
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    conditions, _ = label_encoder(adata_latent, condition_key='batch')
    labels, _ = label_encoder(adata_latent, condition_key='celltype')
    adata_latent.obs['batch'] = conditions.squeeze(axis=1)
    adata_latent.obs['celltype'] = labels.squeeze(axis=1)
    adata_latent.obs['batch'] = adata_latent.obs['batch'].astype('category')
    adata_latent.obs['celltype'] = adata_latent.obs['celltype'].astype('category')
    sc.pp.pca(adata)
    sc.pp.pca(adata_latent)

    adata_latent.write(f'{RES_PATH}/adata_latent.h5ad')
    adata.write(f'{RES_PATH}/adata_original.h5ad')
    scores = metrics(
        adata, 
        adata_latent, 
        'batch', 
        'celltype',
        isolated_labels_asw_=True,
        silhouette_=True,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1=True,
        nmi_=True,
        ari_=True
    )
    
    scores = scores.T
    scores = scores[[
        'NMI_cluster/label', 
        'ARI_cluster/label',
        'ASW_label',
        'ASW_label/batch',
        'PCR_batch', 
        'isolated_label_F1',
        'isolated_label_silhouette',
        'graph_conn',
    ]]

    results = {
        'reference_time': ref_time,
        'query_time': query_time,
        'classification_report': report_full,
        'classification_report_query': report,
        'integration_scores': scores
    }
    return results