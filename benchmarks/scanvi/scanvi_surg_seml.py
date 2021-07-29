import logging
from sacred import Experiment
import seml
import scanpy as sc
import numpy as np
import pandas as pd
import scvi
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scarches.dataset.trvae.data_handling import remove_sparsity
from lataq_reproduce.exp_dict import EXPERIMENT_INFO
from lataq.metrics.metrics import metrics
import time

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

    DATA_DIR = '/storage/groups/ml01/workspace/carlo.dedonno/LATAQ/data'
    RES_PATH = (
        f'/storage/groups/ml01/workspace/carlo.dedonno/'
        f'lataq_reproduce/results/scanvi/{data}'
    )
    EXP_PARAMS = EXPERIMENT_INFO[data]
    FILE_NAME = EXP_PARAMS['file_name']

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

    scvi.data.setup_anndata(source_adata, batch_key=condition_key)

    #TRAINING REFERENCE MODEL
    vae_ref = scvi.model.SCVI(source_adata, **arches_params)
    ref_time = time.time()
    vae_ref.train()
    vae_ref_scan = scvi.model.SCANVI.from_scvi_model(
        vae_ref,
        unlabeled_category="Unknown",
    )
    vae_ref_scan.train(max_epochs=20)
    ref_time = time.time() - ref_time
    vae_ref_scan.save(f'{RES_PATH}/scanvi_model', overwrite=True)
    #save ref time

    #TRAINING QUERY MODEL
    vae_q = scvi.model.SCANVI.load_query_data(
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
    probs = []
    for cell_prob in full_probs:
        probs.append(max(cell_prob))
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
    set_axis_style(ax, labels)
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
    adata_full = target_adata.concatenate(source_adata)
    adata_full.obs[condition_key].cat.rename_categories(["Query", "Reference"], inplace=True)
    preds = vae_q.predict(adata_full)
    full_probs = vae_q.predict(adata_full, soft=True)
    probs = []
    for cell_prob in full_probs:
        probs.append(max(cell_prob))
    probs = np.array(probs)
    checks = np.array(len(adata_full) * ['incorrect'])
    checks[preds == adata_full.obs[cell_type_key[0]]] = 'correct'

    report_full = pd.DataFrame(
        classification_report(
            y_true=adata_full.obs[cell_type_key[0]],
            y_pred=preds,
        )
    ).transpose().add_prefix('full_')

    correct_probs = probs[preds == adata_full.obs[cell_type_key[0]]]
    incorrect_probs = probs[preds != adata_full.obs[cell_type_key[0]]]
    data = [correct_probs, incorrect_probs]
    fig, ax = plt.subplots()
    ax.set_title('Default violin plot')
    ax.set_ylabel('Observed values')
    ax.violinplot(data)
    labels = ['Correct', 'Incorrect']
    set_axis_style(ax, labels)
    plt.savefig(
        f'{RES_PATH}/full_uncertainty.png',
        bbox_inches='tight'
    )

    adata_latent = sc.AnnData(vae_q.get_latent_representation(adata_full))
    adata_latent.obs['celltype'] = adata_full.obs[cell_type_key[0]].tolist()
    adata_latent.obs['batch'] = adata_full.obs[condition_key].tolist()
    adata_latent.obs['predictions'] = preds.tolist()
    adata_latent.obs['checking'] = checks.tolist()
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

    conditions, _ = label_encoder(adata_latent, condition_key='batch')
    labels, _ = label_encoder(adata_latent, condition_key='celltype')
    adata_latent.obs['batch'] = conditions.squeeze(axis=1)
    adata_latent.obs['celltype'] = labels.squeeze(axis=1)

    scores = metrics(
        adata, 
        adata_latent, 
        'batch', 
        'celltype',
        nmi_=True,
        ari_=False,
        silhouette_=False,
        pcr_=True,
        graph_conn_=True,
        isolated_labels_=True,
        hvg_score_=False,
        knn_=True,
        ebm_=True,
    )
    
    scores = scores.T
    scores = scores[['NMI_cluster/label', 
                     #'ARI_cluster/label',
                     #'ASW_label',
                     #'ASW_label/batch',
                     'PCR_batch', 
                     'isolated_label_F1',
                     'isolated_label_silhouette',
                     'graph_conn',
                     'ebm',
                     'knn',
                    ]]

    results = {
        'reference_time': ref_time,
        'query_time': query_time,
        'classification_report': report,
        'classification_report_query': report_full,
        'integration_scores': scores
    }
    return results