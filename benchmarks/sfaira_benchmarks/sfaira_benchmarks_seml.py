import scanpy as sc
import anndata
import seml
import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from scarches.dataset.trvae.data_handling import remove_sparsity
from sacred import Experiment
from scib.metrics import metrics
from lataq_reproduce.utils import label_encoder


ex = Experiment()
seml.setup_logger(ex)

SFAIRA_DATA_PATH = '/storage/groups/ml01/workspace/david.fischer/sfairazero/data/store/h5ad'
DATA_DIR = '/storage/groups/ml01/workspace/felix.fischer.2/lataq/data'


DATA_SET_SPLITS = {
    'bonemarrow': {
        'query': ['homosapiens_bonemarrow_2019_10x3transcriptionprofiling_szabo_009_10.1038/s41467-019-12464-3'],
        'reference': [
            'homosapiens_bonemarrow_2019_10x3transcriptionprofiling_szabo_003_10.1038/s41467-019-12464-3',
            'homosapiens_bonemarrow_2019_10x3transcriptionprofiling_szabo_010_10.1038/s41467-019-12464-3',
            'homosapiens_bonemarrow_2019_10x3transcriptionprofiling_szabo_004_10.1038/s41467-019-12464-3'
        ]
    },
    'brain': {
        'query': ['homosapiens_brain_2017_droncseq_habib_001_10.1038/nmeth.4407'],
        'reference': [
            'homosapiens_brain_2019_dropseq_polioudakis_001_10.1016/j.neuron.2019.06.011',
            'homosapiens_brain_2019_10x3v2_kanton_001_10.1038/s41586-019-1654-9'
        ]
    },
    'ileum': {
        'query': ['homosapiens_ileum_2019_10x3transcriptionprofiling_wang_002_10.1084/jem.20191130'],
        'reference': ['homosapiens_ileum_2019_10x3v2_martin_001_10.1016/j.cell.2019.08.008']
    },
    'kidney': {
        'query': [
            'homosapiens_kidney_2019_droncseq_lake_001_10.1038/s41467-019-10861-2',
            'homosapiens_kidney_2020_10x3v2_liao_001_10.1038/s41597-019-0351-8'
        ],
        'reference': ['homosapiens_kidney_2019_10x3v2_stewart_001_10.1126/science.aat5031']
    },
    'liver': {
        'query': ['homosapiens_liver_2019_celseq2_aizarani_001_10.1038/s41586-019-1373-2'],
        'reference': [
            'homosapiens_liver_2019_10x3v2_ramachandran_001_10.1038/s41586-019-1631-3',
            'homosapiens_liver_2019_10x3v2_popescu_001_10.1038/s41586-019-1652-y'
        ]
    },
    'lung': {
        'query': [
            'homosapiens_lung_2019_10x3transcriptionprofiling_szabo_002_10.1038/s41467-019-12464-3',
            'homosapiens_lung_2019_10x3transcriptionprofiling_szabo_001_10.1038/s41467-019-12464-3',
            'homosapiens_lung_2019_dropseq_braga_001_10.1038/s41591-019-0468-5'
        ],
        'reference': [
            'homosapiens_lung_2019_10x3transcriptionprofiling_szabo_007_10.1038/s41467-019-12464-3',
            'homosapiens_lung_2019_10x3transcriptionprofiling_szabo_008_10.1038/s41467-019-12464-3',
            'homosapiens_lung_2020_smartseq2_travaglini_002_10.1038/s41586-020-2922-4',
            'homosapiens_lung_2020_10x3v2_lukassen_002_10.15252/embj.20105114',
            'homosapiens_lung_2020_10x3v2_lukassen_001_10.15252/embj.20105114',
            'homosapiens_lung_2020_10x3v2_miller_001_10.1016/j.devcel.2020.01.033',
            'homosapiens_lung_2020_10x3v2_travaglini_001_10.1038/s41586-020-2922-4'
        ]
    },
    'lungparenchyma': {
        'query': ['homosapiens_lungparenchyma_2019_10x3transcriptionprofiling_braga_001_10.1038/s41591-019-0468-5'],
        'reference': [
            'homosapiens_lungparenchyma_2019_10x3v2_madissoon_001_10.1186/s13059-019-1906-x',
            'homosapiens_lungparenchyma_2020_None_habermann_001_10.1126/sciadv.aba1972'
        ]
    },
    'lymphnode': {
        'query': ['homosapiens_lymphnode_2019_10x3transcriptionprofiling_szabo_011_10.1038/s41467-019-12464-3'],
        'reference': [
            'homosapiens_lymphnode_2019_10x3transcriptionprofiling_szabo_006_10.1038/s41467-019-12464-3',
            'homosapiens_lymphnode_2019_10x3transcriptionprofiling_szabo_005_10.1038/s41467-019-12464-3',
            'homosapiens_lymphnode_2019_10x3transcriptionprofiling_szabo_012_10.1038/s41467-019-12464-3'
        ]
    },
    'pancreas': {
        'query': ['homosapiens_pancreas_2016_smartseq2_segerstolpe_001_10.1016/j.cmet.2016.08.020'],
        'reference': ['homosapiens_pancreas_2016_indrop_baron_001_10.1016/j.cels.2016.08.011']
    },
    'placenta': {
        'query': ['homosapiens_placenta_2018_10x3v2_ventotormo_001_10.1038/s41586-018-0698-6'],
        'reference': ['homosapiens_placenta_2018_smartseq2_ventotormo_002_10.1038/s41586-018-0698-6']
    },
    'retina': {
        'query': [
            'homosapiens_retina_2019_10x3v3_voigt_001_10.1073/pnas.1914143116',
            'homosapiens_retina_2019_10x3v3_menon_001_10.1038/s41467-019-12780-8'
        ],
        'reference': ['homosapiens_retina_2019_10x3v2_lukowski_001_10.15252/embj.2018100811']
    }
}


@ex.pre_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


def create_anndata_objects(tissue: str) -> (anndata.AnnData, anndata.AnnData):
    """
    Create anndata.AnnData objects for reference and query data set.
    """
    import sfaira
    # create reference data set
    data_store_reference = sfaira.data.load_store(SFAIRA_DATA_PATH, store_format='h5ad')
    data_store_reference = data_store_reference.stores['Homo sapiens']
    data_store_reference.subset(attr_key='id', values=DATA_SET_SPLITS[tissue]['reference'])
    data_store_reference.subset(attr_key='cell_type', excluded_values=['unknown'])
    cell_types_reference = data_store_reference.obs.cell_type.unique().tolist()
    # create query data set
    data_store_query = sfaira.data.load_store(SFAIRA_DATA_PATH, store_format='h5ad')
    data_store_query = data_store_query.stores['Homo sapiens']
    data_store_query.subset(attr_key='id', values=DATA_SET_SPLITS[tissue]['query'])
    data_store_query.subset(attr_key='cell_type', excluded_values=['unknown'])
    data_store_query.subset(attr_key='cell_type', values=cell_types_reference)
    assert data_store_query.n_obs >= 100
    # create adata objects for reference and query data
    adata_reference = anndata.AnnData(
        X=data_store_reference.X,
        obs=data_store_reference.obs,
        var=data_store_reference.var)
    adata_query = anndata.AnnData(
        X=data_store_query.X,
        obs=data_store_query.obs,
        var=data_store_reference.var
    )

    return adata_reference, adata_query


def preprocess_data(adata_reference, adata_query, n_genes: int = 4000) -> (anndata.AnnData, anndata.AnnData):
    """
    Preprocess data for model fitting.

    Preprocessing consits of the following steps:
        1. Select n_genes highly variable genes
        2. log1p tranform data
    """
    highly_variable_genes = sc.pp.highly_variable_genes(sc.pp.log1p(adata_reference, copy=True),
                                                        inplace=False,
                                                        n_top_genes=n_genes)['highly_variable']
    adata_reference = adata_reference[:, highly_variable_genes]
    adata_query = adata_query[:, highly_variable_genes]

    return adata_reference, adata_query


def eval_classification_performance(y_true: pd.Series, y_pred: pd.Series):
    return pd.DataFrame(
        classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=y_true.unique(),
            output_dict=True
        )
    ).transpose()


def eval_integration_performance(adata: anndata.AnnData, adata_latent: anndata.AnnData):
    conditions, _ = label_encoder(adata, condition_key='id')
    labels, _ = label_encoder(adata, condition_key='cell_type')
    
    adata.obs['batch'] = conditions.squeeze(axis=1)
    adata.obs['cell_type'] = labels.squeeze(axis=1)
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    
    adata_latent.obs['batch'] = conditions.squeeze(axis=1)
    adata_latent.obs['cell_type'] = labels.squeeze(axis=1)
    adata_latent.obs['batch'] = adata_latent.obs['batch'].astype('category')
    adata_latent.obs['cell_type'] = adata_latent.obs['cell_type'].astype('category')
    
    sc.pp.pca(adata)
    sc.pp.pca(adata_latent)
    
    scores = metrics(
        adata,
        adata_latent,
        batch_key="batch",
        label_key="cell_type",
        isolated_labels_asw_=True,
        silhouette_=True,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=True,
        nmi_=True,
        ari_=True,
    ).T

    scores = scores[
        [
            "NMI_cluster/label",
            "ARI_cluster/label",
            "ASW_label",
            "ASW_label/batch",
            "PCR_batch",
            "isolated_label_F1",
            "isolated_label_silhouette",
            "graph_conn"
        ]
    ]
    
    return scores


def evaluate_lataq(source_adata, target_adata, model_type):
    from lataq.models import EMBEDCVAE, TRANVAE

    if model_type == 'tranvae':
        tranvae = TRANVAE(
            adata=source_adata,
            cell_type_keys=['cell_type'],
            condition_key='id',
            hidden_layer_sizes=[128, 128, 128],
            use_mmd=False,
        )
    elif model_type == 'embedcvae':
        tranvae = EMBEDCVAE(
            adata=source_adata,
            cell_type_keys=['cell_type'],
            condition_key='id',
            hidden_layer_sizes=[128, 128, 128],
            use_mmd=False,
        )
    else:
        raise ValueError(f'model_type can only be "tranvae" or "embedcvae". You supplied: {model_type}')

    early_stopping_kwargs = {
        "early_stopping_metric": "val_landmark_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    start = time.perf_counter()
    tranvae.train(
        n_epochs=500,
        early_stopping_kwargs=early_stopping_kwargs,
        alpha_epoch_anneal=0.25 * 1e3 if model_type == 'tranvae' else 0.25 * 1e6,
        pretraining_epochs=400,
        clustering_res=2,
        eta=10,
    )
    ref_time = time.perf_counter() - start
    
    y_true = target_adata.obs['cell_type'].copy()
    target_adata.obs['cell_type'] = 'unknown'

    if model_type == "embedcvae":
        tranvae_query = EMBEDCVAE.load_query_data(
            adata=target_adata, reference_model=tranvae, labeled_indices=[], unknown_ct_names=['unknown']
        )
    elif model_type == "tranvae":
        tranvae_query = TRANVAE.load_query_data(
            adata=target_adata, reference_model=tranvae, labeled_indices=[], unknown_ct_names=['unknown']
        )
    else:
        raise ValueError(f'model_type can only be "tranvae" or "embedcvae". You supplied: {model_type}')

    start = time.perf_counter()
    tranvae_query.train(
        n_epochs=100,
        early_stopping_kwargs=early_stopping_kwargs,
        alpha_epoch_anneal=0.25 * 1e3 if model_type == 'tranvae' else 0.25 * 1e6,
        pretraining_epochs=80,
        clustering_res=2,
        eta=10
    )
    y_pred = tranvae_query.classify(target_adata.X, target_adata.obs['id'])['cell_type']['preds']
    query_time = time.perf_counter() - start
    
    target_adata.obs['cell_type'] = y_true
    
    clf_report = eval_classification_performance(y_true=y_true, y_pred=y_pred)
    
    x_latent_query = tranvae_query.get_latent(x=target_adata.X, c=target_adata.obs['id'])
    x_latent_ref = tranvae_query.get_latent(x=source_adata.X, c=source_adata.obs['id'])
    adata_full = source_adata.concatenate(target_adata)
    adata_latent_full = sc.AnnData(x_latent_ref).concatenate(sc.AnnData(x_latent_query))
    integration_scores = eval_integration_performance(adata_full, adata_latent_full)

    return {
        'reference_time': ref_time,
        'query_time': query_time,
        'classification_report_query': clf_report,
        'integration_scores': integration_scores
    }


def evaluate_svm(source_adata, target_adata):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    train_X = source_adata.X
    train_X = np.log1p(train_X)
    train_Y = source_adata.obs['cell_type']
    test_X = target_adata.X
    test_X = np.log1p(test_X)

    clf = CalibratedClassifierCV(LinearSVC())
    start = time.perf_counter()
    clf.fit(train_X, train_Y)
    ref_time = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = clf.predict(test_X)
    prob = np.max(clf.predict_proba(test_X), axis=1)
    query_time = time.perf_counter() - start
    unlabeled = np.where(prob < 0.)
    y_pred[unlabeled] = "unknown"

    clf_report = eval_classification_performance(y_true=target_adata.obs['cell_type'], y_pred=y_pred)

    return {
        'reference_time': ref_time,
        'query_time': query_time,
        'classification_report_query': clf_report,
        'integration_scores': None
    }


def evaluate_sfaira_mlp(target_adata, tissue: str):
    import sfaira

    ui = sfaira.ui.UserInterface(sfaira_repo=True)
    superset_lookup = {'retina': 'eye', 'lungparenchyma': 'lung', 'ileum': 'intestine', 'lymphnode': 'spleen'}
    if tissue in superset_lookup:
        # use superset model if there is no direct model
        tissue = superset_lookup[tissue]
    ui.zoo_celltype.model_id = sorted([
        model for model in ui.zoo_celltype.available_model_ids if model.startswith(f'celltype_human-{tissue}-mlp')
    ])[-1]
    ui.zoo_embedding.model_id = sorted([
        model for model in ui.zoo_embedding.available_model_ids if model.startswith(f'embedding_human-{tissue}-vae')
    ])[-1]
    ui.load_data(target_adata, gene_ens_col='index')
    ui.load_model_celltype()

    start = time.perf_counter()
    ui.predict_celltypes()
    y_pred = ui.data.adata.obs['celltypes_sfaira'].to_numpy()
    query_time = time.perf_counter() - start
    
    clf_report = eval_classification_performance(y_true=target_adata.obs['cell_type'], y_pred=y_pred)

    return {
        'reference_time': 0.,
        'query_time': query_time,
        'classification_report_query': clf_report,
        'integration_scores': None
    }


def evaluate_scanvi(source_adata, target_adata):
    import scarches as sca

    source_adata = sca.dataset.setup_anndata(source_adata, labels_key='cell_type', batch_key='id', copy=True)
    # TRAINING REFERENCE MODEL
    vae_ref = sca.models.SCVI(source_adata)
    ref_time = time.perf_counter()
    vae_ref.train()
    vae_ref_scan = sca.models.SCANVI.from_scvi_model(vae_ref, "unknown")
    vae_ref_scan.train(max_epochs=20)
    ref_time = time.perf_counter() - ref_time
    # TRAINING QUERY MODEL
    true_labels = target_adata.obs['cell_type'].copy()
    # set cell_type to 'unknown'
    target_adata.obs['cell_type'] = 'unknown'
    vae_q = sca.models.SCANVI.load_query_data(
        target_adata, vae_ref_scan, freeze_dropout=True
    )
    vae_q._unlabeled_indices = np.arange(target_adata.n_obs)
    vae_q._labeled_indices = []
    query_time = time.perf_counter()
    vae_q.train(
        max_epochs=100, plan_kwargs=dict(weight_decay=0.0), check_val_every_n_epoch=10
    )
    y_pred = vae_q.predict()
    query_time = time.perf_counter() - query_time
    
    # set cell_type col to real value again
    target_adata.obs['cell_type'] = true_labels
    
    clf_report = eval_classification_performance(y_true=true_labels, y_pred=y_pred)
    
    x_latent_ref = vae_q.get_latent_representation(source_adata)
    x_latent_query = vae_q.get_latent_representation(target_adata)
    adata_full = source_adata.concatenate(target_adata)
    adata_latent_full = sc.AnnData(x_latent_ref).concatenate(sc.AnnData(x_latent_query))
    integration_scores = eval_integration_performance(adata_full, adata_latent_full)

    return {
        'reference_time': ref_time,
        'query_time': query_time,
        'classification_report_query': clf_report,
        'integration_scores': integration_scores
    }


@ex.automain
def run(tissue: str, model: str):
    print(f'Running benchmark for model={model} on tissue={tissue}')

    source_data, target_data = create_anndata_objects(tissue)
    if model in ['scanvi', 'lataq']:
        source_data, target_data = preprocess_data(source_data, target_data)
    print('Shape source data: ', source_data.shape)
    print('Shape target data: ', target_data.shape)
    source_data = remove_sparsity(source_data)
    target_data = remove_sparsity(target_data)

    if model == 'scanvi':
        print(f'Evaluating SCANVI model on tissue={tissue}')
        results = evaluate_scanvi(source_data, target_data)
    elif model == 'lataq_tranvae':
        print(f'Evaluating LATAQ TRANVAE model on tissue={tissue}')
        results = evaluate_lataq(source_data, target_data, model_type='embedcvae')
    elif model == 'lataq_embedcvae':
        print(f'Evaluating LATAQ EMBEDCVAE model on tissue={tissue}')
        results = evaluate_lataq(source_data, target_data, model_type='tranvae')
    elif model == 'sfaira_mlp':
        print(f'Evaluating SFAIRA_MLP model on tissue={tissue}')
        results = evaluate_sfaira_mlp(target_data, tissue)
    elif model == 'svm':
        print(f'Evaluating LINEAR SVM model on tissue={tissue}')
        results = evaluate_svm(source_data, target_data)
    else:
        raise ValueError(f'model: {model} is not implemented')

    results['num_cells_in_reference'] = len(source_data)
    results['num_cells_in_query'] = len(target_data)

    return results
