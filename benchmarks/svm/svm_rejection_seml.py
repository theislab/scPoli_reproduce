import logging
from sacred import Experiment
import seml
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from scarches.dataset.trvae.data_handling import remove_sparsity
from lataq_reproduce.exp_dict import EXPERIMENT_INFO
import pickle

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
        f'lataq_reproduce/results/svm'
    )
    EXP_PARAMS = EXPERIMENT_INFO[data]
    FILE_NAME = EXP_PARAMS['file_name']

    adata = sc.read(f'{DATA_DIR}/{FILE_NAME}')
    condition_key = EXP_PARAMS['condition_key']
    cell_type_key = EXP_PARAMS['cell_type_key']
    reference = EXP_PARAMS['reference']
    query = EXP_PARAMS['query']

    adata = remove_sparsity(adata)
    source_adata = adata[adata.obs.study.isin(reference)].copy()
    target_adata = adata[adata.obs.study.isin(query)].copy()
    logging.info('Data loaded succesfully')

    train_X = source_adata.X
    train_X = np.log1p(train_X)
    train_Y = source_adata.obs[cell_type_key[0]]

    test_X = target_adata.X
    test_X = np.log1p(test_X)
    test_Y = target_adata.obs[cell_type_key[0]]

    Classifier = LinearSVC()
    clf = CalibratedClassifierCV(Classifier)

    clf.fit(train_X, train_Y)

    filename = f"{RES_PATH}/{data}_classifier.sav"
    pickle.dump(clf, open(filename, 'wb'))

    THRESHOLD = 0
    predicted = clf.predict(test_X)
    prob = np.max(clf.predict_proba(test_X), axis=1)
    unlabeled = np.where(prob < THRESHOLD)
    predicted[unlabeled] = 'Unknown'

    report = pd.DataFrame(
        classification_report(
            y_true=test_Y,
            y_pred=predicted,
            labels=np.array(target_adata.obs[cell_type_key[0]].unique().tolist()),
            output_dict=True,
        )
    ).transpose()

    full_X = adata.X
    full_X = np.log1p(full_X)
    full_Y = adata.obs[cell_type_key[0]]

    full_predicted = clf.predict(full_X)
    full_prob = np.max(clf.predict_proba(full_X), axis=1)
    full_unlabeled = np.where(full_prob < THRESHOLD)
    full_predicted[full_unlabeled] = 'Unknown'

    report_full = pd.DataFrame(
        classification_report(
            y_true=full_Y, 
            y_pred=full_predicted,
            output_dict=True,
        )
    ).transpose().add_prefix('full_')

    df_report = pd.concat([report, report_full], axis=1)
    return df_report