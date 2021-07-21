import argparse
from typing import Type
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.metrics import classification_report
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK
from scarches.dataset.trvae.data_handling import remove_sparsity
from lataq.models import TRANVAE
from exp_dict import EXPERIMENT_INFO
np.random.seed(420)

parser = argparse.ArgumentParser(description='TRANVAE testing')
parser.add_argument(
    '--experiment', 
    type=str,
    help='Dataset of test'
)
parser.add_argument(
    '--max_evals', 
    type=int,
    help='Number of tests',
    default=100
)
parser.add_argument(
    '--n_epochs', 
    type=int,
    help='Number of epochs',
    default=50
)
parser.add_argument(
    '--n_pre_epochs', 
    type=int,
    help='Number of pretraining epochs',
    default=10
)
args = parser.parse_args()
print(args)

DATA_DIR = '/storage/groups/ml01/workspace/carlo.dedonno/LATAQ/data'
RESULTS_DIR = '/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hyperopt/results'
REF_PATH = '/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/tmp/ref_model'
EXP_PARAMS = EXPERIMENT_INFO[args.experiment]
FILE_NAME = EXP_PARAMS['file_name']
adata = sc.read(f'{DATA_DIR}/{FILE_NAME}')
condition_key = EXP_PARAMS['condition_key']
cell_type_key = EXP_PARAMS['cell_type_key']
reference = EXP_PARAMS['reference']
query = EXP_PARAMS['query']

adata = remove_sparsity(adata)
source_adata = adata[adata.obs.study.isin(reference)].copy()
target_adata = adata[adata.obs.study.isin(query)].copy()

space = hp.choice('model_params', [{
    #'tau': hp.quniform('tau', 0, 1, 1),
    #'eta': hp.quniform('eta', 0, 1, 1),
    'latent_dim': hp.quniform('latent_dim', 10, 100, 10),
    #'alpha_epoch_anneal': hp.loguniform('alpha_epoch_anneal', 3, 6),
    'loss_metric': hp.choice('loss_metric', ['dist']),#, 'seurat', 'overlap']),
    'clustering_res': hp.quniform('clustering_res', 0.1, 1, 0.1),
    'hidden_layer_sizes': hp.quniform('hidden_layer_sizes', 1, 5, 1)
}])

OPT_PARAMS = [
    'eta', 
    'latent_dim', 
    'alpha_epoch_anneal', 
    'loss_metric', 
    'clustering_res',
    'hidden_layer_sizes'
]

early_stopping_kwargs = {
    "early_stopping_metric": "val_classifier_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

trials = Trials()

params_list = []
results_list = []

def objective(params):
    EPOCHS = args.n_epochs
    PRE_EPOCHS = args.n_pre_epochs
    
    tranvae = TRANVAE(
        adata=source_adata,
        condition_key=condition_key,
        cell_type_keys=cell_type_key,
        hidden_layer_sizes=[128]*int(params['hidden_layer_sizes']),
        latent_dim=int(params['latent_dim']),
        use_mmd=False,
    )
    try:
        tranvae.train(
            n_epochs=EPOCHS,
            early_stopping_kwargs=early_stopping_kwargs,
            pretraining_epochs=PRE_EPOCHS,
            #alpha_epoch_anneal=params['alpha_epoch_anneal'],
            #eta=params['eta'],
            tau=0,
            clustering_res=params['clustering_res'],
            labeled_loss_metric=params['loss_metric'],
            unlabeled_loss_metric=params['loss_metric']
        )
        tranvae.save(REF_PATH, overwrite=True)
        tranvae_query = TRANVAE.load_query_data(
            adata=target_adata,
            reference_model=REF_PATH,
            labeled_indices=[],
        )
        tranvae_query.train(
                n_epochs=EPOCHS,
                early_stopping_kwargs=early_stopping_kwargs,
                pretraining_epochs=PRE_EPOCHS,
                #eta=params['eta'],
                #tau=0,
                weight_decay=0,
                clustering_res=params['clustering_res'],
                labeled_loss_metric=params['loss_metric'],
                unlabeled_loss_metric=params['loss_metric']
            )
        results_dict = tranvae_query.classify(
            adata.X, 
            adata.obs[condition_key], 
            metric=params['loss_metric']
        )
        for i in range(len(cell_type_key)):
            preds = results_dict[cell_type_key[i]]['preds']
            probs = results_dict[cell_type_key[i]]['probs']
            results_dict[cell_type_key[i]]['report'] = classification_report(
                y_true=adata.obs[cell_type_key[i]], 
                y_pred=preds,
                output_dict=True
            )
        params_list.append(params)
        results_list.append(results_dict[cell_type_key[0]]['report'])
        del tranvae, tranvae_query
    except TypeError:
        pass
    return {
        'loss': -results_dict[cell_type_key[-1]]['report']['weighted avg']['f1-score'],
        'status': STATUS_OK,
    }

best = fmin(
    objective,
    space=space,
    algo=tpe.suggest,
    max_evals=args.max_evals,
    trials=trials
)

t = pd.concat(
    [pd.DataFrame(params_list), pd.DataFrame(results_list)],
    axis=1
)
t = t[OPT_PARAMS + ['accuracy', 'macro avg', 'weighted avg']]
for col in ['macro avg', 'weighted avg']:
    t = pd.concat(
        [
            t.drop([col], axis=1), 
            t[col].apply(pd.Series).add_prefix(f'{col} ')
        ], 
        axis=1
    )
t.to_pickle(f'{RESULTS_DIR}/tranvae_{args.experiment}.pickle')