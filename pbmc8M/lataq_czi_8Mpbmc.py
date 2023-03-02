import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import json
import time

from scarches.dataset.trvae.data_handling import remove_sparsity
from lataq.models import EMBEDCVAE
from lataq.exp_dict import EXPERIMENT_INFO

print('Default model with norm-constrained embedding')
print('load data')



ref_adata = sc.read('/lustre/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/data/pbmc_raw.h5ad')


id_key = 'sample_ID_lataq'
condition_key = id_key
cell_type_key = ['cell_type_for_integration']




print('Initialize network')
lataq_model = EMBEDCVAE(
    adata=ref_adata,
    condition_key=condition_key,
    cell_type_keys=cell_type_key,
    hidden_layer_sizes=[128]*3,
    latent_dim=30,
    embedding_dim=20,
    inject_condition=['encoder', 'decoder']
)


PARAMS = {
    'EPOCHS': 50,                                      #TOTAL TRAINING EPOCHS
    'N_PRE_EPOCHS': 40,                                #EPOCHS OF PRETRAINING WITHOUT LANDMARK LOSS
    #'DATA_DIR': '../../lataq_reproduce/data',          #DIRECTORY WHERE THE DATA IS STORED
    #'DATA': 'pancreas',                                #DATA USED FOR THE EXPERIMENT
    'EARLY_STOPPING_KWARGS': {                         #KWARGS FOR EARLY STOPPING
        "early_stopping_metric": "val_landmark_loss",  ####value used for early stopping
        "mode": "min",                                 ####choose if look for min or max
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    },
    'LABELED_LOSS_METRIC': 'dist',           
    'UNLABELED_LOSS_METRIC': 'dist',
    'LATENT_DIM': 50,
    'ALPHA_EPOCH_ANNEAL': 1e3,
    'CLUSTERING_RES': 2,
    'HIDDEN_LAYERS': 4,
    'ETA': 1,
}

print('train')
tic = time.perf_counter()
# need to time the training as well.
lataq_model.train(
    n_epochs=100,
    pretraining_epochs=45,
    early_stopping_kwargs=PARAMS['EARLY_STOPPING_KWARGS'],
    alpha_epoch_anneal=PARAMS['ALPHA_EPOCH_ANNEAL'],
    eta=PARAMS['ETA'],
    clustering_res=PARAMS['CLUSTERING_RES'],
    labeled_loss_metric=PARAMS['LABELED_LOSS_METRIC'],
    unlabeled_loss_metric=PARAMS['UNLABELED_LOSS_METRIC'],
    weight_decay=0,
    use_stratified_sampling=False,
)
toc = time.perf_counter()


time_mapping = toc - tic
times = {}
times['czi_pbmc_integration_lataq'] = time_mapping

with open('results/czi_pbmc_lataq_time_100epochs_2hidden_sampleidlataq_10k.json', 'w') as fp:
    json.dump(times, fp)
print('output written to json')

lataq_model.save('models/czi_8M_pbmc_lataq_100epochs_2hidden_sampleidlataq_10k/', overwrite=True)
print('model saved')
