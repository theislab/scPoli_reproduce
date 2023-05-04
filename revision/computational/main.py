import argparse
import numpy as np
import scanpy as sc
from scarches.models.scpoli import scPoli
import time

parser = argparse.ArgumentParser(description='number of obs')
parser.add_argument('--n_obs', dest='n_obs', type=int, help='Number of observations')
args = parser.parse_args()

adata = sc.read('./../data/brain_concatenated_sparse.h5ad')
if args.n_obs>0:
    adata = sc.pp.subsample(adata, n_obs=args.n_obs, copy=True)

    HIDDEN_LAYER_WIDTH = int(np.sqrt(adata.shape[1]))
    condition_key = 'study'
    cell_type_key = ['cell_type']

    start = time.time()
    scpoli_model = scPoli(
        adata=adata,
        condition_key=condition_key,
        cell_type_keys=cell_type_key,
        hidden_layer_sizes=[HIDDEN_LAYER_WIDTH],
        embedding_dim=5,
    )

    scpoli_model.train(
        n_epochs=100,
        pretraining_epochs=80,
        use_early_stopping=False,
        eta=1,
        train_frac=1,
    )
    print(time.time() - start)