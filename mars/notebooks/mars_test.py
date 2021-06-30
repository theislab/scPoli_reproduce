# Maybe needs some tweaks. I ran it directly in their repo, since it was not installable. Also may update the datasets to match the other scripts


import torch
import os
import numpy as np
import pandas as pd
import scanpy.api as sc
from anndata import AnnData

import warnings

warnings.filterwarnings('ignore')

from args_parser import get_parser
from model.mars import MARS
from model.experiment_dataset import ExperimentDataset
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report


experiment = "brain"
test_nr = 2


def celltype_to_numeric(adata, obs_key):
    """Adds ground truth clusters data."""
    annotations = list(adata.obs[obs_key])
    annotations_set = sorted(set(annotations))

    mapping = {a: idx for idx, a in enumerate(annotations_set)}

    truth_labels = [mapping[a] for a in annotations]
    adata.obs['truth_labels'] = pd.Categorical(values=truth_labels)

    return adata, mapping

# ------------------------------------------------------------------------------------ Hyperparams
params, unknown = get_parser().parse_known_args()
params.cuda = True
params.pretrain_batch = 128
print('PARAMS:', params)
if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
params.device = device

# ------------------------------------------------------------------------------------ Data Processing
if experiment == "pancreas":
    adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
    condition_key = "study"
    cell_type_key = "cell_type"
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
        reference = ['Pancreas inDrop', 'Pancreas SS2', 'Pancreas CelSeq2', 'Pancreas CelSeq', 'Pancreas Fluidigm C1']
        query = []
if experiment == "pbmc":
    adata_all = sc.read(os.path.expanduser(
        f'~/Documents/benchmarking_datasets/Immune_ALL_human_wo_villani_rqr_normalized_hvg.h5ad'))
    condition_key = 'condition'
    cell_type_key = 'final_annotation'
    if test_nr == 1:
        reference = ['10X']
        query = ['Oetjen', 'Sun', 'Freytag']
    elif test_nr == 2:
        reference = ['10X', 'Oetjen']
        query = ['Sun', 'Freytag']
    elif test_nr == 3:
        reference = ['10X', 'Oetjen', 'Sun']
        query = ['Freytag']
    elif test_nr == 4:
        reference = ['10X', 'Oetjen', 'Sun', 'Freytag']
        query = []
if experiment == "brain":
    adata_all = sc.read(
        os.path.expanduser(f'~/Documents/benchmarking_datasets/mouse_brain_subsampled_normalized_hvg.h5ad'))
    condition_key = "study"
    cell_type_key = "cell_type"
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

# Load Data
adata = adata_all.raw.to_adata()

# Create Int Mapping for celltypes
adata, celltype_id_map = celltype_to_numeric(adata, cell_type_key)
cell_type_name_map = {v: k for k, v in celltype_id_map.items()}

# Preprocess data
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10, zero_center=True)

# Make labeled Datasets for Mars
annotated = []
labels = []
batches = []
for batch in reference:
    labeled_adata = adata[adata.obs.study.isin([batch])].copy()
    y_labeled = np.array(labeled_adata.obs['truth_labels'], dtype=np.int64)
    annotated.append(ExperimentDataset(labeled_adata.X,
                                       labeled_adata.obs_names,
                                       labeled_adata.var_names,
                                       batch,
                                       y_labeled
                                       ))
    labels += labeled_adata.obs[cell_type_key].tolist()
    batches += labeled_adata.obs[condition_key].tolist()

# Make Unlabeled Datasets for Mars
unlabeled_adata = adata[adata.obs.study.isin(query)].copy()
y_unlabeled = np.array(unlabeled_adata.obs['truth_labels'], dtype=np.int64)
unannnotated = ExperimentDataset(
    unlabeled_adata.X,
    unlabeled_adata.obs_names,
    unlabeled_adata.var_names,
    'query',
    y_unlabeled
)
labels += unlabeled_adata.obs[cell_type_key].tolist()
batches += unlabeled_adata.obs[condition_key].tolist()
n_clusters = len(np.unique(unannnotated.y))

# Make pretrain Dataset
pretrain = ExperimentDataset(
    adata.X,
    adata.obs_names,
    adata.var_names,
    'Pretrain'
)

# ------------------------------------------------------------------------------------ Model Init and Training
mars = MARS(
    n_clusters,
    params,
    annotated,
    unannnotated,
    pretrain,
    hid_dim_1=1000,
    hid_dim_2=100
)
ref_time = time.time()
adata, landmarks, scores = mars.train(evaluation_mode=True)
ref_time = time.time() - ref_time
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

print('QUERY ACC:', np.mean(np.array(predictions)[adata.obs['experiment'] == 'query'] == np.array(labels_after)[adata.obs['experiment'] == 'query']))
print('FULL ACC:', np.mean(np.array(predictions) == np.array(labels_after)))

print('{}: Acc {}, F1_score {}, NMI {}, Adj_Rand {}, Adj_MI {}'.format(
    unannnotated.metadata,
    scores['accuracy'], scores['f1_score'],
    scores['nmi'],
    scores['adj_rand'],
    scores['adj_mi'])
)

text_file = open(os.path.expanduser(f'~/Documents/tranvae_paper/mars/{experiment}/{test_nr}_ref_acc_report.txt'), "w")
n = text_file.write(classification_report(y_true=np.array(labels_after)[adata.obs['experiment'] == 'query'],
                                          y_pred=np.array(predictions)[adata.obs['experiment'] == 'query']))
text_file.close()
text_file_t = open(os.path.expanduser(f'~/Documents/tranvae_paper/mars/{experiment}/{test_nr}_ref_runtime.txt'), "w")
m = text_file_t.write(str(ref_time))
text_file_t.close()

adata_mars = AnnData(adata.obsm['MARS_embedding'])
adata_mars.obs['celltype'] = labels_after
adata_mars.obs['predictions'] = predictions
adata_mars.obs['batch'] = batches

sc.pp.neighbors(adata_mars, n_neighbors=30, use_rep='X')
sc.tl.leiden(adata_mars)
sc.tl.umap(adata_mars)
sc.pl.umap(adata_mars,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/tranvae_paper/mars/{experiment}/{test_nr}_umap_mars_surg.png'), bbox_inches='tight')
adata_mars.write_h5ad(filename=os.path.expanduser(f'~/Documents/tranvae_paper/mars/{experiment}/{test_nr}_full_data.h5ad'))
