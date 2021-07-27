import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import itemfreq, entropy
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

def label_encoder(adata, encoder=None, condition_key='condition'):
    """Encode labels of Annotated `adata` matrix.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       encoder: Dict or None
            dictionary of encoded labels. if `None`, will create one.
       condition_key: String
            column name of conditions in `adata.obs` data frame.

       Returns
       -------
       labels: `~numpy.ndarray`
            Array of encoded labels
       label_encoder: Dict
            dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    if encoder is None:
        encoder = {k: v for k, v in zip(sorted(unique_conditions), np.arange(len(unique_conditions)))}

    labels = np.zeros(adata.shape[0])
    if not set(unique_conditions).issubset(set(encoder.keys())):
        print("Warning: Labels in adata is not a subset of label-encoder!")
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), encoder


def remove_sparsity(adata):
    """
        If ``adata.X`` is a sparse matrix, this will convert it in to normal matrix.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
    """
    if sparse.issparse(adata.X):
        new_adata = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True))
        return new_adata

    return adata


def __entropy_from_indices(indices, n_cat):
    return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)), base=n_cat)


def entropy_batch_mixing(adata, label_key="batch",
                         n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    """Computes Entory of Batch mixing metric for ``adata`` given the batch column name.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        n_pools: int
            Number of EBM computation which will be averaged.
        n_samples_per_pool: int
            Number of samples to be used in each pool of execution.
        Returns
        -------
        score: float
            EBM score. A float between zero and one.
    """
    adata = remove_sparsity(adata)
    n_cat = len(adata.obs[label_key].unique().tolist())
    print(f'Calculating EBM with n_cat = {n_cat}')
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = neighbors.kneighbors(adata.X, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: adata.obs[label_key].values[i])(indices)

    entropies = np.apply_along_axis(__entropy_from_indices, axis=1, arr=batch_indices, n_cat=n_cat)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])
    print("EBM:", score)
    return score


def knn_purity(adata, label_key="celltype", n_neighbors=30):
    """Computes KNN Purity metric for ``adata`` given the batch column name.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        Returns
        -------
        score: float
            KNN purity score. A float between 0 and 1.
    """
    adata = remove_sparsity(adata)
    labels = LabelEncoder().fit_transform(adata.obs[label_key].to_numpy())

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = nbrs.kneighbors(adata.X, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: labels[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[labels == i]) for i in np.unique(labels)
    ]  # per cell-type purity
    knn_p_score = np.mean(res)
    print("KNN-P:", knn_p_score)
    return knn_p_score



def nmi_helper(adata, group1, group2, method="arithmetic"):
    """
    Normalized mutual information NMI based on 2 different cluster assignments `group1` and `group2`
    params:
        adata: Anndata object
        group1: column name of `adata.obs` or group assignment
        group2: column name of `adata.obs` or group assignment
        method: NMI implementation
            'max': scikit method with `average_method='max'`
            'min': scikit method with `average_method='min'`
            'geometric': scikit method with `average_method='geometric'`
            'arithmetic': scikit method with `average_method='arithmetic'`
            'Lancichinetti': implementation by A. Lancichinetti 2009 et al.
            'ONMI': implementation by Aaron F. McDaid et al. (https://github.com/aaronmcdaid/Overlapping-NMI) Hurley 2011
        nmi_dir: directory of compiled C code if 'Lancichinetti' or 'ONMI' are specified as `method`. Compilation should be done as specified in the corresponding README.
    return:
        normalized mutual information (NMI)
    """
    adata = remove_sparsity(adata)

    if isinstance(group1, str):
        group1 = adata.obs[group1].tolist()
    elif isinstance(group1, pd.Series):
        group1 = group1.tolist()

    labels = adata.obs[group2].values
    labels_encoded = LabelEncoder().fit_transform(labels)
    group2 = labels_encoded

    if len(group1) != len(group2):
        raise ValueError(f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})')

    # choose method
    if method in ['max', 'min', 'geometric', 'arithmetic']:
        nmi_value = normalized_mutual_info_score(group1, group2, average_method=method)
    else:
        raise ValueError(f"Method {method} not valid")

    return nmi_value


def asw_c(adata, label_key="celltype"):
    score = silhouette(adata, group_key=label_key, metric='euclidean')
    print("ASW_C:", score)
    return score

def asw_b(adata, label_key="celltype", batch_key="batch"):
    _, sil_clus = silhouette_batch(adata, batch_key=batch_key, group_key=label_key,
                                   metric='euclidean', verbose=False)
    score = sil_clus['silhouette_score'].mean()
    print("ASW_B:", score)
    return score


def silhouette(adata, group_key, metric='euclidean', scale=True):
    """
    wrapper for sklearn silhouette function values range from [-1, 1] with 1 being an ideal fit, 0 indicating overlapping clusters and -1 indicating misclassified cells
    """
    adata = remove_sparsity(adata)
    labels = adata.obs[group_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)
    asw = silhouette_score(adata.X, labels_encoded, metric=metric)
    if scale:
        asw = (asw + 1)/2
    return asw


def silhouette_batch(adata, batch_key, group_key, metric='euclidean', verbose=True, scale=True):
    """
    Silhouette score of batch labels subsetted for each group.
    params:
        batch_key: batches to be compared against
        group_key: group labels to be subsetted by e.g. cell type
        metric: see sklearn silhouette score
        embed: name of column in adata.obsm
    returns:
        all scores: absolute silhouette scores per group label
        group means: if `mean=True`
    """
    adata = remove_sparsity(adata)
    glob_batches = adata.obs[batch_key].values
    batch_enc = LabelEncoder()
    batch_enc.fit(glob_batches)
    sil_all = pd.DataFrame(columns=['group', 'silhouette_score'])

    for group in adata.obs[group_key].unique():
        adata_group = adata[adata.obs[group_key] == group]
        if adata_group.obs[batch_key].nunique() == 1:
            continue
        batches = batch_enc.transform(adata_group.obs[batch_key])
        sil_per_group = silhouette_samples(adata_group.X, batches, metric=metric)
        # take only absolute value
        sil_per_group = [abs(i) for i in sil_per_group]
        if scale:
            # scale s.t. highest number is optimal
            sil_per_group = [1 - i for i in sil_per_group]
        d = pd.DataFrame({'group': [group] * len(sil_per_group), 'silhouette_score': sil_per_group})
        sil_all = sil_all.append(d)
    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby('group').mean()

    if verbose:
        print(f'mean silhouette per cell: {sil_means}')
    return sil_all, sil_means