
from lataq.models import TRANVAE
from scarches.dataset.trvae.data_handling import remove_sparsity
from sklearn.metrics import classification_report

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

adata = remove_sparsity(adata)
source_adata = adata[adata.obs.study.isin(reference)].copy()
target_adata = adata[adata.obs.study.isin(query)].copy()

space = hp.choice(
    "model_params",
    [
        {
            ## architecture
            #'tau': hp.quniform('tau', 0, 1, 1),
            "eta": hp.quniform("eta", 0, 1, 1),
            "latent_dim": hp.quniform("latent_dim", 10, 100, 1),
            "alpha_epoch_anneal": hp.loguniform("alpha_epoch_anneal", 1e3, 1e6),
            "loss_metric": hp.choice(
                "loss_metric", ["dist"]
            ),  # , 'seurat', 'overlap']),
            "clustering_res": hp.uniform("clustering_res", 0.1, 2),
            "hidden_layer_sizes": hp.quniform("hidden_layer_sizes", 1, 4, 1),
        }
    ],
)
OPT_PARAMS = [
    "eta",
    "latent_dim",
    "alpha_epoch_anneal",
    "loss_metric",
    "clustering_res",
]

trials = Trials()

params_list = []
results_list = []


def objective(params):
    EPOCHS = 50
    PRE_EPOCHS = 10

    tranvae = TRANVAE(
        adata=source_adata,
        condition_key=condition_key,
        cell_type_keys=cell_type_key,
        hidden_layer_sizes=[128] * int(params["hidden_layer_sizes"]),
        latent_dim=int(params["latent_dim"]),
        use_mmd=False,
    )
    tranvae.train(
        n_epochs=EPOCHS,
        early_stopping_kwargs=early_stopping_kwargs,
        pretraining_epochs=PRE_EPOCHS,
        alpha_epoch_anneal=params["alpha_epoch_anneal"],
        eta=params["eta"],
        tau=0,
        clustering_res=params["clustering_res"],
        labeled_loss_metric=params["loss_metric"],
        unlabeled_loss_metric=params["loss_metric"],
    )
    ref_path = f"../tmp/ref_model"
    tranvae.save(ref_path, overwrite=True)
    tranvae_query = TRANVAE.load_query_data(
        adata=target_adata,
        reference_model=f"../tmp/ref_model",
        labeled_indices=[],
    )
    tranvae_query.train(
        n_epochs=EPOCHS,
        early_stopping_kwargs=early_stopping_kwargs,
        pretraining_epochs=PRE_EPOCHS,
        eta=params["eta"],
        tau=0,
        weight_decay=0,
        clustering_res=params["clustering_res"],
        labeled_loss_metric=params["loss_metric"],
        unlabeled_loss_metric=params["loss_metric"],
    )
    results_dict = tranvae_query.classify(
        adata.X, adata.obs[condition_key], metric=params["loss_metric"]
    )
    for i in range(len(cell_type_key)):
        preds = results_dict[cell_type_key[i]]["preds"]
        results_dict[cell_type_key[i]]["probs"]
        results_dict[cell_type_key[i]]["report"] = classification_report(
            y_true=adata.obs[cell_type_key[i]], y_pred=preds, output_dict=True
        )
    params_list.append(params)
    results_list.append(results_dict[cell_type_key[0]]["report"])

    return {
        "loss": -results_dict[cell_type_key[0]]["report"]["weighted avg"]["f1-score"],
        "status": STATUS_OK,
    }


best = fmin(
    objective, space=space, algo=tpe.suggest, max_evals=args.max_evals, trials=trials
)
