import os
import pickle

import numpy as np
import scanpy as sc
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# Experiment Params
Threshold = 0
# experiments = ["pancreas","pbmc","lung","scvelo","brain"]
experiments = ["pbmc", "tumor"]
test_nrs = [10]
cell_type_key = "cell_type"
for experiment in experiments:
    for test_nr in test_nrs:
        if experiment == "pancreas":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_pancreas_shrinked.h5ad"
                )
            )
            condition_key = "study"
            if test_nr == 1:
                reference = ["Pancreas inDrop"]
                query = [
                    "Pancreas SS2",
                    "Pancreas CelSeq2",
                    "Pancreas CelSeq",
                    "Pancreas Fluidigm C1",
                ]
            elif test_nr == 2:
                reference = ["Pancreas inDrop", "Pancreas SS2"]
                query = ["Pancreas CelSeq2", "Pancreas CelSeq", "Pancreas Fluidigm C1"]
            elif test_nr == 3:
                reference = ["Pancreas inDrop", "Pancreas SS2", "Pancreas CelSeq2"]
                query = ["Pancreas CelSeq", "Pancreas Fluidigm C1"]
            elif test_nr == 4:
                reference = [
                    "Pancreas inDrop",
                    "Pancreas SS2",
                    "Pancreas CelSeq2",
                    "Pancreas CelSeq",
                ]
                query = ["Pancreas Fluidigm C1"]
            elif test_nr == 5:
                reference = [
                    "Pancreas inDrop",
                    "Pancreas SS2",
                    "Pancreas CelSeq2",
                    "Pancreas CelSeq",
                    "Pancreas Fluidigm C1",
                ]
                query = []
            elif test_nr == 10:
                reference = [
                    "inDrop1",
                    "inDrop2",
                    "inDrop3",
                    "inDrop4",
                    "fluidigmc1",
                    "smartseq2",
                    "smarter",
                ]
                query = ["celseq", "celseq2"]
        if experiment == "pbmc":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_pbmc_shrinked.h5ad"
                )
            )
            condition_key = "condition"
            if test_nr == 1:
                reference = ["Oetjen"]
                query = ["10X", "Sun", "Freytag"]
            elif test_nr == 2:
                reference = ["Oetjen", "10X"]
                query = ["Sun", "Freytag"]
            elif test_nr == 3:
                reference = ["Oetjen", "10X", "Sun"]
                query = ["Freytag"]
            elif test_nr == 4:
                reference = ["Oetjen", "10X", "Sun", "Freytag"]
                query = []
            elif test_nr == 10:
                reference = ["Oetjen", "10X", "Sun"]
                query = ["Freytag"]
        if experiment == "brain":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_brain_shrinked.h5ad"
                )
            )
            condition_key = "study"
            if test_nr == 1:
                reference = ["Rosenberg"]
                query = ["Saunders", "Zeisel", "Tabula_muris"]
            elif test_nr == 2:
                reference = ["Rosenberg", "Saunders"]
                query = ["Zeisel", "Tabula_muris"]
            elif test_nr == 3:
                reference = ["Rosenberg", "Saunders", "Zeisel"]
                query = ["Tabula_muris"]
            elif test_nr == 4:
                reference = ["Rosenberg", "Saunders", "Zeisel", "Tabula_muris"]
                query = []
            elif test_nr == 10:
                reference = ["Rosenberg", "Saunders"]
                query = ["Zeisel", "Tabula_muris"]
        if experiment == "scvelo":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_scvelo_shrinked.h5ad"
                )
            )
            condition_key = "study"
            if test_nr == 1:
                reference = ["12.5"]
                query = ["13.5", "14.5", "15.5"]
            elif test_nr == 2:
                reference = ["12.5", "13.5"]
                query = ["14.5", "15.5"]
            elif test_nr == 3:
                reference = ["12.5", "13.5", "14.5"]
                query = ["15.5"]
            elif test_nr == 4:
                reference = ["12.5", "13.5", "14.5", "15.5"]
                query = []
            elif test_nr == 10:
                reference = ["12.5", "13.5"]
                query = ["14.5", "15.5"]
        if experiment == "lung":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_lung_shrinked.h5ad"
                )
            )
            condition_key = "condition"
            if test_nr == 1:
                reference = ["Dropseq_transplant", "10x_Biopsy"]
                query = ["10x_Transplant"]
            elif test_nr == 10:
                reference = ["Dropseq_transplant", "10x_Biopsy"]
                query = ["10x_Transplant"]
        if experiment == "tumor":
            adata = sc.read(
                os.path.expanduser(
                    f"~/Documents/benchmarking_datasets/benchmark_tumor_shrinked.h5ad"
                )
            )
            condition_key = "study"
            if test_nr == 10:
                reference = [
                    "breast",
                    "colorectal",
                    "liver2",
                    "liver1",
                    "lung1",
                    "lung2",
                    "multiple",
                    "ovary",
                    "pancreas",
                    "skin",
                ]
                query = ["melanoma1", "melanoma2", "uveal melanoma"]

        adata_ref = adata[adata.obs.study.isin(reference)].copy()
        adata_query = adata[adata.obs.study.isin(query)].copy()

        train_X = adata_ref.X
        train_X = np.log1p(train_X)
        train_Y = adata_ref.obs[cell_type_key]

        test_X = adata_query.X
        test_X = np.log1p(test_X)
        test_Y = adata_query.obs[cell_type_key]

        Classifier = LinearSVC()
        clf = CalibratedClassifierCV(Classifier)

        clf.fit(train_X, train_Y)

        filename = os.path.expanduser(
            f"~/Documents/svm_benchmarks/batchwise/{experiment}/{test_nr}_classifier.sav"
        )
        pickle.dump(clf, open(filename, "wb"))

        """
        filename = os.path.expanduser(
            f"~/Documents/svm_benchmarks/batchwise/{experiment}/{test_nr}_classifier.sav")
        clf = pickle.load(open(filename, 'rb'))
        """

        predicted = clf.predict(test_X)
        prob = np.max(clf.predict_proba(test_X), axis=1)
        unlabeled = np.where(prob < Threshold)
        predicted[unlabeled] = "Unknown"
        acc = np.mean(predicted == test_Y)

        text_file = open(
            os.path.expanduser(
                f"~/Documents/svm_benchmarks/batchwise/{experiment}/{test_nr}_acc_report_updated.txt"
            ),
            "w",
        )
        n = text_file.write(
            classification_report(
                y_true=test_Y,
                y_pred=predicted,
                labels=np.array(adata_query.obs[cell_type_key].unique().tolist()),
            )
        )
        text_file.close()

        full_X = adata.X
        full_X = np.log1p(full_X)
        full_Y = adata.obs[cell_type_key]

        full_predicted = clf.predict(full_X)
        full_prob = np.max(clf.predict_proba(full_X), axis=1)
        full_unlabeled = np.where(full_prob < Threshold)
        full_predicted[full_unlabeled] = "Unknown"
        text_file_f = open(
            os.path.expanduser(
                f"~/Documents/svm_benchmarks/batchwise/{experiment}/{test_nr}_acc_report_full.txt"
            ),
            "w",
        )
        m = text_file_f.write(
            classification_report(y_true=full_Y, y_pred=full_predicted)
        )
        text_file_f.close()
