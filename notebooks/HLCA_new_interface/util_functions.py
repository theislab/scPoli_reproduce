import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# From HLCA repo
HLCA_LABEL_RELATIONS = os.path.expanduser("~/repositories/HLCA_reproducibility/supporting_files/metadata_harmonization/HLCA_cell_type_reference_mapping_20211103.csv")

# Note: Got from HLCA notebbok reuslts

manual_to_fine_grained = {'NK cells': ['NK cells'],
 'Neuroendocrine': ['Neuroendocrine'],
 'CD8 T cells': ['CD8 T cells'],
 'Suprabasal': ['Suprabasal'],
 'Alveolar fibroblasts': ['Alveolar fibroblasts'],
 'Plasma cells': ['Plasma cells'],
 'Mast cells': ['Mast cells'],
 'Alveolar macrophages': ['Alveolar macrophages'],
 'SMG mucous': ['SMG mucous'],
 'AT1': ['AT1'],
 'Classical monocytes': ['Classical monocytes'],
 'Mesothelium': ['Mesothelium'],
 'DC2': ['DC2'],
 'Non-classical monocytes': ['Non-classical monocytes'],
 'DC1': ['DC1'],
 'Deuterosomal': ['Deuterosomal'],
 'AT2 proliferating': ['AT2 proliferating'],
 'Pericytes': ['Pericytes'],
 'EC venous pulmonary': ['EC venous pulmonary'],
 'AT2': ['AT2'],
 'EC venous systemic': ['EC venous systemic'],
 'Plasmacytoid DCs': ['Plasmacytoid DCs'],
 'CD4 T cells': ['CD4 T cells'],
 'SMG serous': ['SMG serous (nasal)', 'SMG serous (bronchial)'],
 'Secretory': ['Club (non-nasal)',
  'Club (nasal)',
  'Goblet (nasal)',
  'Goblet (bronchial)',
  'Goblet (subsegmental)',
  'Transitional Club-AT2'],
 'Basal': ['Basal resting', 'Suprabasal'],
 'Lymphatic EC': ['Lymphatic EC mature',
  'Lymphatic EC proliferating',
  'Lymphatic EC differentiating'],
 'Goblet': ['Goblet (nasal)', 'Goblet (bronchial)', 'Goblet (subsegmental)'],
 'Fibroblasts': ['Peribronchial fibroblasts',
  'Adventitial fibroblasts',
  'Alveolar fibroblasts',
  'Pericytes',
  'Subpleural fibroblasts'],
 'Club': ['Club (non-nasal)', 'Club (nasal)'],
 'Rare': ['Ionocyte', 'Tuft', 'Neuroendocrine'],
 'Interstitial macrophages': ['Monocyte-derived Mφ',
  'Interstitial Mφ perivascular'],
 'Multiciliated': ['Multiciliated (nasal)', 'Multiciliated (non-nasal)'],
 'EC capillary': ['EC aerocyte capillary', 'EC general capillary'],
 'Mature B cells': ['B cells'],
 'NK_ITGAD+': ['NK cells'],
 'CD4T cells TRM': ['CD4 T cells'],
 'Arterial Pulmonary': ['EC arterial'],
 'Arterial Systemic': ['EC arterial'],
 'CD4 T cells MAIT': ['CD4 T cells'],
 'CD8 T cells TRM': ['CD8 T cells'],
 'CD4 T cells naive': ['CD4 T cells'],
 'Airway smooth muscle': ['Smooth muscle'],
 'NK_XCL1+': ['NK cells'],
 'CD8 T cells GZMK+': ['CD8 T cells'],
 'Plasmablasts': ['Plasma cells'],
 'Vascular smooth muscle': ['Smooth muscle'],
 'Naive B cells': ['B cells'],
 'CD8 T cells ME': ['CD8 T cells'],
 'Erythrocytes': ['New'],
 'Basal proliferating': ['Basal resting', 'Suprabasal'],
 'Schwann myelinating': ['New'],
 'Gamma-delta T cells': ['New'],
 'DC activated': ['Migratory DCs'],
 'Megakaryocytes': ['New'],
 'NKT cells': ['New'],
 'Chondrocytes': ['New'],
 'Regulatory T cells': ['New'],
 'ILCs': ['New'],
 'Mφ proliferating': ['Alveolar Mφ proliferating',
  'Interstitial Mφ perivascular',
  'Monocyte-derived Mφ',
  'Alveolar macrophages'],
 'Schwann nonmyelinating': ['New']}

# Note: for compatibility with Carlo
manual_to_fine_grained['Club'] += 'Transitional Club-AT2'
# My division
manual_to_fine_grained['EC aerocyte capillary'] =  ['EC aerocyte capillary']
manual_to_fine_grained['EC general capillary'] =  ['EC general capillary']


thienpont_2018_acceptable_preds = {'Alveolar epithelium': ['AT1', 'AT2', 'AT2 proliferating'],
 'B cell lineage': ['B cells', 'Plasma cells'],
 'Cancer': ['New'],
 'Endothelial': ['EC arterial',
  'EC aerocyte capillary',
  'EC general capillary',
  'EC venous systemic',
  'EC venous pulmonary',
  'Lymphatic EC mature',
  'Lymphatic EC proliferating',
  'Lymphatic EC differentiating'],
 'Epithelial': ['Basal resting',
  'Suprabasal',
  'Deuterosomal',
  'Multiciliated (nasal)',
  'Multiciliated (non-nasal)',
  'Club (non-nasal)',
  'Club (nasal)',
  'Goblet (nasal)',
  'Goblet (bronchial)',
  'Goblet (subsegmental)',
  'Transitional Club-AT2',
  'Ionocyte',
  'Tuft',
  'Neuroendocrine',
  'SMG serous (nasal)',
  'SMG serous (bronchial)',
  'SMG mucous',
  'SMG duct',
  'AT1',
  'AT2',
  'AT2 proliferating'],
 'Erythroblast': ['New'],
 'Fibroblast lineage': ['Peribronchial fibroblasts',
  'Adventitial fibroblasts',
  'Alveolar fibroblasts',
  'Pericytes',
  'Subpleural fibroblasts',
  'Myofibroblasts',
  'Smooth muscle',
  'Fibromyocytes',
  'SM activated stress response'],
 'Mast cells': ['Mast cells'],
 'Myeloid': ['DC1',
  'DC2',
  'Migratory DCs',
  'Plasmacytoid DCs',
  'Alveolar macrophages',
  'Alveolar Mφ CCL3+',
  'Alveolar Mφ MT-positive',
  'Alveolar Mφ proliferating',
  'Monocyte-derived Mφ',
  'Interstitial Mφ perivascular',
  'Classical monocytes',
  'Non-classical monocytes',
  'Mast cells'],
 'T cell lineage': ['CD4 T cells',
  'CD8 T cells',
  'T cells proliferating',
  'NK cells']
}

thienpont_2018_manual_to_fig_label = {
    "T_cell": "T/NK cells",
    "Myeloid": "Myeloid",
    "B_cell": "B/Plasma cells",
    "Cancer": "Cancer",
    "Fibroblast": "Fibroblasts/SM",
    "Alveolar": "Alveolar",
    "Erythroblast": "Erythroblast",
    "Epithelial": "Epithelial",
    "EC": "Endothelium",
    "Mast_cell": "Mast cells",
}


healthy_manual_to_fig_label = {'NK cells': 'NK cells',
 'Neuroendocrine': 'Neuroendocrine',
 'CD8 T cells': 'CD8 T cells',
 'Suprabasal': 'Suprabasal',
 'Alveolar fibroblasts': 'Alveolar fibroblasts',
 'Plasma cells': 'Plasma cells',
 'Mast cells': 'Mast cells',
 'Alveolar macrophages': 'Alveolar macrophages',
 'SMG mucous': 'SMG mucous',
 'AT1': 'AT1',
 'Classical monocytes': 'Classical monocytes',
 'Mesothelium': 'Mesothelium',
 'DC2': 'DC2',
 'Non-classical monocytes': 'Non-classical monocytes',
 'DC1': 'DC1',
 'Deuterosomal': 'Deuterosomal',
 'AT2 proliferating': 'AT2 proliferating',
 'Pericytes': 'Pericytes',
 'EC venous pulmonary': 'EC venous pulmonary',
 'AT2': 'AT2',
 'EC venous systemic': 'EC venous systemic',
 'Plasmacytoid DCs': 'Plasmacytoid DCs',
 'CD4 T cells': 'CD4 T cells',
 'Mature B cells': 'B cells',
 'NK_ITGAD+': 'NK cells',
 'CD4T cells TRM': 'CD4 T cells',
 'Arterial Pulmonary': 'EC arterial',
 'Arterial Systemic': 'EC arterial',
 'CD4 T cells MAIT': 'CD4 T cells',
 'CD8 T cells TRM': 'CD8 T cells',
 'CD4 T cells naive': 'CD4 T cells',
 'Airway smooth muscle': 'Smooth muscle',
 'NK_XCL1+': 'NK cells',
 'CD8 T cells GZMK+': 'CD8 T cells',
 'Plasmablasts': 'Plasma cells',
 'Vascular smooth muscle': 'Smooth muscle',
 'Naive B cells': 'B cells',
 'CD8 T cells ME': 'CD8 T cells',
 'Basal proliferating': 'Basal',
 'Mφ proliferating': 'Macrophages',
 'SMG serous': 'SMG serous',
 'Secretory': 'Secretory',
 'Basal': 'Basal',
 'Lymphatic EC': 'Lymphatic EC',
 'Goblet': 'Goblet',
 'Fibroblasts': 'Fibroblasts',
 'Club': 'Club',
 'Rare': 'Rare',
 'Interstitial macrophages': 'Interstitial macrophages',
 'Multiciliated': 'Multiciliated',
 'EC aerocyte capillary': 'EC aerocyte capillary',
 'EC general capillary': 'EC general capillary',
 'Erythrocytes': 'Erythrocytes',
 'Schwann myelinating': 'Schwann myelinating',
 'Gamma-delta T cells': 'Gamma-delta T cells',
 'DC activated': 'Migratory DCs',
 'Megakaryocytes': 'Megakaryocytes',
 'NKT cells': 'NKT cells',
 'Chondrocytes': 'Chondrocytes',
 'Regulatory T cells': 'Regulatory T cells',
 'ILCs': 'ILCs',
 'Schwann nonmyelinating': 'Schwann nonmyelinating'}

healthy_ct_ordered = ['Basal',
 'Suprabasal',
 'Deuterosomal',
 'Multiciliated',
 'Club',
 'Secretory',
 'Goblet',
 'Rare',
 'Neuroendocrine',
 'SMG serous',
 'SMG mucous',
 'AT1',
 'AT2',
 'AT2 proliferating',
 'EC arterial',
 'EC aerocyte capillary',
 'EC general capillary',
 'EC venous systemic',
 'EC venous pulmonary',
 'Lymphatic EC',
 'Fibroblasts',
 'Alveolar fibroblasts',
 'Pericytes',
 'Smooth muscle',
 'Mesothelium',
 'B cells',
 'Plasma cells',
 'CD4 T cells',
 'CD8 T cells',
 'NK cells',
 'DC1',
 'DC2',
 'Migratory DCs',
 'Plasmacytoid DCs',
 'Alveolar macrophages',
 'Macrophages',
 'Interstitial macrophages',
 'Classical monocytes',
 'Non-classical monocytes',
 'Mast cells',
 'Chondrocytes',
 'Erythrocytes',
 'Gamma-delta T cells',
 'ILCs',
 'Megakaryocytes',
 'NKT cells',
 'Regulatory T cells',
 'Schwann myelinating',
 'Schwann nonmyelinating']


cancer_manual_to_fig_label = {'T cell lineage': 'T/NK cells',
 'Myeloid': 'Myeloid',
 'B cell lineage': 'B/Plasma cells',
 'Cancer': 'Cancer',
 'Fibroblast lineage': 'Fibroblasts/SM',
 'Alveolar epithelium': 'Alveolar',
 'Erythroblast': 'Erythroblast',
 'Epithelial': 'Epithelial',
 'Endothelial': 'Endothelium',
 'Mast cells': 'Mast cells'}

cancer_ct_ordered = [
    'Alveolar', 
    'B/Plasma cells', 
    'Endothelium', 
    'Epithelial', 
    'Fibroblasts/SM', 
    'Mast cells', 
    'Myeloid', 
    'T/NK cells', 
    'Cancer', 
    'Erythroblast', 
]


def get_celltype_mappings(dataset_name, csv_file_path=HLCA_LABEL_RELATIONS):
    LEVELS = ['Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5']
    main_mapping = pd.read_csv(csv_file_path, skiprows=1)
    main_mapping = main_mapping.replace(">", np.nan)
    main_mapping["ann_finest_level"] = main_mapping[LEVELS[-1]]
    for level in LEVELS[::-1]:
        main_mapping["ann_finest_level"] = main_mapping["ann_finest_level"].combine_first(main_mapping[level])
    main_mapping = main_mapping.fillna('-')
    main_mapping = main_mapping.replace("|", np.nan)
    for level in LEVELS:
        main_mapping[level] = main_mapping[level].ffill()
    main_mapping = main_mapping.replace("-", np.nan)
    final_mapping = main_mapping[[dataset_name, 'ann_finest_level']].dropna(subset=[dataset_name])
    return final_mapping


def get_prediction_performance(adata_obs, pred_col, uncert_col, valid_col, uncert_threshold):
    def get_status(uncert, pred, valid):
        if uncert < uncert_threshold:
            return 'Correct_certain' if pred in valid else 'Incorrect_certain'
        else:
            if 'New' in valid:
                return 'Correct_uncertain'
            else:
                if pred in valid:
                    return 'Incorrect_because_uncertain'
                else:
                    return 'Incorrect_and_uncertain'

    return np.vectorize(get_status)(adata_obs[uncert_col].to_numpy(), adata_obs[pred_col].to_numpy(), adata_obs[valid_col].to_numpy())


def plot_hlca_results(adata_obs, annotation_col, result_col, manual_to_fig_label, ct_ordered):
    plot_df = adata_obs[[annotation_col, result_col]].copy()
    
    plot_df['Annotations'] = plot_df[annotation_col].map(manual_to_fig_label)
    plot_df['status'] = plot_df[result_col].map({'Correct_certain': 'Correct', 'Incorrect_certain': 'Incorrect', 'Correct_uncertain': 'Unknown', 
                                                 'Incorrect_because_uncertain': 'Unknown', 'Incorrect_and_uncertain': 'Unknown'})
    
    total_n_per_ct = plot_df['Annotations'].value_counts()
    total_n_per_ct["Overall"] = plot_df.shape[0]

    perc_correct = (pd.crosstab(plot_df['Annotations'], plot_df['status']).loc[ct_ordered, :])
    perc_correct.loc["Overall", :] = (
        plot_df['status'].value_counts() / total_n_per_ct["Overall"] * 100
    )
    perc_correct = perc_correct.div(perc_correct.sum(axis=1), axis="rows") * 100
    
    with plt.rc_context(
        {
            "figure.figsize": (0.4 * len(perc_correct), 3),
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    ):
        fig, ax = plt.subplots()
        perc_correct.plot(kind="bar", stacked=True, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        
        ax.legend(handles[::-1], labels[::-1], loc=(1.01, 0.60), frameon=False)

        plt.xticks(
            ticks=range(len(ct_ordered) + 1),
            labels=[f"{ct} ({total_n_per_ct[ct]})" for ct in ct_ordered + ["Overall"]],
        )
        ax.set_ylabel("% of cells")
        plt.grid(False)
        plt.show()


def plot_hlca_healthy_results(adata_obs, annotation_col, result_col):
    plot_hlca_results(adata_obs, annotation_col, result_col, healthy_manual_to_fig_label, healthy_ct_ordered)


def plot_hlca_cancer_results(adata_obs, annotation_col, result_col):
    plot_hlca_results(adata_obs, annotation_col, result_col, cancer_manual_to_fig_label, cancer_ct_ordered)
        

def get_uncert_threshold_data(adata_obs, pred_col, uncert_col, valid_col, method_name, MAX_THRESHOLD=2, STEP=0.01):
    df = adata_obs[[pred_col, uncert_col, valid_col]].copy()
    seen_indicator = df[valid_col].map(lambda x: 'New' not in x)
    seen_df = df[seen_indicator].copy()
    unseen_df = df[~seen_indicator].copy()

    result_df = []
    for threshold in np.arange(0, MAX_THRESHOLD, STEP):
        results_seen = get_prediction_performance(seen_df, pred_col, uncert_col, valid_col, threshold)
        results_seen = pd.Series(results_seen).value_counts(normalize=True)
        tp = results_seen['Correct_certain'] if 'Correct_certain' in results_seen else 0.
        results_unseen = get_prediction_performance(unseen_df, pred_col, uncert_col, valid_col, threshold)
        results_unseen = pd.Series(results_unseen).value_counts(normalize=True)
        fu = results_unseen['Incorrect_certain'] if 'Incorrect_certain' in results_unseen else 0.

        result_df.append((threshold, tp, fu, method_name))
    result_df = pd.DataFrame(result_df, columns=['threshold', 'Seen Cell True Prediction', 'Unseen Cell Assignment', 'Method'])
    return result_df


def plot_uncert_threshold(result_df, xlim=(0., 1.), ylim=(0., 1.)):
    sns.set(rc={'figure.figsize':(5,5)})
    with sns.axes_style("whitegrid"):
        X = result_df.copy()
        sns.lineplot(x='Unseen Cell Assignment', y='Seen Cell True Prediction', 
                     hue='Method', data=X, drawstyle='steps-post')
        plt.xticks(rotation=90)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
