EXPERIMENT_INFO = {
    "pancreas": {
        "file_name": "pancreas.h5ad",
        "condition_key": "study",
        "cell_type_key": ["cell_type"],
        "reference": [
            "inDrop1",
            "inDrop2",
            "inDrop3",
            "inDrop4",
            "fluidigmc1",
            "smartseq2",
            "smarter",
        ],
        "query": ["celseq", "celseq2"],
    },
    "pbmc": {
        "file_name": "pbmc.h5ad",
        "condition_key": "study",
        "cell_type_key": ["cell_type"],
        "reference": [
            "Oetjen",
            "10X",
            "Sun",
        ],
        "query": [
            "Freytag",
        ],
    },
    "brain": {
        "file_name": "brain.h5ad",
        "condition_key": "study",
        "cell_type_key": ["cell_type"],
        "reference": [
            "Rosenberg",
            "Saunders",
        ],
        "query": ["Zeisel", "Tabula_muris"],
    },
    "scvelo": {
        "file_name": "scvelo.h5ad",
        "condition_key": "study",
        "cell_type_key": ["cell_type"],
        "reference": [
            "12.5",
            "13.5",
        ],
        "query": ["14.5", "15.5"],
    },
    "lung": {
        "file_name": "lung.h5ad",
        "condition_key": "study",
        "cell_type_key": ["cell_type"],
        "reference": [
            "Dropseq_transplant",
            "10x_Biopsy",
        ],
        "query": [
            "10x_Transplant",
        ],
    },
    "tumor": {
        "file_name": "tumor.h5ad",
        "condition_key": "study",
        "cell_type_key": ["cell_type"],
        "reference": [
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
        ],
        "query": ["melanoma1", "melanoma2", "uveal melanoma"],
    },
    "lung_h_sub": {
        "file_name": "adata_lung_subsampled.h5ad",
        "condition_key": "study",
        "cell_type_key": ["ann_level_1", "ann_level_2"],
        "reference": ["Stanford_Krasnow_bioRxivTravaglini", "Misharin_new"],
        "query": [
            "Vanderbilt_Kropski_bioRxivHabermann_vand",
            "Sanger_Teichmann_2019VieiraBraga",
        ],
    },
}
