#!/bin/bash

#DATA=("pancreas" "pbmc" "scvelo" "tumor" "lung" "brain")
DATA=("brain")
for d in "${DATA[@]}"; do Rscript seurat_script_labeltransfer.R "$d"; done