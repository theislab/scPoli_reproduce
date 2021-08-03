#!/bin/bash

DATA=("pancreas" "pbmc" "scvelo" "tumor" "lung" "brain")

for d in "${DATA[@]}"; do Rscript seurat_script.R "$d"; done