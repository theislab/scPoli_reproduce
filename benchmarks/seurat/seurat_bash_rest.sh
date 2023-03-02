#!/bin/bash

DATA=("pancreas" "pbmc" "scvelo" "lung")
for d in "${DATA[@]}"; do Rscript seurat_script_labeltransfer.R "$d"; done