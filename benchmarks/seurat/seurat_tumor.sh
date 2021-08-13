#!/bin/bash

DATA=("tumor")
for d in "${DATA[@]}"; do Rscript seurat_script_labeltransfer.R "$d"; done