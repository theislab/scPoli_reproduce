#!/bin/bash

DATA=("brain")
for d in "${DATA[@]}"; do Rscript seurat_script_labeltransfer.R "$d"; done