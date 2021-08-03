#!/bin/bash

DATA=("pancreas" "pbmc" "scvelo" "tumor" "lung" "brain")

for d in "${DATA[@]}"; do Rscript symphony_script.R "$d"; done