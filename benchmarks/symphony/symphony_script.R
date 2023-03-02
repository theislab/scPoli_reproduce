#! /usr/bin/Rscript
options(show.error.locations = TRUE)

args <- commandArgs(trailingOnly = TRUE)
print(args)
data <- args[1]

.libPaths(.libPaths()[2])
library("symphony")
library("anndata")
source('symphony_fun.R')
options(future.globals.maxSize = (2000000*1024^2))

DATA_DIR = paste(
    '/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/data/',
    data,
    '.h5ad',
    sep=""
)
RES_PATH =  paste(
    '/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/results/seurat/',
    data,
    sep=""
)

if (data == 'pancreas') {
    reference <- c(
        "inDrop1", 
        "inDrop2", 
        "inDrop3", 
        "inDrop4", 
        "fluidigmc1", 
        "smartseq2", 
        "smarter"
    )
    query <- c(
        'celseq',
        'celseq2'
    )
    batch_key <- "study"
    ct_key <- "cell_type"
} else if (data == 'pbmc') {
    reference <- c(
        "Oetjen", 
        "10X", 
        "Sun"
    )
    query <- c(
        'Freytag'
    )
    batch_key <- "study"
    ct_key <- "cell_type"
} else if (data == 'brain') {
    reference <- c(
        'Rosenberg', 
        'Saunders'
    )
    query <- c(
        'Zeisel', 
        'Tabula_muris'
    )
    batch_key <- "study"
    ct_key <- "cell_type"
} else if (data == 'scvelo') {
    reference <- c(
        '12.5', 
        '13.5'
    )
    query <- c(
        '14.5', 
        '15.5'
    )
    batch_key <- "study"
    ct_key <- "cell_type"
} else if (data == 'lung') {
    reference <- c(
        'Dropseq_transplant',
        '10x_Biopsy'
    )
    query <- c(
        '10x_Transplant'
    )
    batch_key <- "study"
    ct_key <- "cell_type"
} else if (data == 'tumor') {
    reference <- c(
        'breast', 
        'colorectal', 
        'liver2', 
        'liver1', 
        'lung1', 
        'lung2', 
        'multiple', 
        'ovary',
        'pancreas', 
        'skin'
    )
    query <- c(
        'melanoma1',
        'melanoma2',
        'uveal melanoma'
    )
    batch_key <- "study"
    ct_key <- "cell_type"
}
batches <- c(reference, query)

project_symphony(DATA_DIR, batch_key, query)