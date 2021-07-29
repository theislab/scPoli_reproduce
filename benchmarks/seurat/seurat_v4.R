library("Seurat")
library("SeuratDisk")
library("future")
options(future.globals.maxSize = (2000000*1024^2))

nHVGs <- 4000
verbose <- TRUE
ncpu <- 4
dimr <- 150
dimqm <- 50
savedir = "~/Documents/seurat_benchmarks/brain/"
datadir <- "~/Documents/benchmarking_datasets/benchmark_brain_shrinked.h5ad"
reference <- c('Rosenberg', 'Saunders')
query <- c('Zeisel', 'Tabula_muris')
batch_key <- "study"
ct_key <- "cell_type"

seudatdir <- paste(substr(datadir, 1, nchar(datadir)-4),"h5seurat",sep="")
batches <- c(reference, query)

Convert(datadir, dest= "h5seurat", assay = "RNA", overwrite  = T, verbose = verbose)
seudata <- LoadH5Seurat(seudatdir, assays = "RNA", verbose = verbose)
print(seudata@assays)

seudata_batches <- SplitObject(seudata, split.by = batch_key)
seudata_batches <- seudata_batches[batches]
seu_integ_feat <- SelectIntegrationFeatures(object.list = seudata_batches, 
                                            nfeatures = nHVGs,
                                            fvf.nfeatures = nHVGs,
                                            verbose = verbose)
seudata_ref <- seudata_batches[reference]
seudata_que <- seudata_batches[query]

integ_anchors <- FindIntegrationAnchors(object.list = seudata_ref,
                                        dims = 1:dimr,
                                        anchor.features = seu_integ_feat,
                                        normalization.method = 'LogNormalize', 
                                        reduction = 'cca', 
                                        verbose = verbose)

seudata_ref_integ <- IntegrateData(anchorset = integ_anchors,
                                   normalization.method = "LogNormalize",
                                   features = seu_integ_feat,
                                   new.assay.name = "reference", 
                                   dims = 1:dimr, 
                                   verbose = verbose)
seudata_ref_integ <- ScaleData(seudata_ref_integ, 
                               features = seu_integ_feat, 
                               verbose = verbose)
seudata_ref_integ <- RunPCA(seudata_ref_integ,
                            npcs = dimqm,
                            features = seu_integ_feat, 
                            reduction.name = "pca", 
                            verbose = verbose)
seudata_ref_integ <- FindNeighbors(seudata_ref_integ, 
                                   reduction = "pca",
                                   dims = 1:dimqm,
                                   features = seu_integ_feat, 
                                   graph.name = "snn", 
                                   verbose = verbose)
seudata_ref_integ <- RunSPCA(seudata_ref_integ,
                             features = seu_integ_feat,
                             npcs = dimqm,
                             graph = "snn", 
                             verbose = verbose)
seudata_ref_integ <- FindNeighbors(seudata_ref_integ, 
                                   reduction = "spca",
                                   features = seu_integ_feat,
                                   dims = 1:dimqm,
                                   graph.name = "spca.annoy.neighbors", 
                                   k.param = dimqm,
                                   cache.index = TRUE,
                                   return.neighbor = TRUE,
                                   l2.norm = TRUE,
                                   verbose = verbose)

transfer_anchors <- list()
for(i in 1:length(seudata_que)){
  transfer_anchors[[i]] <- FindTransferAnchors(reference = seudata_ref_integ,
                                               query = seudata_que[[i]],
                                               reference.assay = "reference",
                                               features = seu_integ_feat, 
                                               normalization.method = 'LogNormalize',
                                               reference.reduction = "spca",
                                               reference.neighbors = "spca.annoy.neighbors",
                                               dims = 1:dimqm,
                                               k.anchor = 12,
                                               n.trees = 70,
                                               verbose = verbose)
  seudata_que[[i]] <- IntegrateEmbeddings(anchorset = transfer_anchors[[i]],
                                          reference = seudata_ref_integ,
                                          query = seudata_que[[i]],
                                          reductions = "pcaproject",
                                          dims = 1:dimqm,
                                          new.reduction.name = "qrmapping",
                                          reuse.weights.matrix = FALSE,
                                          verbose = verbose)
  predictions <- TransferData(anchorset = transfer_anchors[[i]],
                              refdata = seudata_ref_integ$cell_type,
                              dims = 1:dimqm)
  seudata_que[[i]] <- AddMetaData(seudata_que[[i]], metadata = predictions)

}
seudata_que[[1]]$prediction.match <- seudata_que[[1]]$predicted.id == seudata_que[[1]]$cell_type
table(seudata_que[[1]]$prediction.match)
seudata_que[[2]]$prediction.match <- seudata_que[[2]]$predicted.id == seudata_que[[2]]$cell_type
table(seudata_que[[2]]$prediction.match)
print(seudata_que[[1]])
print(seudata_que[[2]])
print(seudata_ref_integ)
SaveH5Seurat(seudata_ref_integ, filename = paste(savedir,"ref.h5Seurat",sep=""))
Convert(paste(savedir,"ref.h5Seurat",sep=""), dest = "h5ad")

SaveH5Seurat(seudata_que[[1]], filename = paste(savedir,"q1.h5Seurat",sep=""))
Convert(paste(savedir,"q1.h5Seurat",sep=""), dest = "h5ad")

SaveH5Seurat(seudata_que[[2]], filename = paste(savedir,"q2.h5Seurat",sep=""))
Convert(paste(savedir,"q2.h5Seurat",sep=""), dest = "h5ad")
