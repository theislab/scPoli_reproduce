library("Seurat")
library("SeuratDisk")
library("future")

options(future.globals.maxSize = (2000000*1024^2))

nHVGs <- 4000
verbose <- TRUE
ncpu <- 4
dimr <- 150
dimqm <- 50
savedir = "~/Documents/seurat_benchmarks/scvelo/"
datadir <- "~/Documents/benchmarking_datasets/benchmark_scvelo_shrinked.h5ad"
reference <- c('12.5', '13.5')
query <- c('14.5', '15.5')
batch_key <- "study"
ct_key <- "cell_type"

seudatdir <- paste(substr(datadir, 1, nchar(datadir)-4),"h5seurat",sep="")
batches <- c(reference, query)

Convert(datadir, dest= "h5seurat", assay = "RNA", overwrite  = T)
seudata <- LoadH5Seurat(seudatdir, assays = "RNA")

seudata.list <- SplitObject(seudata, split.by = batch_key)
seudata.list <- seudata.list[batches]

for (i in 1:length(seudata.list)) {
  seudata.list[[i]] <- NormalizeData(seudata.list[[i]])
  seudata.list[[i]] <- FindVariableFeatures(seudata.list[[i]], 
                                            selection.method = "vst", 
                                            nfeatures = nHVGs)
  seudata.list[[i]] <- ScaleData(seudata.list[[i]])
  seudata.list[[i]] <- RunPCA(seudata.list[[i]])
}

reference.list <- seudata.list[reference]
query.list <- seudata.list[query]

start.time <- Sys.time()

seudata.anchors <- FindIntegrationAnchors(object.list = reference.list,
                                          reduction = "rpca",
                                          dims = 1:dimr)

seudata.integrated <- IntegrateData(anchorset = seudata.anchors,
                                    dims = 1:dimr)

seudata.integrated <- ScaleData(seudata.integrated)
seudata.integrated <- RunPCA(seudata.integrated, npcs = dimqm)
seudata.integrated <- FindNeighbors(seudata.integrated, 
                                   reduction = "pca",
                                   dims = 1:dimqm,
                                   graph.name = "snn")
end.time <- Sys.time()
time.taken.ref <- end.time - start.time

fileConn<-file(paste(savedir,"ref_time.txt",sep=""))
writeLines(as.character(time.taken.ref), fileConn)
close(fileConn)

start.time <- Sys.time()
for(i in 1:length(query.list)){
  query.anchors <- FindTransferAnchors(reference = seudata.integrated,
                                       query = query.list[[i]],
                                       reference.reduction = "pca", 
                                       reference.neighbors = "snn",
                                       dims = 1:dimqm)
  
  query.list[[i]] <- IntegrateEmbeddings(anchorset = query.anchors,
                                         reference = seudata.integrated,
                                         query = query.list[[i]],
                                         reductions = "pcaproject",
                                         dims = 1:dimqm,
                                         new.reduction.name = "qrmapping",
                                         reuse.weights.matrix = FALSE)
  
  predictions <- TransferData(anchorset = query.anchors,
                              refdata = seudata.integrated$cell_type,
                              dims = 1:dimqm)
  query.list[[i]] <- AddMetaData(query.list[[i]], metadata = predictions)

}
end.time <- Sys.time()
time.taken.q <- end.time - start.time

fileConnq<-file(paste(savedir,"q_time.txt",sep=""))
writeLines(as.character(time.taken.q), fileConnq)
close(fileConnq)

print(seudata.integrated)
SaveH5Seurat(seudata.integrated, filename = paste(savedir,"ref.h5Seurat",sep=""))
Convert(paste(savedir,"ref.h5Seurat",sep=""), dest = "h5ad")

for(i in 1:length(query.list)){
  query.list[[i]]$prediction.match <- query.list[[i]]$predicted.id == query.list[[i]]$cell_type
  print(table(query.list[[i]]$prediction.match))
  print(query.list[[i]])
  query.name <- paste("q", i,".h5seurat",sep="")
  SaveH5Seurat(query.list[[i]], filename = paste(savedir, query.name,sep=""))
  Convert(paste(savedir, query.name,sep=""), dest = "h5ad")
}
