library(Seurat)
library(anndata)

# For multiple queries integrates these queries and then projects the integrated query object
# to the reference.
# .X should have unnormalized counts.
project_seurat <- function(adata_file, batch_col, query_names, dim=50)
{

  ad <- read_h5ad(adata_file)

  se <- CreateSeuratObject(counts=t(ad$X))
  VariableFeatures(se) <- rownames(se)
  se[[batch_col]] <- ad$obs[[batch_col]]

  datas <- SplitObject(se, split.by = batch_col)
  rm(se)

  for (i in 1:length(datas))
  {
    datas[[i]] <- NormalizeData(datas[[i]], verbose = FALSE)
  }

  q_mask <- names(datas) %in% query_names
  print(names(datas))  
  print(q_mask)
  refs <- datas[!q_mask]
  queries <- datas[q_mask]
  rm(datas)

  print("Integrating reference:")
  print(names(refs))
  anchors_refs <- FindIntegrationAnchors(object.list = refs, dims = 1:dim)
  rm(refs)
  ref <- IntegrateData(anchorset = anchors_refs, dims = 1:dim)
  rm(anchors_refs)

  if(length(queries) > 1)
  {
    print("Integrating query:")
    print(names(queries))
    anchors_qs <- FindIntegrationAnchors(object.list = queries, dims = 1:dim)
    rm(queries)
    query <- IntegrateData(anchorset = anchors_qs, dims = 1:dim)
    rm(anchors_qs)
  }
  else
  {
    query <- queries[[1]]
    rm(queries)
  }

  ref <- ScaleData(ref)
  ref <- RunPCA(ref, npcs = dim)
  ref <- FindNeighbors(ref, reduction = "pca", dims = 1:dim, graph.name = "snn")
  ref <- RunSPCA(ref, npcs = dim, graph = "snn")
  ref <- FindNeighbors(ref, reduction = "spca", dims = 1:dim, graph.name = "spca.nn",
                       k.param = 50, cache.index = TRUE, return.neighbor = TRUE, l2.norm = TRUE)

  print("Mapping query to reference.")
  anchors_query <- FindTransferAnchors(reference = ref, query = query, reference.reduction = "spca",
                                       reference.neighbors = "spca.nn", dims = 1:dim)

  query <- IntegrateEmbeddings(anchorset = anchors_query, reference = ref, query = query, reductions = "pcaproject",
                               dims = 1:dim, new.reduction.name = "qrmapping", reuse.weights.matrix = FALSE)
  rm(anchors_query)

  latent <- rbind(Embeddings(ref, reduction = "spca"), Embeddings(query, reduction = "qrmapping"))
  ad$obsm[["X_seurat"]] <- latent[ad$obs_names,]

  ad$write_h5ad(adata_file)

}