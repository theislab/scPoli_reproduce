library(Seurat)
library(anndata)
library(Matrix)

# For multiple queries just projects every query to the reference.
# .X should have unnormalized counts.
project_only_seurat <- function(adata_file, batch_col, query_names, label_col=NULL, dim=50)
{

  print("Reading data.")

  ad <- read_h5ad(adata_file)

  # Sort ad by batch with queries last
  batch_names <- as.character(unique(ad$obs[[batch_col]]))
  is_query <- batch_names %in% query_names
  batch_names <- c(sort(batch_names[!is_query]), sort(batch_names[is_query]))
  batch_order <- order(factor(ad$obs[[batch_col]], levels = batch_names))
  ad <- ad[batch_order, ]

  se <- CreateSeuratObject(counts=t(as(ad$X, "CsparseMatrix")))
  VariableFeatures(se) <- rownames(se)
  se[[batch_col]] <- ad$obs[[batch_col]]

  do_transfer <- !is.null(label_col)

  if (do_transfer) {
      se[[label_col]] <- ad$obs[[label_col]]
      n_ref <- sum(!(se[[]][[batch_col]] %in% query_names))
  }

  datas <- SplitObject(se, split.by = batch_col)
  rm(se)

  for (i in 1:length(datas))
  {
    datas[[i]] <- NormalizeData(datas[[i]], verbose = FALSE)
  }

  q_mask <- names(datas) %in% query_names

  refs <- datas[!q_mask]
  queries <- datas[q_mask]
  rm(datas)

  print("Integrating reference:")
  print(names(refs))
  anchors_refs <- FindIntegrationAnchors(object.list = refs, dims = 1:dim)
  rm(refs)
  ref <- IntegrateData(anchorset = anchors_refs, dims = 1:dim)
  rm(anchors_refs)


  ref <- ScaleData(ref)
  ref <- RunPCA(ref, npcs = dim)
  ref <- FindNeighbors(ref, reduction = "pca", dims = 1:dim, graph.name = "snn")
  ref <- RunSPCA(ref, npcs = dim, graph = "snn")
  ref <- FindNeighbors(ref, reduction = "spca", dims = 1:dim, graph.name = "spca.nn",
                       k.param = 50, cache.index = TRUE, return.neighbor = TRUE, l2.norm = TRUE)

  latent <- Embeddings(ref, reduction = "spca")

  if (do_transfer) {
      pred_labels <- rep(NA, n_ref)
      pred_scores <- rep(NA, n_ref)
  }

  for (i in 1:length(queries))
  {
    query <- queries[i]
    print("Mapping query to reference:")
    print(names(query))
    query <- query[[1]]
    anchors_query <- FindTransferAnchors(reference = ref, query = query, reference.reduction = "spca",
                                         reference.neighbors = "spca.nn", dims = 1:dim)

    if (do_transfer) {
        query <- TransferData(anchorset = anchors_query, reference = ref, query = query, refdata = list(label = label_col)
        )

        pred_labels <- c(pred_labels, query$predicted.label)
        pred_scores <- c(pred_scores, query$predicted.label.score)
    }

    query <- IntegrateEmbeddings(anchorset = anchors_query, reference = ref, query = query, reductions = "pcaproject",
                                 dims = 1:dim, new.reduction.name = "qrmapping", reuse.weights.matrix = FALSE)
    rm(anchors_query)

    latent <- rbind(latent, Embeddings(query, reduction = "qrmapping"))
  }

  ad$obsm[["X_seurat"]] <- latent[ad$obs_names,]

  if (do_transfer) {
      ad$obs$pred_label <- pred_labels
      ad$obs$pred_score <- pred_scores
  }

  ad$write_h5ad(adata_file)

}