library(symphony)
library(anndata)

# .X should have unnormalized counts.
project_symphony <- function(adata_file, batch_col, query_names, dim=20)
{

  ad <- read_h5ad(adata_file)

  q_mask <- ad$obs[[batch_col]] %in% query_names

  ref_X <- t(ad$X[!q_mask,])
  query_X <- t(ad$X[q_mask,])

  ref_meta <- ad$obs[!q_mask,][batch_col]
  query_meta <- ad$obs[q_mask,][batch_col]

  print("Integrating reference:")
  print(unique(ref_meta[[1]]))

  reference <- buildReference(
      ref_X,
      ref_meta,
      vars = batch_col,
      K = 100,
      verbose = TRUE,
      do_umap = FALSE,
      do_normalize = TRUE,
      vargenes_method = 'vst',
      topn = 2000,
      d = dim
  )

  print("Projecting query:")
  print(unique(query_meta[[1]]))

  query <- mapQuery(
      query_X,
      query_meta,
      reference,
      do_normalize = TRUE,
      do_umap = FALSE
  )

  latent <- t(cbind(reference[["Z_corr"]], query[["Z"]]))
  ad$obsm[["X_symphony"]] <- latent[ad$obs_names,]

  ad$write_h5ad(adata_file)

}