#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for dimensionality reduction & clustering.

@author: E. N. Aslinger
"""

from warnings import warn
import scanpy as sc
import scipy.sparse as sp
try:
    import rapids_singlecell as rsc
    import cupyx
except Exception:
    rsc = None
    warn("Cannot import rapids_singlecell.")
import numpy as np


def cluster(adata, col_celltype="leiden", seed=0,
            n_comps=None, resolution=1, min_dist=0.5, spread=1,
            kws_pca=None, kws_neighbors=None, layer="scaled",
            kws_cluster=None, kws_umap=None, plot=True, inplace=False):
    """Cluster omics data."""
    kws_pca, kws_neighbors, kws_cluster, kws_umap = [
        {} if x is None else x for x in [
            kws_pca, kws_neighbors,
            kws_cluster, kws_umap]]  # empty dictionaries if no kws
    if inplace is False:
        adata = adata.copy()
    if layer is not None:
        print(f"\t***Storing layer {layer} in `.X`...")
        adata.X = adata.layers[layer].copy()
    else:
        print("\t**Using layer already stored in `.X`")
    if rsc is not None:
        print("\t***Moving `.X` to GPU for `rapids`...")
        if hasattr(adata.X, "get"):
            adata.X = adata.X.get()
        adata.X = sp.csr_matrix(np.asarray(adata.X) if hasattr(
            adata.X, "toarray") else adata.X)
        # adata.X = cupyx.scipy.sparse.csr_matrix(adata.X)
        rsc.get.anndata_to_GPU(adata)  # move to GPU
    else:
        if layer is not None:
            adata.X = adata.layers[layer].copy()
    if kws_pca is not False:
        print(f"\t***Calculating PCA with {n_comps} components...")
        (rsc if rsc else sc).pp.pca(adata, n_comps=n_comps, **kws_pca)  # PCA
        if plot is True:
            sc.pl.pca_variance_ratio(adata, log=True)
    n_n = " with " + str(kws_neighbors["n_neighbors"]) + " neighbors" if (
        "n_neighbors" in kws_neighbors) else ""
    print(f"\t***Building neighborhood{n_n}...")
    (rsc if rsc else sc).pp.neighbors(adata, random_state=seed,
                                      **kws_neighbors)  # neighbors
    print(f"\t***Embedding UMAP with minimum distance {min_dist} & "
          f"spread {spread}...")
    (rsc if rsc else sc).tl.umap(adata, min_dist=min_dist, spread=spread,
                                 random_state=seed, **kws_umap)  # UMAP
    print(f"\t***Performing Leiden clustering with resolution {resolution}...")
    (rsc if rsc else sc).tl.leiden(
        adata, key_added=col_celltype,
        resolution=resolution, random_state=seed, **kws_cluster)  # cluster
    if rsc is not None:
        # adata.X = adata.X.get()
        rsc.get.anndata_to_CPU(adata)  # move back to CPU
    return adata
