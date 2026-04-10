#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for dimensionality reduction & clustering.

@author: E. N. Aslinger
"""

from warnings import warn
import scanpy as sc
# import scipy.sparse as sp
from scipy import sparse
try:
    import rapids_singlecell as rsc
    # import cupyx
except Exception:
    rsc = None
    warn("Cannot import rapids_singlecell.")
import numpy as np


def cluster(adata, col_celltype="leiden", seed=0,
            n_comps=None, resolution=1, min_dist=0.5, spread=1,
            kws_pca=None, kws_neighbors=None, layer="log1p",
            kws_cluster=None, kws_umap=None, plot=True,
            use_rapids=True, use_highly_variable=True, inplace=False):
    """Cluster omics data."""
    if inplace is False:
        adata = adata.copy()
    use_rapids = rsc is not None and use_rapids is True
    kws_pca, kws_neighbors, kws_cluster, kws_umap = [
        {} if x is None else {**x} if isinstance(x, dict) else x for x in [
            kws_pca, kws_neighbors, kws_cluster, kws_umap]]  # empty if no kws
    if kws_pca is not False:
        if "use_highly_variable" in kws_pca or "mask_var" in kws_pca:
            raise ValueError(
                "The 'use_highly_variable' keyword argument "
                "should be passed directly to the `cluster()` function, "
                "and any 'mask_var' argument specification should be passed "
                "via 'use_highly_variable`.  Neither should be "
                "contained in the `kws_pca` dictionary.")
        kws_pca["use_highly_variable"] = use_highly_variable
        if use_highly_variable is False or isinstance(use_highly_variable, (
                list, np.ndarray, tuple)):
            kws_pca["mask_var"] = None if (
                use_highly_variable is False) else use_highly_variable
    if layer is not None:
        print(f"\t***Storing layer '{layer}' in `.X`...")
        adata.X = adata.layers[layer].copy()
    else:
        print("\t***Using layer already stored in `.X`")
    if use_rapids is True:
        print("\t***Moving `.X` to GPU for `rapids`...")
        if hasattr(adata.X, "get"):
            adata.X = adata.X.get()
        # adata.X = sp.csr_matrix(np.asarray(adata.X) if hasattr(
        #     adata.X, "toarray") else adata.X)
        if (sparse.isspmatrix_csc(adata.X) or sparse.isspmatrix_csr(
                adata.X)) is False:
            adata.X = sparse.csr_matrix(adata.X)
        # adata.X = cupyx.scipy.sparse.csr_matrix(adata.X)
        rsc.get.anndata_to_GPU(adata)  # move to GPU
    else:
        if layer is not None:
            adata.X = adata.layers[layer].copy()
    if kws_pca is not False:
        print(f"\t***Calculating PCA with {n_comps} components "
              f"(seed={seed})...")
        (rsc if use_rapids else sc).pp.pca(
            adata, n_comps=n_comps, random_state=seed, **kws_pca)  # PCA
        if plot is True:
            sc.pl.pca_variance_ratio(adata, log=True)
    else:
        print("\t***Using pre-existing PCA as `kws_pca=False`...")
    n_n = " with " + str(kws_neighbors["n_neighbors"]) + " neighbors" if (
        "n_neighbors" in kws_neighbors) else ""
    print(f"\t***Building neighborhood{n_n} (seed={seed})...")
    (rsc if use_rapids else sc).pp.neighbors(
        adata, random_state=seed, **kws_neighbors)  # neighbors
    print(f"\t***Embedding UMAP with minimum distance {min_dist} & "
          f"spread {spread} (seed={seed})...")
    (rsc if use_rapids else sc).tl.umap(
        adata, min_dist=min_dist, spread=spread,
        random_state=seed, **kws_umap)  # UMAP
    print(f"\t***Performing Leiden clustering with resolution {resolution}"
          f" (seed={seed})...")
    (rsc if use_rapids else sc).tl.leiden(
        adata, key_added=col_celltype,
        resolution=resolution, random_state=seed, **kws_cluster)  # cluster
    if use_rapids is True:
        # adata.X = adata.X.get()
        rsc.get.anndata_to_CPU(adata)  # move back to CPU
    return adata
