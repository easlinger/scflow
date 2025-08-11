#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for dimensionality reduction & clustering.

@author: E. N. Aslinger
"""

from warnings import warn
import scanpy as sc
try:
    import rapids_singlecell as rsc
except Exception:
    rsc = None
    warn("Cannot import rapids_singlecell.")


def cluster(adata, col_celltype="leiden", seed=0,
            n_comps=None, resolution=1, min_dist=0.5,
            kws_pca=None, kws_neighbors=None, layer="log1p",
            kws_cluster=None, kws_umap=None, plot=True, inplace=False):
    """Cluster omics data."""
    kws_pca, kws_neighbors, kws_cluster, kws_umap = [
        {} if x is None else x for x in [
            kws_pca, kws_neighbors,
            kws_cluster, kws_umap]]  # empty dictionaries if no kws
    if inplace is False:
        adata = adata.copy()
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    if rsc is not None:
        rsc.get.anndata_to_GPU(adata)
    if kws_pca is not False:
        print(f"***Calculating PCA with {n_comps} components...")
        (rsc if rsc else sc).pp.pca(adata, n_comps=n_comps, **kws_pca)  # PCA
        if plot is True:
            sc.pl.pca_variance_ratio(adata, log=True)
    print("***Constructing neighborhood...")
    (rsc if rsc else sc).pp.neighbors(adata, random_state=seed,
                                      **kws_neighbors)  # neighbors
    print(f"***Embedding UMAP with minimum distance {min_dist}...")
    (rsc if rsc else sc).tl.umap(adata, min_dist=min_dist, **kws_umap)  # UMAP
    print(f"***Performing Leiden clustering with resolution {resolution}...")
    (rsc if rsc else sc).tl.leiden(
        adata, key_added=col_celltype,
        resolution=resolution, random_state=seed, **kws_cluster)  # cluster
    if rsc is not None:
        rsc.get.anndata_to_CPU(adata)  # move backj to cpu
    return adata
