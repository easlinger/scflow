#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for dimensionality reduction & clustering.

@author: E. N. Aslinger
"""

import scanpy as sc


def cluster(adata, col_celltype="leiden",
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
    if kws_pca is not False:
        print("***Calculating PCA...")
        sc.pp.pca(adata, n_comps=n_comps, **kws_pca)  # PCA
        if plot is True:
            sc.pl.pca_variance_ratio(adata, log=True)
    print("***Constructing neighborhood...")
    sc.pp.neighbors(adata, **kws_neighbors)  # neighbors
    print("***Embedding UMAP...")
    sc.tl.umap(adata, min_dist=min_dist, **kws_umap)  # UMAP
    print(f"***Performing Leiden clustering with resolution {resolution}...")
    sc.tl.leiden(adata, key_added=col_celltype,
                 resolution=resolution, **kws_cluster)  # Leiden
    return adata
