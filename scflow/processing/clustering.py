#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for dimensionality reduction & clustering.

@author: E. N. Aslinger
"""

import os
import warnings
import anndata
import scanpy as sc


def cluster(adata, col_celltype="leiden", n_comps=None,
            kws_pca=None, kws_neighbors=None,
            kws_cluster=None, kws_umap=None, plot=True, inplace=False):
    """Cluster omics data."""
    kws_pca, kws_neighbors, kws_cluster, kws_umap = [
        {} if x is None else x for x in [
            kws_pca, kws_neighbors,
            kws_cluster, kws_umap]]  # empty dictionaries if no kws
    if inplace is False:
        adata = adata.copy()
    if kws_pca is not False:
        sc.pp.pca(adata, n_comps=n_comps, **kws_pca)  # PCA
        sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, **kws_neighbors)  # neighbors
    sc.tl.umap(adata, **kws_umap)  # UMAP
    sc.tl.leiden(adata, key_added=col_celltype, **kws_cluster)  # Leiden
    return adata
