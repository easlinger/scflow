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


def cluster(adata, key_added="leiden", n_comps=None,
            kws_pca=None, kws_neighbors=None, inplace=False):
    """Cluster omics data."""
    kws_pca, kws_neighbors, kws_cluster = [{} if x is None else x for x in [
        kws_pca, kws_neighbors, kws_cluster]]  # empty dictionaries if no kws
    if inplace is False:
        adata = adata.copy()
    sc.pp.pca(adata, n_comps=n_comps, **kws_pca)  # PCA
    sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, **kws_neighbors)  # neighbors
    sc.tl.leiden(adata, key_added=key_added, **kws_cluster)  # clustering
    return adata
