#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data manipulation.

@author: E. N. Aslinger
"""

import scanpy as sc
import pertpy as pt


def create_pseudobulk(adata, col_groupings, col_celltype,
                      layer="counts", mode="sum"):
    """Create pseudobulk data from single-cell data."""
    adata = adata.copy()
    if isinstance(col_groupings, str):
        col_groupings = [col_groupings]
    # TODO: PULL out into separate pseudo-bulk function
    sep = " *** "
    adata.X = adata.layers[layer].copy()
    if any([adata.obs[x].apply(lambda x: sep in x).any(
            ) for x in [i for i in col_groupings if i]]):
        raise ValueError(f"{sep} = a reserved separator but"
                         "found in `col_condition` or `col_covariate`")
    adata.obs.loc[:, "PBULK_GROUPING"] = adata.obs[col_groupings[0]]
    if len(col_groupings) > 1:
        for x in col_groupings[1:]:  # make combined columns
            adata.obs.loc[:, "PBULK_GROUPING"] = adata.obs[
                "PBULK_GROUPING"].astype(str) + sep + adata.obs[x].astype(str)
    col_target = "PBULK_GROUPING"  # sample | condition | covariate
    p_s = pt.tl.PseudobulkSpace()
    adata = p_s.compute(
        adata, target_col=col_target, groups_col=col_celltype,
        layer_key=layer, mode=mode)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata)
    adata.X = adata.layers["counts"].copy()
    for i, x in enumerate(col_groupings):  # re-split grouping columns
        adata.obs.loc[:, x] = adata.obs[col_target].apply(
            lambda x: x.split(sep)[i])
    adata.obs = adata.obs.dropna(how="all", axis=1)
    return adata
