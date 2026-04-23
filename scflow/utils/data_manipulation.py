#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data manipulation.

@author: E. N. Aslinger
"""

import scanpy as sc
import pertpy as pt
import decoupler as dc


def create_pseudobulk(adata, col_groupings, col_celltype,
                      layer="counts", mode="sum", method="pertpy",
                      min_cells=0, min_counts=0,
                      perform_preprocessing=True,
                      target_sum=1e4, max_value=10, zero_center=True):
    """Create pseudobulk data from single-cell data."""
    # adata = adata.copy()
    print(f">>>Pseudo-bulking data with grouping columns {col_groupings} "
          f"and cell type column {col_celltype} (`mode='{mode}'`)")
    if isinstance(col_groupings, str):
        col_groupings = [col_groupings]
    # TODO: PULL out into separate pseudo-bulk function
    sep = " *** "
    # adata.X = adata.layers[layer].copy()
    if any([adata.obs[x].apply(lambda x: sep in x).any(
            ) for x in [i for i in col_groupings if i]]):
        raise ValueError(f"{sep} = a reserved separator but"
                         "found in `col_condition` or `col_covariate`")
    if method == "pertpy":
        adata = adata.copy()
        adata.obs.loc[:, "PBULK_GROUPING"] = adata.obs[col_groupings[0]]
        if len(col_groupings) > 1:
            print(f"\t***Creating combined group column with {col_groupings}")
            for x in col_groupings[1:]:  # make combined columns
                adata.obs.loc[:, "PBULK_GROUPING"] = adata.obs[
                    "PBULK_GROUPING"].astype(str) + sep + adata.obs[
                        x].astype(str)
            print(adata.obs["PBULK_GROUPING"].head())
        col_target = "PBULK_GROUPING"  # sample | condition | covariate
        print(f"\t***Using `pertpy` pseudo-bulking on layer {layer}...")
        if min_cells != 0 or min_counts != 0:
            raise ValueError("Cannot perform `min_cells` or `min_counts` "
                             "filtering with `pertpy` method")
        p_s = pt.tl.PseudobulkSpace()
        adata = p_s.compute(
            adata, target_col=col_target, groups_col=col_celltype,
            layer_key=layer, mode=mode)
    else:
        print(f"\t***Using `decoupler` pseudo-bulking on layer '{layer}'...")
        col_target = None
        grps = col_celltype + list(col_groupings[1:] if len(
            col_groupings) > 1 else [])
        if any((i not in [col_groupings[0]] + grps for i in adata.obs)):
            print(f"\t***Subsetting `.obs` to {[col_groupings[0]] + grps}...")
            adata = adata.copy()
            adata.obs = adata.obs[[col_groupings[0]] + grps]
        adata = dc.pp.pseudobulk(
            adata=adata, sample_col=col_groupings[0],
            groups_col=grps, mode="sum", layer=layer)
        if min_cells != 0 or min_counts != 0:
            try:
                dc.pl.filter_samples(
                    adata=adata, groupby=col_groupings, min_cells=min_cells,
                    min_counts=min_counts, figsize=(5, 8))
            except Exception as err:
                print(f"\t`dc.pl.filter_samples()` failed: {err}")
            dc.pp.filter_samples(
                adata=adata, groupby=col_groupings, min_cells=min_cells,
                min_counts=min_counts, figsize=(5, 8))
            try:
                dc.pl.obsbar(
                    adata=adata, y=col_groupings[0],
                    hue=col_groupings[1] if len(col_groupings) > 1 else None,
                    figsize=(6, 3))
            except Exception as err:
                print(f"\t`dc.pl.obsbar()` failed: {err}")
    adata.layers["counts"] = adata.X.copy()
    if perform_preprocessing is True:
        print(f"\t***Total count-normalizing pseudobulk data "
              f"(`target_sum={target_sum}`)...")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        print("\t***Log-transforming pseudo-bulked data...")
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()
        print(f"\t***Scaling pseudo-bulked data (`max_value={max_value}`, "
              f"`zero_center={zero_center}`)...")
        sc.pp.scale(adata, max_value=max_value, zero_center=zero_center)
        adata.layers["scaled"] = adata.X.copy()
        print("\t***Performing PCA on pseudo-bulked data...")
        sc.pp.pca(adata)
        print("\t***Swapping back to counts layer...")
        try:  # back to counts layer
            dc.pp.swap_layer(adata=adata, key="counts", inplace=True)
        except Exception:  # alternative method
            adata.X = adata.layers["counts"].copy()
    if col_target is not None:
        for i, x in enumerate(col_groupings):  # re-split grouping columns
            adata.obs.loc[:, x] = adata.obs[col_target].apply(
                lambda x: x.split(sep)[i])
    adata.obs = adata.obs.dropna(how="all", axis=1)
    return adata
