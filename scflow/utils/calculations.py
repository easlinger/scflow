#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for miscellaneous calculations.

@author: E. N. Aslinger
"""


def is_outlier(data, metric: str, nmads: int):
    """Detect outliers (modified from SC Best Practices)."""
    outlier = (data < np.median(data) - nmads * median_abs_deviation(
        data)) | (np.median(data) + nmads * median_abs_deviation(data) < data)
    return outlier


def calculate_outliers(adata, drop_outliers=False, inplace=False):
    """
    Find (and, optionally, drop) outliers.

    Find outliers given a dictionary (`nmads`) keyed by column names
    to test for outliers and the respective `nmads` to use for each
    (e.g., {"log1p_total_counts": 5, "log1p_n_genes_by_counts": 5,
            "pct_counts_in_top_20_genes": 5, "pct_counts_mt": 3}).
    """
    if inplace is False:
        adata = adata.copy()
    obs_cols, var_cols = [], []
    for x in nmads:
        if x in adata.obs:
            adata.obs[f"outlier_{x}"] = is_outlier(adata.obs[x], nmads[x])
            obs_cols += [f"outlier_{x}"]
        elif x in adata.var:
            adata.var[f"outlier_{x}"] = is_outlier(adata.var[x], nmads[x])
            var_cols += [f"outlier_{x}"]
    if len(var_cols) > 0:
        adata.var["outlier"] = adata.var[var_cols].T.any()
    if len(obs_cols) > 0:
        adata.obs["outlier"] = adata.obs[obs_cols].T.any()
    if verbose is True:
        if len(var_cols) > 0:
            print(f"\n\n{'=' * 80}\n\nOutliers(var){'=' * 80}\n\n"
                  f"{adata.var.outlier.value_counts()}")
        if len(obs_cols) > 0:
            print(f"\n\n{'=' * 80}\n\nOutliers (obs){'=' * 80}\n\n"
                  f"{adata.obs.outlier.value_counts()}")
    if drop_outliers is True:
        if len(obs_cols) > 0:
            if verbose is True:
                print("\n\n***Dropping outliers (obs)...")
            adata = adata[~adata.obs["outlier"]]
        if len(var_cols) > 0:
            if verbose is True:
                print("\n\n***Dropping outliers (var)...")
            adata.var = adata.var[~adata.var["outlier"]]
    return adata
