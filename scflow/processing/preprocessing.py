#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for preprocessing.

@author: E. N. Aslinger
"""

import os
import warnings
import anndata
import scanpy as sc


def preprocess(adata, min_max_genes=None, min_max_cells=None,
               col_sample=None, layer_counts="counts", layer_log1p="log1p",
               layer_scaled="scaled", doublet_detection=False,
               vars_regress_out=None, target_sum=1e4, max_fraction=0.05,
               exclude_highly_expressed=False, n_top_genes=2000, max_mt=None,
               zero_center=True, max_value=None, inplace=True):
    """Filter, normalize, and perform QC on scRNA-seq data."""
    if isinstance(min_max_genes, str):
        if min_max_genes.lower() not in ["min", "max"]:
            raise ValueError(
                "`min_max_genes, if a string, must be 'min' or 'max'")
    if inplace is False:
        adata = adata.copy()
    if layer_counts not in adata.layers:
        adata.layers[layer_counts] = adata.X.copy()  # assume is counts layer
        warnings.warn("`layer_counts` not found in `adata.layers`. "
                      "Assuming current `adata.X` is integer counts.")
    adata.X = adata.layers[layer_counts].copy()  # ensure using counts layer

    # Highly-Expressed Genes
    sc.pl.highest_expr_genes(adata, n_top=20)

    # Detect Special Genes
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "Mt-", "mt-"))
    adata.var["ribo"] = adata.var_names.str.startswith((
        "RPS", "RPL", "rps", "rpl", "Rpl", "Rps"))
    adata.var["hb"] = adata.var_names.str.contains((
        "^HB[^(P)]", "^Hb[^(p)]", "^Hb[^(P)]", "^hb[^(p)]"))

    # Quality Control
    qc_vars = [i for i in ["mt", "ribo", "hb"] if adata.obs[i].sum() > 0]
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=qc_vars, inplace=True, log1p=log1p)
    sc.pl.violin(
        adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4, multi_panel=True, use_raw=False)  # QC violin plot
    if col_sample is not None:  # QC violin plot by sample
        sc.pl.violin(
            adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4, multi_panel=True, groupby=col_sample, use_raw=False)
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts",
                  color="pct_counts_mt")  # QC scatter plot

    # Filter Genes & Cells
    if min_max_genes is not None:  # filter cells by gene counts
        # if min_max_genes is True or isinstance(min_max_genes, str):
        #     min_max_genes = scflow.tl.calculate_outliers(
        #         adata.obs["n_genes"], inplace=True)
        #     if isinstance(min_max_genes, str):
        #         if min_max_genes.lower() == "min":  # only lower bound
        #             min_max_genes = [min_max_genes[0], None]
        #         elif min_max_genes.lower() == "max":  # only upper bound
        #             min_max_genes = [None, min_max_genes[1]]
        if min_max_genes[0] is not None:
            sc.pp.filter_cells(adata, min_genes=min_max_genes[0])
        if min_max_genes[1] is not None:
            sc.pp.filter_cells(adata, max_genes=min_max_genes[1])
    if min_max_cells is not None:  # filter genes by cell counts
        # if min_max_cells is True or isinstance(min_max_cells, str):
        #     min_max_cells = scflow.tl.calculate_outliers(
        #         adata.var["n_cells"], inplace=True)
        #     if isinstance(min_max_cells, str):
        #         if min_max_cells.lower() == "min":  # only lower bound
        #             min_max_cells = [min_max_cells[0], None]
        #         elif min_max_cells.lower() == "max":  # only upper bound
        #             min_max_cells = [None, min_max_cells[1]]
        if min_max_cells[0] is not None:
            sc.pp.filter_genes(adata, min_cells=min_max_cells[0])
        if min_max_cells[1] is not None:
            sc.pp.filter_genes(adata, max_cells=min_max_cells[1])
        if max_mt is not None:  # filter by maximum mitochondrial count
            adata = adata[adata.obs["pct_mt"] <= max_mt]

    # Doublet Detection
    if doublet_detection is True:
        sc.pp.scrublet(adata, batch_key=col_sample)

    # Normalization & Regress Out (Optional)
    sc.pp.normalize_total(adata, target_sum=target_sum,
                          exclude_highly_expressed=exclude_highly_expressed,
                          max_fraction=max_fraction)
    sc.pp.log1p(adata)
    sc.pp.regress_out(
        adata, vars_regress_out)  # e.g., ["total_counts", "pct_counts_mt"]
    adata.layers[layer_log1p] = adata.X.copy()

    # HVGs
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, batch_key=col_sample)
    sc.pl.highly_variable_genes(adata)  # plot HVGs

    # Scale
    if zero_center is not None or max_value is not None:
        sc.pp.scale(adata, zero_center=zero_center, max_value=max_value)
        adata.layers[layer_scaled] = adata.X.copy()

    return adata
