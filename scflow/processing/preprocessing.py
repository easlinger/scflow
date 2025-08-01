#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for preprocessing.

@author: E. N. Aslinger
"""

import os
import warnings
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd


def preprocess(adata, min_max_genes=None, min_max_cells=None,
               col_sample=None, layer_counts="counts", layer_log1p="log1p",
               layer_scaled="scaled", doublet_detection=False,
               normalize=True, min_max_counts=None,
               vars_regress_out=None, target_sum=1e4, max_fraction=0.05,
               exclude_highly_expressed=False, n_top_genes=2000, max_mt=None,
               zero_center=True, max_value=None, plot_qc=True, inplace=False):
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
    try:
        adata.var_names_make_unique()
    except Exception as err:
        print(err)
    try:
        adata.obs_names_make_unique()
    except Exception as err:
        print(err)

    # Highly-Expressed Genes
    if plot_qc is True:
        sc.pl.highest_expr_genes(adata, n_top=20)

    # Quality Control
    adata = perform_qc(adata, plot_qc=plot_qc, col_sample=col_sample)

    # Filter Genes & Cells
    if min_max_counts is not None:
        if min_max_counts[0] is not None:
            adata = adata[adata.obs["total_counts"] >= min_max_counts[0]]
        if min_max_counts[1] is not None:
            adata = adata[adata.obs["total_counts"] <= min_max_counts[1]]
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
            adata = adata[adata.obs["pct_counts_mt"] <= max_mt]

    # Doublet Detection
    if doublet_detection is True:
        sc.pp.scrublet(adata, batch_key=col_sample)

    # Normalization & Regress Out (Optional)
    if normalize is True:
        sc.pp.normalize_total(
            adata, target_sum=target_sum,
            exclude_highly_expressed=exclude_highly_expressed,
            max_fraction=max_fraction)
        sc.pp.log1p(adata)
    if vars_regress_out is not None:
        sc.pp.regress_out(
            adata, vars_regress_out)  # e.g. ["total_counts", "pct_counts_mt"]
    adata.layers[layer_log1p] = adata.X.copy()

    # HVGs
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, batch_key=col_sample)
    if plot_qc is True:
        sc.pl.highly_variable_genes(adata)  # plot HVGs

    # Scale
    if zero_center is not None or max_value is not None:
        sc.pp.scale(adata, zero_center=zero_center, max_value=max_value)
        adata.layers[layer_scaled] = adata.X.copy()

    return adata


def perform_qc(adata, qc_vars=None, plot_qc=True, col_sample=None):
    """Perform QC."""
    # Detect Special Genes
    try:
        adata.var_names_make_unique()
    except Exception as err:
        print(err)
    try:
        adata.obs_names_make_unique()
    except Exception as err:
        print(err)
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "Mt-", "mt-"))
    adata.var["ribo"] = adata.var_names.str.startswith((
        "RPS", "RPL", "rps", "rpl", "Rpl", "Rps"))
    adata.var["hb"] = adata.var_names.str.contains(
        r"^hb[^p]", case=False, regex=True)
    if qc_vars is None:
        qc_vars = [i for i in ["mt", "ribo", "hb"] if adata.var[i].sum() > 0]
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=qc_vars, inplace=True, log1p=True)
    if plot_qc is True:
        sc.pl.violin(
            adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4, multi_panel=True, use_raw=False)  # QC violin plot
        if col_sample is not None:  # QC violin plot by sample
            sc.pl.violin(
                adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                jitter=0.4, multi_panel=True,
                groupby=col_sample, use_raw=False)
        adata.var["n_cells_by_counts"].hist()
        print(adata.var["n_cells_by_counts"].describe())
        plt.title("# of Cells in which Each Gene Has Non-Zero Expression "
                  "(n_cells_by_counts)")
        sc.pl.scatter(adata, "total_counts", "n_genes_by_counts",
                      color="pct_counts_mt")  # QC scatter plot
    return adata


def perform_qc_multi(adatas, plot_qc=False, col_gene="gene",
                     col_sample="sample", col_batch="batch",
                     percentiles=None,
                     plot=True, legend="brief", figsize=None):
    """Get (and optionally plot) QC values for multiple samples."""
    percentiles = list(percentiles) if percentiles is not None else [
        0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    mets = ["pct_counts_mt", "total_counts", "n_genes_by_counts"]
    qcs, n_cells_by_counts = {}, {}
    for x in adatas:
        tmp = perform_qc(adatas[x].copy(), plot_qc=plot_qc)
        qcs[x] = tmp.obs[mets].reset_index(drop=True)
        n_cells_by_counts[x] = tmp.var[["n_cells_by_counts"]].rename_axis([
            col_gene])
        if col_batch in tmp.obs:
            qcs[x] = qcs[x].assign(
                batch=tmp.obs[col_batch][0]).rename({
                    "batch": col_batch}).rename_axis([
                        "cell_id"]).set_index(col_batch, append=True)
            n_cells_by_counts[x] = n_cells_by_counts[x].assign(
                batch=tmp.obs[col_batch][0]).rename({
                    "batch": col_batch}).set_index(col_batch, append=True)
    n_cells_by_counts = pd.concat(n_cells_by_counts, keys=list(qcs.keys()),
                                  names=[col_sample])
    qcs = pd.concat(qcs, keys=list(qcs.keys()),
                    names=[col_sample]).rename_axis("Variable", axis=1)
    figs = {}
    if plot is True:
        figs["obs"] = plt.subplots(1, len(mets),
            figsize=(20, 20) if figsize is None else figsize, sharey=False)
        for i, m in enumerate(mets):
            sns.stripplot(
                data=n_cells_by_counts.reset_index(), y="n_cells_by_counts",
                x=col_batch, hue=col_sample,
                legend=legend, ax=figs["obs"][1][i])
        figs["var"] = plt.subplots(
            figsize=(20, 20) if figsize is None else figsize)
        sns.stripplot(
            data=n_cells_by_counts.reset_index(), y="n_cells_by_counts",
            x=col_batch, hue=col_sample, legend=legend, ax=figs["var"][1])
    descriptives = qcs.groupby([col_sample, col_batch] if (
        col_batch is not None) else col_sample).describe(
            percentiles=percentiles).rename_axis([
                "Variable", "Metric"], axis=1).stack(0)  # percentiles~sample
    descriptives = pd.concat([descriptives, n_cells_by_counts.groupby([
        col_sample, col_batch] if (
            col_batch is not None) else col_sample).describe(
                percentiles=percentiles).rename_axis([
                    "Variable", "Metric"], axis=1).stack(0)]).sort_index()
    return qcs, n_cells_by_counts, descriptives, figs
