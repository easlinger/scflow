#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data reading, concatenation, integration, etc.

@author: E. N. Aslinger
"""

import os
import anndata
import scanpy as sc
import scflow


def read_scrna(file_path, **kws_read):
    """Read scRNA-seq data."""
    if os.path.splitext(file_path)[1] == ".h5ad":
        var_names = kws_read.pop("var_names", None)
        rna = sc.read_h5ad(file_path, **kws_read)
        if var_names is not None:
            if rna.var.index.names[0] != var_names:
                rna.var.loc[:, var_names] = rna.var[
                    var_names].astype("string")
                rna.var = rna.var.reset_index().set_index(var_names)
                rna.var_names = [str(i) for i in rna.var.index.values]
    elif os.path.splitext(file_path)[1] == ".mtx":
        rna = sc.read_10x_mtx(os.path.dirname(file_path), **kws_read)
    elif os.path.isdir(file_path):
        rna = sc.read_10x_mtx(file_path, **kws_read)
    elif os.path.slitext(file_path)[1] == ".h5":
        rna = sc.read_hdf(file_path, **kws_read)
    else:
        raise ValueError("`file_path` not a valid/recognized input.")
    return rna


def integrate(adata, kws_pp=None, kws_cluster=None,
              col_sample="sample", col_batch=None, axis="obs",
              join="outer", merge=None, uns_merge=None,
              index_unique=None, fill_value=None, pairwise=False,
              basis="X_pca", plot_qc=False, verbose=True, **kwargs):
    """
    Integrate scRNA-seq anndata objects with `Harmony`.

    Args:
        adata (list or dict or AnnData): List or dict
            (keyed by sample IDs) of `AnnData` objects,
            or a single `AnnData` object with a column (name(s) passed
            to `col_sample` argument) containing the IDs
            of the samples.
            If passing a list instead of a dict, if `col_sample` is
            found in the `.obs` for a given object in the `adata` list
            and has one unique value in that column,
            the sample ID will be inferred from that value; if the
            column is missing for any object or
            has more than one value in it,
            a sample ID will be created ('sample_#').
        kws_pp (dict or None, optional): Dictionary containing
            preprocessing keyword arguments to be passed to
            `scflow.pp.preprocess`. If sample-specific, key the
            dictionary by sample ID and put individual dictionaries
            in each item.
        kws_cluster (dict or None, optional): Dictionary containing
            clustering keyword arguments to be passed to
            `scflow.pp.cluster`. If sample-specific, key the
            dictionary by sample ID and put individual dictionaries
            in each item.
        col_sample (str or list, optional): Name of existing column(s)
            in `adata` object(s) containing the sample ID (if present)
            and/or the name of the column to be created in the
            integrated object to contain the sample IDs (as inferred
            from the objects or keys of `adata` if provided as a dict).
            Pass a list of keys if providing a list to `adata` and if
            not all the objects in the list have the sample column name
            in `.obs` containing their sample IDs; in this case,
            the i_th string in the `col_sample` list should correspond
            to the sample column name in the `.obs` attribute of the
            i_th `AnnData` object provided in `adata` (i.e., same order
            as `adata`), and the first string in the `col_sample` list
            will be used as the column name in the integrated object.
        col_batch (str or list, optional): Like `col_sample`. If
            specified, integration will be performed with respect to
            both `col_sample` and `col_batch`.
    """
    if isinstance(adata, (list, dict)):  # if not already concatenated
        ids = col_sample if isinstance(col_sample, list) else [
            col_sample] * len(adata)  # in case sample-specific ID columns
        sample_ids = list(adata.keys()) if isinstance(adata, dict) else [
            x.obs[ids[i]].iloc[0] if ids[i] in x.obs.columns and len(
                x.obs[ids[i]].unique()) == 1 else f"sample_{i}"
            for i, x in enumerate(
                adata)]  # IDs from dict keys or .obs (if possible) or create
        if isinstance(adata, list):  # convert to dictionary if list
            adata = dict(zip(sample_ids, adata))
        if isinstance(col_sample, list):
            col_sample = col_sample[0]  # if >1 columns, use 1st as final name
        for x in adata:
            adata[x].var_names_make_unique()
            adata[x].obs_names_make_unique()
        if kws_pp is not None:
            print("\n\n")
            if isinstance(kws_pp, dict) and any((
                    x in kws_pp for x in adata)) is False:
                kws_pp = dict(zip(sample_ids, [kws_pp] * len(
                    sample_ids)))  # assume same keywords for all samples
            for x in adata:
                if x in kws_pp:
                    if verbose is True:
                        print(f"***Preprocessing {x}: {kws_pp[x]}...")
                    adata[x] = scflow.pp.preprocess(adata[x], **{
                        "plot_qc": plot_qc, **kws_pp[x], "inplace": True})
        if kws_cluster is not None:
            print("\n\n")
            if isinstance(kws_cluster, dict) and any((
                    x in kws_cluster for x in adata)) is False:
                kws_cluster = dict(zip(sample_ids, [kws_cluster] * len(
                    sample_ids)))  # assume same keywords for all samples
            for x in adata:
                if x in kws_cluster:
                    if verbose is True:
                        print(f"***Clustering {x}: {kws_cluster[x]}...")
                    adata[x] = scflow.pp.cluster(adata[x], **{
                        "plot": plot_qc, **kws_cluster[x], "inplace": True})
        adata = anndata.concat(
            adata, axis=axis, join=join, merge=merge, uns_merge=uns_merge,
            label=col_sample, index_unique=index_unique,
            fill_value=fill_value, pairwise=pairwise)  # concatenate
    col_covs = col_sample if col_batch is None else [col_sample, col_batch]
    if verbose is True:
        ccs = col_covs if isinstance(col_covs, str) else " & ".join(col_covs)
        print(f"\n\n***Integrating with respect to {ccs}...")
    sc.external.pp.harmony_integrate(
        adata, col_covs, basis=basis,
        adjusted_basis=f"{basis}_harmony", **kwargs)  # Harmony integration
    adata.obsm["X_pca_old"] = adata.obsm["X_pca"].copy()
    adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"].copy()
    return adata
