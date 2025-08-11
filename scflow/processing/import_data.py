#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data reading, concatenation, integration, etc.

@author: E. N. Aslinger
"""

import os
import anndata
import scanpy as sc
from scipy import sparse
from warnings import warn
try:
    import rapids_singlecell as rsc
    # import cupy
    # import rmm
    # from rmm.allocators.cupy import rmm_cupy_allocator
    # from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
    # rmm.reinitialize(managed_memory=False, pool_allocator=True)
    # rmm.reinitialize(managed_memory=True)
    # cupy.cuda.set_allocator(rmm_cupy_allocator)
    warn("Cannot import rapids_singlecell.")
except Exception:
    rsc = None
import numpy as np
import scflow

layer_log1p = "log1p"


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
              join="outer", merge=None, uns_merge=None, n_top_genes=2000,
              layer_log1p=layer_log1p, n_comps=None, kws_pca_final=None,
              index_unique="_", fill_value=None, pairwise=False,
              basis="X_pca", drop_non_hvgs=False,
              plot_qc=False, out_file=None,
              use_rapids=True, verbose=True, layer="scaled", **kwargs):
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
    pkg = sc if rsc is None or use_rapids is False else rsc
    if rsc is None:
        use_rapids = False
    if kws_pca_final is None:
        kws_pca_final = {}
    if isinstance(adata, (list, dict)):
        ix_0 = 0 if isinstance(adata, list) else list(adata.keys())[0]
    if isinstance(adata, (list, dict)) and isinstance(
            adata[ix_0], anndata.AnnData):  # if needs concatenate
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
                        print(f"\n>>>Preprocessing {x}: {kws_pp[x]}...")
                    adata[x] = scflow.pp.preprocess(adata[x], **{
                        "plot_qc": plot_qc, **kws_pp[x],
                        "use_rapids": use_rapids, "inplace": True})
                    # if rsc is not None:
                    #     adata[x].X = cupy_csr_matrix(adata[x].X)
        if kws_cluster is not None:
            print("\n\n")
            if isinstance(kws_cluster, dict) and any((
                    x in kws_cluster for x in adata)) is False:
                kws_cluster = dict(zip(sample_ids, [kws_cluster] * len(
                    sample_ids)))  # assume same keywords for all samples
            for x in adata:
                if x in kws_cluster:
                    if verbose is True:
                        print(f"\n>>>Clustering {x}: {kws_cluster[x]}...")
                    adata[x] = scflow.pp.cluster(adata[x], **{
                        "plot": plot_qc, **kws_cluster[x], "inplace": True})
        else:  # just do PCA if not full clustering
            for x in adata:
                pkg.pp.pca(adata[x], n_comps=n_comps)
        for x in adata:  # iterate data to convert to sparse (save memory)
            if isinstance(adata[x].X, np.ndarray) or not sparse.issparse(
                    adata[x].X):
                adata[x].X = sparse.csr_matrix(adata[x].X)  # to sparse matrix
        fx_concat = anndata.concat  # function to use to concatenate
        first_args = [adata]  # positional argument to concatenate
    elif isinstance(adata, (list, dict)):  # on disk concatenation?
        fx_concat = anndata.experimental.concat_on_disk  # function
        if isinstance(adata, list):  # retrieve sample names if needed
            sample_ids = []
            for x in adata:
                sid = sc.read(adata[x]).obs[col_sample]
                if len(sid.unique()) > 1:
                    raise ValueError(f"Sample {adata[x]} has non-unique valu"
                                     f"es in sample column '{col_sample}'")
                sample_ids += [sid.iloc[0]]
            adata = dict(zip(sample_ids, adata))  # convert list to dictionary
        first_args = [adata, out_file]  # positional arguments to concatenate
    if isinstance(adata, (list, dict)):  # if needs concatenation
        print("\n>>>Concatenating data...")
        adata = fx_concat(
            *first_args, axis=axis, join=join,
            merge=merge, uns_merge=uns_merge,
            label=col_sample, index_unique=index_unique,
            pairwise=pairwise, fill_value=fill_value)  # concatenate
        if adata is None:  # if wrote to file instead of doing in memory...
            adata = sc.read_h5ad(out_file)
    if verbose is True:
        print("\n>>>Finding HVGs for overall data...")
    if use_rapids is True:  # make sure proper matrix format for `rapids`
        # adata.X = cupy_csr_matrix(adata.X)
        # adata.X = sparse.csr_matrix(adata.X)
        for x in adata.layers:
            if (sparse.isspmatrix_csc(adata.layers[x]) or (
                    sparse.isspmatrix_csr(adata.layers[x]))) is False:
                adata.layers[x] = sparse.csr_matrix(adata.layers[x].copy())
        if (sparse.isspmatrix_csc(adata.X) or sparse.isspmatrix_csr(
                adata.X)) is False:
            adata.X = sparse.csr_matrix(adata.X)
        rsc.get.anndata_to_GPU(adata)
    # pkg.pp.highly_variable_genes(
    #     adata, n_top_genes=n_top_genes,  # batch_key=col_sample,
    #     flavor="cell_ranger")  # find highly variable
    # if verbose is True:
    #     sc.pl.highly_variable_genes(adata)  # plot HVGs
    if drop_non_hvgs is True:  # only retain HVGs?
        if verbose is True:
            print(f"\n>>>Subsetting to top {n_top_genes} HVGs...")
        adata.raw = adata  # save full data in .raw
        # pkg.pp.filter_highly_variable(adata)
        adata = adata[:, adata.var.highly_variable].copy()
    # if verbose is True:
    #     print("\n>>>Computing PCA for Combined Dataset...")
    # pkg.pp.pca(adata, **kws_pca_final, copy=False)  # PCA on all data
    col_covs = col_sample if col_batch is None else [col_sample, col_batch]
    if verbose is True:
        ccs = col_covs if isinstance(col_covs, str) else " & ".join(col_covs)
        print(f"\n>>>Integrating with respect to {ccs}...")
    if layer is not None:  # layer for harmonypy
        adata.X = adata.layers[layer].copy()
    fxi = rsc.pp.harmony_integrate if (
        use_rapids is True) else sc.external.pp.harmony_integrate  # fx?
    fxi(adata, col_covs, basis=basis,
        adjusted_basis=f"{basis}_harmony", **kwargs)  # Harmony integration
    adata.obsm["X_pca_old"] = adata.obsm["X_pca"].copy()
    adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"].copy()
    adata.X = adata.layers[layer_log1p].copy()  # set back to log1p
    if "n_genes_by_counts" not in adata.var:  # re-perform QC if needed
        adata = scflow.pp.perform_qc(
            adata, plot_qc=verbose, inplace=True, use_rapids=use_rapids)
    if use_rapids is True:
        rsc.get.anndata_to_CPU(adata)  # move backj to cpu
    return adata
