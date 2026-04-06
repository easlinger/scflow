#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data reading, concatenation, integration, etc.

@author: E. N. Aslinger
"""

import os
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import scvi
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
except Exception:
    warn("Cannot import rapids_singlecell.")
    rsc = None
try:
    from scib_metrics.benchmark import (
        Benchmarker, BioConservation, BatchCorrection)
    # import jax
except Exception:
    pass
try:
    import scanorama
except Exception:
    pass
import pandas as pd
import numpy as np
import scflow

layer_log1p = "log1p"
layer_counts = "counts"
layer_scaled = "scaled"
KWS_SCVI = [
    "max_epochs", "accelerator", "devices", "train_size", "validation_size",
    "shuffle_set_split", "batch_size", "datasplitter_kwargs", "plan_kwargs",
    "datamodule", "benchmark", "default_root_dir", "enable_checkpointing",
    "checkpointing_monitor", "num_sanity_val_steps", "enable_model_summary",
    "early_stopping", "early_stopping_monitor", "early_stopping_min_delta",
    "early_stopping_patience", "early_stopping_warmup_epochs", "logger",
    "early_stopping_mode", "enable_progress_bar", "progress_bar_refresh_rate",
    "simple_progress_bar", "log_every_n_steps", "learning_rate_monitor"
]
KWS_SCANVI = ["n_samples_per_label", "check_val_every_n_epoch",
              "adversarial_classifier"]


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


def integrate(adata, redo_qc_allowed=False, kws_pp=None, kws_cluster=None,
              col_sample="sample", col_batch=None, col_subject=None,
              axis="obs", join="outer", merge=None,
              uns_merge=None, n_top_genes=None, min_cells=None,
              layer_log1p=layer_log1p, layer_counts=layer_counts,
              layer_scaled=layer_scaled, zero_center=True, max_value=10,
              target_sum=1e4, n_comps=None, kws_pca_final=None,
              index_unique="_", fill_value=None, pairwise=False,
              basis="X_pca", drop_non_hvgs=False, retain_original=False,
              plot_qc=False, out_file=None, vars_regress_out=None,
              use_rapids=True, verbose=True, layer=None,
              out_file_processed=None, flavor="harmony",
              col_celltype=None, conserve_memory=False, **kwargs):
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
        out_file (str or None, optional): File path to write
            CONCATENATED (not fully processed/integrated) object
            in case using the on-disk concatenate option to save
            memory by writing to disk.
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
        min_cells (None or int, optional): Re-filter genes by minimum
            number of cells with non-zero expression (if not None).
            Useful if you're concatenating multiple samples in which
            you didn't filter genes in case they were expressed enough
            in some samples but not others.
    """
    pkg = sc if rsc is None or use_rapids is False else rsc
    new_pca_key = kwargs.pop("new_pca_key", f"X_pca_{flavor.lower()}")
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
        # Preprocessing
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

        # Clustering
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

    # On-Disk Setup
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
    else:
        print("\n>>>File appears concatenated already. Proceeding...")
        pass  # already concatenated

    # Concatenation
    if isinstance(adata, (list, dict)):  # if needs concatenation
        print("\n>>>Concatenating data...")
        if merge is not None or uns_merge is not None:
            warn("\n`merge`/`uns_merge` other than None not yet supported\n")
        adata = fx_concat(
            *first_args, axis=axis, join=join,
            # merge=merge, uns_merge=uns_merge,
            label=col_sample, index_unique=index_unique,
            pairwise=pairwise, fill_value=fill_value)  # concatenate
        if adata is None:  # if wrote to file instead of doing in memory...
            adata = sc.read_h5ad(out_file)

    # To GPU (Optionally)
    pkg = rsc if use_rapids is True else sc
    if use_rapids is True:  # make sure proper matrix format for `rapids`
        for x in adata.layers:
            if (sparse.isspmatrix_csc(adata.layers[x]) or (
                    sparse.isspmatrix_csr(adata.layers[x]))) is False:
                adata.layers[x] = sparse.csr_matrix(adata.layers[x].copy())
        if (sparse.isspmatrix_csc(adata.X) or sparse.isspmatrix_csr(
                adata.X)) is False:
            adata.X = sparse.csr_matrix(adata.X)
        rsc.get.anndata_to_GPU(adata)

    # Re-Filter Genes by `min_cells` (Optionally) & Re-Normalize
    adata.X = adata.layers[layer_counts].copy()  # back to counts layer
    try:
        adata = scflow.pp.classify_gene_types(adata)
        qc_vars = [i for i in ["mt", "ribo", "hb"] if adata.var[i].sum() > 0]
        qc_new = pkg.pp.calculate_qc_metrics(
            adata, qc_vars=qc_vars, layer=layer_counts, inplace=False)
        adata.obs = adata.obs.join(qc_new[0], rsuffix="_integrated")
        adata.var = adata.var.join(qc_new[1], rsuffix="_integrated")
    except Exception as err:
        print(f"\n\nCalculating new QC metrics failed: {err}\n\n")
    if min_cells is not None:
        print(f"\n>>>Filtering genes expressed in < {min_cells} cells...")
        pkg.pp.filter_genes(adata, min_cells=min_cells)
    print("\n>>>Re-normalizing & log-transforming data...")
    pkg.pp.normalize_total(adata, target_sum=target_sum)
    pkg.pp.log1p(adata)
    if vars_regress_out is not None:
        print(f"\n>>>Regressing out {vars_regress_out}...")
        pkg.pp.regress_out(adata, vars_regress_out)
    adata.layers[layer_log1p] = adata.X.copy()
    print("\n>>>Scaling data...")
    pkg.pp.scale(adata, zero_center=zero_center, max_value=max_value)
    adata.layers[layer_scaled] = adata.X.copy()
    if out_file_processed is not None:
        print(f"\n>>>Writing processed file to {out_file_processed}...")
        adata.write_h5ad(out_file_processed)

    # Find (& Optionally Subset by) HVGs; (Optionally) Store Original Object
    print(f"\n>>>Finding HVGs for data using `{layer_log1p}` layer" + str(
        " & dropping non-HVGs..." if drop_non_hvgs is True else "..."))
    adata_original = adata.copy() if all([
        drop_non_hvgs, retain_original]) else None  # in case dropping non-HVG
    pkg.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, layer=layer_log1p,
        flavor="cell_ranger", batch_key=col_sample,
        subset=drop_non_hvgs)  # find (optionally subset to) highly variable
    if verbose is True:
        try:
            sc.pl.highly_variable_genes(adata)  # plot HVGs
        except Exception as err:
            print(f"Failed to plot HVGs: {err}")

    # Batch Keys/Covariates
    col_covs = col_sample if col_batch is None else [col_subject if (
        col_subject is not None) else col_sample, col_batch]
    if verbose is True:
        ccs = col_covs if isinstance(col_covs, str) else " & ".join(col_covs)
        print(f"\n>>>Integrating with respect to {ccs} ({flavor.upper()})...")

    # Harmony
    if flavor.lower() == "harmony":
        print(f"\t***Using {layer_log1p} layer for Harmony...")
        adata.X = adata.layers[layer_log1p].copy()
        fxi = rsc.pp.harmony_integrate if (
            use_rapids is True) else sc.external.pp.harmony_integrate  # fx?
        fxi(adata, col_covs, basis=basis,
            adjusted_basis=f"{basis}_harmony", **kwargs)  # Harmony

    # scVI or scANVI
    elif flavor.lower() in ["scvi", "scanvi"]:
        print(f"\t***Using {layer_counts} layer for {flavor}...")
        kws_setup = dict(layer=layer_counts, batch_key=col_sample)
        kss = ["size_factor_key", "categorical_covariate_keys",
               "continuous_covariate_keys"]
        for k in [i for i in kss if i in kwargs]:
            kws_setup[k] = kwargs.pop(k)  # extract setup arguments
        ckws_pr = [str(kws_setup[i]) for i in [
            "categorical_covariate_keys", "continuous_covariate_keys"] if (
                i in kws_setup)]  # covariate keyword arguments?
        if len(ckws_pr) > 0:
            print(f"\t***Using {', '.join(ckws_pr)} as covariates...")
        if "categorical_covariate_keys" not in kws_setup:
            kws_setup["categorical_covariate_keys"] = None if (
                isinstance(col_covs, str)) else col_covs[1]
        kws_train = {}
        for k in pd.unique([i for i in KWS_SCVI if i in kwargs]):
            kws_train[k] = kwargs.pop(k)  # extract shared training arguments
        if flavor.lower() == "scanvi":  # scANVI setup
            kws_train_scanvi = {**kws_train}  # start with shared arguments
            for k in [i for i in KWS_SCANVI if i in kwargs]:
                kws_train_scanvi[k] = kwargs.pop(k)
            unlabeled = kwargs.pop("unlabeled_category", "Unlabeled")
            if "use_minified" in kwargs:
                kws_setup["use_minified"] = kwargs.pop(x)
        if conserve_memory is False:
            adata = adata.copy()
        scvi.model.SCVI.setup_anndata(adata, **kws_setup)  # setup data
        kws_train_scvi = {**kws_train}  # start with shared arguments
        for k in [i for i in ["load_sparse_tensor", "early_stopping"] if (
                i in kwargs)]:
            kws_train_scvi[k] = kwargs.pop(k)
        print(f"\t***Setting up scVI model: {kwargs}...")
        model = scvi.model.SCVI(adata, **kwargs)  # scVI or scANVI model
        print(f"\t***Training scVI: {kws_train_scvi}...")
        model.train(**kws_train_scvi)  # train model
        if flavor.lower() == "scanvi":
            print(f"\t***Setting up scANVI model: {kwargs}...")
            vimodel = scvi.model.SCANVI.from_scvi_model(
                model, adata=adata, labels_key=col_celltype,
                unlabeled_category=unlabeled, **kwargs)
            print(f"\t***Traning scANVI: {kws_train_scanvi}...")
            vimodel.train(**kws_train_scanvi)
            adata.obsm[new_pca_key] = vimodel.get_latent_representation(adata)
            if col_celltype is not None:
                try:
                    adata.obs.loc[:, "annotation_scanvi"] = vimodel.predict(
                        adata)
                except Exception as err:
                    print(err)
            if verbose is True:
                try:
                    dff = adata.obs.groupby([
                        col_celltype, "annotation_scanvi"]).size().unstack(
                            fill_value=0)
                    conf_mat = dff / dff.sum(axis=1).values[:, np.newaxis]
                    plt.figure(figsize=(8, 8))
                    _ = plt.pcolor(conf_mat)
                    _ = plt.xticks(np.arange(0.5, len(dff.columns), 1),
                                   dff.columns, rotation=90)
                    _ = plt.yticks(np.arange(0.5, len(dff.index), 1),
                                   dff.index)
                    plt.xlabel("Predicted")
                    plt.ylabel("Observed")
                except Exception as err:
                    print(f"\nCould not plot scANVI confusion matrix: {err}")
        # if use_rapids is True:
        #     model.to_device("cpu")
        adata.obsm[new_pca_key] = model.get_latent_representation()

    # Scanorama
    elif flavor.lower() == "scanorama":
        batch_cats = adata.obs[col_sample].cat.categories
        adata_list = [adata[adata.obs[col_batch] == b].copy()
                      for b in batch_cats]
        scanorama.integrate_scanpy(adata_list)
        adata.obsm["Scanorama"] = np.zeros((adata.shape[0], adata_list[
            0].obsm[new_pca_key].shape[1]))
        for i, b in enumerate(batch_cats):
            adata.obsm["Scanorama"][adata.obs[col_batch] == b] = adata_list[
                i].obsm[new_pca_key]
    else:
        raise ValueError(f"`flavor='{flavor}'` not valid.")

    # Final Cleanup & Re-Calculate QC
    if use_rapids is True:
        rsc.get.anndata_to_CPU(adata)  # move back to cpu
    if adata_original is not None:  # join new information to original object?
        adata_original.obsm[new_pca_key] = adata.obsm[new_pca_key].copy()
        for x in adata.obs:
            if x not in adata_original.obs or (all(adata.obs[
                    x] == adata_original.obs[x]) is False):
                print(f"\t***{'Overw' if x in adata_original.obs else 'W'}"
                      f"riting `.obs['{x}']` to original object...")
                adata_original.obs[x] = adata.obs[x].copy()
        for x in adata.uns:
            print(f"\t***{'Overw' if x in adata_original.uns else 'W'}riting "
                  f"`.uns['{x}']` to original object...")
            adata_original.uns[x] = adata.uns[x].copy()
        for x in [i for i in adata.var if i not in adata_original.var]:
            print(f"\t***Joining `.var['{x}']` to original object as"
                  f" `.var['{x}_{flavor.lower()}']`...")
            adata_original.var[f"{x}_{flavor.lower()}"] = adata.var[x].copy()
        adata_original.uns[f"var_{flavor}"] = adata.var.copy()
        # adata = adata_original
        adata.raw = adata_original
    if new_pca_key is not None:  # new PCA -> X_pca default key
        adata.obsm["X_pca_old"] = adata.obsm["X_pca"].copy()
        adata.obsm["X_pca"] = adata.obsm[new_pca_key].copy()
    print(f"\n>>>Setting `.X` back to {layer_counts} layer...")
    adata.X = adata.layers[layer_counts].copy()  # set back to counts layer
    if verbose is True:
        try:
            sc.pl.pca(adata, color=col_covs)
        except Exception as err:
            print(f"\n\n PCA plotting failed: {err}")
    return adata


def benchmark_integration(adata, col_sample, pca_keys=None, col_celltype=None,
                          pre_integrated_embedding_obsm_key="X_pca_old",
                          n_jobs=-1, precision="float32", **kwargs):
    """Benchmark integration results."""
    if pca_keys is None:
        pca_keys = ["X_pca_old", "X_scVI", "X_scANVI", "X_pca_harmony"]
    pca_keys = [i for i in pca_keys if i in adata.obsm]
    adata = adata.copy()
    adata.X = adata.X.astype(precision)
    for key in adata.obsm:
        if hasattr(adata.obsm[key], "astype"):
            adata.obsm[key] = adata.obsm[key].astype(precision)
    for layer_key in adata.layers.keys():
        adata.layers[layer_key] = adata.layers[layer_key].astype(precision)
    # jax.config.update("jax_enable_x64", True)
    biocons = BioConservation(isolated_labels=False)
    bmr = Benchmarker(adata, batch_key=col_sample, label_key=col_celltype,
                      embedding_obsm_keys=pca_keys, n_jobs=n_jobs,
                      bio_conservation_metrics=biocons,
                      batch_correction_metrics=BatchCorrection(), **kwargs)
    bmr.benchmark()
    bmr.plot_results_table(min_max_scale=True)
    bmr.plot_results_table(min_max_scale=False)
    results = dict(zip(["Scaled", "Unscaled"], [bmr.get_results(
        min_max_scale=x) for x in [True, False]]))  # results table
    for x in results:
        print(f"\n\n{'=' * 80}\n{x} Benchmarking Results\n{'=' * 80}",
              results[x])
    return results, bmr
