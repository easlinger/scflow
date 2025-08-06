#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for annotating single-cell data.

@author: E. N. Aslinger
"""

import os
from warnings import warn
import celltypist
try:
    from cell_type_mapper.cli.from_specified_markers import (
        FromSpecifiedMarkersRunner)
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import (
        AbcProjectCache)
except ModuleNotFoundError:
    pass
import scanpy as sc
import pandas as pd
import numpy as np


def run_celltypist(adata, model, layer="log1p", col_celltype=None,
                   col_celltypist_suffix="", **kwargs):
    """Run CellTypist (specify `col_celltype` to plot)."""
    plot = kwargs.pop(
        "plot", col_celltype is not None and col_celltype in adata.obs)
    if col_celltype is not None and col_celltype not in adata.obs:
        warn("Can't plot: `col_celltype` {col_celltype} not in `adata.obs`.")
    if layer is not None:
        adata = adata.copy()  # TODO: not memory efficient...
        adata.X = adata.layers[layer]
    predictions = celltypist.annotate(
        adata, model=model, **kwargs)  # CellTypist
    obs_add = predictions.predicted_labels["predicted_labels"].to_frame(
        f"predicted_labels{col_celltypist_suffix}")  # cell-level predictions
    if plot is True:
        celltypist.dotplot(predictions, use_as_reference=col_celltype,
                           use_as_prediction="predicted_labels")  # plot
    if "majority_voting" in kwargs and kwargs[
            "majority_voting"] is True:  # majority voting labels >
        obs_add = obs_add.join(predictions.predicted_labels[
            "majority_voting"].to_frame(
                f"majority_voting{col_celltypist_suffix}"))
        mv_pcol = f"majority_voting_probabilities{col_celltypist_suffix}"
        obs_add = obs_add.join(predictions.probability_matrix.apply(
            lambda x: np.nan if predictions.predicted_labels.loc[x.name][
                "majority_voting"] == "Heterogeneous" else x[
                    predictions.predicted_labels.loc[x.name][
                        "majority_voting"]], axis=1).to_frame(
                            mv_pcol)) # MV probability
    col_overlap = adata.obs.columns.intersection(obs_add.columns)
    if len(col_overlap) > 0:  # drop any pre-existing/overlapping columns
        warn(f"{col_overlap} already in `adata.obs`. Overwriting.")
        adata.obs = adata.obs.drop(list(col_overlap), axis=1)
    adata.obs = adata.obs.join(obs_add)
    return predictions, adata


def run_mapbraincells(file_adata, map_my_cells_source="WHB-10X",
                      dir_scratch="scratch", dir_resources="resources",
                      validate_output_file="scratch/tmp.h5ad",
                      map_my_cells_region_keys=None,
                      map_my_cells_cell_keys=None,
                      key_drop="CCN20230722",
                      max_gb=10, chunk_size=10000,
                      n_processors=1, seed=233211, **kwargs):
    """
    Run Map My Cells (Allen Brain Atlas).

    See https://github.com/AllenInstitute/cell_type_mapper/blob/update\
        /docs/250304/examples/mapping_to_subset_of_abc_atlas_data.ipynb

    - Make sure to run the following bash commands after activating
      the conda environment you will use.

    - Clone the cell_type_mapper repo into your home directory:

    >>> cd
    >>> git clone git@github.com:AllenInstitute/cell_type_mapper.git

    - Navigate to the folder containing this notebook.

    - Install ABC Atlas (from same directory as this notebook):

    >>> pip install -U git+https://github.com/alleninstitute/\
abc_atlas_access >& <dir_scratch>/junk.txt

      Replace <dir_scratch> with the value passed to the
      `dir_scratch` argument.

    - Pull lookup files (from the same directory as this notebook):

    >>> cd resources
    >>> wget https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com\
    /mapmycells/WMB-10X/20240831/mouse_markers_230821.json
    >>> wget https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com\
    /mapmycells/WMB-10X/20240831/precomputed_stats_ABC_\
    revision_230821.h5

    Note: To use GPU + Torch, you may need to alter the file
    cell_type_mapper/src/cell_type_mapper/cell_by_gene/cell_by_gene.py
    line `np.where(np.logical_not(np.isfinite(data)))[0]`

    to read instead

    >>>  try:
    >>>      nan_rows = np.where(
    >>>          np.logical_not(np.isfinite(data.cpu().numpy())))[0]
    >>>  except Exception:
    >>>      nan_rows = np.where(np.logical_not(np.isfinite(data)))[0]

    And potentially in `_correlation_dot_gpu()` in
    `distance_utils.py` change

    >>> try:
    >>>     correlation = torch.matmul(arr0, arr1)
    >>> except RuntimeError as err:
    >>>     if "CUBLAS_STATUS_NOT_INITIALIZED" in str(err):
    >>>         arr0_cpu = arr0.cpu()
    >>>         arr1_cpu = arr1.cpu()
    >>>         correlation = torch.matmul(arr0_cpu, arr1_cpu).to(
    >>>             arr0.device)
    >>> else:
    >>>     raise

    """
    # Make Needed Directories
    os.makedirs(dir_scratch, exist_ok=True)
    os.makedirs(dir_resources, exist_ok=True)

    # Keep Threads from Competing (numpy)
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Genes to EnsemblIDs?
    if validate_output_file is not None:
        os.system("python -m cell_type_mapper.cli.validate_h5ad --input_path "
                  f"{file_adata} --layer counts --output_json {dir_scratch}"
                  f"/out.json --valid_h5ad_path {validate_output_file}")
        file_adata_use = validate_output_file
    else:
        file_adata_use = file_adata

    # Configuration
    baseline_precomp_path = os.path.join(
        dir_resources, "precomputed_stats_ABC_revision_230821.h5")
    baseline_marker_path = os.path.join(
        dir_resources, "mouse_markers_230821.json")
    baseline_json_output_path = os.path.join(
        dir_scratch, "baseline_json_mapping_output.json")
    baseline_csv_output_path = os.path.join(
        dir_scratch, "baseline_csv_mapping_output.csv")
    baseline_mapping_config = {
        "query_path": file_adata_use, "tmp_dir": dir_scratch,
        "extended_result_path": str(baseline_json_output_path),
        "csv_result_path": str(baseline_csv_output_path),
        "max_gb": max_gb, "cloud_safe": False, "verbose_stdout": False,
        "type_assignment": {
            "normalization": "raw",
            "n_processors": n_processors,
            "chunk_size": chunk_size,
            "bootstrap_iteration": 100,
            "bootstrap_factor": 0.5,
            "rng_seed": seed
        },
        "precomputed_stats": {"path": str(baseline_precomp_path)},
        "query_markers": {"serialized_lookup": str(baseline_marker_path)},
        "drop_level": None,
        **kwargs
    }

    # Subset by Region (if Desired)
    if map_my_cells_region_keys is not None or (
            map_my_cells_cell_keys is not None):  # subset by region
        abc_cache = AbcProjectCache.from_cache_dir(dir_scratch)
        abc_cache.load_latest_manifest()
        # abc_cache.list_metadata_files(directory=map_my_cells_source)
        abc_cache.get_directory_metadata(
            directory=map_my_cells_source.split("-10X")[0] + "-taxonomy")
        abc_cache.get_metadata_path(
            directory=map_my_cells_source, file_name="cell_metadata")
        taxonomy_df = abc_cache.get_metadata_dataframe(
            directory=map_my_cells_source.split("-10X")[0] + "-taxonomy",
            file_name="cluster_to_cluster_annotation_membership")
        alias_to_truth = dict()
        for cell in taxonomy_df.to_dict(orient="records"):
            alias = cell["cluster_alias"]
            level = cell["cluster_annotation_term_set_label"]
            if alias not in alias_to_truth:
                alias_to_truth[alias] = dict()
            alias_to_truth[alias][level] = cell["cluster_annotation_term_label"]
        cell_metadata = abc_cache.get_metadata_dataframe(
            directory=map_my_cells_source, file_name="cell_metadata")
        if map_my_cells_region_keys is not None:  # subset by region keys
            cell_metadata = cell_metadata[pd.concat([
                cell_metadata.region_of_interest_acronym == i
                for i in map_my_cells_region_keys], axis=1).T.any()]  # subset
        if map_my_cells_cell_keys is not None:  # subset by cell label pattern
            ckeys = cell_metadata.feature_matrix_label.str.contains("|".join(
                map_my_cells_cell_keys))
            cell_metadata = cell_metadata[ckeys]
        valid_classes = set([alias_to_truth[x][f"{key_drop}_CLAS"]
                            for x in cell_metadata.cluster_alias.values])
        classes_to_drop = list(set([alias_to_truth[x][
            f"{key_drop}_CLAS"] for x in alias_to_truth if alias_to_truth[x][
                f"{key_drop}_CLAS"] not in valid_classes]))
        nodes_to_drop = [("class", x) for x in classes_to_drop]
        baseline_mapping_config.update({
            # "drop_level": f"{key_drop}_SUPT",
            "nodes_to_drop": nodes_to_drop})
        print("=======Nodes Being Dropped=======")
        for pair in nodes_to_drop[:4]:
            print(pair)
    print(f"{'=' * 80}\nConfiguration\n{'=' * 80}\n{baseline_mapping_config}")

    # Run Mapper
    mapping_runner = FromSpecifiedMarkersRunner(
        args=[], input_data=baseline_mapping_config)
    mapping_runner.run()

    # Output & Clean Up
    adata = sc.read_h5ad(file_adata)
    cellmap = pd.read_csv(os.path.join(
        dir_scratch, "baseline_csv_mapping_output.csv"),
                          skiprows=4).set_index("cell_id").rename_axis(
                              adata.obs.index.names)  # read annotation output
    cellmap.columns = [f"cellmap_{i}" for i in cellmap]
    if len(adata.obs.columns.intersection(cellmap.columns)) > 0:
        adata.obs = adata.obs.drop(list(adata.obs.columns.intersection(
            cellmap.columns)), axis=1)
    adata.obs = adata.obs.join(cellmap).loc[adata.obs.index]  # join
    for x in ["cellmap_class_name", "cellmap_subclass_name"]:
        adata.obs.loc[:, f"{x}_original"] = adata.obs[x].copy()
        adata.obs.loc[:, f"{x}"] = adata.obs[x].apply(
            lambda x: " ".join(x.split(" ")[1:]) if all((
                i in [str(i) for i in np.arange(0, 10)] for i in x.split(
                    " ")[0])) else x)  # drop pointless #s in cell types
    if validate_output_file is not None:
        try:
            os.system(f"rm {validate_output_file}")  # remove temporary h5ad
        except Exception:
            warn(f"Could not clean up output file {validate_output_file}.")
    return adata
