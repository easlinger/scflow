# __init__.py
# pylint: disable=unused-import

import os
import anndata
from anndata import AnnData
import scanpy as sc
from mudata import MuData
import celltypist
import scflow
# from scflow import Regression
# from scflow.processing import import_data
import pandas as pd
import numpy as np

layer_log1p = "log1p"  # TODO: MOVE THESE TO CONSTANTS MODULE
layer_counts = "counts"


class Rna(object):
    """A class for single-cell RNA-seq data."""

    def __init__(self, file_path=None, col_sample=None, kws_read=None,
                 col_celltype=None, kws_integrate=None, assay=None, **kwargs):
        """
        Initialize class instance.

        Args:
            file_path (PathLike or list or dict or AnnData): Path or
                object containing data (or a list or dict thereof,
                to integrate multiple datasets/samples, with the dict
                keyed by sample names if desired).
                If the desired path is a barcodes/feature/matrix
                directory, pass the
                path to the .mtx file (and don't forget to pass the
                `prefix` argument in a `kws_read` dictionary if there
                are characters in front of the file
                names, such as 'patientA_matrix.mtx') or the directory
                path (less reliable, typically).
            col_sample (str or None, optional): Column in `.obs` that
                contains (or should be made to contain) sample IDs.
            kws_read (dict or list, optional): Dictionary of keyword
                arguments (or list thereof, if passing a list or
                dict of multiple samples to `file_path` and wanting to
                use sample-specific arguments)
                to pass to `scanpy` read function.
            kws_integrate (dict or list, optional): Dictionary of
                keyword arguments to pass to Harmony integration
                function (if passed list or dict to `file_path`).

        """
        if kws_read is None:
            kws_read = {}
        if kws_integrate is None:
            kws_integrate = {}
        if "var_names" in kwargs:
            kws_read["var_names"] = kwargs.pop("var_names")
        if isinstance(file_path, (AnnData, MuData)):
            adata = file_path.copy()
        elif isinstance(file_path, (list, dict)):
            if col_sample is None:
                col_sample = "batch"
            if not isinstance(kws_read, list):  # if not sample-specific kws
                kws_read = [kws_read] * len(file_path)  # use same for all
            kws_read = dict(zip(list(file_path.keys()), kws_read))
            adata = [scflow.pp.read_scrna(file_path[x], **kws_read[x])
                          for x in file_path]
            adata = scflow.pp.integrate(
                adata, col_sample=col_sample,
                **kws_integrate)  # integrate with Harmony
        else:
            adata = scflow.pp.read_scrna(file_path, **kws_read)
        if isinstance(file_path, (AnnData, MuData)):
            if "var_names" in kws_read:
                if assay is not None:
                    if adata[assay].var.index.names[0] != kws_read[
                            "var_names"]:
                        adata.mod[assay].var = adata.mod[assay].reset_index(
                                ).set_index(kws_read["var_names"])
                else:
                    if adata.var.index.names[0] != kws_read["var_names"]:
                        adata.var = adata.var.reset_index().set_index(
                            kws_read["var_names"])
                    adata.var_names = adata.var[kws_read[
                        "var_names"]].to_list()
        self._info = {"col_sample": col_sample, "kws_read": kws_read,
                      "kws_integrate": kws_integrate,
                      "col_celltype": col_celltype}
        self.assay = assay
        self.rna = adata
        # self._adata = adata
        # self._update_rna_from_adata()

    # @property
    # def adata(self):
    #     return self._adata

    # @adata.setter
    # def adata(self, new_adata):
    #     self._adata = new_adata
    #     self._update_rna_from_adata()

    # @property
    # def rna(self):
    #     return self._rna

    # @rna.setter
    # def rna(self, new_rna):
    #     self._rna = new_rna
    #     if isinstance(self._adata, MuData):
    #         self._adata.mod[self.assay] = new_rna
    #     else:
    #         self._adata = new_rna

    # def _update_rna_from_adata(self):
    #     if isinstance(self._adata, MuData):
    #         self._rna = self._adata.mod[self.assay]
    #     else:
    #         self._rna = self._adata

    def plot(self, kind=None, col_celltype=None, genes=None,
             color=None, figsize=None, return_fig=False, **kwargs):
        """Plot."""

        # Process Arguments
        if col_celltype is None:
            col_celltype = self._info["col_celltype"]
        genes_plots = ["dot", "tracks", "matrix", "heat",
                       "stacked_violin", "violin"]
        colors_plots = ["scatter", "umap"]
        if kind is None:  # if no plot kind specified
            kind = ["umap", "heat", "violin"]
        if isinstance(kind, str):  # ensure consistency if just 1 plot kind
            if kind not in kwargs:
                kwargs = {kind: kwargs}  # make kwargs keyed ~ kind
            kind = [kind]  # ensure list, even if just one plot kind
        kind = [k.lower() for k in kind]  # make not case-sensitive
        fig = {}  # to hold plots

        # Iterate Plot Kinds
        for k in kind:
            if k not in kwargs:  # if no plot-specific arguments specified...
                kwargs[k] = {}  # ...just empty to auto-fill later if needed
            f_x = scflow.get_plot_fx(k)  # get the right plot function
            if k in genes_plots and "genes" not in kwargs[k]:  # if needed...
                kwargs[k].update({"genes": genes})  # ...specify "genes"
            if k in colors_plots and "color" not in kwargs[k]:  # if needed...
                kwargs[k].update({"color": color})  # ...specify grouping
            if k in [x"matrix", "violin", "stacked_violin", "heat"] and (
                    "col_celltype" not in kwargs[k]):
                kwargs[k]["col_celltype"] = col_celltype  # specify cell type
            if "violin" not in k and k != "umap":
                kwargs[k]["figsize"] = figsize
            fig[k] = f_x(self.rna, **kwargs[k])
        if return_fig  is True:
            return fig

    def preprocess(self, inplace=True, **kws_pp):
        """Filter, normalize, and perform QC on data."""
        if inplace is True:
            self.rna = scflow.pp.preprocess(
                self.rna, inplace=True, **kws_pp)
        else:
            return scflow.pp.preprocess(self.rna, inplace=False, **kws_pp)

    def cluster(self, col_celltype="leiden", resolution=1, min_dist=0.5,
                use_highly_variable=True, inplace=True, **kws):
        """Perform Leiden clustering."""
        for x in ["kws_cluster", "kws_umap"]:
            if x not in kws:
                kws.update({x: {}})
        kws["kws_cluster"]["resolution"] = resolution
        kws["kws_umap"]["min_dist"] = min_dist
        if "kws_pca" not in kws:
            kws["kws_pca"] = {}
        if kws["kws_pca"] is not False:
            kws["kws_pca"]["use_highly_variable"] = use_highly_variable
        if inplace is True:
            self.rna = scflow.pp.cluster(self.rna, inplace=True, **kws)
            if self._info["col_celltype"] is None:  # update default column
                self._info["col_celltype"] = col_celltype
        else:
            return scflow.pp.cluster(self.rna, inplace=False, **kws)

    def find_markers(self, n_genes=None, rankby_abs=False,
                     col_celltype=None, key_added=None, plot=True,
                     inplace=True):
        """Find marker genes for clusters."""
        if col_celltype is None:
            col_celltype = self._info["col_celltype"]
        if key_added is None:
            key_added = f"rank_genes_groups_{col_celltype}"
        adata = self.rna if inplace is True else self.rna.copy()
        sc.tl.rank_genes_groups(
            adata, col_celltype, n_genes=n_genes,
            rankby_abs=rankby_abs, key_added=key_added, copy=False)
        if plot is True:
            sc.pl.rank_genes_groups(adata, key=key_added)
        if inplace is False:
            return adata

    def get_markers_df(self, key_celltype=None, col_celltype=None,
                       n_genes=None, p_threshold=None,
                       log2fc_threshold=None,
                       log2fc_threshold_abs=False, **kwargs):
        """Get (a subset of) a marker genes dataframe."""
        if col_celltype is None:
            col_celltype = self._info["col_celltype"]
        key = kwargs.pop("key_added", f"rank_genes_groups_{col_celltype}")
        if key_celltype is None:  # default to retrieving for all cell types
            key_celltype = list(self.rna.obs[col_celltype].unique())
        if isinstance(key_celltype, (int, float, str)):
            key_celltype = [key_celltype]  # ensure iterable
        if isinstance(log2fc_threshold, (int, float)) or (
                log2fc_threshold is None):  # if just max bound or no bound...
            log2fc_threshold = [None, log2fc_threshold]  # convert to mix/max
        marker_df = []  # to concatenate later
        for x in key_celltype:
            if log2fc_threshold_abs is False or (
                    log2fc_threshold is None):  # can just use native function
                tmp = sc.get.rank_genes_groups_df(
                    self.rna, x, key=key, pval_cutoff=p_threshold,
                    log2fc_min=log2fc_threshold[0],
                    log2fc_max=log2fc_threshold[1])
            else:  # must manually figure out cutoffs by absolute l2fc
                raise NotImplementedError(
                    "Code done for absolute value LFC cutoff but untested")
                tmp = sc.get.rank_genes_groups_df(
                    self.rna, x, key=key, pval_cutoff=p_threshold,
                    log2fc_min=None, log2fc_max=None)
                tmp = tmp[tmp["logfoldchanges"].abs() >= log2fc_threshold]
            tmp = tmp.sort_values("pvals_adj")  # ensure sorted by p-values
            if n_genes is not None:
                tmp = tmp.iloc[:(n_genes + 1)]  # top n_genes if wanted
            marker_df += [tmp]  # add to overall list of dfs
        marker_df = pd.concat(marker_df, keys=key_celltype, names=[
            col_celltype]).reset_index(1, drop=True).set_index(
                "names", append=True)  # concatenate across cell types
        return marker_df

    def annotate(self, annotation_guide, col_celltype=None, layer="log1p",
                 col_celltype_new=None, overwrite=False, **kwargs):
        """
        Annotate a clustering result.

        Provide a string representing a CellTypist model .pkl or a
        dictionary where items are sets ({...}) of canonical/a priori
        markers keyed by cell types
        (`reference_markers` in `scanpy.tl.marker_gene_overlap`).
        """
        if col_celltype is None:
            col_celltype = self._info["col_celltype"]
        if col_celltype_new is None:
            col_celltype_new = f"{col_celltype}_annotated"
        if col_celltype_new in self.rna.obs and overwrite is False:
            raise ValueError(f"Cannot overwrite column {col_celltype_new} "
                             "unless overwrite=True")
        # CellTypist Method (annotation_guide is a string)
        if isinstance(annotation_guide, str) and ".pkl" in annotation_guide:
            suff_ct = "" if col_celltype_new == "" else f"_{col_celltype_new}"
            adata = self.rna.copy()  # TODO: not memory efficient...
            if layer is not None:
                adata.X = adata.layers[layer]
            predictions = celltypist.annotate(
                adata, model=annotation_guide, **kwargs)  # CellTypist
            self.rna.obs = self.rna.obs.join(
                predictions.predicted_labels["predicted_labels"].to_frame(
                    f"predicted_labels{suff_ct}"), lsuffix="_o"
                )  # individual cell-level predictions to .obs
            celltypist.dotplot(predictions, use_as_reference=col_celltype,
                               use_as_prediction="predicted_labels")  # plot
            if "majority_voting" in kwargs and kwargs[
                    "majority_voting"] is True:  # majority voting labels >
                self.rna.obs = self.rna.obs.join(
                    predictions.predicted_labels["majority_voting"].to_frame(
                        f"majority_voting{suff_ct}"), lsuffix="_o")
                probs = predictions.probability_matrix.apply(
                    lambda x: np.nan if predictions.predicted_labels.loc[
                        x.name]["majority_voting"] == "Heterogeneous" else x[
                            predictions.predicted_labels.loc[x.name][
                                "majority_voting"]], axis=1)  # MV probability
                self.rna.obs = self.rna.obs.join(probs.to_frame(
                    f"majority_voting_probabilities{suff_ct}"), lsuffix="_o")
            return predictions
        # Marker Overlap Method
        elif marker_genes_dict is not None:
            key = kwargs.pop("key_added", f"rank_genes_groups_{col_celltype}")
            new = f"marker_gene_overlap_{key.split('rank_genes_groups_')[1]}"
            marker_matches = sc.tl.marker_gene_overlap(
                self.rna, annotation_guide, key=key,
                key_added=new, **kwargs)  # detect marker overlap
            new_labels = dict(marker_matches.apply(
                lambda x: " | ".join(np.array(marker_matches.index.values)[
                    np.where(x == max(x))[0]])))  # find where most overlap
            self.rna.obs.loc[:, col_celltype_new] = self.rna.obs[
                col_cell_type].replace(new_labels)  # replace with best match
            return marker_matches
        else:
            NotImplementedError("")
