# __init__.py
# pylint: disable=unused-import

from warnings import warn
from anndata import AnnData
import scanpy as sc
from scipy.sparse import issparse
from mudata import MuData
import celltypist
import scflow
# from scflow import Regression
# from scflow.processing import import_data
import pandas as pd
import numpy as np

layer_log1p = "log1p"  # TODO: MOVE THESE TO CONSTANTS MODULE
layer_counts = "counts"
layer_scaled = "scaled"


class Rna(object):
    """A class for single-cell RNA-seq data."""

    def __init__(self, file_path=None, col_sample=None, col_batch=None,
                 kws_read=None, col_celltype=None, kws_integrate=None,
                 assay=None, **kwargs):
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
                col_sample = "sample"
            if not isinstance(kws_read, list):  # if not sample-specific kws
                kws_read = [kws_read] * len(file_path)  # use same for all
            kws_read = dict(zip(list(file_path.keys()), kws_read))
            adata = [file_path[x] if isinstance(file_path[x], (
                AnnData, MuData)) else scflow.pp.read_scrna(
                    file_path[x], **kws_read[x]) for x in file_path]  # read
            adata = scflow.pp.integrate(
                adata, col_sample=col_sample, col_batch=col_batch,
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
        self._info = {"col_sample": col_sample,
                      "col_batch": col_batch,
                      "kws_read": kws_read,
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

    def plot(self, kind=None, col_celltype=None, genes=None, layer=None,
             color=None, subset=None,
             figsize=None, return_fig=False, **kwargs):
        """Plot."""

        # Process Arguments
        adata = self.rna
        if subset is not None:
            adata = adata[subset]
        if col_celltype is None:
            col_celltype = self._info["col_celltype"]
        gby = ["matrix", "violin", "stacked_violin", "heat", "dot"]
        genes_plots = ["dot", "tracks", "matrix", "heat",
                       "stacked_violin", "violin"]
        colors_plots = ["scatter", "umap"]
        if kind is None:  # if no plot kind specified
            kind = ["umap", "heat", "violin"]
        if isinstance(kind, str):  # ensure consistency if just 1 plot kind
            if kind not in kwargs:
                kwargs = {kind: kwargs}  # make kwargs keyed ~ kind
            kind = [kind]  # ensure list, even if just one plot kind
        else:  # sort & validate shared & plot-specific kwargs (multi-plot)
            k_shared = [k for k in kwargs if k not in kind]  # passed directly
            if len(k_shared) > 0:  # if any plot kind non-specific kws...
                kws_shared = dict(zip(k_shared, [
                    kwargs[k] for k in k_shared]))  # extract plot-nonspecific
                kwargs = dict(zip(kind, [kwargs[k] if (
                    k in kwargs) else {} for k in kind]))  # plot-specific
                for s in kws_shared:  # iterate kwargs not passed by plot kind
                    for k in kind:  # add shared kwargs under each plot kind
                        if s not in kwargs[k]:  # if kw not specified for kind
                            kwargs[k][s] = kws_shared[s]  # add under kind kws
                        else:
                            warn(f"Shared kwarg {s} already in plot-specific "
                                 f"arguments for {k}. Keeping {kwargs[k][s]}")
            kind = [k.lower() for k in kind]  # make not case-sensitive
        fig = {}  # to hold plots
        for k in kind:  # iterate plot kinds
            f_x = scflow.get_plot_fx(k)  # get the right plot function
            kwargs[k]["layer"] = layer
            if k in genes_plots and "genes" not in kwargs[k]:  # if needed...
                kwargs[k].update({"genes": genes})  # ...specify "genes"
            if k in colors_plots and "color" not in kwargs[k]:  # if needed...
                kwargs[k].update({"color": color})  # ...specify grouping
            if k in gby and "col_celltype" not in kwargs[k]:
                kwargs[k]["col_celltype"] = col_celltype  # specify cell type
            if "violin" not in k and k != "umap":
                kwargs[k]["figsize"] = figsize
            if subset is not None and "dendrogram" in kwargs[k] and kwargs[
                    k]["dendrogram"] is True and "col_celltype" in kwargs[k]:
                adata = adata.copy()
                sc.tl.dendrogram(adata, kwargs[k]["col_celltype"])
            fig[k] = f_x(adata, **kwargs[k])
        if return_fig is True:
            return fig

    def get_gex_matrix(self, genes=None, subset=None):
        """Get gene expression matrix."""
        adata = self.rna if subset is None else self.rna[subset]
        g_original = [genes] if isinstance(genes, str) else genes
        genes = None if genes is None else list(set(
            genes).intersection(adata.var_names))  # valid gene names
        if g_original is not None and len(g_original) > genes:
            warn(f"Genes not found: {set(genes).difference(g_original)}")
        expr = adata.X if genes is None else adata.var[:, genes].X
        expr = expr.toarray() if issparse(expr) else expr
        return pd.DataFrame(expr, columns=genes)

    def preprocess(self, inplace=True, **kws_pp):
        """Filter, normalize, and perform QC on data."""
        if inplace is True:
            self.rna = scflow.pp.preprocess(
                self.rna, inplace=True, **kws_pp)
        else:
            return scflow.pp.preprocess(self.rna, inplace=False, **kws_pp)

    def cluster(self, col_celltype="leiden", layer="log1p",
                resolution=1, min_dist=0.5,
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

    def annotate(self, annotation_guide, marker_genes_dict=None,
                 col_celltype=None, layer=None,
                 col_celltype_new=None, overwrite=False,
                 inplace=True, **kwargs):
        """
        Annotate a clustering result.

        Provide a string representing a CellTypist model .pkl or a
        dictionary where items are sets ({...}) of canonical/a priori
        markers keyed by cell types
        (`reference_markers` in `scanpy.tl.marker_gene_overlap`).
        """
        # TODO: Overwriting columns unsophisticated (will fail in some cases)
        if col_celltype is None:
            col_celltype = self._info["col_celltype"]
        if col_celltype_new is None:
            col_celltype_new = f"{col_celltype}_annotated"
        if col_celltype_new in self.rna.obs and overwrite is False:
            raise ValueError(f"Cannot overwrite column {col_celltype_new} "
                             "unless overwrite=True")
        adata = self.rna.copy() if inplace is False else self.rna
        # CellTypist Method (annotation_guide is a string)
        if isinstance(annotation_guide, str) and ".pkl" in annotation_guide:
            suff_ct = "" if col_celltype_new == "" else f"_{col_celltype_new}"
            if layer != layer_log1p:
                warn("Layer for CellTypist should be log-normalized, total "
                     "count-normalized (target_sum=10000) layer. "
                     f"Specified layer = {layer} (expected = {layer_log1p})")
                if layer is None:
                    layer = layer_log1p
                    warn(f"Changing layer to {layer_log1p}")
            predictions, adata = scflow.pp.run_celltypist(
                adata, annotation_guide, col_celltypist_suffix=suff_ct,
                layer=layer, col_celltype=col_celltype, **kwargs)
            if inplace is True:
                self.rna = adata
                return predictions
            else:
                return adata, predictions
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
                col_celltype].replace(new_labels)  # replace with best match
            return marker_matches
        else:
            NotImplementedError("")


def run_mapmycells():
    """Run Map My Cells (Brain Atlas)."""
    # Write Object & Rectify Gene Names (to EnsemblIDs)
    if overwrite is True or not os.path.exists(file_new):
        os.makedirs("data", exist_ok=True)
        # self.rna.X = self.rna.layers["counts"]
        self.rna.var_names_make_unique()
        if "ENSMUSG00000118396" in self.rna.var_names and (
                "Iqcf3" in self.rna.var_names):
            self.rna = self.rna[:, list(set(self.rna.var_names).difference(
                ["ENSMUSG00000118396"]))]  # drop duplicate gene to avoid error?
        self.rna.write_h5ad(file_new)
    else:
        raise ValueError("Must be able to write to use My Cell Mapper")
    # self.rna.var_names = var_names_orig
    # self.rna.write_h5ad("scratch/tmp.h5ad")
    # self.rna.var_names = var_names_orig
    out_file = "scratch/tmp.h5ad"
    os.system(
        f"python -m cell_type_mapper.cli.validate_h5ad --input_path {file_new} "
        "--layer counts --output_json scratch/out.json "
        f"--valid_h5ad_path {out_file}")

    # Configuration
    baseline_precomp_path = "resources/precomputed_stats_ABC_revision_230821.h5"
    baseline_marker_path = "resources/mouse_markers_230821.json"
    baseline_json_output_path = "scratch/baseline_json_mapping_output.json"
    baseline_csv_output_path = "scratch/baseline_csv_mapping_output.csv"
    baseline_mapping_config = {
        "query_path": out_file, "tmp_dir": "scratch",
        "extended_result_path": str(baseline_json_output_path),
        "csv_result_path": str(baseline_csv_output_path),
        "max_gb": 10, "cloud_safe": False, "verbose_stdout": False,
        "type_assignment": {
            "normalization": "raw",
            "n_processors": n_processors,
            "chunk_size": 10000,
            "bootstrap_iteration": 100,
            "bootstrap_factor": 0.5,
            "rng_seed": 233211
        },
        "precomputed_stats": {"path": str(baseline_precomp_path)},
        "query_markers": {"serialized_lookup": str(baseline_marker_path)},
        "drop_level": None,
    }

    # Subset by Region (if Desired)
    if map_my_cells_region_keys is not None:  # subset by region
        abc_cache = AbcProjectCache.from_cache_dir("scratch")
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
        subset_cells = cell_metadata[pd.concat([
            cell_metadata.region_of_interest_acronym == i
            for i in map_my_cells_region_keys], axis=1).T.any()]  # subset meta
        valid_classes = set([alias_to_truth[x]["CCN20230722_CLAS"]
                            for x in subset_cells.cluster_alias.values])
        classes_to_drop = list(set([alias_to_truth[x][
            "CCN20230722_CLAS"] for x in alias_to_truth if alias_to_truth[x][
                "CCN20230722_CLAS"] not in valid_classes]))
        nodes_to_drop = [("class", x) for x in classes_to_drop]
        baseline_mapping_config.update({
            "nodes_to_drop": nodes_to_drop,
            "drop_level": "CCN20230722_SUPT"})
        print("=======Nodes Being Dropped=======")
        for pair in nodes_to_drop[:4]:
            print(pair)

    # Run Mapper
    mapping_runner = FromSpecifiedMarkersRunner(
        args=[], input_data=baseline_mapping_config)
    mapping_runner.run()

    # Programmatic Runner
    # config_path = "scratch/config.json"
    # with open(config_path, "w") as f:
    #     json.dump(baseline_mapping_config, f, indent=2)
    # os.system("python -m cell_type_mapper.cli.from_specified_markers "
    #           f"--input_json {config_path}")

    # Output & Clean Up
    cellmap = pd.read_csv(
        "scratch/baseline_csv_mapping_output.csv", skiprows=4).set_index(
            "cell_id").rename_axis(self.rna.obs.index.names)
    cellmap.columns = [f"cellmap_{i}" for i in cellmap]  # cellmap_ column prefix
    self.rna.obs = self.rna.obs.join(cellmap).loc[self.rna.obs.index]  # join
    for x in ["cellmap_class_name", "cellmap_subclass_name"]:
        self.rna.obs.loc[:, f"{x}"] = self.rna.obs[x].apply(
            lambda x: " ".join(x.split(" ")[1:]) if all((
                i in [str(i) for i in np.arange(0, 10)] for i in x.split(
                    " ")[0])) else x)  # drop pointless #s in front of cell types
    os.system(f"rm {out_file}")  # remove temporary h5ad input
