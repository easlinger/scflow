import functools
import scanpy as sc
try:
    import senepy
except ModuleNotFoundError:
    pass


def run_senepy(adata, species="Human", tissue=None, celltype=None,
               overlap_threshold=0, literature_sources=None,
               sengpt_sources=True, plot=True,
               col_celltype=None, col_senscore="senscore",
               use_translator=False, identifiers=None, figsize=None,
               swap_axes=True, standard_scale="var",
               plot_layer=None, inplace=True):
    """Run `senepy`."""
    if figsize is None:
        figsize = (20, 20)
    adata = adata.copy() if inplace is False else adata
    will_merge = tissue is None or celltype is None or len(
        tissue) > 1 or len(celltype) > 1  # need to merge hubs?
    genes, hubs, figs = find_senescence_genes(
        adata, species=species, tissue=tissue, celltype=celltype,
        overlap_threshold=overlap_threshold,
        literature_sources=literature_sources, figsize=figsize,
        sengpt_sources=sengpt_sources, plot=plot,
        swap_axes=swap_axes, col_celltype=col_celltype,
        standard_scale=standard_scale, plot_layer=plot_layer
    )  # find tissue- and/or cell type-specific genes
    if use_translator is True:
        translator = senepy.translator(hub=hubs.hubs, data=adata)
    else:
        translator = None
    hubhubs = functools.reduce(lambda i, j: i + j, [
        hubs.hubs[x] for x in hubs.hubs]) if will_merge is True else hubs.hubs
    adata.obs.loc[:, col_senscore] = senepy.score_all_cells(
        adata, hubhubs, identifiers=identifiers,
        translator=translator)
    return adata, genes, figs


def find_senescence_genes(adata, species="Human", tissue=None, celltype=None,
                          overlap_threshold=0, literature_sources=None,
                          sengpt_sources=True, col_celltype=None,
                          plot=True, figsize=None, plot_layer=None,
                          swap_axes=True, standard_scale="var"):
    """Use `senepy` to find tissue- and/or cell type-specific genes."""
    figs = {}  # to hold figures
    hubs = senepy.load_hubs(species=species)  # load hub
    metadata = hubs.metadata  # extract metadata
    tissue, celltype = [[x] if isinstance(x, str) else None if (
        x is None) else x for x in [tissue, celltype]]  # ensure iterable
    literature_sources = [literature_sources] if isinstance(
        literature_sources, str) else literature_sources
    # TODO: Allow tissue-specific cell type specification
    if tissue is not None:  # filter by tissue if specified
        metadata = metadata[metadata.tissue.isin(tissue)]
    if celltype is not None:  # filter by cell type if specified
        metadata = metadata[metadata.cell.isin(celltype)]
    if tissue is None or celltype is None or len(
            tissue) > 1 or len(celltype) > 1:  # if >1 tissue or cell type...
        hub_key = str("" if tissue is None else "_".join(tissue) + "|") + str(
            "" if celltype is None else "_".join(celltype))
        hubs.merge_hubs(metadata, new_name=hub_key,
                        overlap_threshold=overlap_threshold)  # ...merge hubs
    genes = []
    print(metadata)
    if literature_sources is not False:  # if want to include literature genes
        if literature_sources is not None:  # filter literature genes ~ source?
            genes += [x for x in hubs.literature_markers if (
                hubs.literature_markers[x] in literature_sources)]
        else:  # otherwise, include all literature markers
            genes += list(hubs.literature_markers.keys())
    if sengpt_sources is not False:
        genes += hubs.senGPT
    genes = [g for g in genes if g in adata.var_names]  # present in adata
    if plot is True and col_celltype is not None:
        figs["heat"] = sc.pl.heatmap(
            adata, genes, col_celltype, swap_axes=swap_axes, figsize=figsize,
            layer=plot_layer, standard_scale=standard_scale)
        figs["dot"] = sc.pl.dotplot(
            adata, genes, col_celltype, swap_axes=swap_axes,
            standard_scale=standard_scale, layer=plot_layer)
    return genes, hubs, figs
