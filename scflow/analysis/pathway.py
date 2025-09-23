#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for running pathway/enrichment analysis.

@author: E. N. Aslinger
"""

# import pdb
from warnings import warn
import functools
import matplotlib.pyplot as plt
import scanpy as sc
import decoupler as dc
import gseapy as gp
import pandas as pd
import numpy as np
import scflow


def run_decoupler(adata, col_celltype, col_condition=None, col_sample=None,
                  resource="MSigDB", species="human", query=None,
                  # query="collection == 'reactome_pathways'",
                  use_hvg=True, inplace=False):
    """Run dc on DEGs."""
    if inplace is False:
        adata = adata.copy()
    rsrc = dc.op.progeny(organism=species.lower()) if (
        resource == "progeny") else dc.op.hallmark(
            organism=species.lower()) if (
                resource == "hallmark") else dc.op.resource(
                    resource, organism=species.lower())  # retrieve resource
    if query is not None:
        rsrc = rsrc.query(query)
    if resource == "MSigDB":
        rsrc = rsrc[~rsrc.duplicated(("geneset", "genesymbol") if (
            "geneset" in rsrc) else ("genesymbol"))]  # drop duplicates
        gs_size = rsrc.groupby("geneset").size()
        gsea_genesets = gs_size.index[(gs_size > 15) & (gs_size < 500)]
        rsrc = rsrc[rsrc["geneset"].isin(gsea_genesets)]
    if col_condition is not None:
        adata.obs.loc[:, "group_dc"] = adata.obs[col_condition].astype(
            "string") + "_" + adata.obs[col_celltype]
    else:
        adata.obs.loc[:, "group_dc"] = adata.obs[col_celltype].copy()
    sc.tl.rank_genes_groups(adata, "group_dc",
                            method="t-test", key_added="t-test")
    t_stats_all = sc.get.rank_genes_groups_df(
        adata, None, key="t-test").set_index("names")
    use_hvg = 1000
    if use_hvg not in [None, False]:
        if use_hvg is not True:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=use_hvg, batch_key=col_sample)
        t_stats_all = t_stats_all.loc[adata.var["highly_variable"]]
    gsea_results = {}
    for g in adata.obs["group_dc"].unique():
        t_stats = t_stats_all[t_stats_all.group == g].sort_values(
            "scores", key=np.abs, ascending=False)[["scores"]].rename_axis([
                "Condition"], axis=1)
        # scores, norm, pvals = dc.mt.gsea(t_stats.T, rsrc)
        gsea_results[g] = dc.mt.gsea(t_stats.T, rsrc)
        # gsea_results[g] = pd.concat({"score": scores.T, "norm": norm.T,
        #                              "pval": pvals.T}, axis=1).droplevel(
        #                                  level=1, axis=1).sort_values("pval")
    # gsea_results = pd.concat(gsea_results, names=[col_celltype])
    return gsea_results


def run_decoupler_ulm(adata, col_celltype, col_condition=None,
                      resource="progeny", query=None, species="human",
                      top_n=3, plot=True, title=None, p_threshold=0.05,
                      title_position=0.6, brackets=True,
                      title_cb="Z-Scaled Scores", inplace=False, **kws_fig):
    """Run ULM."""
    fig = None
    if inplace is False:
        adata = adata.copy()
    if col_condition is not None:
        grps = list(adata.obs[col_condition].unique())
        out = dict(zip(grps, [run_decoupler_ulm(
            adata[adata.obs[col_condition] == g], col_celltype,
            resource=resource, species=species, top_n=top_n, query=query,
            plot=False, inplace=True) for g in grps]))
        if plot is True:
            cmap = kws_fig.pop("cmap", "RdBu_r")
            gridspec_kws, gks = {}, ["top", "bottom", "left",
                                     "right", "wspace", "hspace"]
            for x in [i for i in gks if i in kws_fig]:
                gridspec_kws[x] = kws_fig.pop(x)
            gridspec_kws = {**dict(hspace=1.5, top=0.95), **gridspec_kws}
            non_empty_out = dict(zip(out, [[v for v in out[g][2] if len(
                out[g][2][v]) > 0] for g in out]))
            non_empty_out = [i for i in non_empty_out if len(
                non_empty_out[i]) > 0]  # filter if no significant pathways
            fig, axes = plt.subplots(
                *scflow.pl.square_grid(len(non_empty_out)), **kws_fig,
                gridspec_kw=gridspec_kws)
            for i, g in enumerate(out):
                sgv = list(out[g][0].obs.columns.difference(functools.reduce(
                    lambda u, v: u + v, [out[g][2][q] for q in out[g][2]])))
                out[g][0].obs = out[g][0].obs[sgv]
                # dndr = len(out[g][0].obs[col_celltype].unique()) > 2
                dndr = False
                non_empty = [v for v in out[g][2] if len(out[g][2][v]) > 0]
                gtmp = dict(zip(non_empty, [
                    out[g][2][v] for v in non_empty])) if (
                        brackets is True) else functools.reduce(
                            lambda i, j: i + j, [out[g][2][k]
                                                 for k in non_empty])
                if len(gtmp) < 1:
                    warn(f"No significant pathways for {g}")
                    continue
                print(gtmp)
                m_p = sc.pl.matrixplot(
                    adata=out[g][0], var_names=gtmp,
                    groupby=col_celltype, dendrogram=dndr,
                    var_group_rotation=45,
                    standard_scale="var", show=False, use_raw=False,
                    vmin=0, vmax=1, colorbar_title=title_cb,
                    cmap=cmap, ax=axes.flatten()[i])
                axes.flatten()[i].set_title(g, y=title_position)
                if g != list(out.keys())[-1]:
                    axes.flatten()[i].legend().set_visible(False)
                if i != len(out) - 1 and "color_legend_ax" in m_p:
                    m_p["color_legend_ax"].remove()
            if len(axes.flatten()) > len(non_empty_out):
                for a in axes.flatten()[len(non_empty_out):]:
                    a.set_visible(False)
            if title is not None:  # title?
                fig.suptitle(title)
            fig.show()
            try:
                for x in out:
                    pthwys = functools.reduce(lambda i, j: i + j, [
                        out[x][2][i] for i in out[x][2]])
                    if len(pthwys) == 0:
                        print(f"\n\nNo significant contrasts for {x}")
                        continue
                    scflow.pl.plot_violin(
                        out[x][0], genes=pthwys, col_celltype=col_celltype,
                        figsize=(15, 15), hspace=0.5 * len(pthwys) / 10,
                        wspace=1, title=x, rotation=45)
            except Exception as err:
                print(err)
        return out, fig
    rsrc = dc.op.progeny(organism=species.lower()) if (
        resource == "progeny") else dc.op.hallmark(
            organism=species.lower()) if (
                resource == "hallmark") else dc.op.resource(
                    resource, organism=species.lower())  # retrieve resource
    if query is not None:
        rsrc = rsrc.query(query)
    dc.mt.ulm(data=adata, net=rsrc, tmin=3)
    score = dc.pp.get_obsm(adata, key="score_ulm")
    dcdf = dc.tl.rankby_group(
        adata=score, groupby=col_celltype,
        reference="rest", method="t-test_overestim_var")
    dcdf = dcdf[dcdf["stat"] > 0]
    if p_threshold is not None:
        dcdf = dcdf[dcdf["padj"] < p_threshold]
    ctypes_dict = dcdf.groupby("group").head(top_n).groupby("group")[
        "name"].apply(lambda x: list(x)).to_dict()
    if plot is True:
        sgv = list(score.obs.columns.difference(functools.reduce(
            lambda u, v: u + v, [ctypes_dict[q] for q in ctypes_dict])))
        score.obs = score.obs[sgv]
        if len(ctypes_dict) < 1:
            warn("No significant pathways found.")
        else:
            fig = sc.pl.matrixplot(
                adata=score, var_names=ctypes_dict, groupby=col_celltype,
                dendrogram=True, standard_scale="var",
                colorbar_title=title_cb, cmap="RdBu_r")
            if title is not None:  # title?
                fig.fig_title = title
            fig.show()
    return score, dcdf, ctypes_dict, fig


def run_decoupler_aucell(adata, col_covariates=None, resource="progeny",

                         query=None, species="human", wspace=0.5,
                         col_wrap=True, layer=None, inplace=False, **kws_fig):
    """Run AUCell."""
    if isinstance(col_covariates, str):
        col_covariates = [col_covariates]
    if inplace is False:
        adata = adata.copy()
    rsrc = dc.op.progeny(organism=species.lower()) if (
        resource == "progeny") else dc.op.hallmark(
            organism=species.lower()) if (
                resource == "hallmark") else dc.op.resource(
                    resource, organism=species.lower())  # retrieve resource
    if query is not None:
        rsrc = rsrc.query(query)
    dc.mt.aucell(adata, rsrc, layer=layer, verbose=True)  # run AUCell
    pathways = list(adata.obsm["score_aucell"].columns)
    adata.obs[pathways] = adata.obsm["score_aucell"]
    if col_covariates is not None:
        if col_wrap is True:
            col_wrap = scflow.pl.square_grid(
                len(col_covariates + pathways))[1]
        sc.pl.umap(adata, color=col_covariates + pathways, frameon=False,
                   ncols=col_wrap, wspace=wspace)
    return adata


def run_enrichr(df_degs, col_grouping=None, gene_sets=None,
                p_threshold=0.05, cutoff=0.5, top_term=100, species="Human",
                fx_replace_string=None, plot_top_n=10, title=None,
                font_family="serif", fontsize_suptitle="xx-large",
                fontsize_title=10, labelsize=None,
                plot=True, show_ring=True, **kwargs):
    """Run Enrichr on DEGs."""
    gridspec_kws, gks = {}, ["top", "bottom", "left",
                             "right", "wspace", "hspace"]
    for x in [i for i in gks if i in kwargs]:
        gridspec_kws[x] = kwargs.pop(x)
    if "log_fc" not in df_degs and "logfoldchanges" in df_degs:
        df_degs.loc[:, "log_fc"] = df_degs["logfoldchanges"]
    gridspec_kws = {**dict(wspace=1.5, top=0.95), **gridspec_kws}
    names = gp.get_library_name()
    fig, axes = None, None
    labelsize = {} if labelsize is None else labelsize
    labelsize["x"] = labelsize["x"] if "x" in labelsize else 10
    labelsize["y"] = labelsize["y"] if "y" in labelsize else 8
    if gene_sets is None:
        gene_sets = names  # use all gene sets if unspecified
    gene_sets = [gene_sets] if isinstance(gene_sets, str) else {
        **gene_sets} if isinstance(gene_sets, dict) else list(gene_sets)
    if not isinstance(gene_sets, dict):  # if same sets for up/down
        gene_sets = {"up": gene_sets, "down": gene_sets}
    for x in gene_sets:
        gmiss = list(set(gene_sets[x]).difference(set(names)))
        if len(gmiss) > 0:
            warn(f"Some gene sets ({gmiss}) not found in Enrichr library. "
                 "See `gseapy.get_library_name()` for available gene sets.")
            gene_sets[x] = list(set(gene_sets[x]).intersection(set(names)))
    # ixs = [i for i in list([col_grouping] if isinstance(
    #     col_grouping, str) else col_grouping) if i in df_degs.index.names]
    print(gene_sets)
    if col_grouping is None:
        col_grouping = "dummy_col"
        df_degs = df_degs.assign(
            dummy_col=1)  # make generalizable for no group (iterable)
    if isinstance(col_grouping, list) and len(col_grouping) > 1:
        if len(col_grouping) > 2:
            raise NotImplementedError("> 2 grouping columns not supported.")
        if col_grouping[0] in df_degs.index.names:  # if grouping column in ix
            df_degs = df_degs.reset_index(col_grouping[0])
        grps = df_degs[col_grouping[0]].unique()
        kws = dict(col_grouping=col_grouping[1], gene_sets=gene_sets,
                   p_threshold=p_threshold, cutoff=cutoff, top_term=top_term,
                   species=species, fx_replace_string=fx_replace_string,
                   font_family=font_family, plot_top_n=plot_top_n,
                   fontsize_suptitle=fontsize_suptitle,
                   fontsize_title=fontsize_title,
                   labelsize=labelsize, plot=False, **kwargs)
        results = dict(zip(grps, [run_enrichr(
            df_degs[df_degs[col_grouping[0]] == g],
            title=g, **kws) for g in grps]))  # recurse over grouping 1 values
        pathways = dict(zip(results, [results[x][0] for x in results]))
        res_pathways = pd.concat([results[x][1] for x in results],
                                 keys=results, names=[col_grouping[0]])
        res_pathways_top = res_pathways.reset_index().groupby(
            col_grouping + ["Direction"]).apply(lambda x: x.sort_values(
                "Adjusted P-value", ascending=True).head(
                    plot_top_n), include_groups=False)  # top n
        grps_1, grps_2 = [res_pathways_top.reset_index(u)[u].unique(
            ) for u in col_grouping]
        if plot is True:
            fig, axes = plt.subplots(len(grps_1), len(grps_2), figsize=(
                20, 25), squeeze=False)
            for i, x in enumerate(grps_1):
                for j, y in enumerate(grps_2):
                    if y in res_pathways_top.loc[x].index:
                        gp.dotplot(
                            res_pathways_top.loc[x].loc[y].reset_index(),
                            column="Adjusted P-value", x="Direction",
                            y="Term Short", ax=axes[i, j], size=20,
                            top_term=100, title=x if x != "INDICATOR" else "",
                            xticklabels_rot=45, yticklabels_rot=45,
                            show_ring=True, marker="o", cutoff=0.5)  # dotplot
                        axes[i, j].tick_params(axis="y", labelsize=8)  # y
                        axes[i, j].tick_params(axis="x", labelsize=8)  # x
                        axes[i, j].title.set_fontsize(6)  # title
                    else:
                        axes[i, j].set_visible(False)
                        axes[i, j].set_title(y)  # title
                if len(axes[i, :]) > len(grps_1):
                    for a in axes[i, :].flatten()[len(grps_1):]:
                        a.set_visible(False)
            kws_title = dict(fontsize=fontsize_suptitle, fontproperties=dict(
                family=font_family))  # suptitle
            if title is not None:
                fig.suptitle(title, **kws_title)  # suptitle
        return pathways, res_pathways, (fig, axes)
    else:
        if col_grouping in df_degs.index.names:  # if grouping column in ix
            df_degs = df_degs.reset_index(col_grouping)
        pathways, res_pathways = {}, {}
        for x in df_degs[col_grouping].unique():  # iterate
            pathways[x], res_pathways[x] = {}, {}
            for i in ["up", "down"]:
                degs_tmp = df_degs[df_degs[col_grouping] == x]
                degs_tmp = list(degs_tmp[(degs_tmp.log_fc > 0) if (
                    i == "up") else (degs_tmp.log_fc < 0)].index.values)
                if len(degs_tmp) > 0:
                    pathways[x][i] = gp.enrichr(
                        gene_list=degs_tmp, gene_sets=gene_sets[i],
                        organism=species, cutoff=cutoff, **kwargs)
                    res_pathways[x][i] = pathways[x][i].results[
                        pathways[x][i].results[
                            "Adjusted P-value"] < p_threshold]
            res_pathways[x] = pd.concat(res_pathways[x], names=["Direction"])
    # pdb.set_trace()
    res_pathways = pd.concat(res_pathways, names=[col_grouping])  # all groups
    res_pathways.loc[:, "Term Short"] = res_pathways.Term.apply(
        lambda x: fx_replace_string(x)) if (
            fx_replace_string is not None) else res_pathways.Term  # shortened
    cts_i = res_pathways.reset_index()[col_grouping].unique()
    res_pathways_top = res_pathways.reset_index().groupby([
        col_grouping, "Direction"]).apply(lambda x: x.sort_values(
            "Adjusted P-value", ascending=True).head(plot_top_n),
                                          include_groups=False)  # top n
    if plot is True:
        fig, axes = plt.subplots(*scflow.pl.square_grid(cts_i), figsize=(
            20, 25), gridspec_kw=gridspec_kws, squeeze=False)
        for i, x in enumerate(cts_i):
            gp.dotplot(
                res_pathways_top.loc[x].reset_index(),
                column="Adjusted P-value", x="Direction", y="Term Short",
                ax=axes.flatten()[i], size=20, top_term=top_term,
                title=x if x != "INDICATOR" else "", xticklabels_rot=45,
                yticklabels_rot=45, show_ring=show_ring,
                marker="o", cutoff=cutoff)  # dotplot
            axes.flatten()[i].tick_params(axis="y", labelsize=labelsize["y"],
                                          labelfontfamily=font_family)  # y
            axes.flatten()[i].tick_params(axis="x", labelsize=labelsize["x"],
                                          labelfontfamily=font_family)  # x
            axes.flatten()[i].title.set_fontsize(fontsize_title)  # title
        if len(axes.flatten()) > len(cts_i):
            for a in axes.flatten()[len(cts_i):]:
                a.set_visible(False)
        kws_title = dict(fontsize=fontsize_suptitle,
                         fontproperties=dict(family=font_family))  # suptitle
        if title is not None:
            fig.suptitle(title, **kws_title)  # suptitle
    res_pathways = res_pathways[["Term Short", "Genes"] + list(
        res_pathways.columns.difference(["Genes", "Term Short"]))]
    return pathways, res_pathways, (fig, axes)
