#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E. N. Aslinger
"""
Functions for basic plotting.

Use the convention plot_{kind} for function names to enable dynamic
retrieval of plot functions.

@author: E. N. Aslinger
"""

import matplotlib.pyplot as plt
import scanpy as sc
import math
import numpy as np
import pandas as pd
import scflow


def get_kws_gridspec(kwargs):
    """Get gridspec keywords from kwargs."""
    gss = ["left", "bottom", "right", "top", "wspace", "hspace"]
    if any((k in kwargs for k in gss)):
        kws_gs = {}
        for k in [i for i in gss if i in kwargs]:
            kws_gs[k] = kwargs.pop(k)
    else:
        kws_gs = None if "gridspec_kw" not in kwargs else kwargs.pop(
            "gridspec_kw")
    return kws_gs


def square_grid(n_subs):
    """Create a square grid for plotting."""
    n_subs = len(n_subs) if isinstance(n_subs, (
        list, tuple, np.ndarray, pd.Series)) else n_subs
    root = int(math.sqrt(n_subs))
    rows, cols = root, root
    while rows * cols < n_subs:
        if cols <= rows:
            cols += 1
        else:
            rows += 1
    if rows == 1 and cols > 2:
        rows, cols = 2, math.ceil(n_subs / 2)
    if cols == 1 and rows > 2:
        cols, rows = 2, math.ceil(n_subs / 2)
    if rows == 2 and cols == 2 and n_subs == 3:
        rows, cols = 1, 3
    return rows, cols


def plot_scatter(adata, var_x, var_y, title=None, **kwargs):
    """Make a scatterplot."""
    fig = sc.pl.scatter(adata, x=var_x, y=var_y, title=title, **kwargs)
    return fig


def plot_heat(adata, genes=None, col_celltype=None, title=None, **kwargs):
    """Plot a gene expression heat map."""
    if title is True:  # construct default title
        title = "Gene Expression"
        scale = (col_celltype if kwargs[
            "standard_scale"] == "obs" else "Gene") if (
                "standard_scale" in kwargs) else None
        title += f" (Scaled by {scale})" if scale else f" by {col_celltype}"
    if "standard_scale" in kwargs and kwargs["standard_scale"] == "group":
        kwargs["standard_scale"] = "obs"  # convert if used matrixplot style
    fontsize = kwargs.pop("fontsize", None)
    show = kwargs["show"] if "show" in kwargs else True
    kws_gs = get_kws_gridspec(kwargs)
    if title is True or isinstance(
            title, str) or kws_gs is not None:  # return figure if need
        kwargs["show"] = False
    fig = sc.pl.heatmap(adata, genes, col_celltype, **kwargs)
    if fig is not None:
        fig = {"scanpy": fig, "fig": fig["heatmap_ax"].get_figure()}
    if title is not None:
        fig["fig"].suptitle(title, fontsize=fontsize)
    if kws_gs is not None:
        fig["fig"].subplots_adjust(**kws_gs)
    if show is True:
        plt.show()
    return fig


def plot_dot(adata, genes=None, col_celltype=None, return_fig=True,
             title=None, **kwargs):
    """Plot a gene expression dot plot."""
    fig = sc.pl.dotplot(adata, genes, col_celltype,
                        return_fig=return_fig, **kwargs)
    if title is not None:  # title?
        fig.fig_title = title
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig


def plot_matrix(adata, genes=None, col_celltype=None,
                return_fig=True, title=None, **kwargs):
    """Plot a gene expression matrix plot."""
    kws_gs = get_kws_gridspec(kwargs)
    if "standard_scale" in kwargs and kwargs["standard_scale"] == "obs":
        kwargs["standard_scale"] = "group"  # convert if used heatmap style
    fontsize = kwargs.pop("fontsize", None)
    show_later = fontsize is not None or kws_gs is not None and (
        "show" not in kwargs or kwargs["show"] is True)
    kwargs["show"] = False if show_later is True else kwargs.get("show", True)
    fig = sc.pl.matrixplot(adata, genes, col_celltype,
                           return_fig=return_fig, **kwargs)
    if fig is not None:
        fig = {"returned": fig}
        fig["scanpy"] = fig["returned"].get_axes(),
        fig["fig"] = fig["returned"].get_axes()["mainplot_ax"].get_figure()
    if title is not None:
        fig["fig"].suptitle(title, fontsize=fontsize)
    if kws_gs is not None:
        fig["fig"].subplots_adjust(**kws_gs)
    # if show_later is True:
    #     fig["returned"].show()
    return fig


def plot_violin(adata, genes=None, col_celltype=None,
                col_wrap=True, title=None,
                fontsize=None, font="serif", figsize=None, **kwargs):
    """Plot a gene expression violin plot."""
    if isinstance(genes, str):
        genes = [genes]
    genes = list(genes)
    fontsize = {} if fontsize is None else fontsize
    fontsize = {**{"title": "large", "subtitle": 10,
                   "x": 8, "y": 8}, **fontsize}
    for u in ["supx", "supy"]:
        if u not in fontsize:
            fontsize[u] = fontsize["x" if u == "supx" else "y"]
    if title is not None and col_wrap is None:
        col_wrap = 1  # workaround so title displays
    if col_wrap is True:
        col_wrap = scflow.pl.square_grid(genes)[1]
    if col_wrap not in [None, False]:
        kws_fig = {"figsize": figsize, "squeeze": False}
        kss = ["sharex", "sharey", "width_ratios", "height_ratios",
               "subplot_kw", "gridspec_kw"]
        for x in kss:
            kws_fig[x] = kwargs.pop(x, False if "share" in x else None)
        gss = ["left", "bottom", "right", "top", "wspace", "hspace"]
        if any((k in kwargs for k in gss)):
            kws_fig["gridspec_kw"] = {} if "gridspec_kw" not in kwargs or (
                kwargs["gridspec_kw"] is None) else kws_fig["gridspec_kw"]
            for x in gss:
                kws_fig["gridspec_kw"][x] = kwargs.pop(x, None)
        fig, axes = plt.subplots(round(len(genes) / col_wrap),
                                 col_wrap, **kws_fig)  # facet grid setup
        kws = {**kwargs, "show": False}  # to avoid showing during iterations
        for i, g in zip(axes.flatten(), genes):  # iterate components to plot
            sc.pl.violin(adata, g, col_celltype, ax=i, **kws)
        i.set_title(g)
        i.title.set_fontsize(fontsize["subtitle"])
        for q in ["x", "y"]:
            i.tick_params(axis=q, labelsize=fontsize[q], labelfontfamily=font)
        if len(axes.flatten()) > len(genes):
            for ax in axes.flatten()[len(genes):]:  # turn off unused axes
                ax.set_visible(False)
    else:
        fig = sc.pl.violin(adata, genes, col_celltype, **kwargs)
        fig = plt.gcf()
    if title is not None:  # title?
        fig.suptitle(title)
    if fontsize["supy"] is None:
        fig.supylabel("")
    else:
        fig.supylabel(fig.get_supylabel(), fontsize=fontsize["supy"],
                      fontproperties=dict(family=font))
    if "show" not in kwargs or kwargs["show"] is True:
        plt.show()
    return fig


def plot_violin_by_group(adata, gene, col_celltype=None, col_condition=None,
                         col_wrap=True, title=None, font="serif",
                         figsize=None, fontsize=None, **kwargs):
    """Plot a gene expression violin plot by groups."""
    if title is not None and col_wrap is None:
        col_wrap = 1  # workaround so title displays
    fontsize = {} if fontsize is None else fontsize
    fontsize = {**{"title": "large", "subtitle": 10,
                   "x": 8, "y": 8}, **fontsize}
    for u in ["supx", "supy"]:
        if u not in fontsize:
            fontsize[u] = fontsize["x" if u == "supx" else "y"]
    for u in ["xlabel", "ylabel"]:
        if u not in fontsize:
            fontsize[u] = fontsize["x" if u == "supx" else "y"]
    # TODO: 2 conditions, rows/columns
    conds = adata.obs[col_condition].cat.categories if isinstance(
        adata.obs[col_condition].dtype,
        pd.CategoricalDtype) else pd.unique(adata.obs[col_condition])
    if col_wrap is True:
        col_wrap = square_grid(len(conds))[1]
    kws_fig = {"figsize": figsize, "squeeze": False}
    kss = ["sharex", "sharey", "width_ratios", "height_ratios",
           "subplot_kw", "gridspec_kw"]
    for x in kss:
        kws_fig[x] = kwargs.pop(x, False if "share" in x else None)
    gss = ["left", "bottom", "right", "top", "wspace", "hspace"]
    if any((k in kwargs for k in gss)):
        kws_fig["gridspec_kw"] = {} if "gridspec_kw" not in kwargs or (
            kwargs["gridspec_kw"] is None) else kws_fig["gridspec_kw"]
        for x in gss:
            kws_fig["gridspec_kw"][x] = kwargs.pop(x, None)
    fig, axes = plt.subplots(round(len(conds) / col_wrap),
                             col_wrap, **kws_fig)  # facet grid setup
    kws = {**kwargs, "show": False}  # to avoid showing during iterations
    for i, g in zip(axes.flatten(), conds):  # iterate components to plot
        sc.pl.violin(adata[adata.obs[col_condition] == g], gene,
                     col_celltype, ax=i, **kws)
        i.set_title(g)
        i.title.set_fontsize(fontsize["subtitle"])
        if fontsize["ylabel"] is None:
            i.set_ylabel("")
        else:
            i.set_ylabel(i.get_ylabel(), fontsize=fontsize["ylabel"],
                         fontproperties=dict(family=font))
        for q in ["x", "y"]:
            i.tick_params(axis=q, labelsize=fontsize[q], labelfontfamily=font)
    if len(axes.flatten()) > len(conds):
        for ax in axes.flatten()[len(conds):]:  # turn off unused axes
            ax.set_visible(False)
    if title is not None:  # title?
        fig.suptitle(title, fontsize=fontsize["title"],
                     fontproperties=dict(family=font))
    if fontsize["supy"] is None:
        fig.supylabel("")
    else:
        fig.supylabel(fig.get_supylabel(), fontsize=fontsize["supy"],
                      fontproperties=dict(family=font))
    if "show" not in kwargs or kwargs["show"] is True:
        plt.show()
    return fig


def plot_stacked_violin(adata, genes=None, col_celltype=None,
                        return_fig=True, title=None, **kwargs):
    """Plot a gene expression stacked violin plot."""
    fig = sc.pl.stacked_violin(adata, genes, col_celltype,
                               return_fig=return_fig, **kwargs)
    if title is not None:  # title?
        fig.fig_title = title
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig


def plot_umap(adata, color=None, return_fig=True, title=None,
              col_wrap=4, **kwargs):
    """Plot a UMAP."""
    fig = sc.pl.umap(adata, color=color, ncols=col_wrap,
                     return_fig=return_fig, **kwargs)
    if title is not None:  # title?
        fig.suptitle(title)
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig


def plot_clustermap(adata, color=None, return_fig=True, title=None, **kwargs):
    """Plot a cluster map."""
    fig = sc.pl.clustermap(adata, color=color, **kwargs)
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig
