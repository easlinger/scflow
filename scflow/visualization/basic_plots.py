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
    show = kwargs["show"] if "show" in kwargs else True
    if title is not None:
        kwargs["show"] = False  # so can return axes
    fig = sc.pl.heatmap(adata, genes, col_celltype, **kwargs)
    fig["heatmap_ax"].set_title(title)
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
    fig = sc.pl.matrixplot(adata, genes, col_celltype,
                           return_fig=return_fig, **kwargs)
    if title is not None:  # title?
        fig.fig_title = title
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig


def plot_violin(adata, genes=None, col_celltype=None,
                col_wrap=None, title=None, figsize=None, **kwargs):
    """Plot a gene expression violin plot."""
    if col_wrap is not None:
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
    else:
        fig = sc.pl.violin(adata, genes, col_celltype, **kwargs)
        fig = plt.gcf()
    if title is not None:  # title?
        fig.suptitle(title)
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


def plot_umap(adata, color=None, return_fig=True, title=None, **kwargs):
    """Plot a UMAP."""
    fig = sc.pl.umap(adata, color=color,
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
