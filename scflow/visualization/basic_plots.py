#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E. N. Aslinger
"""
Functions for basic plotting.

Use the convention plot_{kind} for function names to enable dynamic
retrieval of plot functions.

@author: E. N. Aslinger
"""

import scanpy as sc


def plot_scatter(adata, color=None, **kwargs):
    """Make a scatterplot."""
    fig = sc.pl.scatter(color=color, **kwargs)
    return fig


def plot_heat(adata, genes=None, col_celltype=None, **kwargs):
    """Plot a gene expression heat map."""
    fig = sc.pl.heatmap(adata, genes, col_celltype, **kwargs)
    return fig


def plot_dot(adata, genes=None, col_celltype=None, return_fig=True, **kwargs):
    """Plot a gene expression dot plot."""
    fig = sc.pl.dotplot(adata, genes, col_celltype,
                        return_fig=return_fig, **kwargs)
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig


def plot_matrix(adata, genes=None, col_celltype=None,
                return_fig=True, **kwargs):
    """Plot a gene expression matrix plot."""
    fig = sc.pl.dotplot(adata, genes, col_celltype,
                        return_fig=return_fig, **kwargs)
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig


def plot_violin(adata, genes=None, col_celltype=None, **kwargs):
    """Plot a gene expression violin plot."""
    fig = sc.pl.violin(adata, genes, col_celltype, **kwargs)
    return fig


def plot_stacked_violin(adata, genes=None, col_celltype=None,
                        return_fig=True, **kwargs):
    """Plot a gene expression stacked violin plot."""
    fig = sc.pl.stacked_violin(adata, genes, col_celltype,
                               return_fig=return_fig, **kwargs)
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig


def plot_umap(adata, color=None, return_fig=True, **kwargs):
    """Plot a UMAP."""
    fig = sc.pl.umap(adata, color=color,
                     return_fig=return_fig, **kwargs)
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig


def plot_clustermap(adata, color=None, return_fig=True, **kwargs):
    """Plot a cluster map."""
    fig = sc.pl.clustermap(adata, color=color, **kwargs)
    if ("show" not in kwargs or kwargs["show"] is True) and (
            return_fig is True):
        fig.show()
    return fig
