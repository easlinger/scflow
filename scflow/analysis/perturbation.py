#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for analyzing perturbation effects.

@author: E. N. Aslinger
"""

import matplotlib.pyplot as plt
import seaborn as sns
# import jax
from warnings import warn
from PIL import Image
import tempfile
import os
import scanpy as sc
import pertpy as pt
import pandas as pd
import numpy as np
import scflow


def analyze_perturbation_distance(adata, col_condition, key_pca="X_pca",
                                  layer="log1p", cmap="Reds", figsize=None,
                                  show_progressbar=True, kwargs_plot=None,
                                  inplace=True, **kwargs):
    """Analyze the perturbation effect of treatment(s)."""
    distance = pt.tl.Distance("mse", obsm_key="X_pca")
    if layer is not None:
        warn(f"Changing layer to {layer}...")
        adata.X = adata.layers[layer].copy()  # change layer
    if kwargs_plot is None:
        kwargs_plot = {}
    if isinstance(col_condition, str):
        col_condition = [col_condition]
    dfs = []
    # Run Distance Calculation
    for x in col_condition:  # iterate condition columns in case > 1e
        dfs += [distance.pairwise(
            adata, groupby=x, show_progressbar=show_progressbar, **kwargs)]
    # Ensure Same Heatmap Scale
    g_max, g_min = [f(*[x.max(axis=None) for x in dfs]) for f in [max, min]]
    # Create Figures
    figs = {}
    for i, x in zip(col_condition, dfs):
        f_s = figsize if figsize else (len(adata.obs[i].unique()) * 2, ) * 2
        figs[i], a_x = plt.subplots(figsize=f_s)
        sns.heatmap(x, annot=True, fmt=".2f", vmin=g_min, vmax=g_max,
                    cmap=cmap, ax=a_x, **kwargs_plot)
        figs[i].show()
        plt.show()
    return dfs, figs


def analyze_composition(adata, col_celltype, col_condition, col_sample=None,
                        formula=None, generate_sample_level=True,
                        key_modality="coda", reference_cell_type="automatic",
                        absence_threshold=0.1, est_fdr=0.05, plot_facets=True,
                        palette="tab20", label_rotation=90, full_hmc=False,
                        level_order=None, figsize=None, seed=0, **kwargs):
    """Analyze perturbation-related shifts in cell type composition."""
    if figsize is None:
        figsize = (20, 20)
    figs = {}
    if isinstance(col_condition, str):
        col_condition = [col_condition]
    if len(col_condition) > 1:
        print(f"\n\n***Analyzing cell type composition, {col_condition[0]} "
              f"= main group with {col_condition[1:]} as covariates.")
    if formula is None:
        # formula = "(" + ", ".join(col_condition) + ")" if len(
        #     col_condition) > 1 else col_condition
        formula = " + ".join(col_condition) if len(
            col_condition) > 1 else col_condition[0]
    sccoda_model = pt.tl.Sccoda()
    sccoda_data = sccoda_model.load(
        adata, type="cell_level", generate_sample_level=generate_sample_level,
        cell_type_identifier=col_celltype, sample_identifier=col_sample,
        covariate_obs=col_condition)
    sccoda_data = sccoda_model.prepare(
        sccoda_data, modality_key=key_modality, formula=formula,
        reference_cell_type=reference_cell_type,
        automatic_reference_absence_threshold=absence_threshold)
    sccoda_model = pt.tl.Sccoda()
    for c in col_condition:
        figs["box"] = sccoda_model.plot_boxplots(
            sccoda_data, modality_key=key_modality, feature_name=c,
            add_dots=True, plot_facets=plot_facets, palette=palette,
            level_order=level_order, return_fig=True)
        if label_rotation not in [None, False]:  # rotate axis labels?
            for a in figs["box"].fig.axes:
                a.tick_params(axis="x", labelrotation=label_rotation)
        plt.show()
    # kws_nuts = {
    #     "rng_key": jax.random.key(seed) if full_hmc is True else seed}
    kws_nuts = {"rng_key": seed}
    for x in [i for i in ["num_warmup", "num_samples"] if i in kwargs]:
        kws_nuts[x] = kwargs.pop(x)
    f_x = sccoda_model.run_hmc if full_hmc is True else sccoda_model.run_nuts
    f_x(sccoda_data, **kws_nuts, modality_key=key_modality)  # MCMC
    if est_fdr not in [None, False]:
        sccoda_model.set_fdr(sccoda_data, modality_key=key_modality,
                             est_fdr=est_fdr)  # FDR
    sccoda_model.summary(sccoda_data, modality_key=key_modality,
                         extended=True)
    credible_effects = sccoda_model.credible_effects(
        sccoda_data, modality_key=key_modality, est_fdr=est_fdr)
    cred_tmp = credible_effects.unstack(0).replace(
        False, "").replace(True, "*")
    print(f"\n\n{'=' * 50}   Credible Effects   {'=' * 50}\n\n",
          f"{cred_tmp}\n\n{'=' * 122}\n\n\n")
    # try:
    #     figs["effects"]   = sccoda_model.plot_draw_effects(
    #         sccoda_data, col_condition, modality_key=key_modality,
    #         show_legend=None, show_leaf_effects=True, tight_text=False,
    #         show_scale=False, figsize=figsize, dpi=100,
    #         save=False, return_fig=True)
    # except Exception as e:
    #     print(e)
    # def _run_comps(sccoda_data, sccoda_model, adata, reference,
    #                col_condition, formula, key_modality="coda",
    #                est_fdr=0.1):
    #     """Run comparisons (adapted from `pertpy` tutorial)."""
    #     comparison_groups = [g for g in adata.obs[col_condition[
    #         0]].unique() if g != reference]
    #     effect_df = pd.DataFrame(
    #         {"log2-fold change": [], "Cell Type": [], "Reference": [],
    #          "Comp. Group": [], "Final Parameter": []})
    #     for comp_group in comparison_groups:
    #         print(sccoda_data[key_modality].varm)
    #         group_effects = sccoda_data[key_modality].varm[
    #             f"effect_df_{formula}[{comp_group}]"][[
    #                 "log2-fold change", "Final Parameter"]]
    #         group_effects = group_effects[
    #             group_effects["Final Parameter"] != 0]
    #         group_effects["Cell Type"] = group_effects.index
    #         group_effects["Reference"] = reference
    #         group_effects["Comp. Group"] = comp_group
    #         effect_df = pd.concat([effect_df, group_effects])
    #     if not effect_df.empty:
    #         fig = sccoda_model.plot_effects_barplot(
    #             sccoda_data, return_fig=True, show=False)
    #         fig.set_size_inches(12, 4)
    #         fig.show()
    #     else:
    #         print(f"No significant effects for reference {reference}")
    #     return effect_df

    # credible_effects = pd.DataFrame({
    #     "log2-fold change": [], "Cell Type": [], "Reference": [],
    #     "Comp. Group": [], "Final Parameter": []})
    # for reference in adata.obs[col_condition[0]].unique():
    #     effect_df = _run_comps(sccoda_data, sccoda_model, adata, reference,
    #                            col_condition, formula,
    #                            key_modality=key_modality, est_fdr=est_fdr)
    #     credible_effects = pd.concat([credible_effects, effect_df])
    return sccoda_model, sccoda_data, credible_effects, figs


def analyze_composition_tree(adata, col_celltype, col_covariates, col_sample,
                             layer="counts", key_control=None,
                             col_celltype_hierarchy=None,
                             dendrogram_key=None, est_fdr=0.05,
                             model_type="cell_level", formula=None,
                             reference_cell_type="automatric",
                             inplace=False, seed=0, figsize=None, **kwargs):
    """Analyze shifts in cell type composition with Tasccoda."""
    if inplace is False:
        adata = adata.copy()
    if figsize is None:
        figsize = (800, 800)
    plot_kws = {"show_legend": False, "show_leaf_effects": True}
    for x in plot_kws:
        plot_kws[x] = kwargs.pop(x) if x in kwargs else plot_kws[x]
    figs = {}
    key_added = kwargs.pop("key_added", "tree")
    # key_dend = kwargs.pop("dendrogram_key", f"dendrogram_{col_celltype}")
    adata.obs = adata.obs.assign(**{col_celltype: pd.Categorical(
        adata.obs[col_celltype])})  # to categorical
    # if key_dend not in adata.uns:
    #     sc.tl.dendrogram(adata, col_celltype, inplace=True,
    #                      key_added="dendrogram_cell_label")
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    if isinstance(col_covariates, str):
        col_covariates = [col_covariates]
    if key_control is None:
        key_control = {}
        for i in col_covariates:
            if pd.api.types.is_categorical_dtype(adata.obs[i]):
                key_control[i] = adata.obs[i].cat.categories[0]
    if isinstance(key_control, list):
        key_control = dict(zip(col_covariates, key_control))
    if formula is None:
        formula = " + ".join(col_covariates) if len(
            col_covariates) > 1 else col_covariates[0]
    coda_key = kwargs.pop("modality_key_2", "coda")
    kws_prep = {"formula": formula}
    kprep = ["automatic_reference_absence_threshold", "pen_args"]
    for x in [i for i in kprep if i in kwargs]:
        kws_prep[x] = kwargs.pop(x)
    if "add_level_name" not in kwargs:
        kwargs["add_level_name"] = False
    kws_nuts = {"rng_key": seed}
    for x in [i for i in ["num_warmup", "num_samples"] if i in kwargs]:
        kws_nuts[x] = kwargs.pop(x)
    tasccoda_model = pt.tl.Tasccoda()
    adata = tasccoda_model.load(
        adata, type=model_type, cell_type_identifier=col_celltype,
        levels_orig=col_celltype_hierarchy,
        sample_identifier=col_sample, covariate_obs=col_covariates,
        key_added=key_added,
        dendrogram_key=dendrogram_key, **kwargs)
    tasccoda_model.plot_draw_tree(adata[coda_key])
    # tasccoda_model.plot_boxplots(adata, modality_key="coda_LP",
    #                              feature_name="Health", figsize=(20, 8))
    # plt.show()
    adata = tasccoda_model.prepare(
        adata, modality_key=coda_key, tree_key=key_added,
        reference_cell_type=reference_cell_type, **kws_prep)
    tasccoda_model.run_nuts(adata, modality_key=coda_key, **kws_nuts)  # MCMC
    if est_fdr not in [None, False]:
        tasccoda_model.set_fdr(adata, modality_key=coda_key,
                               est_fdr=est_fdr)  # FDR
    tasccoda_model.summary(adata, modality_key=coda_key, extended=True)
    try:
        figs["barplot"] = tasccoda_model.plot_effects_barplot(
            adata, modality_key=coda_key,
            covariates=col_covariates, return_fig=True)
    except Exception as err:
        warn(f"Tasccoda plot bar failed!\n{err}")
    figs["effects"] = {}
    try:
        cov_keys = [i.split("_node")[0] for i in adata[coda_key].uns[
            "scCODA_params"]["node_df"].reset_index()["Covariate"].unique()]
        for x in cov_keys:
            try:
                figs["effects"][x] = tasccoda_model.plot_draw_effects(
                    adata, x, modality_key=coda_key, tree=key_added,
                    **plot_kws, return_fig=True)
                tree, treestyle = figs["effects"][x]
                with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False) as tmp:
                    filename = tmp.name
                tree.render(filename, w=figsize[0], units="px",
                            tree_style=treestyle)
                img = Image.open(filename)
                fig, a_x = plt.subplots()
                figs["effects"][f"{x}_figax"] = fig, a_x
                fig.suptitle(x)
                a_x.imshow(img)
                a_x.axis("off")
                os.remove(filename)
            except Exception as err:
                warn(f"Tasccoda plot effects failed for {x}!\n{err}")
    except Exception as err:
        warn(f"Tasccoda plot effects failed!\n{err}")
    return tasccoda_model, adata, figs


def run_deg_edgr(adata, col_condition, col_covariate=None, formula=None,
                 key_treatment=None, key_control=None,
                 col_sample=None, col_celltype=None,  # for pseudo-bulking
                 log2fc_thresh=0, n_top_vars=8,
                 layer="counts", layer_counts="counts",
                 fig_title=None, xlabel_rotation=None, legend_loc=None,
                 kws_subplots=None, layer_plot=None,
                 kws_xticks=None, **kwargs):
    """Run edgeR differential gene expression testing."""
    figs, kws_subplots = {}, kws_subplots if kws_subplots else {}
    ksub = [k for k in ["wspace", "hspace", "left",
                        "bottom", "right", "top"] if k in kwargs]
    kws_xticks = {"rotation": xlabel_rotation, **dict(**kws_xticks if (
        kws_xticks) else {})} if xlabel_rotation or kws_xticks else kws_xticks
    kws_subplots.update(dict(zip(ksub, [kwargs.pop(k) for k in ksub])))
    if formula is None:
        formula = "~" + "+".join([col_condition, col_covariate]) if (
            col_covariate is not None) else f"~{col_condition}"
    if col_celltype is not None:  # create pseudo-bulk if needed
        mode = kwargs.pop("mode", "sum")
        adata = scflow.tl.create_pseudobulk(
            adata, [i for i in [
                col_sample, col_condition, col_covariate] if i],
            col_celltype, layer=layer_counts, mode=mode)
        adata.layers[layer_counts] = adata.X.copy()  # save scaled layer
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        # sc.pp.scale(adata, max_value=10)
        # sc.pp.pca(adata)
        # adata.layers[layer] = adata.X.copy()  # save scaled layer
        # adata.X = adata.layers[layer_counts].copy()  # change layer
    if key_treatment is None and key_control is None:
        raise ValueError("Specify `key_treatment` or `key_control`.")
    if key_treatment is None:
        key_treatment = [i for i in adata.obs[
            col_condition].unique() if i != key_control]
        if len(key_treatment) != 1:
            raise ValueError("Specify `key_treatment` if >2 categories.")
        key_treatment = key_treatment[0]
    if key_control is None:
        key_control = [i for i in adata.obs[
            col_condition].unique() if i != key_control]
        if len(key_control) != 1:
            raise ValueError("Specify `key_control` if >2 categories.")
        key_control = key_control[0]
    keytc = [key_treatment, key_control] if isinstance(
        key_treatment, str) else key_treatment + [key_control]
    if not adata.obs[col_condition].isin(keytc).all():
        print(f"***Subsetting adata by {key_treatment}, {key_control}...")
        adata = adata[adata.obs[col_condition].isin(
            keytc)].copy()  # subset to tx & control
    edgr = pt.tl.EdgeR(adata, design=formula, layer=layer)  # set up edgeR
    edgr.fit()  # fit edgeR
    if col_covariate is not None:  # contrasts
        # Infer `key_control` or `key_treatment` If Needed
        # Run Contrasts
        res_df = edgr.test_contrasts(edgr.contrast(
            column=col_condition, baseline=key_control,
            group_to_compare=key_treatment))
        figs["volcano"] = edgr.plot_volcano(
            res_df, log2fc_thresh=log2fc_thresh, return_fig=True)  # volcano
        if fig_title is not None:
            figs["volcano"].suptitle(fig_title)
        figs["paired"] = edgr.plot_paired(
            adata, results_df=res_df, n_top_vars=n_top_vars, layer=layer_plot,
            groupby=col_condition, pairedby=col_covariate, return_fig=True,
            show_legend=False if legend_loc is not None else True)  # paired
        if fig_title is not None:
            figs["paired"].suptitle(fig_title)
        if kws_xticks:
            for a in figs["paired"].axes:
                a.tick_params(axis="x", rotation=xlabel_rotation)
        if len(kws_subplots) > 0:
            figs["paired"].subplots_adjust(**kws_subplots)
        if legend_loc not in [None, False]:
            handles, labels = figs["paired"].axes[
                0].get_legend_handles_labels()
            figs["paired"].legend(handles, labels, loc=legend_loc)
        res_df = res_df.assign(abs_log_fc=np.abs(res_df["log_fc"]))
    else:
        # res_df = None
        res_df = edgr.compare_groups(
            adata, column=col_condition, baseline=key_control,
            groups_to_compare=key_treatment,  layer=layer)
        figs["mcf"] = edgr.plot_multicomparison_fc(
            res_df, figsize=(12, 1.5), return_fig=True)
        if fig_title is not None:
            figs["mcf"].suptitle(fig_title)
    return res_df, figs


def run_deg_pydeseq(adata, col_condition, col_covariate=None, n_top_vars=15,
                    col_celltype=None, col_sample=None, formula=None,
                    key_control=None, key_treatment=None,
                    threshold_l2fc=0, layer="counts", figsize=None, **kwargs):
    """Run PyDESeq2 differential gene expression testing."""
    figs = {}
    key_control_cov, key_treatment_cov = key_control.pop(
        col_covariate, None), key_treatment.pop(col_covariate, None)
    key_control, key_treatment = key_control[col_condition], key_treatment[
        col_condition]
    if figsize is None:
        figsize = (12, 1.5)
    to_compare = [key_treatment] if isinstance(
        key_treatment, str) else key_treatment
    if formula is None:
        formula = "~" + " + ".join([col_condition, col_covariate]) if (
            col_covariate is not None) else f"~{col_condition}"
        if col_covariate is not None:
            formula_int = formula + " + " + "*".join([
                col_condition, col_covariate])
    print(f"***Using formula: {formula}...")
    if col_celltype is not None:  # create pseudo-bulk if needed
        print("***Pseudo-bulking...")
        mode = kwargs.pop("mode", "sum")
        adata = scflow.tl.create_pseudobulk(
            adata, col_sample, col_celltype, layer="counts", mode=mode)
    pds2 = pt.tl.PyDESeq2(adata=adata, design=formula)
    pds2.fit()
    res_df = pds2.test_contrasts(pds2.contrast(
        column=col_condition, baseline=key_control,
        group_to_compare=key_treatment))
    figs["lfc"] = pds2.plot_fold_change(res_df, n_top_vars=n_top_vars)
    print(res_df.head(n_top_vars))
    pds2.plot_volcano(res_df, log2fc_thresh=threshold_l2fc)
    res_df_contr = pds2.compare_groups(
        adata, column=col_condition, baseline=key_control,
        groups_to_compare=to_compare)
    print(res_df_contr)
    pds2.plot_volcano(res_df_contr, log2fc_thresh=threshold_l2fc)
    pds2b = pt.tl.PyDESeq2(adata=adata, design=formula_int)
    edgr = pt.tl.EdgeR(adata, design=formula, layer=layer)
    edgr.plot_multicomparison_fc(res_df_contr, figsize=figsize)
    ctl = {col_condition: key_control, col_covariate: key_control_cov}
    txs = {col_condition: key_treatment, col_covariate: key_treatment_cov}
    mix1 = {col_condition: key_treatment, col_covariate: key_control_cov}
    mix2 = {col_condition: key_control, col_covariate: key_treatment_cov}
    interaction_contrast = (pds2b.cond(**txs) - pds2b.cond(**mix1)) - (
        pds2b.cond(**ctl) - pds2b.cond(**mix2))
    print(f"Interaction:\n{interaction_contrast}")
    # interaction_res_df = pds2b.test_contrasts(interaction_contrast)
    gen_ctr = pds2b.cond(**{col_covariate: key_treatment_cov}) - pds2.cond(
        **{col_covariate: key_control_cov})
    print(gen_ctr)
    # return pds2, interaction_contrast, gen_ctr
    interaction_res_df = pds2b.test_contrasts(
        {f"{key_treatment}_specific": interaction_contrast,
         "General": gen_ctr})
    print(interaction_res_df)
    pds2b.plot_volcano(interaction_res_df, log2fc_thresh=threshold_l2fc)
    edgr.plot_multicomparison_fc(interaction_res_df, figsize=figsize)
    # return res_df, res_df_contr, interaction_res_df, pds2, pds2b, figs
    return res_df, res_df_contr, pds2, figs


def run_mixscape(adata, col_condition, col_guide,
                 key_control=None, col_sample=None,
                 key_pca="X_pca", inplace=False):
    """Run Mixscape."""
    raise NotImplementedError()
    if inplace is False:
        adata = adata.copy()
    else:
        if "original" in adata.layers:
            raise ValueError("'original' reserved layer for `run_mixscape`")
    if key_control is None:
        key_control = adata.obs[col_condition].unique()[0]
        warn(f"\nNo control specified. Using {key_control}.\n")
    ms_pt = pt.tl.Mixscape()
    ms_pt.perturbation_signature(
        adata, pert_key=col_condition, control=key_control,
        split_by=col_sample, use_rep=key_pca, copy=False)
    adata.layers["original"] = adata.X.copy()  # so can switch back after
    adata.X = adata.layers["X_pert"]  # change layer
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, metric="cosine")
    sc.tl.umap(adata)
    ms_pt.mixscape(adata=adata, control=key_control,
                   labels=col_guide, layer="X_pert")
    # TODO: Continue coding
    adata.X = adata.layers["original"].copy()
    _ = adata.layers.pop("original")


def run_dialogue_mcp(adata, col_celltype, col_counts="total_counts",
                     col_condition=None, col_sample=None, col_confounder=None,
                     n_mpcs=3, layer="counts", cmap="coolwarm",
                     mlm_threshold=0.7, p_threshold=0.05):
    """Run Dialogue to find multi-cellular programs."""
    adata = adata.copy()
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    if col_sample is not None:  # only cell types represented in all samples
        isecs = pd.crosstab(adata.obs[col_celltype], adata.obs[col_sample])
        adata[adata.obs[col_celltype].isin(isecs[isecs.T.all()].index.values)]
    d_l = pt.tl.Dialogue(sample_id=col_sample, celltype_key=col_celltype,
                         n_counts_key=col_counts, n_mpcs=n_mpcs)
    cols_mcp = [f"mcp_{i}" for i in np.arange(n_mpcs)]
    adata, mcps, w_s, ct_subs = d_l.calculate_multifactor_PMD(
        adata, normalize=True)
    sc.pl.umap(
        adata, color=cols_mcp + ([col_condition] if col_condition else []),
        ncols=1, cmap=cmap, vcenter=0)  # plot MCPs on UMAP
    d_l.test_association(adata, "path_str")
    for x in cols_mcp:
        d_l.plot_split_violins(adata, split_key="path_str",
                               celltype_key=col_celltype, mcp=x)
        plt.set_title(x)
        d_l.plot_pairplot(adata, celltype_key=col_celltype,
                          color=col_condition, mcp=x,
                          sample_id=col_sample)
    extrema_genes = d_l.get_extrema_MCP_genes(ct_subs)
    if col_confounder is not None:
        all_results, new_mcps = d_l.multilevel_modeling(
            ct_subs=ct_subs, mcp_scores=mcps,
            ws_dict=w_s, confounder=col_confounder)
        mlm_genes = {}
        for x in adata.obs[col_celltype].unique():
            mlm_genes[x] = {}
            for m in cols_mcp:
                mlm_genes[x][m] = d_l.get_mlm_mcp_genes(
                    celltype=x, results=all_results, MCP=m,
                    threshold=mlm_threshold)
                # extract significantly different genes from extrema
                sig_genes = extrema_genes[m][x][extrema_genes[m][
                    x]["pvals_adj"] < p_threshold]
                up_genes = sig_genes[sig_genes["logfoldchanges"] > 0]["names"]
                down_g = sig_genes[sig_genes["logfoldchanges"] < 0]["names"]
                print("\n\n{'=' * 80}\n{x} | {m}\n\n")
                for i, direc in enumerate(["down", "up"]):
                    direc2 = ["decreased", "increased"][i]
                    glist = [down_g, up_genes][i]
                    diff = set(mlm_genes[x][m][f"{direc}_genes"]).difference(
                        set(glist))
                    inter = set(mlm_genes[x][m][
                        f"{direc}_genes"]).intersection(set(glist))
                    print(f"DIALOGUE {direc2} genes that are also in the "
                          f"extrema {direc2} list: {inter}")
                    print(f"DIALOGUE {direc2} genes that are not in the "
                          f"extrema {direc2} list: {diff}")
# all of the Dialogue genes from the multilevel model are also in this list
        res_mlm = [all_results, new_mcps, mlm_genes]
    else:
        res_mlm = None
    return d_l, adata, mcps, w_s, ct_subs, extrema_genes, res_mlm
