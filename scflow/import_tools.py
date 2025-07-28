#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E. N. Aslinger

import importlib


def get_plot_fx(kind):
    """Get the function for a plot type string (e.g., 'heat')."""
    module = importlib.import_module("scflow.pl.basic_plots")
    return getattr(module, f"plot_{kind}")
