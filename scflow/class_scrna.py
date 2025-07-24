# __init__.py
# pylint: disable=unused-import

import os
import anndata
import scanpy as sc
import scflow
# from scflow import Regression
# from scflow.processing import import_data


class Rna(object):
    """A class for single-cell RNA-seq data."""

    def __init__(self, file_path=None, kws_read=None,
                 kws_integrate=None, **kwargs):
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
        if isinstance(file_path, anndata.AnnData):
            self.rna = file_path.copy()
        elif isinstance(file_path, (list, dict)):
            if not isinstance(kws_read, list):  # if not sample-specific kws
                kws_read = [kws_read] * len(file_path)  # use same for all
            kws_read = dict(zip(list(file_path.keys()), kws_read))
            self.rna = [scflow.pp.read_scrna(file_path[x], **kws_read[x])
                        for x in file_path]
            self.rna = scflow.pp.integrate(self.rna, **kws_integrate)
        else:
            self.rna = scflow.pp.read_scrna(file_path, **kws_read)
        else:
            raise ValueError("`file_path` not a valid/recognized input.")

