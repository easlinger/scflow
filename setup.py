#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E. N. Aslinger

from setuptools import setup, find_packages
import pathlib

must_test = []  # tests required
must_have = ["scanpy", "pertpy", "pandas", "numpy"]  # dependencies
pkg_was_here = pathlib.Path(__file__).parent.resolve()

with open("README.md") as info:
    readme = info.read()

setup(name="scflow",
      version="0.0.0",
      description="Process and analyze scRNA-seq and spatial omics data.",
      url="http://github.com/easlinger/glmax",
      author="Elizabeth N. Aslinger",
      packages=find_packages(),
      install_requires=must_have,
      test_suite="tests",
      tests_require=["pytest"],
      long_description=readme,
      zip_safe=False,
      python_requires=">=3.10.4")
