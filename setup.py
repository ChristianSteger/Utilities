# Description: Setup file
#
# Installation of package: python -m pip install .
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
from setuptools import setup

setup(
    name="utilities",
    version="0.1",
    description="Various utilities for processing climate model data with"
    			"Python.",
    author="Christian R. Steger",
    author_email="christian.steger@env.ethz.ch",
    packages=["utilities"]
)
