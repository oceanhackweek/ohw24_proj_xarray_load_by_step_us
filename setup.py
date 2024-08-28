#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name="load_by_step",
    version="0.0.1",
    description=("A xarray accessor to retrieve large quantities of data from a"
                 "THREDDS server splitting the request in smaller requests"),
    packages=["load_by_step"],
    install_requires=["numpy", "xarray", "pydantic", "tqdm"],
)
