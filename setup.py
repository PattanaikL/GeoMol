#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="GeoMol",
    version="1.0.0",
    author="Lagnajit Pattanaik",
    author_email="lagnajit@mit.com",
    description="Machine learning tools for molecule conformer generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PattanaikL/GeoMol",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    license="MIT License",
    python_requires='>=3.7',
)
