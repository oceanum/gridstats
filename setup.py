#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "bottleneck",
    "cmocean",
    "dask",
    "distributed",
    "flox",
    "matplotlib",
    "nco",
    "pyyaml",
    "simplekml",
    "tqdm",
    "zarr",
    "oncore @ git+https://oceanum-dev:REDACTED@gitlab.com/oceanum/oncore@v0.1.28",
    "intake-oceanum @ git+https://oceanum-dev:REDACTED@gitlab.com/oceanum/intake/intake-oceanum@v1.6.0",
    # "onstats @ git+https://oceanum-dev:REDACTED@gitlab.com/oceanum/onstats@master",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

setup(
    author="Oceanum Developers",
    author_email="developers@oceanum.science",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    description="Library for datasets onstats",
    entry_points={"console_scripts": ["onstats=onstats.cli:main"]},
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="onstats",
    name="onstats",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.com/oceanum/onstats",
    version="1.3.0",
    zip_safe=False,
)
