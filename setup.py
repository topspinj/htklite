#!/usr/bin/env python

from setuptools import setup, find_packages
from pkg_resources import parse_requirements, RequirementParseError

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    with open('requirements.txt') as f:
        req_list = parse_requirements(f.read())
except RequirementParseError:
    raise
requirements = [str(req) for req in req_list]

setup(
    name="htklite",
    version="0.0.1",
    author="Jill Cates",
    author_email="jill@biosymetrics.com",
    description="A lightweight version of HistomicsTK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/topspinj/htklite",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
        
    ],
)