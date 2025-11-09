#!/usr/bin/env python

from setuptools import setup, find_packages

# This file exists for backward compatibility
# Configuration is in pyproject.toml

setup(
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
)
