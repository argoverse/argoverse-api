#!/usr/bin/env python
"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import platform

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

if platform.system() == "Windows":
    print("Argoverse currently does not support Windows, please use Linux/Mac OS")
    exit(1)

setup()
