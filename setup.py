#!/usr/bin/env python
"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import platform
import sys
from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


if platform.system() == "Windows":
    print("Argoverse currently does not support Windows, please use Linux/Mac OS")
    sys.exit(1)

setup(
    name="argoverse",
    version="1.0.0",
    description="",
    long_description=long_description,
    url="https://www.argoverse.org",
    author="Argo AI",
    author_email="argoverse-api@argo.ai",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="self-driving-car dataset-tools",
    packages=find_packages(exclude=["tests", "integration_tests", "map_files"]),
    package_data={"argoverse": ["argoverse/visualization/data/**/*"]},
    include_package_data=True,
    python_requires=">= 3.6",
    install_requires=[
        "colour",
        "descartes",
        "imageio",
        "matplotlib",
        "motmetrics==1.1.3",
        "numpy",
        "opencv-python>=4.1.0.25",
        "pandas>=0.23.1",
        "pillow",
        "imageio",
        "pyntcloud @ git+https://github.com/daavoo/pyntcloud#egg=pyntcloud-0.1.0",
        "scipy>=1.4.0",
        "shapely",
        "sklearn",
        "typing_extensions",
        "h5py",
        "lap",
    ],
    # for older pip version, use with --process-dependency-links
    dependency_links=["git+https://github.com/daavoo/pyntcloud#egg=pyntcloud-0.1.0"],
)
