# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List

from recommonmark.transform import AutoStructify

import sphinx.ext.apidoc

_BADGE_REGEX = re.compile(r"^(?:\[!\[[^[]*\](\([^)]+\)\]\([^)]+\)))", re.M)
_TOC_REGEX = re.compile(r"(## Table of Contents.+- \[Installation\][^\n]*\n)", re.S)
_NOTEBOOK_REGEX = re.compile(r"(?:./)?(demo_usage/.+.ipynb)", re.M)

_GITHUB_ROOT = "https://github.com/argoai/argoverse-api/tree"

_CONTRIBUTING = "CONTRIBUTING.md"
_LICENSE = "LICENSE"

sphinx_root = Path(__file__).parent
project_root = sphinx_root.parent
demo_dir = project_root / "demo_usage"
doc_dir = project_root / "docs"
autodoc_dir = sphinx_root / "source"
index_md = autodoc_dir / "index.md"

project = "argoverse"
copyright = "2019, Argo AI, LLC"
author = "Argo AI, LLC"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.imgmath",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "recommonmark",
]
templates_path: List[str] = []
exclude_patterns: List[str] = []
source_suffix: Dict[str, str] = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme: str = "sphinx_rtd_theme"
html_static_path: List[str] = []


# -- Extension configuration -------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org", None),
    "mayavi": ("https://docs.enthought.com/mayavi/mayavi", None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/version/0.24", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# Extra configuration


def setup(app: sphinx.application.Sphinx) -> None:
    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": lambda url: f"{_GITHUB_ROOT}/{url}",
            "enable_auto_toc_tree": True,
            "enable_eval_rst": True,
            "auto_toc_tree_section": "Table of Contents",
        },
        True,
    )

    app.add_transform(AutoStructify)


# Sphinx source generation


# Directories and setup
for path in (doc_dir, autodoc_dir):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

os.makedirs(autodoc_dir)

# Copy over the images used in the README
shutil.copytree(project_root / "images", autodoc_dir / "images")

# Symlink the CONTRIBUTING.md
(autodoc_dir / _CONTRIBUTING).symlink_to(project_root / _CONTRIBUTING)

# Symlink the LICENSE
(autodoc_dir / (_LICENSE + ".txt")).symlink_to(project_root / _LICENSE)

# Copy the README over with some formatting and linking changes
with open(project_root / "README.md", "r") as readme_file:
    readme_content = readme_file.read()

# Sphinx renders the title text of badges, which looks very weird
readme_content = _BADGE_REGEX.sub(r"[![]\1", readme_content)

# Add an API Reference link to the table of contents
readme_content = _TOC_REGEX.sub(r"\1- [API Reference](./argoverse)\n", readme_content)

# Make the *.ipynb links go to GitHub
readme_content = _NOTEBOOK_REGEX.sub(_GITHUB_ROOT + r"/\1", readme_content)

with open(index_md, "w") as index_file:
    index_file.write(readme_content)

# Run the API doc tool to generated the RST files for the code
sphinx.ext.apidoc.main(["-fo", str(autodoc_dir), str(project_root / "argoverse")])
