# <Copyright 2018-2019, Argo AI, LLC. Released under the MIT license.>
"""Pickle utility functions."""

import os
import pickle as pkl
from typing import Any, Dict, Union

_PathLike = Union[str, "os.PathLike[str]"]


def load_pkl_dictionary(pkl_fpath: _PathLike) -> Any:
    """Load a Python dictionary from a file serialized by pickle.

    Args:
        pkl_fpath: Path to pickle file.

    Returns:
        Deserialized Python dictionary.
    """
    with open(pkl_fpath, "rb") as f:
        return pkl.load(f)


def save_pkl_dictionary(pkl_fpath: _PathLike, dictionary: Dict[Any, Any]) -> None:
    """Save a Python dictionary to a file using pickle.

    Args:
        pkl_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """

    if not os.path.exists(os.path.dirname(pkl_fpath)):
        os.makedirs(os.path.dirname(pkl_fpath))

    with open(pkl_fpath, "wb") as f:
        pkl.dump(dictionary, f)
