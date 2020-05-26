# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Pickle utility test functions."""

import pathlib
import pickle as pkl
from typing import Any, Dict

import numpy as np

from argoverse.utils.pkl_utils import load_pkl_dictionary, save_pkl_dictionary

_TEST_DIR = pathlib.Path(__file__).parent


def dictionaries_are_equal(dict_a: Dict[Any, Any], dict_b: Dict[Any, Any]) -> None:
    """Compare the keys and corresponding values of two dictionaries.

    Args:
        dict_a: Python dictionary
        dict_b: Python dictionary
    """
    # make sure key sets are identical
    assert dict_a.keys() == dict_b.keys()

    for k, v in dict_a.items():
        if isinstance(v, np.ndarray):
            assert np.allclose(v, dict_b[k])
        else:
            assert v == dict_b[k]


def test_save_pickle_from_disk() -> None:
    """Save a dictionary to a pickle file.

    The file should contain the same Python dictionary as above:
    {'a': 1, 'b':'2', 'c':[9,8,7,6,5,'d','c','b','a'], 'd': np.array([True,False,True]) }
    """
    pkl_fpath = _TEST_DIR / "test_data/pkl_test_file.pkl"
    intended_dict = {"a": 1, "b": "2", "c": [9, 8, 7, 6, 5, "d", "c", "b", "a"], "d": np.array([True, False, True])}
    save_pkl_dictionary(pkl_fpath, intended_dict)

    with open(pkl_fpath, "rb") as f:
        loaded_pkl_dict = pkl.load(f)
    dictionaries_are_equal(intended_dict, loaded_pkl_dict)


def test_load_pickle_from_disk() -> None:
    """Load a pickle file from disk.

    The file should contain the following Python dictionary:
    {'a': 1, 'b':'2', 'c':[9,8,7,6,5,'d','c','b','a'], 'd': np.array([True,False,True]) }

    We demonstrate that we can load Numpy arrays and lists from Pickle files.
    """
    gt_dict = {"a": 1, "b": "2", "c": [9, 8, 7, 6, 5, "d", "c", "b", "a"], "d": np.array([True, False, True])}
    pkl_fpath = _TEST_DIR / "test_data/pkl_test_file.pkl"
    loaded_pkl_dict = load_pkl_dictionary(pkl_fpath)
    dictionaries_are_equal(gt_dict, loaded_pkl_dict)
