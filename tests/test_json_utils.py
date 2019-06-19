# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Unit tests for JSON utility functions."""
import json
import pathlib

from argoverse.utils.json_utils import read_json_file, save_json_dict

_TEST_DIR = pathlib.Path(__file__).parent


def test_read_json_file() -> None:
    """Test reading from a JSON file.

    Load a file that has the following dictionary saved as JSON:
    dict = {'a':1,'b':'2','c':[9,8,7,6,5,'d','c','b','a']}
    """
    fpath = _TEST_DIR / "test_data/json_read_test_file.json"
    json_data = read_json_file(fpath)

    gt_dict = {"a": 1, "b": "2", "c": [9, 8, 7, 6, 5, "d", "c", "b", "a"]}
    assert gt_dict == json_data


def test_save_json_dict() -> None:
    """Test saving a dictionary to a JSON file."""
    json_fpath = _TEST_DIR / "test_data/json_save_test_file.json"
    intended_dict = {"a": 1, "b": None, "c": 9999, "d": "abc"}
    save_json_dict(json_fpath, intended_dict)

    # verify it was saved correctly
    with open(json_fpath, "rb") as f:
        loaded_dict = json.load(f)

    assert intended_dict == loaded_dict

    # make sure key sets are identical
    assert set(intended_dict.keys()) == set(loaded_dict.keys())

    for k in intended_dict.keys():
        assert intended_dict[k] == loaded_dict[k]
