# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""JSON utility functions."""

import json
import os
from typing import Any, Dict, List, Union


def read_json_file(fpath: Union[str, "os.PathLike[str]"]) -> Any:
    """Load dictionary from JSON file.

    Args:
        fpath: Path to JSON file.

    Returns:
        Deserialized Python dictionary.
    """
    with open(fpath, "rb") as f:
        return json.load(f)


def save_json_dict(
    json_fpath: Union[str, "os.PathLike[str]"],
    dictionary: Union[Dict[Any, Any], List[Any]],
) -> None:
    """Save a Python dictionary to a JSON file.

    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)
