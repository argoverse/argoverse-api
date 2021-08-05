# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Point cloud loading utility functions."""

import os
from pathlib import Path
from typing import Dict, Final, Optional, Union

import numpy as np

_PathLike = Union[str, "os.PathLike[str]"]
DTYPE_MAP: Final[Dict[str, type]] = {
    "int": int,
    "float": np.float32,
    "long": np.uint64,
    "uchar": np.uint8,
    "ushort": np.uint16,
}


def _load_ply(ply_fpath: _PathLike) -> np.ndarray:
    """Read a .ply file from a binary file.

    Helper function to load the Polygon File Format (PLY):
    https://en.wikipedia.org/wiki/PLY_(file_format).

    Args:
        ply_fpath (_PathLike): File path to the ply file.

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: Structured array containing the point attributes.
    """
    with Path(ply_fpath).open("rb") as f:
        ply = f.readlines()
    if len(ply) < 3:
        raise ValueError("Malformed PLY file.")
    
    format = ply[1].split()[1].decode()
    num_elems = ply[2].split()[2].decode()

    header = []
    start, offset = 3, 0
    for i, line in enumerate(ply[start:]):
        decoded_line = line.decode().strip("\n")
        if "end_header" in decoded_line:
            offset = i
            break
        header.append(decoded_line.split(" "))
    attr = b"".join(ply[start + offset + 1:])

    dtype = [(name, DTYPE_MAP[d]) for (_, d, name) in header]
    return np.frombuffer(attr, dtype)

def load_ply(ply_fpath: _PathLike) -> np.ndarray:
    """Load a point cloud file from a filepath.

    Args:
        ply_fpath: Path to a PLY file

    Returns:
        arr: Array of shape (N, 3)
    """
    points = _load_ply(ply_fpath)
    return np.stack((points["x"], points["y"], points["z"]), axis=1)

def load_ply_by_attrib(ply_fpath: _PathLike, attrib_spec: str = "xyzil") -> Optional[np.ndarray]:
    """Load a point cloud file from a filepath.

    Args:
        ply_fpath: Path to a PLY file
        attrib_spec: string of C characters, each char representing a desired point attribute
            x -> point x-coord
            y -> point y-coord
            z -> point z-coord
            i -> point intensity/reflectance
            l -> laser number of laser from which point was returned

    Returns:
        arr: Array of shape (N, C). If attrib_str is invalid, `None` will be returned
    """
    possible_attributes = ["x", "y", "z", "i", "l"]
    if not all([a in possible_attributes for a in attrib_spec]):
        return None

    data = _load_ply(ply_fpath)
    attrib_dict = {
        "x": np.array(data["x"]),
        "y": np.array(data["y"]),
        "z": np.array(data["z"]),
        "i": np.array(data["intensity"]),
        "l": np.array(data["laser_number"]),
    }

    # return only the requested point attributes
    attrib_arrs = [attrib_dict[a] for a in attrib_spec]
    # join arrays of the same shape along new dimension
    return np.stack(attrib_arrs, axis=1)
