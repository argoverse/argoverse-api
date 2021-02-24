# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Point cloud loading utility functions."""

import os
from typing import Optional, Union

import numpy as np
import pyntcloud

_PathLike = Union[str, "os.PathLike[str]"]


def load_ply(ply_fpath: _PathLike) -> np.ndarray:
    """Load a point cloud file from a filepath.

    Args:
        ply_fpath: Path to a PLY file

    Returns:
        arr: Array of shape (N, 3)
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]

    return np.concatenate((x, y, z), axis=1)


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

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))

    attrib_dict = {
        "x": np.array(data.points.x),
        "y": np.array(data.points.y),
        "z": np.array(data.points.z),
        "i": np.array(data.points.intensity),
        "l": np.array(data.points.laser_number),
    }

    # return only the requested point attributes
    attrib_arrs = [attrib_dict[a] for a in attrib_spec]
    # join arrays of the same shape along new dimension
    return np.stack(attrib_arrs, axis=1)
