# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Point cloud loading utility functions."""

import os
from typing import Union

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
