
"""
Utility to convert JSON-loaded map entities to Numpy objects.
"""

from typing import Dict, List

import numpy as np

def point_arr_from_points_list_dict(points_dict: List[Dict[str, float]]) -> np.ndarray:
    """Convert a list of dictionaries containing vertices of a 3d polyline or polygon into a Nx3 Numpy array."""
    arr = np.vstack([np.array([v["x"], v["y"], v["z"]]).reshape(1, 3) for v in points_dict])
    assert arr.shape[1] == 3
    return arr
