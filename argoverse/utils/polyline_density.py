# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

from typing import Tuple

import numpy as np

from argoverse.utils.interpolate import interp_arc

NUM_PTS_PER_TRAJ = 50


def get_polyline_length(polyline: np.ndarray) -> float:
    """Calculate the length of a polyline.

    Args:
        polyline: Numpy array of shape (N,2)

    Returns:
        The length of the polyline as a scalar
    """
    assert polyline.shape[1] == 2
    return float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())


def interpolate_polyline_to_ref_density(polyline_to_interp: np.ndarray, ref_polyline: np.ndarray) -> np.ndarray:
    """
    Interpolate a polyline so that its density matches the density of a reference polyline.

    ::

             ref_l2             query_l2
        ----------------  =  --------------
        NUM_PTS_PER_TRAJ      num_interp_pts

    Args:
        polyline_to_interp: Polyline to interpolate -- numpy array of shape (M,2)
        ref_polyline: Reference polyline -- numpy array of shape (N,2)

    Returns:
        Interpolated polyline -- numpy array of shape (K,2)
    """
    ref_l2 = get_polyline_length(ref_polyline)
    query_l2 = get_polyline_length(polyline_to_interp)
    num_interp_pts = int(query_l2 * NUM_PTS_PER_TRAJ / ref_l2)
    dense_interp_polyline = interp_arc(num_interp_pts, polyline_to_interp[:, 0], polyline_to_interp[:, 1])
    return dense_interp_polyline


def traverse_polyline_by_specific_dist(polyline_to_walk: np.ndarray, l2_dist_to_walk: float) -> Tuple[np.ndarray, bool]:
    """
    Walk a distance along a polyline, and return the points along which you walked.

    Assumption: polyline is much longer than the distance to walk.

    Args:
        polyline_to_walk: Numpy array of shape (N,2)
        l2_dist_to_walk: Distance to traverse

    Returns:
        Tuple of (polyline, success flag)
    """
    MAX_NUM_PTS_TO_WALK = 100
    dense_polyline_to_walk = interp_arc(MAX_NUM_PTS_TO_WALK, polyline_to_walk[:, 0], polyline_to_walk[:, 1])

    for i in range(MAX_NUM_PTS_TO_WALK):
        l2 = get_polyline_length(dense_polyline_to_walk[:i])
        if l2 > l2_dist_to_walk:
            # break from for-loop execution and return
            return dense_polyline_to_walk[:i], True

    return dense_polyline_to_walk, False
