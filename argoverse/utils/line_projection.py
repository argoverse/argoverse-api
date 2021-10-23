# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from argoverse.utils.interpolate import interp_arc
from argoverse.utils.polyline_density import interpolate_polyline_to_ref_density


def project_to_line_seq(
    trajectory: NDArray[np.float64], lines: Sequence[NDArray[np.float64]], interpolate_more: bool = True
) -> Tuple[float, NDArray[np.float64]]:
    """Project a trajectory onto a line sequence.

    Args:
        trajectory: An array of shape (N,2) that we will project onto a polyline.
        lines: A sequence of lines.
        interpolate_more: True by default.

    Returns:
        Projected distance along the centerline, Polyline centerline_trajectory.
    """
    linestrings: List[NDArray[np.float64]] = []
    for line in lines:
        if interpolate_more:
            linestrings += [interp_arc(100, line[:, 0], line[:, 1])]
        else:
            linestrings += [line]

    centerline_linestring = np.vstack([*linestrings])
    return project_to_line(trajectory, centerline_linestring)


def project_to_line(
    trajectory: NDArray[np.float64],
    center_polyline: NDArray[np.float64],
    enforce_same_density: bool = False,
) -> Tuple[float, NDArray[np.float64]]:
    """Project a trajectory onto a polyline.

    Args:
        trajectory: An array of shape (N,2) that we will project onto a polyline.
        center_polyline: An array of shape (N,2) to project onto.
        enforce_same_density: False by default, if set to True, centerline polyline is interpolated to match the point
                              density of the trajectory.

    Returns:
        Projected distance along the centerline, Polyline centerline_trajectory.
    """
    if enforce_same_density:
        center_polyline = interpolate_polyline_to_ref_density(center_polyline, trajectory)

    centerline_trajectory: List[NDArray[np.float64]] = []
    for i, pt in enumerate(trajectory):
        closest_idx = np.linalg.norm(center_polyline - pt, axis=1).argmin()
        closest_pt = center_polyline[closest_idx]
        centerline_trajectory += [closest_pt]

    centerline_trajectory = np.array(centerline_trajectory)
    trajectory_diff_norm = np.linalg.norm(np.diff(centerline_trajectory, axis=0), 1)
    dist_along_centerline = float(np.sum(trajectory_diff_norm))

    return dist_along_centerline, centerline_trajectory
