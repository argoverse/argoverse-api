# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Utility function to visualize a 3d point cloud from a bird's eye view with Matplotlib ("mpl")."""

from typing import Optional

import numpy as np
from matplotlib.axes import Axes

__all__ = ["draw_point_cloud_bev"]

AXIS_INDEX = [0, 1]
AXIS_LIMITS = [
    [-80, 80],
    [-90, 90],
    [-10, 50],
]  # X axis range  # Y axis range  # Z axis range
AXIS_NAME = ["X", "Y", "Z"]


def draw_point_cloud_bev(
    axes: Axes,
    pointcloud: np.ndarray,
    color: str = "w",
    x_lim_3d: Optional[float] = None,
    y_lim_3d: Optional[float] = None,
    z_lim_3d: Optional[float] = None,
) -> None:
    """Draw a pointcloud from a bird's eye view perspective.

    Convenient method for drawing various point cloud projections with Matplotlib.
    The point cloud is rendered in a 2D projection, the bird's eye view ("bev").

    Reference: http://eamonbracht.com/lidar.html

    Args:
        axes: Matplotlib axes
        pointcloud: Numpy array of shape (N,3) representing point cloud
        color: A color string (e.g. 'w')
        x_lim_3d: X axis limit
        y_lim_3d: Y axis limit
        z_lim_3d: Z axis limit
    """
    points = 1.0
    point_size = 0.01 * (1.0 / points)

    axes.scatter(*np.transpose(pointcloud[:, AXIS_INDEX]), s=point_size, c=[color], cmap="gray")
    axes.set_xlabel(f"{AXIS_NAME[AXIS_INDEX[0]]} axis")
    axes.set_ylabel(f"{AXIS_NAME[AXIS_INDEX[1]]} axis")
    if len(AXIS_INDEX) > 2:
        axes.set_xlim3d(*AXIS_LIMITS[AXIS_INDEX[0]])
        axes.set_ylim3d(*AXIS_LIMITS[AXIS_INDEX[1]])
        axes.set_zlim3d(*AXIS_LIMITS[AXIS_INDEX[2]])
        axes.set_zlabel(f"{AXIS_NAME[AXIS_INDEX[2]]} axis")
    else:
        axes.set_xlim(*AXIS_LIMITS[AXIS_INDEX[0]])
        axes.set_ylim(*AXIS_LIMITS[AXIS_INDEX[1]])

    # User specified limits
    if x_lim_3d is not None:
        axes.set_xlim3d(x_lim_3d)
    if y_lim_3d is not None:
        axes.set_ylim3d(y_lim_3d)
    if z_lim_3d is not None:
        axes.set_zlim3d(z_lim_3d)

    axes.set_facecolor((0, 0, 0))
