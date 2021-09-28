# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Collection of functions that handle filtering a point cloud to a 2D polygon.

For filtering to 3D polygons, please see cuboid_interior.py or iou_3d.py instead.
"""

from typing import Optional, cast

import numpy as np
from shapely.geometry import Point, Polygon


def rotate_polygon_about_pt(pts: np.ndarray, rotmat: np.ndarray, center_pt: np.ndarray) -> np.ndarray:
    """Rotate a polygon about a point with a given rotation matrix.

    Args:
        pts: Array of shape (N, 3) representing a polygon or point cloud
        rotmat: Array of shape (3, 3) representing a rotation matrix
        center_pt: Array of shape (3,) representing point about which we rotate the polygon

    Returns:
        rot_pts: Array of shape (N, 3) representing a ROTATED polygon or point cloud
    """
    pts -= center_pt
    rot_pts = pts.dot(rotmat.T)
    rot_pts += center_pt
    return rot_pts


def filter_point_cloud_to_polygon(polygon: np.ndarray, point_cloud: np.ndarray) -> Optional[np.ndarray]:
    """Filter a point cloud to the points within a polygon.

    Args:
        polygon: Array of shape (K, 2) representing points of a polygon
        point_cloud: Array of shape (N, 2) or (N, 3) representing points in a point cloud

    Returns:
        interior_pts: Array of shape (N, 3) representing filtered points.
        Returns None if no point falls within the polygon.
    """
    is_inside = np.zeros(point_cloud.shape[0], dtype=bool)
    for pt_idx, lidar_pt in enumerate(point_cloud):
        is_inside[pt_idx] = point_inside_polygon(
            polygon.shape[0], polygon[:, 0], polygon[:, 1], lidar_pt[0], lidar_pt[1]
        )

    if is_inside.sum() == 0:
        return None
    else:
        interior_pts = point_cloud[is_inside]
        return interior_pts


def point_inside_polygon(
    n_vertices: int,
    poly_x_pts: np.ndarray,
    poly_y_pts: np.ndarray,
    test_x: float,
    test_y: float,
) -> bool:
    """Check whether a point is inside a polygon.

    Args:
        n_vertices: number of vertices in the polygon
        vert_x_pts, Array containing the x-coordinates of the polygon's vertices.
        vert_y_pts: Array containing the y-coordinates of the polygon's vertices.
        test_x, test_y: the x- and y-coordinate of the test point

    Returns:
        inside: boolean, whether point lies inside polygon
    """
    assert poly_x_pts.shape == poly_y_pts.shape
    poly_arr = np.hstack([poly_x_pts.reshape(-1, 1), poly_y_pts.reshape(-1, 1)])
    assert poly_arr.shape[0] == n_vertices
    polygon = Polygon(poly_arr)
    pt = Point([test_x, test_y])
    return cast(bool, polygon.contains(pt))
