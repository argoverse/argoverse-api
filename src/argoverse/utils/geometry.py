# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Geometric utilities for manipulation point clouds, rigid objects, and vector geometry."""

from typing import Final, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from argoverse.utils.constants import NAN, PI

# Unit polygon (vertices in {-1, 0, 1}) with counter-clockwise (CCW) winding order.
# +---+---+
# | 1 | 0 |
# +---+---+
# | 2 | 3 |
# +---+---+
UNIT_POLYGON_2D: Final[np.ndarray] = np.array(
    [[+1.0, +1.0, 0.0], [-1.0, +1.0, 0.0], [-1.0, -1.0, 0.0], [+1.0, -1.0, 0.0]]
)

UNIT_POLYGON_2D_EDGES: Final[np.ndarray] = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

UNIT_POLYGON_3D: Final[np.ndarray] = np.array(
    [
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
    ]
)

UNIT_POLYGON_3D_EDGES: Final[np.ndarray] = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
)


def wrap_angles(angles: np.ndarray, period: float = PI) -> np.ndarray:
    """Map angles (in radians) from domain [-∞, ∞] to [0, π). This function is
        the inverse of `np.unwrap`.
    Returns:
        Angles (in radians) mapped to the interval [0, π).
    """

    # Map angles to [0, ∞].
    angles = np.abs(angles)

    # Calculate floor division and remainder simultaneously.
    divs, mods = np.divmod(angles, period)

    # Select angles which exceed specified period.
    angle_complement_mask = np.nonzero(divs)

    # Take set complement of `mods` w.r.t. the set [0, π].
    # `mods` must be nonzero, thus the image is the interval [0, π).
    angles[angle_complement_mask] = period - mods[angle_complement_mask]
    return angles


def cart2hom(cart: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates into Homogenous coordinates.

    This function converts a set of points in R^N to its homogeneous representation in R^(N+1).

    Args:
        cart (np.ndarray): (M,N) Array of points in Cartesian space.

    Returns:
        np.ndarray: (M,N+1) Array in Homogeneous space.
    """
    M = cart.shape[0]
    N = cart.shape[1]
    hom: np.ndarray = np.ones((M, N + 1))
    hom[:, :N] = cart
    return hom


def hom2cart(hom: np.ndarray) -> np.ndarray:
    """Convert Homogenous coordinates into Cartesian coordinates.

    This function converts a set of points in R^(N+1) to its Cartesian representation in R^N.

    Args:
        hom (np.ndarray): (M,N+1) Array of points in Homogeneous space.

    Returns:
        np.ndarray: (M,N) Array in Cartesian space.
    """
    hom[:, :3] /= hom[:, 3:4]
    return hom[:, :3]


def cart2range(
    cart: np.ndarray,
    fov: np.ndarray,
    dims: np.ndarray = np.array([64, 1024]),
) -> Tuple[np.ndarray, np.ndarray]:
    fov_bottom, fov_top = np.abs(fov).transpose()

    az, el, r = cart2sph(*cart.transpose())

    v = 0.5 * (-az / PI + 1.0)
    u = 1.0 - (el + fov_bottom) / (fov_bottom + fov_top)

    perm = np.argsort(r)

    uv = np.stack((u, v), axis=-1)[perm] * dims
    uv = np.clip(uv, 0, dims - 1).astype(int)

    range_im = np.full(dims, NAN)
    pos_im = np.full((dims.tolist() + [3]), NAN)

    u, v = uv.transpose()
    range_im[u, v] = r[perm]
    pos_im[u, v] = cart[perm]
    return range_im, pos_im


def crop_points(
    points: np.ndarray,
    lower_bound_inclusive: np.ndarray,
    upper_bound_exclusive: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ndim = points.shape[-1]
    lb_dim = lower_bound_inclusive.shape[0]
    ub_dim = upper_bound_exclusive.shape[0]

    assert ndim == lb_dim
    assert ndim == ub_dim

    lower_bound_condition = np.greater_equal(points, 0)
    upper_bound_condition = np.less(points, upper_bound_exclusive)
    grid_bound_reduction = np.logical_and(lower_bound_condition, upper_bound_condition).all(axis=-1)
    return points, grid_bound_reduction


def annotations2polygons(annotations: pd.DataFrame, unit_polygon: np.ndarray) -> np.ndarray:
    # Grab centers (xyz) of the annotations.
    center_xyz = annotations[["x", "y", "z"]].to_numpy()

    # Grab dimensions (length, width, height) of the annotations.
    dims_lwh = annotations[["length", "width", "height"]].to_numpy()

    # Construct unit polygons.
    scaled_polygons = unit_polygon[None] * np.divide(dims_lwh[:, None], 2.0)

    # Get scalar last quaternions.
    # NOTE: SciPy follows scaler *last* while AV2 uses scaler *first* ordering.
    quat_xyzw = annotations[["qx", "qy", "qz", "qw"]].to_numpy()

    # Repeat transformations by number of polygon vertices to vectorize SO3 transformation in SciPy.
    quat_xyzw = np.repeat(quat_xyzw[:, None], scaled_polygons.shape[1], axis=1).reshape(-1, 4)

    # Get SO3 transformation.
    ego_SO3_obj = R.from_quat(quat_xyzw)

    # Apply ego_SO3_obj to the scaled polygons.
    polygons_xyz = ego_SO3_obj.apply(scaled_polygons.reshape(-1, 3)).reshape(-1, 4, 3)

    # Translate by the annotations centers.
    polygons_xyz += center_xyz[:, None]

    return polygons_xyz


def filter_interior_pts(bbox: np.ndarray, pc_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Args:
       bbox: Numpy array pf shape (8,3) representing 3d cuboid vertices, ordered
                as shown below.
       pc_raw: Numpy array of shape (N,3), representing a point cloud
    Returns:
       segment: Numpy array of shape (K,3) representing 3d points that fell
                within 3d cuboid volume.
       is_valid: Numpy array of shape (N,) of type bool
    https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
    ::
            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.
    """
    # get 3 principal directions (edges) of the cuboid
    u = bbox[2] - bbox[6]
    v = bbox[2] - bbox[3]
    w = bbox[2] - bbox[1]

    # point x lies within the box when the following
    # constraints are respected

    # IN BETWEEN

    # do i need to check the other direction as well?
    valid_u1 = np.logical_and(u.dot(bbox[2]) <= pc_raw.dot(u), pc_raw.dot(u) <= u.dot(bbox[6]))
    valid_v1 = np.logical_and(v.dot(bbox[2]) <= pc_raw.dot(v), pc_raw.dot(v) <= v.dot(bbox[3]))
    valid_w1 = np.logical_and(w.dot(bbox[2]) <= pc_raw.dot(w), pc_raw.dot(w) <= w.dot(bbox[1]))

    valid_u2 = np.logical_and(u.dot(bbox[2]) >= pc_raw.dot(u), pc_raw.dot(u) >= u.dot(bbox[6]))
    valid_v2 = np.logical_and(v.dot(bbox[2]) >= pc_raw.dot(v), pc_raw.dot(v) >= v.dot(bbox[3]))
    valid_w2 = np.logical_and(w.dot(bbox[2]) >= pc_raw.dot(w), pc_raw.dot(w) >= w.dot(bbox[1]))

    valid_u = np.logical_or(valid_u1, valid_u2)
    valid_v = np.logical_or(valid_v1, valid_v2)
    valid_w = np.logical_or(valid_w1, valid_w2)

    is_valid = np.logical_and(np.logical_and(valid_u, valid_v), valid_w)
    segment_pc = pc_raw[is_valid]
    return segment_pc, is_valid
