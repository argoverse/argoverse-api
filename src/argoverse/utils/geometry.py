# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Geometric utilities for manipulation point clouds, rigid objects, and vector geometry."""

from typing import Tuple

import numpy as np

from argoverse.utils.constants import NAN, PI


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
    points: np.ndarray, lower_bound_inclusive: np.ndarray, upper_bound_exclusive: np.ndarray
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
