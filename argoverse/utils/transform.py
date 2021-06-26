# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Utility functions for converting quaternions to 3d rotation matrices.

Unit quaternions are a way to compactly represent 3D rotations
while avoiding singularities or discontinuities (e.g. gimbal lock).

If a quaternion is not normalized beforehand to be unit-length, we will
re-normalize it on the fly.
"""

import logging

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def yaw_to_quaternion3d(yaw: float) -> np.ndarray:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.

    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians

    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat()
    return np.array([qw, qx, qy, qz])


def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation-matrix to a quaternion in Argo's scalar-first notation (w, x, y, z)."""
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = quat_scipy2argo(quat_xyzw)
    return quat_wxyz


def quat2rotmat(q: np.ndarray) -> np.ndarray:
    """Normalizes a quaternion to unit-length, then converts it into a rotation matrix.

    Note that libraries such as Scipy expect a quaternion in scalar-last [x, y, z, w] format,
    whereas at Argo we work with scalar-first [w, x, y, z] format, so we convert between the
    two formats here. We use the [w, x, y, z] order because this corresponds to the
    multidimensional complex number `w + ix + jy + kz`.

    Args:
        q: Array of shape (4,) representing (w, x, y, z) coordinates

    Returns:
        R: Array of shape (3, 3) representing a rotation matrix.
    """
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, atol=1e-12):
        logger.info("Forced to re-normalize quaternion, since its norm was not equal to 1.")
        if np.isclose(norm, 0.0):
            raise ZeroDivisionError("Normalize quaternioning with norm=0 would lead to division by zero.")
        q /= norm

    quat_xyzw = quat_argo2scipy(q)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def quat_argo2scipy(q: np.ndarray) -> np.ndarray:
    """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    return q_scipy


def quat_scipy2argo(q: np.ndarray) -> np.ndarray:
    """Re-order Scipy's scalar-last [x,y,z,w] quaternion order to Argoverse's scalar-first [w,x,y,z]."""
    x, y, z, w = q
    q_argo = np.array([w, x, y, z])
    return q_argo


def quat_argo2scipy_vectorized(q: np.ndarray) -> np.ndarray:
    """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    return q[..., [1, 2, 3, 0]]


def quat_scipy2argo_vectorized(q: np.ndarray) -> np.ndarray:
    """Re-order Scipy's scalar-last [x,y,z,w] quaternion order to Argoverse's scalar-first [w,x,y,z]."""
    return q[..., [3, 0, 1, 2]]
