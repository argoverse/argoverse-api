# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Tests for quaternion conversion utility functions.

We mainly verify that Scipy's quaternion to rotation matrix
utility yields the correct result given the swapped argument order.
"""

import numpy as np
import pytest

from argoverse.utils.transform import quat2rotmat, rotmat2quat, yaw_to_quaternion3d

EPSILON = 1e-10


def quat2rotmat_numpy(q: np.ndarray) -> np.ndarray:
    """Sanity check for Scipy function. Convert a quaternion into a matrix.

    Note, this function normalizes given quarternions to unit length.

    Args:
        q: Array of shape (4,) representing (w, x, y, z) quaternion coordinates

    Returns:
        R: Array of shape (3, 3) representing the rotation matrix
    """
    if (np.linalg.norm(q) - 1.0) > EPSILON:
        q /= np.linalg.norm(q)
    w, x, y, z = q

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
        ]
    )


def test_quat2rotmat_1() -> None:
    """Test receiving a quaternion in (w, x, y, z) from a camera extrinsic matrix."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    R = quat2rotmat(q)
    assert np.allclose(R, quat2rotmat_numpy(q))
    R_gt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R_gt, R)


def test_quat2rotmat_2() -> None:
    """Test receiving a quaternion in (w, x, y, z) from a camera extrinsic matrix."""
    q = np.array(
        [
            0.4962730586309743,
            -0.503110985154011,
            0.4964713836540661,
            -0.5040918101963521,
        ]
    )
    R = quat2rotmat(q)
    assert np.allclose(R, quat2rotmat_numpy(q))

    R_gt = np.array(
        [
            [-1.18477579e-03, 7.73955092e-04, 9.99998999e-01],
            [-9.99894783e-01, -1.44584330e-02, -1.17346213e-03],
            [1.44575103e-02, -9.99895172e-01, 7.91003660e-04],
        ]
    )
    assert np.allclose(R_gt, R)


def test_quat2rotmat_3() -> None:
    """Test receiving a quaternion in (w, x, y, z) from a camera extrinsic matrix."""
    q = np.array(
        [
            0.6115111374269877,
            -0.6173269265351116,
            -0.3480540121107544,
            0.3518806604959585,
        ]
    )
    R = quat2rotmat(q)
    assert np.allclose(R, quat2rotmat_numpy(q))

    R_gt = np.array(
        [
            [5.10076811e-01, -6.31658748e-04, -8.60128623e-01],
            [8.60084113e-01, -9.82506691e-03, 5.10057631e-01],
            [-8.77300364e-03, -9.99951533e-01, -4.46825914e-03],
        ]
    )
    assert np.allclose(R_gt, R)


def test_quat2rotmat_4() -> None:
    """Test receiving a quaternion in (w, x, y, z) from an object trajectory."""
    q = np.array(
        [
            0.0036672729619914197,
            -1.3748614058859026e-05,
            -0.00023389080405946338,
            0.9999932480847505,
        ]
    )
    R = quat2rotmat(q)
    assert np.allclose(R, quat2rotmat_numpy(q))

    R_gt = np.array(
        [
            [-9.99973102e-01, -7.33448997e-03, -2.92125253e-05],
            [7.33450283e-03, -9.99972993e-01, -4.67677610e-04],
            [-2.57815596e-05, -4.67879290e-04, 9.99999890e-01],
        ]
    )
    assert np.allclose(R_gt, R)


def test_quat2rotmat_5() -> None:
    """Test receiving a quaternion in (w, x, y, z) from an object trajectory."""
    q = np.array(
        [
            0.9998886199825181,
            -0.002544078377693514,
            -0.0028621717588219564,
            -0.01442509159370476,
        ]
    )
    R = quat2rotmat(q)
    assert np.allclose(R, quat2rotmat_numpy(q))

    R_gt = np.array(
        [
            [0.99956745, 0.02886153, -0.00565031],
            [-0.02883241, 0.99957089, 0.00517016],
            [0.0057971, -0.00500502, 0.99997067],
        ]
    )
    assert np.allclose(R_gt, R)


def test_invalid_quaternion_zero_norm() -> None:
    """Ensure that passing a zero-norm quaternion raises an error, as normalization would divide by 0."""
    q = np.array([0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ZeroDivisionError) as e_info:
        quat2rotmat(q)


def test_quaternion_renormalized() -> None:
    """Make sure that a quaternion is correctly re-normalized.

    Normalized and unnormalized quaternion variants should generate the same 3d rotation matrix.
    """
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    R1 = quat2rotmat(q1)

    q2 = np.array([0.0, 0.0, 0.0, 2.0])
    R2 = quat2rotmat(q2)

    assert np.allclose(R1, R2)


def test_rotmat2quat() -> None:
    """Ensure `rotmat2quat()` correctly converts rotation matrices to scalar-first quaternions."""
    num_trials = 1000
    for trial in range(num_trials):

        # generate random rotation matrices by sampling quaternion elements from normal distribution
        # https://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices

        q = np.random.randn(4)
        q /= np.linalg.norm(q)

        R = quat2rotmat(q)
        q_ = rotmat2quat(R)

        # Note: A unit quaternion multiplied by 1 or -1 represents the same 3d rotation
        # https://math.stackexchange.com/questions/2016282/negative-quaternion
        # https://math.stackexchange.com/questions/1790521/unit-quaternion-multiplied-by-1
        assert np.allclose(q, q_) or np.allclose(-q, q_)


def test_yaw_to_quaternion3d() -> None:
    """Ensure yaw_to_quaternion3d() outputs correct values.

    Compare `yaw_to_quaternion3d()` output (which relies upon Scipy) to manual
    computation of the 3d rotation matrix, followed by a rotation matrix -> quaternion conversion.
    """

    def rot3d_z(angle_rad: float) -> np.ndarray:
        """Generate 3d rotation matrix about the z-axis, from a rotation angle in radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        R = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )
        return R

    num_trials = 1000
    angle_samples_rad = np.random.rand(num_trials) * 2 * np.pi
    for angle_rad in angle_samples_rad:

        R = rot3d_z(angle_rad)
        q = rotmat2quat(R)

        q_ = yaw_to_quaternion3d(angle_rad)

        assert isinstance(q_, np.ndarray)
        assert q_.size == 4

        # Note: A unit quaternion multiplied by 1 or -1 represents the same 3d rotation
        # https://math.stackexchange.com/questions/2016282/negative-quaternion
        # https://math.stackexchange.com/questions/1790521/unit-quaternion-multiplied-by-1
        assert np.allclose(q, q_) or np.allclose(q, -q_)
