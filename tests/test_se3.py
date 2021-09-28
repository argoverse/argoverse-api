# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Tests to verify the SE3 class works correctly."""

import time
from typing import Any, Callable, Union

import numpy as np

from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat


def get_yaw_angle_rotmat(theta: Union[float, int, np.ndarray]) -> np.ndarray:
    """Build simple test cases that rotate points by yaw angle. Points are rotated about the z-axis in the xy-plane."""
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R


def test_SE3_constructor() -> None:
    """Test making an arbitrary SE2 transformation for a pedestrian cuboid."""
    # x, y, z of cuboid center
    t = np.array([-34.7128603513203, 5.29461762417753, 0.10328996181488])

    # quaternion has order (w,x,y,z)
    q = np.array([0.700322174885275, 0.0, 0.0, -0.713826905743933])
    R = quat2rotmat(q)

    dst_se3_src = SE3(rotation=R.copy(), translation=t.copy())

    T_mat_gt = np.eye(4)
    T_mat_gt[:3, :3] = R
    T_mat_gt[:3, 3] = t

    assert np.allclose(dst_se3_src.rotation, R)
    assert np.allclose(dst_se3_src.translation, t)
    assert np.allclose(dst_se3_src.transform_matrix, T_mat_gt)


def test_SE3_transform_point_cloud_identity() -> None:
    """Test taking a point cloud and performing an SE3 transformation.

    Since the transformation is the identity, the points should not be affected.
    """
    pts = np.array([[1.0, 1.0, 1.1], [1.0, 1.0, 2.1], [1.0, 1.0, 3.1]])
    dst_se3_src = SE3(rotation=np.eye(3), translation=np.zeros(3))
    transformed_pts = dst_se3_src.transform_point_cloud(pts.copy())

    assert np.allclose(transformed_pts, pts)


def test_SE3_transform_point_cloud_by_quaternion() -> None:
    """Test rotating points by a given quaternion, and then adding translation vector to each point."""
    pts = np.array([[1.0, 1.0, 1.1], [1.0, 1.0, 2.1], [1.0, 1.0, 3.1], [1.0, 1.0, 4.1]])

    # x, y, z of cuboid center
    t = np.array([-34.7128603513203, 5.29461762417753, 0.10328996181488])

    # quaternion has order (w,x,y,z)
    q = np.array([0.700322174885275, 0.0, 0.0, -0.713826905743933])
    R = quat2rotmat(q)

    dst_se3_src = SE3(rotation=R.copy(), translation=t.copy())
    transformed_pts = dst_se3_src.transform_point_cloud(pts.copy())

    gt_transformed_pts = pts.dot(R.T)
    gt_transformed_pts += t
    print(gt_transformed_pts)

    assert np.allclose(transformed_pts, gt_transformed_pts)
    gt_transformed_pts_explicit = np.array(
        [
            [-33.73214043, 4.2757023, 1.20328996],
            [-33.73214043, 4.2757023, 2.20328996],
            [-33.73214043, 4.2757023, 3.20328996],
            [-33.73214043, 4.2757023, 4.20328996],
        ]
    )
    assert np.allclose(transformed_pts, gt_transformed_pts_explicit)


def test_SE3_transform_point_cloud_by_yaw_angle() -> None:
    """Test rotating points by yaw angle of pi/4 in the xy plane, then adding a translation vector to them all."""
    pts = np.array([[1.0, 0.0, 4.0], [1.0, 0.0, 3.0]])
    theta = np.pi / 4
    R = get_yaw_angle_rotmat(theta)
    t = np.array([1.0, 2.0, 3.0])

    dst_SE3_src = SE3(rotation=R.copy(), translation=t.copy())

    transformed_pts = dst_SE3_src.transform_point_cloud(pts.copy())
    gt_transformed_pts = np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2, 4.0], [np.sqrt(2) / 2, np.sqrt(2) / 2, 3.0]])
    gt_transformed_pts += t
    assert np.allclose(transformed_pts, gt_transformed_pts)


def test_SE3_inverse_transform_point_cloud_identity() -> None:
    """Test taking a transformed point cloud and undoing the transformation.

    Since the transformation was the identity, the points should not be affected.
    """
    transformed_pts = np.array([[1.0, 1.0, 1.1], [1.0, 1.0, 2.1], [1.0, 1.0, 3.1], [1.0, 1.0, 4.1]])
    dst_se3_src = SE3(rotation=np.eye(3), translation=np.zeros(3))
    pts = dst_se3_src.inverse_transform_point_cloud(transformed_pts.copy())
    assert np.allclose(pts, transformed_pts)


def test_SE3_inverse_transform_point_cloud() -> None:
    """Test taking a transformed point cloud and undoing the transformation to recover the original points."""
    transformed_pts = np.array(
        [
            [-33.73214043, 4.2757023, 1.20328996],
            [-33.73214043, 4.2757023, 2.20328996],
            [-33.73214043, 4.2757023, 3.20328996],
            [-33.73214043, 4.2757023, 4.20328996],
        ]
    )
    # x, y, z of cuboid center
    t = np.array([-34.7128603513203, 5.29461762417753, 0.10328996181488])

    # quaternion has order (w,x,y,z)
    q = np.array([0.700322174885275, 0.0, 0.0, -0.713826905743933])
    R = quat2rotmat(q)

    dst_se3_src = SE3(rotation=R, translation=t)
    pts = dst_se3_src.inverse_transform_point_cloud(transformed_pts)
    gt_pts = np.array([[1.0, 1.0, 1.1], [1.0, 1.0, 2.1], [1.0, 1.0, 3.1], [1.0, 1.0, 4.1]])

    assert np.allclose(pts, gt_pts)


def test_SE3_chaining_transforms() -> None:
    """Test chaining two transformations to restore 3 points back to their original position."""
    theta0 = np.pi / 4
    R0 = get_yaw_angle_rotmat(theta0)

    theta1 = np.pi
    R1 = get_yaw_angle_rotmat(theta1)

    t0 = np.zeros(3)
    t1 = np.zeros(3)

    fr2_se3_fr1 = SE3(rotation=R0, translation=t0)
    fr1_se3_fr0 = SE3(rotation=R1, translation=t1)

    fr2_se3_fr0 = fr2_se3_fr1.right_multiply_with_se3(fr1_se3_fr0)

    pts = np.array([[1.0, 0.0, 4.0], [1.0, 0.0, 3.0]])
    transformed_pts = fr2_se3_fr0.transform_point_cloud(pts.copy())
    gt_transformed_pts = np.array(
        [
            [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 4.0],
            [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 3.0],
        ]
    )

    assert np.allclose(transformed_pts, gt_transformed_pts)

    combined_R = get_yaw_angle_rotmat(theta0 + theta1)
    assert np.allclose(fr2_se3_fr0.rotation, combined_R)
    assert np.allclose(fr2_se3_fr0.translation, np.zeros(3))


def test_SE3_transform_point_cloud_optimized() -> None:
    """Ensure that our transform_point_cloud() implementation is faster than a naive implementation.

    We use 1k trials, with a point cloud of 100k points. The chosen SE(3) transformation is arbitrary.
    """

    def transform_point_cloud_slow(point_cloud: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Slow, unoptimized method -- allocated unnecessary array for homogeneous coordinates."""
        # convert to homogeneous
        num_pts = point_cloud.shape[0]
        homogeneous_pts = np.hstack([point_cloud, np.ones((num_pts, 1))])
        transformed_point_cloud = homogeneous_pts.dot(transform_matrix.T)
        return transformed_point_cloud[:, :3]

    # number of trials
    n_trials = 1000
    num_pts = 100000
    pts_src = np.random.randn(num_pts, 3)

    # x, y, z of cuboid center
    t = np.array([-34.7128603513203, 5.29461762417753, 0.10328996181488])

    # quaternion has order (w,x,y,z)
    q = np.array([0.700322174885275, 0.0, 0.0, -0.713826905743933])
    R = quat2rotmat(q)

    dst_se3_src = SE3(rotation=R.copy(), translation=t.copy())

    def measure_time_elapsed(n_trials: int, fn: Callable, *args: Any) -> float:
        """Compute time in seconds to execute a function `n_trials` times."""
        start = time.time()
        for i in range(n_trials):
            pts_dst = fn(*args)
        end = time.time()
        duration = end - start
        return duration

    optimized_duration = measure_time_elapsed(n_trials, dst_se3_src.transform_point_cloud, pts_src)

    unoptimized_duration = measure_time_elapsed(
        n_trials,
        transform_point_cloud_slow,
        pts_src,
        dst_se3_src.transform_matrix,
    )

    assert optimized_duration < unoptimized_duration
