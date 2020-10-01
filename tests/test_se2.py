# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Module for testing `SE2`."""

import numpy as np
import pytest

from argoverse.utils.se2 import SE2


def rotation_matrix_from_rotation(theta: float) -> np.ndarray:
    """Return rotation matrix corresponding to rotation theta.

    Args:
        theta: rotation amount in radians.

    Returns:
        2 x 2 np.ndarray rotation matrix corresponding to rotation theta.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


def test_SE2_constructor() -> None:
    """Test for construction of an arbitrary SE2."""
    theta = 2 * np.pi / 7.0
    rotation_matrix = rotation_matrix_from_rotation(theta)
    translation_vector = np.array([-86.5, 0.99])
    dst_se2_src = SE2(rotation=rotation_matrix.copy(), translation=translation_vector.copy())

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    T_mat_gt = np.array(
        [[cos_theta, -sin_theta, translation_vector[0]], [sin_theta, cos_theta, translation_vector[1]], [0, 0, 1.0]]
    )

    assert np.allclose(dst_se2_src.rotation, rotation_matrix)
    assert np.allclose(dst_se2_src.translation, translation_vector)
    assert np.allclose(dst_se2_src.transform_matrix, T_mat_gt)

    with pytest.raises(ValueError):
        SE2(np.array([1]), translation_vector)

    with pytest.raises(ValueError):
        SE2(rotation_matrix, np.array([1, 2, 3]))


def test_SE2_transform_point_cloud_identity() -> None:
    """Test that transformation by an Identity SE2 does not change pointclouds."""
    pts = np.array([[0.5, 0], [1, -0.5], [1.5, 0], [2, -1]])
    dst_se2_src = SE2(rotation=np.eye(2), translation=np.zeros(2))
    transformed_pts = dst_se2_src.transform_point_cloud(pts.copy())

    assert np.allclose(transformed_pts, pts)

    with pytest.raises(ValueError):
        dst_se2_src.transform_point_cloud(np.random.rand(1))

    with pytest.raises(ValueError):
        dst_se2_src.transform_point_cloud(np.random.rand(1, 3))


def test_SE2_transform_point_cloud_pi_radians() -> None:
    """Test for validity of results of transformation."""
    pts = np.array([[0.5, 0], [1, -0.5], [1.5, 0], [2, -1]])
    theta = np.pi
    rotation_matrix = rotation_matrix_from_rotation(theta)
    translation_vector = np.array([2.0, 2.0])
    dst_se2_src = SE2(rotation=rotation_matrix, translation=translation_vector)
    transformed_pts = dst_se2_src.transform_point_cloud(pts)

    gt_transformed_pts = np.array([[1.5, 2.0], [1, 2.5], [0.5, 2], [0, 3.0]])

    assert np.allclose(transformed_pts, gt_transformed_pts)


def test_SE2_inverse_transform_point_cloud_identity() -> None:
    """Test that inverse transforming by Identity does not affect the pointclouds."""
    transformed_pts = np.array([[0.5, 0], [1, -0.5], [1.5, 0], [2, -1]])
    dst_se2_src = SE2(rotation=np.eye(2), translation=np.zeros(2))
    pts = dst_se2_src.inverse_transform_point_cloud(transformed_pts.copy())
    assert np.allclose(pts, transformed_pts)

    with pytest.raises(ValueError):
        dst_se2_src.transform_point_cloud(np.random.rand(1, 3))


def test_SE2_inverse_transform_point_cloud_pi_radians() -> None:
    """Test for validity of inverse transformation by an SE2."""
    transformed_pts = np.array([[1.5, 2.0], [1, 2.5], [0.5, 2], [0, 3.0]])
    theta = np.pi
    rotation_matrix = rotation_matrix_from_rotation(theta)
    translation_vector = np.array([2.0, 2.0])
    dst_se2_src = SE2(rotation=rotation_matrix, translation=translation_vector)
    pts = dst_se2_src.inverse_transform_point_cloud(transformed_pts)
    gt_pts = np.array([[0.5, 0], [1, -0.5], [1.5, 0], [2, -1]])

    assert np.allclose(pts, gt_pts)


def test_SE2_chaining_transforms() -> None:
    """Test for correctness of SE2 chaining / composing."""
    theta = np.pi
    rotation_matrix = rotation_matrix_from_rotation(theta)
    translation_vector = np.array([0, 1])
    fr2_se2_fr1 = SE2(rotation=rotation_matrix, translation=translation_vector)
    fr1_se2_fr0 = SE2(rotation=rotation_matrix, translation=translation_vector)

    fr2_se2_fr0 = fr2_se2_fr1.right_multiply_with_se2(fr1_se2_fr0)

    pts = np.array([[1, 0], [2, 0], [3, 0]])
    transformed_pts = fr2_se2_fr0.transform_point_cloud(pts.copy())
    assert np.allclose(pts, transformed_pts)
    assert np.allclose(fr2_se2_fr0.transform_matrix, np.eye(3))


def test_SE2_inverse() -> None:
    """Test for numerical correctess of the inverse functionality."""
    src_pts_gt = np.array([[1, 0], [2, 0]])

    dst_pts_gt = np.array([[-2, -1], [-2, 0]])
    theta = np.pi / 2.0
    rotation_matrix = rotation_matrix_from_rotation(theta)
    translation_vector = np.array([-2, -2])

    dst_se2_src = SE2(rotation=rotation_matrix, translation=translation_vector)
    src_se2_dst = dst_se2_src.inverse()

    dst_pts = dst_se2_src.transform_point_cloud(src_pts_gt.copy())
    src_pts = src_se2_dst.transform_point_cloud(dst_pts_gt.copy())

    assert np.allclose(dst_pts, dst_pts_gt)
    assert np.allclose(src_pts, src_pts_gt)
    gt_inv_mat = np.linalg.inv(dst_se2_src.transform_matrix)
    assert np.allclose(src_se2_dst.transform_matrix, gt_inv_mat)
