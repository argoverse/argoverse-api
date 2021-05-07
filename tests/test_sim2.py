import copy

import numpy as np
import pytest

from argoverse.utils.se2 import SE2
from argoverse.utils.sim2 import Sim2


def test_constructor() -> None:
    """Sim(2) to perform p_b = bSa * p_a"""
    bRa = np.eye(2)
    bta = np.array([1, 2])
    bsa = 3.0
    bSa = Sim2(R=bRa, t=bta, s=bsa)
    assert isinstance(bSa, Sim2)
    assert np.allclose(bSa.R_, bRa)
    assert np.allclose(bSa.t_, bta)
    assert np.allclose(bSa.s_, bsa)


def test_is_eq() -> None:
    """ Ensure object equality works properly (are equal). """
    bSa = Sim2(R=np.eye(2), t=np.array([1, 2]), s=3.0)
    bSa_ = Sim2(R=np.eye(2), t=np.array([1.0, 2.0]), s=3)
    assert bSa == bSa_


def test_not_eq_translation() -> None:
    """ Ensure object equality works properly (not equal translation). """
    bSa = Sim2(R=np.eye(2), t=np.array([2, 1]), s=3.0)
    bSa_ = Sim2(R=np.eye(2), t=np.array([1.0, 2.0]), s=3)
    assert bSa != bSa_


def test_not_eq_rotation() -> None:
    """ Ensure object equality works properly (not equal rotation). """
    bSa = Sim2(R=np.eye(2), t=np.array([2, 1]), s=3.0)
    bSa_ = Sim2(R=-1 * np.eye(2), t=np.array([2.0, 1.0]), s=3)
    assert bSa != bSa_


def test_not_eq_scale() -> None:
    """ Ensure object equality works properly (not equal scale). """
    bSa = Sim2(R=np.eye(2), t=np.array([2, 1]), s=3.0)
    bSa_ = Sim2(R=np.eye(2), t=np.array([2.0, 1.0]), s=1.0)
    assert bSa != bSa_


def test_rotation() -> None:
    """ Ensure rotation component is returned properly. """
    R = np.array([[0, -1], [1, 0]])
    t = np.array([1, 2])
    bSa = Sim2(R=R, t=t, s=3.0)

    expected_R = np.array([[0, -1], [1, 0]])
    assert np.allclose(expected_R, bSa.rotation)


def test_translation() -> None:
    """ Ensure translation component is returned properly. """
    R = np.array([[0, -1], [1, 0]])
    t = np.array([1, 2])
    bSa = Sim2(R=R, t=t, s=3.0)

    expected_t = np.array([1, 2])
    assert np.allclose(expected_t, bSa.translation)


def test_scale() -> None:
    """ Ensure the scale factor is returned properly. """
    bRa = np.eye(2)
    bta = np.array([1, 2])
    bsa = 3.0
    bSa = Sim2(R=bRa, t=bta, s=bsa)
    assert bSa.scale == 3.0


def test_compose():
    """ Ensure we can compose two Sim(2) transforms together. """
    scale = 2.0
    imgSw = Sim2(R=np.eye(2), t=np.array([1.0, 3.0]), s=scale)

    scale = 0.5
    wSimg = Sim2(R=np.eye(2), t=np.array([-2.0, -6.0]), s=scale)

    # identity
    wSw = Sim2(R=np.eye(2), t=np.zeros((2,)), s=1.0)
    assert wSw == imgSw.compose(wSimg)


def test_inverse():
    """ """
    scale = 2.0
    imgSw = Sim2(R=np.eye(2), t=np.array([1.0, 3.0]), s=scale)

    scale = 0.5
    wSimg = Sim2(R=np.eye(2), t=np.array([-2.0, -6.0]), s=scale)

    assert imgSw == wSimg.inverse()
    assert wSimg == imgSw.inverse()


def test_matrix() -> None:
    """ Ensure 3x3 matrix is formed correctly"""
    bRa = np.array([[0, -1], [1, 0]])
    bta = np.array([1, 2])
    bsa = 3.0
    bSa = Sim2(R=bRa, t=bta, s=bsa)

    bSa_expected = np.array([[0, -1, 1], [1, 0, 2], [0, 0, 1 / 3]])
    assert np.allclose(bSa_expected, bSa.matrix)


def test_matrix_homogenous_transform() -> None:
    """ Ensure 3x3 matrix transforms homogenous points as expected."""
    expected_img_pts = np.array([[6, 4], [4, 6], [0, 0], [1, 7]])

    world_pts = np.array([[2, -1], [1, 0], [-1, -3], [-0.5, 0.5]])
    scale = 2.0
    imgSw = Sim2(R=np.eye(2), t=np.array([1.0, 3.0]), s=scale)

    # convert to homogeneous
    world_pts_h = np.hstack([world_pts, np.ones((4, 1))])

    # multiply each (3,1) homogeneous point vector w/ transform matrix
    img_pts_h = (imgSw.matrix @ world_pts_h.T).T
    # divide (x,y,s) by s
    img_pts = img_pts_h[:, :2] / img_pts_h[:, 2].reshape(-1, 1)
    assert np.allclose(expected_img_pts, img_pts)


def test_transform_from_forwards() -> None:
    """ """
    expected_img_pts = np.array([[6, 4], [4, 6], [0, 0], [1, 7]])

    world_pts = np.array([[2, -1], [1, 0], [-1, -3], [-0.5, 0.5]])
    scale = 2.0
    imgSw = Sim2(R=np.eye(2), t=np.array([1.0, 3.0]), s=scale)

    img_pts = imgSw.transform_from(world_pts)
    assert np.allclose(expected_img_pts, img_pts)


def test_transform_from_backwards() -> None:
    """ """
    img_pts = np.array([[6, 4], [4, 6], [0, 0], [1, 7]])

    expected_world_pts = np.array([[2, -1], [1, 0], [-1, -3], [-0.5, 0.5]])
    scale = 0.5
    wSimg = Sim2(R=np.eye(2), t=np.array([-2.0, -6.0]), s=scale)

    world_pts = wSimg.transform_from(img_pts)
    assert np.allclose(expected_world_pts, world_pts)


def rotmat2d(theta: float) -> np.ndarray:
    """ Convert angle `theta` (in radians) to a 2x2 rotation matrix."""
    s = np.sin(theta)
    c = np.cos(theta)
    R = np.array([[c, -s], [s, c]])
    return R


def test_transform_point_cloud() -> None:
    """Guarantee we can implement the SE(2) inferface, w/ scale=1.0

    Sample 1000 random 2d rigid body transformations (R,t) and ensure
    that 2d points are transformed equivalently with SE(2) or Sim(3) w/ unit scale.
    """
    for sample in range(1000):
        # generate random 2x2 rotation matrix
        theta = np.random.rand() * 2 * np.pi
        R = rotmat2d(theta)
        t = np.random.randn(2)

        pts_b = np.random.randn(25, 2)

        aTb = SE2(copy.deepcopy(R), copy.deepcopy(t))
        aSb = Sim2(copy.deepcopy(R), copy.deepcopy(t), s=1.0)

        pts_a = aTb.transform_point_cloud(copy.deepcopy(pts_b))
        pts_a_ = aSb.transform_point_cloud(copy.deepcopy(pts_b))

        assert np.allclose(pts_a, pts_a_, atol=1e-5)


def test_cannot_set_zero_scale() -> None:
    """ Ensure that an exception is thrown if Sim(2) scale is set to zero."""
    R = np.eye(2)
    t = np.arange(2)
    s = 0.0

    with pytest.raises(ZeroDivisionError) as e_info:
        aSb = Sim2(R, t, s)
