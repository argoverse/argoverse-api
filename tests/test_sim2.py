import numpy as np

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
