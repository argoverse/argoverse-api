# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Tests for frustum clipping functions."""

import numpy as np

from argoverse.utils.frustum_clipping import (
    clip_segment_v3_plane_n,
    cuboid_to_2d_frustum_bbox,
    fit_plane_to_point_cloud,
    form_left_clipping_plane,
    form_low_clipping_plane,
    form_near_clipping_plane,
    form_right_clipping_plane,
    form_top_clipping_plane,
    generate_frustum_planes,
    plane_point_side_v3,
)


def test_plane_point_side_v3_behind_plane() -> None:
    """Check if a point is in direction of plane normal or on other side."""
    p = np.array([1, 1, 1, 0])
    v = np.array([-1, -1, -1])
    sign = plane_point_side_v3(p, v)
    assert sign < 0


def test_plane_point_side_v3_on_plane() -> None:
    """Check if point is in direction of plane normal or on other side."""
    p = np.array([1, 1, 1, 0])
    v = np.array([0, 0, 0])
    sign = plane_point_side_v3(p, v)
    assert sign == 0


def test_plane_point_side_v3_point_in_front_of_plane() -> None:
    """Check if point is in direction of plane normal or on other side."""
    p = np.array([1, 1, 1, 0])
    v = np.array([2, 2, 2])
    sign = plane_point_side_v3(p, v)
    assert sign > 0


def test_fit_plane_to_point_cloud() -> None:
    """Given a plane with slope +2/1 for +y/+z, find slowly tilting normal away from the plane.

        +y
       /|
      / |
     /  |
    ------ + z
    """
    pc = np.array([[0, 2, 1], [1, 2, 1], [0, 0, 0]])
    a, b, c, d = fit_plane_to_point_cloud(pc)

    assert np.isclose(d, 0)  # touching origin
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    gt_normal = np.array([0.0, -1.0, 2.0])
    gt_normal /= np.linalg.norm(gt_normal)

    # correct y sign if needed, slope is all what matters
    # (ratio between +y/+z should be -1/2)
    if normal[1] > 0:
        normal *= -1

    assert np.allclose(gt_normal, normal)


def test_clip_segment_v3_plane_n_all_outside() -> None:
    r"""Test clipping line segments when all points are outside the view.

    normals point into this frustum
    \\   //    \\  /
         \\     //
            \\//

        o---------o line segment
    """
    # p1, p2: pair of 3d vectors defining a line segment.
    # planes: a sequence of (4 floats): `(x, y, z, d)`.
    p1 = np.array([-1, -0.5, 4.0])
    p2 = np.array([1.0, -0.5, 3.0])

    planes = [np.array([-1.0, 2.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0, 0.0])]

    # Returns 2 vector triplets (representing the clipped segment)
    # or (None, None) meaning the segment is entirely outside the frustun.
    p1_clip, p2_clip = clip_segment_v3_plane_n(p1, p2, planes)

    assert p1_clip == None  # noqa (ignore pycodestyle E711)
    assert p2_clip == None  # noqa (ignore pycodestyle E711)
    print(p1_clip, p2_clip)


def test_clip_segment_v3_plane_n_clip_twice() -> None:
    r"""Test clipping line segments twice.

    The normals point into this frustum

    \\    /    \\  /
       o-\\ --//--o  line segment
           \\//
    """
    # p1, p2: pair of 3d vectors defining a line segment.
    # planes: a sequence of (4 floats): `(x, y, z, d)`.
    p1 = np.array([2, 0.5, 0.0])
    p2 = np.array([-2, 0.5, 0.0])

    planes = [np.array([-1.0, 2.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0, 0.0])]

    # Returns 2 vector triplets (representing the clipped segment)
    # or (None, None) meaning the segment is entirely outside the frustun.
    p1_clip, p2_clip = clip_segment_v3_plane_n(p1, p2, planes)
    assert np.allclose(p1_clip, np.array([1, 0.5, 0.0]))
    assert np.allclose(p2_clip, np.array([-1, 0.5, 0.0]))


def test_clip_segment_v3_plane_n_subsumed_in_frustum() -> None:
    r"""Test clipping line segments that are subsumed in the frustum.

    The normals point into this frustum
    \\   // o---o  \\  /
         \\      //
            \\//

    Line segment is entirely inside the frustum this time, so stays intact.
    """
    # p1, p2: pair of 3d vectors defining a line segment.
    # planes: a sequence of (4 floats): `(x, y, z, d)`.
    p1 = np.array([1.0, 2, 0.0])
    p2 = np.array([-1.0, 2.0, 0.0])

    planes = [np.array([-1.0, 2.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0, 0.0])]

    # Returns 2 vector triplets (representing the clipped segment)
    # or (None, None) meaning the segment is entirely outside the frustun.
    p1_clip, p2_clip = clip_segment_v3_plane_n(p1.copy(), p2.copy(), planes)
    assert np.allclose(p1_clip, p1)
    assert np.allclose(p2_clip, p2)


def test_clip_segment_v3_plane_n() -> None:
    r"""Test clipping line segment. Expect that the bottom point will be clipped to the origin.

    The normals point into this frustum
    \\   //   o   \\  /
         \\   |  //
            \\//
              |
              o line segment half in, half out
    """
    # p1, p2: pair of 3d vectors defining a line segment.
    # planes: a sequence of (4 floats): `(x, y, z, d)`.
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, -1.0, 0.0])

    planes = [np.array([-1.0, 2.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0, 0.0])]

    # Returns 2 vector triplets (representing the clipped segment)
    # or (None, None) meaning the segment is entirely outside the frustun.
    p1_clip, p2_clip = clip_segment_v3_plane_n(p1.copy(), p2.copy(), planes)
    assert np.allclose(p1_clip, p1)
    assert np.allclose(p2_clip, np.zeros(3))


def test_form_right_clipping_plane() -> None:
    """Test form_right_clipping_plane(). Use 4 points to fit the right clipping plane."""
    fx = 10.0
    img_width = 30
    right_plane = form_right_clipping_plane(fx, img_width)

    Y_OFFSET = 10  # arbitrary extent down the imager
    right = np.array([[0, 0, 0], [img_width / 2.0, 0, fx], [0, Y_OFFSET, 0], [img_width / 2.0, Y_OFFSET, fx]])

    a, b, c, d = fit_plane_to_point_cloud(right)
    right_plane_gt = np.array([a, b, c, d])

    # enforce that plane normal points into the frustum
    # x-component of normal should point in negative direction.
    if right_plane_gt[0] > 0:
        right_plane_gt *= -1

    assert np.allclose(right_plane, right_plane_gt)


def test_form_left_clipping_plane() -> None:
    """Test form_left_clipping_plane(). Use 4 points to fit the left clipping plane."""
    fx = 10.0
    img_width = 30
    left_plane = form_left_clipping_plane(fx, img_width)

    Y_OFFSET = 10
    left = np.array([[0, 0, 0], [-img_width / 2.0, 0, fx], [0, Y_OFFSET, 0], [-img_width / 2.0, Y_OFFSET, fx]])

    a, b, c, d = fit_plane_to_point_cloud(left)
    left_plane_gt = -1 * np.array([a, b, c, d])

    # enforce that plane normal points into the frustum
    if left_plane_gt[0] < 0:
        left_plane_gt *= -1

    assert np.allclose(left_plane, left_plane_gt)


def test_form_top_clipping_plane() -> None:
    """Test form_top_clipping_plane(). Use 3 points to fit the TOP clipping plane."""
    fx = 10.0
    img_height = 45
    top_plane = form_top_clipping_plane(fx, img_height)

    img_width = 1000.0
    top_pts = np.array([[0, 0, 0], [-img_width / 2, -img_height / 2, fx], [img_width / 2, -img_height / 2, fx]])
    a, b, c, d = fit_plane_to_point_cloud(top_pts)
    top_plane_gt = np.array([a, b, c, d])

    # enforce that plane normal points into the frustum
    if top_plane_gt[1] < 0:
        # y-coord of normal should point in pos y-axis dir(down) on top-clipping plane
        top_plane_gt *= -1
    assert top_plane_gt[1] > 0 and top_plane_gt[2] > 0

    assert np.allclose(top_plane, top_plane_gt)


def test_form_low_clipping_plane() -> None:
    """Test form_low_clipping_plane()."""
    fx = 12.0
    img_height = 35
    low_plane = form_low_clipping_plane(fx, img_height)

    img_width = 10000
    low_pts = np.array([[0, 0, 0], [-img_width / 2, img_height / 2, fx], [img_width / 2, img_height / 2, fx]])
    a, b, c, d = fit_plane_to_point_cloud(low_pts)
    low_plane_gt = np.array([a, b, c, d])

    # enforce that plane normal points into the frustum
    # y-coord of normal should point in neg y-axis dir(up) on low-clipping plane
    # z-coord should point in positive z-axis direction (away from camera)
    if low_plane_gt[1] > 0:
        low_plane_gt *= -1
    assert low_plane_gt[1] < 0 and low_plane_gt[2] > 0

    assert np.allclose(low_plane, low_plane_gt)


def test_form_near_clipping_plane() -> None:
    """Test form_near_clipping_plane(). Use 4 points to fit the near clipping plane."""
    img_width = 10
    img_height = 15
    near_clip_dist = 30.0
    near_plane = form_near_clipping_plane(near_clip_dist)

    near = np.array(
        [
            [img_width / 2, 0, near_clip_dist],
            [-img_width / 2, 0, near_clip_dist],
            [img_width / 2, -img_height / 2.0, near_clip_dist],
            [img_width / 2, img_height / 2.0, near_clip_dist],
        ]
    )

    a, b, c, d = fit_plane_to_point_cloud(near)
    near_plane_gt = np.array([a, b, c, d])

    assert np.allclose(near_plane, near_plane_gt)


def test_generate_frustum_planes_ring_cam() -> None:
    """Test generate_frustum_planes() for a ring camera.

    Skew is 0.0.
    """
    near_clip_dist = 6.89  # arbitrary value
    K = np.eye(3)
    # Set "focal_length_x_px_"
    K[0, 0] = 1402.4993697398709

    # Set "focal_length_y_px_"
    K[1, 1] = 1405.1207294310225

    # Set "focal_center_x_px_"
    K[0, 2] = 957.8471720086527

    # Set "focal_center_y_px_"
    K[1, 2] = 600.442948946496

    camera_name = "ring_front_right"
    img_height = 1200
    img_width = 1920
    planes = generate_frustum_planes(K, camera_name, near_clip_dist=near_clip_dist)
    if planes is None:
        assert False
    left_plane, right_plane, near_plane, low_plane, top_plane = planes

    fx = K[0, 0]
    left_plane_gt = np.array([fx, 0.0, img_width / 2.0, 0.0])
    right_plane_gt = np.array([-fx, 0.0, img_width / 2.0, 0.0])
    near_plane_gt = np.array([0.0, 0.0, 1.0, -near_clip_dist])
    low_plane_gt = np.array([0.0, -fx, img_height / 2.0, 0.0])
    top_plane_gt = np.array([0.0, fx, img_height / 2.0, 0.0])

    assert np.allclose(left_plane, left_plane_gt / np.linalg.norm(left_plane_gt))
    assert np.allclose(right_plane, right_plane_gt / np.linalg.norm(right_plane_gt))
    assert np.allclose(low_plane, low_plane_gt / np.linalg.norm(low_plane_gt))
    assert np.allclose(top_plane, top_plane_gt / np.linalg.norm(top_plane_gt))
    assert np.allclose(near_plane, near_plane_gt)


def test_generate_frustum_planes_stereo() -> None:
    """Test generate_frustum_planes() for a stereo camera.

    Skew is 0.0.
    """
    near_clip_dist = 3.56  # arbitrary value
    K = np.eye(3)
    # Set "focal_length_x_px_"
    K[0, 0] = 3666.534329132812

    # Set "focal_length_y_px_"
    K[1, 1] = 3673.5030423482513

    # Set "focal_center_x_px_"
    K[0, 2] = 1235.0158218941356

    # Set "focal_center_y_px_"
    K[1, 2] = 1008.4536901420888

    camera_name = "stereo_front_left"
    img_height = 2056
    img_width = 2464
    planes = generate_frustum_planes(K, camera_name, near_clip_dist=near_clip_dist)
    if planes is None:
        assert False
    left_plane, right_plane, near_plane, low_plane, top_plane = planes

    fx = K[0, 0]
    left_plane_gt = np.array([fx, 0.0, img_width / 2.0, 0.0])
    right_plane_gt = np.array([-fx, 0.0, img_width / 2.0, 0.0])
    near_plane_gt = np.array([0.0, 0.0, 1.0, -near_clip_dist])
    low_plane_gt = np.array([0.0, -fx, img_height / 2.0, 0.0])
    top_plane_gt = np.array([0.0, fx, img_height / 2.0, 0.0])

    assert np.allclose(left_plane, left_plane_gt / np.linalg.norm(left_plane_gt))
    assert np.allclose(right_plane, right_plane_gt / np.linalg.norm(right_plane_gt))
    assert np.allclose(low_plane, low_plane_gt / np.linalg.norm(low_plane_gt))
    assert np.allclose(top_plane, top_plane_gt / np.linalg.norm(top_plane_gt))
    assert np.allclose(near_plane, near_plane_gt)


def test_cuboid_to_2d_frustum_bbox_smokescreen() -> None:
    """This test is currently just a smokescreen for cuboid_to_2d_frustum_bbox().

    Here we are just checking output values, but visual verification is still needed.
    `plot_frustum_planes_and_normals(planes, corners)` can help to do the visual check.
    """
    cuboid_verts_3d = np.array(
        [[2, 0, 10], [-2, 0, 10], [-2, 2, 10], [2, 2, 10], [2, 0, 18], [-2, 0, 18], [-2, 2, 18], [2, 2, 18]]
    )
    K = np.eye(3)
    # Set "focal_length_x_px_"
    K[0, 0] = 3666.534329132812

    # Set "focal_length_y_px_"
    K[1, 1] = 3673.5030423482513

    # Set "focal_center_x_px_"
    K[0, 2] = 1235.0158218941356

    # Set "focal_center_y_px_"
    K[1, 2] = 1008.4536901420888

    img_height = 2056
    img_width = 2464
    fx = K[0, 0]

    near_clip_dist = 0.5
    left_plane = np.array([fx, 0.0, img_width / 2.0, 0.0])
    right_plane = np.array([-fx, 0.0, img_width / 2.0, 0.0])
    near_plane = np.array([0.0, 0.0, 1.0, -near_clip_dist])
    low_plane = np.array([0.0, -fx, img_height / 2.0, 0.0])
    top_plane = np.array([0.0, fx, img_height / 2.0, 0.0])

    planes = [left_plane, right_plane, near_plane, low_plane, top_plane]

    bbox_2d = cuboid_to_2d_frustum_bbox(cuboid_verts_3d, planes, K)

    assert bbox_2d.shape == (4,)

    bbox_2d_gt = np.array([502.0, 1008.0, 1968.0, 1743.0])
    assert np.allclose(bbox_2d, bbox_2d_gt)
