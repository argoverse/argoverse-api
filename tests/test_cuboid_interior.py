# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np

from argoverse.utils.cuboid_interior import (
    extract_pc_in_box3d_hull,
    filter_point_cloud_to_bbox,
    filter_point_cloud_to_bbox_2D_vectorized,
    filter_point_cloud_to_bbox_3D_vectorized,
)

"""
Run it with "pytest tracker_tools_tests.py"
"""


def get_scenario_1():
    """
    Form an arbitrary 3d cuboid, and call it "Custom Cuboid 1".

    Form a Numpy array of shape (N,3), representing a point cloud that might
    be found nearby "Custom Cuboid 1."
    """
    length = 6.0  # x-axis
    width = 4.0  # y-axis
    height = 2.0  # z-axis
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    bbox_3d = np.vstack((x_corners, y_corners, z_corners)).T

    pc_raw = np.array(
        [
            [3.1, 0.0, 0.0],  # outside x-range
            [2.9, 0.0, 0.0],  # inside x-range
            [-2.9, 0.0, 0.0],  # inside x-range
            [-3.1, 0.0, 0.0],  # outside x-range
            [0.0, 2.1, 0.0],  # outside y-range
            [0.0, 1.9, 0.0],  # inside y-range
            [0.0, -1.9, 0.0],  # inside y-range
            [0.0, -2.1, 0.0],  # outside y-range
            [0.0, 0.0, 1.1],  # outside z-range
            [0.0, 0.0, 0.9],  # inside z-range
            [0.0, 0.0, -0.9],  # inside z-range
            [0.0, 0.0, -1.1],  # outside z-range
        ]
    )

    # ground truth segment
    gt_segment = np.array(
        [
            [2.9, 0.0, 0.0],  # inside x-range
            [-2.9, 0.0, 0.0],  # inside x-range
            [0.0, 1.9, 0.0],  # inside y-range
            [0.0, -1.9, 0.0],  # inside y-range
            [0.0, 0.0, 0.9],  # inside z-range
            [0.0, 0.0, -0.9],  # inside z-range
        ]
    )

    gt_is_valid = np.array(
        [
            False,  # outside x-range
            True,  # inside x-range
            True,  # inside x-range
            False,  # outside x-range
            False,  # outside y-range
            True,  # inside y-range
            True,  # inside y-range
            False,  # outside y-range
            False,  # outside z-range
            True,  # inside z-range
            True,  # inside z-range
            False,  # outside z-range
        ]
    )

    return pc_raw, bbox_3d, gt_segment, gt_is_valid


def test_extract_pc_in_box3d_hull():
    """
    """
    pc_raw, bbox_3d, gt_segment, gt_is_valid = get_scenario_1()
    segment, is_valid = extract_pc_in_box3d_hull(pc_raw, bbox_3d)

    assert np.array_equal(segment, gt_segment)
    assert np.array_equal(is_valid, gt_is_valid)


def test_3d_cuboid_interior_test1():
    """
        Generate 6 points just outside the volume of a 3d cuboid, and
        6 points just inside the volume. Verify that we can isolate the
        points that fall within the volume. For us, a 3d cuboid is Numpy
        array of shape (8,3) with the following vertex ordering:

                                5------4
                                |\\    |\\
                                | \\   | \\
                                6--\\--7  \\
                                \\  \\  \\ \\
                        l    \\  1-------0    h
                         e    \\ ||   \\ ||   e
                          n    \\||    \\||   i
                           g    \\2------3    g
                            t      width.(y)  h
                             h.               t.
                              (x)               (z-axis)
        """
    pc_raw, bbox_3d, gt_segment, gt_is_valid = get_scenario_1()
    segment, is_valid = filter_point_cloud_to_bbox_3D_vectorized(bbox_3d, pc_raw)

    assert np.array_equal(segment, gt_segment)
    assert np.array_equal(gt_is_valid, is_valid)


def test_2d_cuboid_interior_test1():
    """Test 2d version of the 3d projection above.  This is a similar
    test except it slices away the z dimension.

    """
    pc_raw, bbox_3d, gt_segment, gt_is_valid = get_scenario_1()

    to_2d = lambda array: array[:, :2]

    pc_raw_2d = np.unique(to_2d(pc_raw), axis=0)
    bbox_2d = to_2d(bbox_3d)[(0, 1, 4, 5), :]
    gt_segment_2d = np.unique(to_2d(gt_segment), axis=0)

    filtered_points, _ = filter_point_cloud_to_bbox_2D_vectorized(bbox_2d, pc_raw_2d)

    assert np.array_equal(filtered_points, gt_segment_2d)

    filtered_points = filter_point_cloud_to_bbox(bbox_2d, pc_raw_2d)

    assert np.array_equal(filtered_points, gt_segment_2d)
