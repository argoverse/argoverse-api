# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import numpy as np
from numpy.testing import assert_almost_equal

from argoverse.utils.centerline_utils import (
    centerline_to_polygon,
    filter_candidate_centerlines,
    get_normal_and_tangential_distance_point,
    get_nt_distance,
    is_overlapping_lane_seq,
)


def temp_test_straight_centerline_to_polygon() -> None:
    """
        Try converting a simple straight polyline into a polygon. Represents
        the conversion from a centerline to a lane segment polygon.

        Note that the returned polygon will ba a Numpy array of
        shape (2N+1,2), with duplicate first and last vertices.
        Dots below signify the centerline coordinates.

                |   .   |
                |   .   |
                |   .   |
                |   .   |
        """
    # create centerline: Numpy array of shape (N,2)
    centerline = np.array([[0, 2.0], [0.0, 0.0], [0.0, -2.0]])

    polygon = centerline_to_polygon(centerline)
    # polygon wraps around with right boundary, then reversed
    # left boundary, then back to start vertex
    gt_polygon = np.array([[-3.8, 2.0], [-3.8, 0.0], [-3.8, -2.0], [3.8, -2.0], [3.8, 0.0], [3.8, 2.0], [-3.8, 2.0]])

    assert np.array_equal(polygon, gt_polygon)


def test_is_overlapping_lane_seq() -> None:
    """Test is_overlapping_lane_seq"""

    lane_seq1 = [1, 2, 3, 4]
    lane_seq2 = [2, 3, 4, 5]

    assert is_overlapping_lane_seq(lane_seq1, lane_seq2)

    lane_seq1 = [1, 2, 3, 4]
    lane_seq2 = [3, 4]

    assert is_overlapping_lane_seq(lane_seq1, lane_seq2)

    lane_seq1 = [1, 2, 3, 4]
    lane_seq2 = [0, 3, 4]

    assert not is_overlapping_lane_seq(lane_seq1, lane_seq2)


def test_get_nt_distance_point() -> None:
    """Compute distances in centerline frame for a point"""
    """Test Case

        0  1 . .   3 . 4 . 5 . 6 . 7
           *                            5
            \
             *                          4
              \
               *                        3
                \
                 *     x                2
                  \
                 . *---*---*---*---*    1

                                        0
    """
    x = 4.0
    y = 2.0
    centerline = [
        (1.0, 5.0),
        (1.5, 4.0),
        (2.0, 3.0),
        (2.5, 2.0),
        (3.0, 1.0),
        (4.0, 1.0),
        (5.0, 1.0),
        (6.0, 1.0),
        (7.0, 1.0),
    ]

    tang_dist, norm_dist = get_normal_and_tangential_distance_point(x, y, centerline)
    assert_almost_equal(tang_dist, 5.472, 3)
    assert_almost_equal(norm_dist, 1.000, 3)


def test_get_nt_distance() -> None:
    """Compute distances in centerline frame for a trajectory"""
    """Test Case

        0  1 . .   3 . 4 . 5 . 6 . 7
           *                            5
            \
        x    *                          4
              \
               *    x                   3
                \
            x    *     x                2
                  \
                 . *---*---*---*---x    1

                            x           0
    """

    xy = np.array([(0.0, 4.0), (3.0, 3.0), (1.0, 2.0), (4.0, 2.0), (5.0, 0.0), (7.0, 1.0)])
    centerline = np.array(
        [(1.0, 5.0), (1.5, 4.0), (2.0, 3.0), (2.5, 2.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0), (6.0, 1.0), (7.0, 1.0)]
    )

    nt_dist = get_nt_distance(xy, centerline)

    expected_nt_dist = np.array([[1.34, 0.44], [2.68, 0.89], [2.68, 1.34], [5.47, 1.0], [6.47, 1.0], [8.47, 0.0]])
    print(nt_dist, expected_nt_dist)
    np.array_equal(nt_dist, expected_nt_dist)


def test_filter_candidate_centerlines() -> None:
    """Test filter candidate centerlines"""

    # Test Case

    # 0              20 24   30       40               60
    #                        *         *                      50
    #                        |         |
    #                        |         |
    #                        |         |
    #                        |         |
    #                        |         |                      40
    #                  (3)   |         |  (2)
    #                        ^         ^
    #                        ^         ^
    #                        |         |
    #                        |         |
    #                        |         |         (1)
    # *--------<<<-----------|---------|--------<<<--------*  30
    #                        | \       |
    #                xxxxxxxx^x \      ^
    #                        ^ x \(5)  ^
    #         (4)            |  x \    |
    # *-------->>>-----------|---x-\---|-------->>>--------*  20
    #                        |    x \  |
    #                        |     x \ |
    #                        |     x  \|
    #                        |     x   |                      10
    #                        |     x   |
    #                        ^     x   ^
    #                        ^     x   ^
    #                        |         |
    #                        *         *                       0
    #

    xy = np.array(
        [
            [35.0, 0.0],
            [35.0, 4.0],
            [35.0, 8.0],
            [35.0, 12.0],
            [35.0, 16.0],
            [33.0, 20.0],
            [31.0, 24.0],
            [29.0, 28.0],
            [25.0, 28.0],
            [21.0, 28.0],
        ]
    )

    cl1 = np.array([(60.0, 30.0), (0.0, 30.0)])
    cl2 = np.array([(40.0, 0.0), (40.0, 50.0)])
    cl3 = np.array([(30.0, 0.0), (30.0, 50.0)])
    cl4 = np.array([(0.0, 20.0), (60.0, 20.0)])
    cl5 = np.array([(40.0, 0.0), (40.0, 10.0), (30.0, 30.0), (0.0, 30.0)])

    candidate_cl = [cl1, cl2, cl3, cl4, cl5]

    filtered_cl = sorted(filter_candidate_centerlines(xy, candidate_cl))
    expected_cl = [cl5]

    for i in range(len(filtered_cl)):
        assert np.allclose(expected_cl[i], filtered_cl[i]), "Filtered centerlines wrong!"
