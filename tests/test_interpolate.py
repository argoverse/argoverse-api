# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Interpolate tools unit tests."""

import matplotlib.pyplot as plt
import numpy as np

from argoverse.utils.interpolate import compute_lane_width, compute_mid_pivot_arc, compute_midpoint_line, interp_arc


def test_compute_lane_width_straight() -> None:
    """
    Compute the lane width of the following straight lane segment
    (waypoints indicated with "o" symbol):

            o   o
            |   |
            o   o
            |   |
            o   o

    We can swap boundaries for this lane, and the width should be identical.
    """
    left_even_pts = np.array([[1, 1], [1, 0], [1, -1]])
    right_even_pts = np.array([[-1, 1], [-1, 0], [-1, -1]])
    lane_width = compute_lane_width(left_even_pts, right_even_pts)
    gt_lane_width = 2.0
    assert np.isclose(lane_width, gt_lane_width)

    lane_width = compute_lane_width(right_even_pts, left_even_pts)
    gt_lane_width = 2.0
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_lane_width_telescoping() -> None:
    """
    Compute the lane width of the following straight lane segment
    (waypoints indicated with "o" symbol):

       o          o
       \\        //
            o        o
            \\     //
             o     o
              \\ //
                o

    We can swap boundaries for this lane, and the width should be identical.
    """
    left_even_pts = np.array([[3, 2], [2, 1], [1, 0], [0, -1]])
    right_even_pts = np.array([[-3, 2], [-2, 1], [-1, 0], [0, -1]])
    lane_width = compute_lane_width(left_even_pts, right_even_pts)
    gt_lane_width = (6.0 + 4.0 + 2.0 + 0.0) / 4
    assert np.isclose(lane_width, gt_lane_width)

    lane_width = compute_lane_width(right_even_pts, left_even_pts)
    gt_lane_width = (6.0 + 4.0 + 2.0 + 0.0) / 4
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_lane_width_curved_width1() -> None:
    """
        Compute the lane width of the following curved lane segment
        Should have width 1 at each pair of boundary waypoints.

          -------boundary
         /  ----boundary
        /  /
        |  |
        |  \\
         \\ -----
          \\-----

        """
    left_even_pts = np.array([[0, 2], [-2, 2], [-3, 1], [-3, 0], [-2, -1], [0, -1]])
    right_even_pts = np.array([[0, 3], [-2, 3], [-4, 1], [-4, 0], [-2, -2], [0, -2]])
    lane_width = compute_lane_width(left_even_pts, right_even_pts)
    gt_lane_width = 1.0
    assert np.isclose(lane_width, gt_lane_width)

    lane_width = compute_lane_width(right_even_pts, left_even_pts)
    gt_lane_width = 1.0
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_lane_width_curved_not_width1() -> None:
    """
        Compute the lane width of the following curved lane segment

          -------
         /  ----
        /  /
        |  |
        |  \\
         \\ -----
          \\-----

        We get waypoint distances of [1,1,1,1,0.707..., 1,1]
        """
    left_even_pts = np.array([[0, 2], [-2, 2], [-3, 1], [-3, 0], [-2.5, -0.5], [-2, -1], [0, -1]])

    right_even_pts = np.array([[0, 3], [-2, 3], [-4, 1], [-4, 0], [-3, -1], [-2, -2], [0, -2]])

    lane_width = compute_lane_width(left_even_pts, right_even_pts)
    gt_lane_width = 0.9581581115980783
    assert np.isclose(lane_width, gt_lane_width)

    lane_width = compute_lane_width(right_even_pts, left_even_pts)
    gt_lane_width = 0.9581581115980783
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_mid_pivot_arc_3pt_cul_de_sac() -> None:
    """
    Make sure we handle the cul-de-sac case correctly.

    When mapping a cul-de-sac, we get a line of points on one boundary,
    and a single point on the other side. This function produces the middle
    arc we get by pivoting around the single point.

    Waypoints are depicted below for the cul-de-sac center, and other boundary.

            o
             \
              \
               \
            O   o
               /
              /
             /
            o
    """
    # Numpy array of shape (3,)
    single_pt = np.array([0, 0])

    # Numpy array of shape (N,3)
    arc_pts = np.array([[0, 1], [1, 0], [0, -1]])

    # centerline_pts: Numpy array of shape (N,3)
    centerline_pts, lane_width = compute_mid_pivot_arc(single_pt, arc_pts)

    gt_centerline_pts = np.array([[0, 0.5], [0.5, 0], [0, -0.5]])
    gt_lane_width = 1.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_mid_pivot_arc_5pt_cul_de_sac() -> None:
    """
    Make sure we handle the cul-de-sac case correctly.

    When mapping a cul-de-sac, we get a line of points on one boundary,
    and a single point on the other side. This function produces the middle
    arc we get by pivoting around the single point.

    Waypoints are depicted below for the cul-de-sac center, and other boundary.

            o
             \
              o
               \
            O   o
               /
              o
             /
            o
    """
    # Numpy array of shape (3,)
    single_pt = np.array([0, 0])

    # Numpy array of shape (N,3)
    arc_pts = np.array([[0, 2], [1, 1], [2, 0], [1, -1], [0, -2]])

    # centerline_pts: Numpy array of shape (N,3)
    centerline_pts, lane_width = compute_mid_pivot_arc(single_pt, arc_pts)

    gt_centerline_pts = np.array([[0, 1], [0.5, 0.5], [1, 0], [0.5, -0.5], [0, -1]])
    gt_lane_width = (2 + 2 + 2 + np.sqrt(2) + np.sqrt(2)) / 5
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_cul_de_sac_right_onept() -> None:
    """
    Make sure that if we provide left and right boundary polylines,
    we can get the correct centerline by averaging left and right waypoints.
    """
    left_ln_bnds = np.array([[0, 2], [1, 1], [2, 0], [1, -1], [0, -2]])
    right_ln_bnds = np.array([[0, 0]])

    centerline_pts, lane_width = compute_midpoint_line(left_ln_bnds, right_ln_bnds, num_interp_pts=5)

    gt_centerline_pts = np.array([[0, 1], [0.5, 0.5], [1, 0], [0.5, -0.5], [0, -1]])
    gt_lane_width = (2 + 2 + 2 + np.sqrt(2) + np.sqrt(2)) / 5

    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_cul_de_sac_left_onept() -> None:
    """
    Make sure that if we provide left and right boundary polylines,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds = np.array([[0, 2], [1, 1], [2, 0], [1, -1], [0, -2]])
    left_ln_bnds = np.array([[0, 0]])

    centerline_pts, lane_width = compute_midpoint_line(left_ln_bnds, right_ln_bnds, num_interp_pts=5)

    gt_centerline_pts = np.array([[0, 1], [0.5, 0.5], [1, 0], [0.5, -0.5], [0, -1]])
    gt_lane_width = (2 + 2 + 2 + np.sqrt(2) + np.sqrt(2)) / 5

    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_straightline_maintain_5_waypts() -> None:
    """
    Make sure that if we provide left and right boundary polylines,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds = np.array([[-1, 4], [-1, 2], [-1, 0], [-1, -2], [-1, -4]])
    left_ln_bnds = np.array([[2, 4], [2, 2], [2, 0], [2, -2], [2, -4]])

    centerline_pts, lane_width = compute_midpoint_line(left_ln_bnds, right_ln_bnds, num_interp_pts=5)

    gt_centerline_pts = np.array([[0.5, 4], [0.5, 2], [0.5, 0], [0.5, -2], [0.5, -4]])
    gt_lane_width = 3.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_straightline_maintain_4_waypts() -> None:
    """
    Make sure that if we provide left and right boundary polylines,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds = np.array([[-1, 4], [-1, 2], [-1, 0], [-1, -2], [-1, -4]])
    left_ln_bnds = np.array([[2, 4], [2, 2], [2, 0], [2, -2], [2, -4]])

    centerline_pts, lane_width = compute_midpoint_line(left_ln_bnds, right_ln_bnds, num_interp_pts=4)

    gt_centerline_pts = np.array([[0.5, 4], [0.5, 4 / 3], [0.5, -4 / 3], [0.5, -4]])
    gt_lane_width = 3.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_straightline_maintain_3_waypts() -> None:
    """
    Make sure that if we provide left and right boundary polylines,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds = np.array([[-1, 4], [-1, 2], [-1, 0], [-1, -2], [-1, -4]])
    left_ln_bnds = np.array([[2, 4], [2, 2], [2, 0], [2, -2], [2, -4]])

    centerline_pts, lane_width = compute_midpoint_line(left_ln_bnds, right_ln_bnds, num_interp_pts=3)

    gt_centerline_pts = np.array([[0.5, 4], [0.5, 0], [0.5, -4]])
    gt_lane_width = 3.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_straightline_maintain_2_waypts() -> None:
    """
    Make sure that if we provide left and right boundary polylines,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds = np.array([[-1, 4], [-1, 2], [-1, 0], [-1, -2], [-1, -4]])
    left_ln_bnds = np.array([[2, 4], [2, 2], [2, 0], [2, -2], [2, -4]])

    centerline_pts, lane_width = compute_midpoint_line(left_ln_bnds, right_ln_bnds, num_interp_pts=2)

    gt_centerline_pts = np.array([[0.5, 4], [0.5, -4]])
    gt_lane_width = 3.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_curved_maintain_4_waypts() -> None:
    """
    Make sure that if we provide left and right boundary polylines,
    we can get the correct centerline by averaging left and right waypoints.

    Note that because of the curve and the arc interpolation, the land width and centerline in the middle points
    will be shifted.
    """
    right_ln_bnds = np.array([[-1, 3], [1, 3], [4, 0], [4, -2]])
    left_ln_bnds = np.array([[-1, 1], [1, 1], [2, 0], [2, -2]])

    centerline_pts, lane_width = compute_midpoint_line(left_ln_bnds, right_ln_bnds, num_interp_pts=4)

    from argoverse.utils.mpl_plotting_utils import draw_polygon_mpl

    fig = plt.figure(figsize=(22.5, 8))
    ax = fig.add_subplot(111)

    draw_polygon_mpl(ax, right_ln_bnds, "g")
    draw_polygon_mpl(ax, left_ln_bnds, "b")
    draw_polygon_mpl(ax, centerline_pts, "r")

    gt_centerline_pts = np.array([[-1, 2], [1, 2], [3, 0], [3, -2]])
    gt_lane_width = 2.0

    assert np.allclose(centerline_pts[0], gt_centerline_pts[0])
    assert np.allclose(centerline_pts[-1], gt_centerline_pts[-1])


def test_interp_arc_straight_line() -> None:
    """ """
    pts = np.array([[-10, 0], [10, 0]])
    interp_pts = interp_arc(t=3, px=pts[:, 0], py=pts[:, 1])
    gt_interp_pts = np.array([[-10, 0], [0, 0], [10, 0]])
    assert np.allclose(interp_pts, gt_interp_pts)

    pts = np.array([[-10, 0], [10, 0]])
    interp_pts = interp_arc(t=4, px=pts[:, 0], py=pts[:, 1])
    gt_interp_pts = np.array([[-10, 0], [-10 / 3, 0], [10 / 3, 0], [10, 0]])
    assert np.allclose(interp_pts, gt_interp_pts)
