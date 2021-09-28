# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import matplotlib.pyplot as plt
import numpy as np

from argoverse.utils.mpl_plotting_utils import (
    animate_polyline,
    draw_polygon_mpl,
    plot_lane_segment_patch,
    plot_nearby_centerlines,
)


def test_draw_polygon_mpl_smokescreen_nolinewidth() -> None:
    """Test drawing polygon with `matplotlib` with no line width."""
    ax = plt.axes([1, 1, 1, 1])
    # polygon: Numpy array of shape (N,2) or (N,3)
    polygon = np.array([[0, 0], [1, 1], [1, 0], [0, 0]])
    color = "r"
    draw_polygon_mpl(ax, polygon, color)
    plt.close("all")


def test_draw_polygon_mpl_smokescreen_with_linewidth() -> None:
    """Test drawing polygon with `matplotlib` with line width."""
    ax = plt.axes([1, 1, 1, 1])
    # polygon: Numpy array of shape (N,2) or (N,3)
    polygon = np.array([[0, 0], [1, 1], [1, 0], [0, 0]])
    color = "r"
    linewidth = 100
    draw_polygon_mpl(ax, polygon, color, linewidth=linewidth)
    plt.close("all")


def test_plot_lane_segment_patch_smokescreen() -> None:
    """Test drawing lane segment with `matplotlib`."""
    ax = plt.axes([1, 1, 1, 1])
    polygon_pts = np.array([[-1, 0], [1, 0], [0, 1]])
    color = "r"
    alpha = 0.9
    plot_lane_segment_patch(polygon_pts, ax, color, alpha)
    plt.close("all")


def test_plot_nearby_centerlines_smokescreen() -> None:
    """Test drawing nearby centerlines with `matplotlib`."""
    ax = plt.axes([1, 1, 1, 1])
    # lane_centerlines: Python dictionary where key is lane ID, value is
    # object describing the lane
    lane_centerlines = {}
    lane_id_1 = 20
    obj_1 = {"centerline": np.array([[0, 0], [1, 1], [2, 2]])}
    lane_centerlines[lane_id_1] = obj_1

    lane_id_1 = 2000
    obj_2 = {"centerline": np.array([[0, -1], [0, -2], [0, -3]])}
    lane_centerlines[lane_id_1] = obj_2

    nearby_lane_ids = [20, 2000]
    color = "g"
    plot_nearby_centerlines(lane_centerlines, ax, nearby_lane_ids, color)
    plt.close("all")


def test_animate_polyline_smokescreen() -> None:
    """Test animating a polyline with `matplotlib`."""
    polyline = np.array([[0, 0], [1, 1], [2, 0], [0, 2]])
    axes_margin = 2
    animate_polyline(polyline, axes_margin, show_plot=False)
    plt.close("all")
