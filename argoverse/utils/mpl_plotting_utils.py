# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Collection of utility functions for Matplotlib."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from descartes.patch import PolygonPatch
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Polygon


def draw_polygon_mpl(
    ax: plt.Axes,
    polygon: np.ndarray,
    color: Union[Tuple[float, float, float], str],
    linewidth: Optional[float] = None,
) -> None:
    """Draw a polygon.

    The polygon's first and last point must be the same (repeated).

    Args:
        ax: Matplotlib axes instance to draw on
        polygon: Array of shape (N, 2) or (N, 3)
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
    """
    if linewidth is None:
        ax.plot(polygon[:, 0], polygon[:, 1], color=color)
    else:
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth)


def draw_polygonpatch_matplotlib(points: Any, color: Union[Tuple[float, float, float], str]) -> None:
    """Draw a PolygonPatch.

    Args:
        points: Unused argument
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
    """
    fig = plt.figure(1, figsize=(10, 10), dpi=90)
    ax = fig.add_subplot(111)

    ext = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0), (0, 0)]
    int = [(0.2, 0.3), (0.3, 0.3), (0.3, 0.4), (0.2, 0.4)]
    polygon = Polygon(ext, [int])
    patch = PolygonPatch(polygon, facecolor=color, alpha=0.5, zorder=2)
    ax.add_patch(patch)


def draw_lane_polygons(
    ax: plt.Axes,
    lane_polygons: np.ndarray,
    color: Union[Tuple[float, float, float], str] = "y",
) -> None:
    """Draw a lane using polygons.

    Args:
        ax: Matplotlib axes
        lane_polygons: Array of (N,) objects, where each object is a (M,3) array
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
    """
    for i, polygon in enumerate(lane_polygons):
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3, zorder=1)


def plot_bbox_2D(
    ax: plt.Axes,
    pts: np.ndarray,
    color: Union[Tuple[float, float, float], str],
    linestyle: str = "-",
) -> None:
    """Draw a bounding box.

    2D bbox vertices should be arranged as::

        0----1
        |    |
        2----3

    i.e. the connectivity is 0->1, 1->3, 3->2, 2->0

    Args:
        ax: Matplotlib axes
        pts: Array of shape (4, 2) representing the 4 points of the bounding box.
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
        linestyle: The linestyle to use
    """
    ax.plot(pts[0:2, 0], pts[0:2, 1], c=color, linestyle=linestyle)
    ax.plot(pts[2:4, 0], pts[2:4, 1], c=color, linestyle=linestyle)
    ax.plot(pts[np.array([1, 3]), 0], pts[np.array([1, 3]), 1], c=color, linestyle=linestyle)
    ax.plot(pts[np.array([0, 2]), 0], pts[np.array([0, 2]), 1], c=color, linestyle=linestyle)


def animate_polyline(polyline: np.ndarray, axes_margin: int = 5, show_plot: bool = True) -> None:
    """Draw and animate a polyline on a plot.

    Args:
        polyline: Array of shape (N, 2) representing the points of the line
        axes_margin: How much margin for the axes
        show_plot: Whether to show the plot after rendering it
    """
    xmin = np.amin(polyline[:, 0]) - axes_margin
    xmax = np.amax(polyline[:, 0]) + axes_margin
    ymin = np.amin(polyline[:, 1]) - axes_margin
    ymax = np.amax(polyline[:, 1]) + axes_margin

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    (ln,) = plt.plot([], [], "ro", animated=True)

    def init() -> Tuple[Line2D]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return (ln,)

    def update(frame: List[Any]) -> Tuple[Line2D]:
        xdata.append(frame[0])
        ydata.append(frame[1])
        ln.set_data(xdata, ydata)
        return (ln,)

    ani = FuncAnimation(fig, update, frames=polyline, init_func=init, blit=True)

    if show_plot:
        plt.show()


def plot_lane_segment_patch(
    polygon_pts: np.ndarray,
    ax: plt.Axes,
    color: Union[Tuple[float, float, float], str] = "y",
    alpha: float = 0.3,
) -> None:
    """Plot a lane segment using a PolygonPatch.

    Args:
        polygon_pts: Array of shape (N, 2) representing the points of the polygon
        ax: Matplotlib axes
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
        alpha: the opacity of the lane segment
    """
    polygon = Polygon(polygon_pts)
    patch = PolygonPatch(polygon, facecolor=color, edgecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(patch)


def plot_nearby_centerlines(
    lane_centerlines: Dict[Any, Any],
    ax: plt.Axes,
    nearby_lane_ids: List[int],
    color: Union[Tuple[int, int, int], str],
) -> None:
    """Plot centerlines.

    Args:
        lane_centerlines: Python dictionary where key is lane ID, value is object describing the lane
        ax: Matplotlib axes
        nearby_lane_ids: List of integers representing lane IDs
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
    """
    for curr_lane_id in nearby_lane_ids:
        centerline = lane_centerlines[curr_lane_id]["centerline"]
        ax.plot(centerline[:, 0], centerline[:, 1], color=color, linestyle="--", alpha=0.4)


def visualize_centerline(centerline: LineString) -> None:
    """Visualize the computed centerline.

    Args:
        centerline: Sequence of coordinates forming the centerline
    """
    line_coords = list(zip(*centerline))
    lineX = line_coords[0]
    lineY = line_coords[1]
    plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
    plt.text(lineX[0], lineY[0], "s")
    plt.text(lineX[-1], lineY[-1], "e")
    plt.axis("equal")
