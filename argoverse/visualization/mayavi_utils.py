# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""A collection of utilities for visualizing 3D data with Mayavi mlab.

Mayavi Examples
----
.. code-block:: python

    # To save a figure:
    mayavi_wrapper.mlab.savefig( png_savepath, size=(1000,1000))

    # To close a figure:
    mayavi_wrapper.mlab.close(fig)

    # To adjust figure's azimuth:
    mayavi_wrapper.mlab.view(azimuth=180)
"""

from typing import Any, Iterable, List, Optional, Tuple, Union, cast

import numpy as np

from argoverse.utils import mayavi_wrapper
from argoverse.utils.frustum_clipping import clip_segment_v3_plane_n

__all__ = [
    "Figure",
    "Point",
    "PointCloud",
    "Number",
    "Color",
    "plot_bbox_3d_mayavi",
    "plot_points_3D_mayavi",
    "plot_3d_clipped_bbox_mayavi",
    "draw_coordinate_frame_at_origin",
    "draw_lidar",
    "mayavi_compare_point_clouds",
    "draw_mayavi_line_segment",
]

#: A stub representing mayavi_wrapper.mlab figure types
Figure = Any

#: A 3D Point
Point = np.ndarray

#: An array of 3D points
PointCloud = np.ndarray

#: Any numeric type
Number = Union[int, float]

#: RGB color created from 0.0 to 1.0 values
Color = Tuple[float, float, float]


def plot_bbox_3d_mayavi(
    fig: Figure,
    corners: np.ndarray,
    colors: Tuple[Color, Color, Color] = ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    line_width: Number = 2,
    draw_text: Optional[str] = None,
    text_scale: Tuple[Number, Number, Number] = (1, 1, 1),
) -> Figure:
    """Plot a 3D bounding box

    We plot the front of the cuboid in blue, and the back in red.
    We plot the sides of the cuboid in green. We draw a line
    segment through the front half of the cuboid, on its bottom face.

    Args:
       fig: Mayavi figure
       corners: Numpy array of shape (N,3)
       colors: RGB values used to color the cuboid
       line_width: Width of the cuboid's lines
       draw_text: Optional text to plot with the cuboid
       text_scale: Scaling factor for the text

    Returns:
        Updated Mayavi figure
    """

    def draw_rect(fig: Figure, selected_corners: np.ndarray, color: Color) -> None:
        prev = selected_corners[-1]
        for corner in selected_corners:
            fig = draw_mayavi_line_segment(fig, [prev, corner], color, line_width)
            prev = corner

    if draw_text:
        mayavi_wrapper.mlab.text3d(
            corners[0, 0],
            corners[0, 1],
            corners[0, 2],
            draw_text,
            scale=text_scale,
            color=colors[0],
            figure=fig,
        )

    # Draw the sides in green
    for i in range(4):
        corner_f = corners[i]  # front corner
        corner_b = corners[i + 4]  # back corner
        fig = draw_mayavi_line_segment(fig, [corner_f, corner_b], colors[2], line_width)

    # Draw front (first 4 corners) in blue
    draw_rect(fig, corners[:4], colors[0])
    # Draw rear (last 4 corners) in red
    draw_rect(fig, corners[4:], colors[1])

    # Draw blue line indicating the front half
    center_bottom_forward = np.mean(corners[2:4], axis=0)
    center_bottom = np.mean(corners[[2, 3, 7, 6]], axis=0)
    fig = draw_mayavi_line_segment(fig, [center_bottom, center_bottom_forward], colors[0], line_width)
    return fig


def plot_points_3D_mayavi(
    points: np.ndarray,
    fig: Figure,
    per_pt_color_strengths: np.ndarray = None,
    fixed_color: Optional[Color] = (1, 0, 0),
    colormap: str = "spectral",
) -> Figure:
    """Visualize points with Mayavi. Scale factor has no influence on point size rendering
    when calling `points3d()` with the mode="point" argument, so we ignore it altogether.
    The parameter "line_width" also has no effect on points, so we ignore it also.

    Args:
       points: The points to visualize
       fig: A Mayavi figure
       per_pt_color_strengths: An array of scalar values the same size as `points`
       fixed_color: Use a fixed color instead of a colormap
       colormap: different green to red jet for 'spectral' or 'gnuplot'

    Returns:
       Updated Mayavi figure
    """
    if len(points) == 0:
        return None

    if per_pt_color_strengths is None or len(per_pt_color_strengths) != len(points):
        # Height data used for shading
        per_pt_color_strengths = points[:, 2]

    mayavi_wrapper.mlab.points3d(
        points[:, 0],  # x
        points[:, 1],  # y
        points[:, 2],  # z
        per_pt_color_strengths,
        mode="point",  # Render each point as a 'point', not as a 'sphere' or 'cube'
        colormap=colormap,
        color=fixed_color,  # Used a fixed (r,g,b) color instead of colormap
        figure=fig,
    )

    return fig


def plot_3d_clipped_bbox_mayavi(
    fig: Figure,
    planes: np.ndarray,
    uv_cam: np.ndarray,
    colors: Tuple[Color, Color, Color] = ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
) -> Figure:
    """
    Args:
       fig: Mayavi figure
       planes:
       uv_cam: 3d points in camera coordinate frame
       colors:

    Returns:
       Updated Mayavi figure
    """

    def draw_rect(fig: Figure, selected_corners: np.ndarray, color: Color) -> None:
        prev = selected_corners[-1]
        for corner in selected_corners:
            clip_prev, clip_corner = clip_segment_v3_plane_n(prev, corner, planes)
            if clip_prev is None or clip_corner is None:
                continue

            fig = draw_mayavi_line_segment(fig, [clip_prev, clip_corner], color, line_width=2)
            prev = corner

    # Draw the sides
    for i in range(4):
        corner_f = uv_cam[i]  # front corner
        corner_b = uv_cam[i + 4]  # back corner

        clip_c_f, clip_c_b = clip_segment_v3_plane_n(corner_f, corner_b, planes)
        if clip_c_f is None or clip_c_b is None:
            continue

        # color green
        fig = draw_mayavi_line_segment(fig, [clip_c_f, clip_c_b], (0, 1, 0), line_width=2)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(fig, uv_cam[:4], colors[1])  # red
    draw_rect(fig, uv_cam[4:], colors[0])  # blue

    return fig


def draw_coordinate_frame_at_origin(fig: Figure) -> Figure:
    """
    Draw the origin and 3 vectors representing standard basis vectors to express
    a coordinate reference frame.

    Args:
       fig: Mayavi figure

    Returns:
       Updated Mayavi figure

    Based on
    --------
    https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
    https://github.com/charlesq34/frustum-pointnets/blob/master/mayavi/viz_util.py

    """
    # draw origin
    mayavi_wrapper.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)
    # Form standard basis vectors e_1, e_2, e_3
    axes = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    # e_1 in red
    mayavi_wrapper.mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    # e_2 in green
    mayavi_wrapper.mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    # e_3 in blue
    mayavi_wrapper.mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )
    return fig


def draw_lidar(
    point_cloud: np.ndarray,
    colormap: str = "spectral",
    fig: Optional[Figure] = None,
    bgcolor: Color = (0, 0, 0),
) -> Figure:
    """Render a :ref:`PointCloud` with a 45 degree viewing frustum from ego-vehicle.

    Creates a Mayavi figure, draws a point cloud. Since the majority of interesting objects and
    scenarios are found closeby to the ground, we want to see the objects near the ground expressed
    in the full range of the colormap. Since returns on power lines, trees, and buildings
    will dominate and dilute the colormap otherwise, we clip the colors so that all points
    beyond a certain z-elevation (height) share the same color at the edge of the colormap.
    We choose anything beyond the 90th percentile as a height outlier.

    Args:
       point_cloud: The pointcloud to render
       fig: A pre-existing Mayavi figure to render to
       bgcolor: The background color
       colormap: "spectral" or "gnuplot" or "jet" are best

    Returns:
       Updated or created Mayavi figure
    """
    if fig is None:
        fig = mayavi_wrapper.mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))

    z_thresh = np.percentile(point_cloud[:, 2], 90)
    thresholded_heights = point_cloud[:, 2].copy()
    # Colors of highest points will be clipped to all lie at edge of colormap
    thresholded_heights[thresholded_heights > z_thresh] = 5

    # draw points
    fig = plot_points_3D_mayavi(
        points=point_cloud,
        fig=fig,
        per_pt_color_strengths=thresholded_heights,
        fixed_color=None,
        colormap=colormap,
    )
    fig = draw_coordinate_frame_at_origin(fig)
    mayavi_wrapper.mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )
    return fig


def mayavi_compare_point_clouds(point_cloud_list: Iterable[np.ndarray]) -> None:
    """
    Useful for visualizing the segmentation of a scene has
    separate objects, each colored differently.

    Args:
       point_cloud_list: A list of :ref:`PointCloud`s to render
    """
    fig = mayavi_wrapper.mlab.figure(bgcolor=(0, 0, 0), size=(2000, 1000))
    colors: List[Color] = [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    for i, point_cloud in enumerate(point_cloud_list):
        if i < 3:
            # use very distinctive colors
            color = colors[i]
        else:
            color = cast(Color, tuple([np.random.rand() for i in range(3)]))

        plot_points_3D_mayavi(fig, point_cloud, color)

    mayavi_wrapper.mlab.view(azimuth=180)
    mayavi_wrapper.mlab.show()


def draw_mayavi_line_segment(
    fig: Figure,
    points: Iterable[np.ndarray],
    color: Color = (1, 0, 0),
    line_width: Optional[Number] = 1,
    tube_radius: Optional[Number] = None,
) -> Figure:
    """Draw a single line segment with Mayavi.

    Args:
       fig: The Mayavi figure to render to
       points: The points representing the line segment to draw
       color: The color of the line segment
       line_width: The width of the line
       tube_radius: The radius if rendering the line segment as a tube

    Returns:
       Mayavi figure
    """
    mayavi_wrapper.mlab.plot3d(
        [point[0] for point in points],
        [point[1] for point in points],
        [point[2] for point in points],
        color=color,
        tube_radius=tube_radius,
        line_width=line_width,
        figure=fig,
    )
    return fig
