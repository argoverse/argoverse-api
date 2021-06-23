# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Smokescreen unit tests to make sure our Mayavi utility functions work."""

try:
    import mayavi.mlab

    MISSING_MAYAVI = False

    from argoverse.visualization.mayavi_utils import (
        draw_coordinate_frame_at_origin,
        draw_lidar,
        plot_bbox_3d_mayavi,
        plot_points_3D_mayavi,
    )
except ImportError:
    MISSING_MAYAVI = True

import pathlib
from pathlib import Path

import numpy as np
import pytest

_TEST_DIR: Path = pathlib.Path(__file__).parent.parent / "tests"

skip_if_not_mayavi = pytest.mark.skipif(MISSING_MAYAVI, reason="mayavi not installed")


@skip_if_not_mayavi  # type: ignore
def test_mayavi_import_basic() -> None:
    """
    To test if Mayavi is installed correctly, generate lines around
    surface of a torus and render them.
    """
    n_mer, n_long = 6, 11
    pi = np.pi
    dphi = pi / 1000.0
    phi = np.arange(0.0, 2 * pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    z = np.sin(n_long * mu / n_mer) * 0.5

    l = mayavi.mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap="Spectral")


@skip_if_not_mayavi  # type: ignore
def test_plot_bbox_3d_mayavi_no_text() -> None:
    """
    To visualize the result, simply add the following line to this function:
        mayavi.mlab.show()

    Plot two cuboids, with front in blue and rear in red, and sides in green.

        5------4
        |\\    |\\
        | \\   | \\
        6--\\--7  \\
        \\  \\  \\ \\
    l    \\  1-------0    h
     e    \\ ||   \\ ||   e
      n    \\||    \\||   i
       g    \\2------3    g
        t      width.     h
         h.               t.
        """
    fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(2000, 1000))
    line_width = 9
    cuboid_1_verts = np.array(
        [
            [2, 0, 1],  # 0
            [1, 1, 1],  # 1
            [1, 1, 0],  # 2
            [2, 0, 0],  # 3
            [4, 2, 1],  # 4
            [3, 3, 1],  # 5
            [3, 3, 0],  # 6
            [4, 2, 0],
        ]
    )  # 7

    cuboid_2_verts = np.array(
        [
            [1, 1, 1.5],  # 0
            [0, 1, 1.5],  # 1
            [0, 1, 0],  # 2
            [1, 1, 0],  # 3
            [1, 3, 1.5],  # 4
            [0, 3, 1.5],  # 5
            [0, 3, 0],  # 6
            [1, 3, 0],
        ]
    )  # 7

    for cuboid_verts in [cuboid_1_verts, cuboid_2_verts]:
        fig = plot_bbox_3d_mayavi(fig, cuboid_verts, line_width=line_width)
    mayavi.mlab.close(fig)


@skip_if_not_mayavi  # type: ignore
def test_plot_bbox_3d_mayavi_drawtext() -> None:
    """
    To visualize the result, simply add the following line to this function:
    .. code-block:: python

        mayavi.mlab.show()

    Plot two cuboids, with front in blue and rear in red, and sides in green.

        5------4
        |\\    |\\
        | \\   | \\
        6--\\--7  \\
        \\  \\  \\ \\
    l    \\  1-------0    h
     e    \\ ||   \\ ||   e
      n    \\||    \\||   i
       g    \\2------3    g
        t      width.     h
         h.               t.
    """
    fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(2000, 1000))
    linewidth = 9
    cuboid_1_verts = np.array(
        [
            [2, 0, 1],  # 0
            [1, 1, 1],  # 1
            [1, 1, 0],  # 2
            [2, 0, 0],  # 3
            [4, 2, 1],  # 4
            [3, 3, 1],  # 5
            [3, 3, 0],  # 6
            [4, 2, 0],
        ]
    )  # 7

    cuboid_2_verts = np.array(
        [
            [1, 1, 1.5],  # 0
            [0, 1, 1.5],  # 1
            [0, 1, 0],  # 2
            [1, 1, 0],  # 3
            [1, 3, 1.5],  # 4
            [0, 3, 1.5],  # 5
            [0, 3, 0],  # 6
            [1, 3, 0],
        ]
    )  # 7

    for cuboid_verts in [cuboid_1_verts, cuboid_2_verts]:
        fig = plot_bbox_3d_mayavi(
            fig,
            cuboid_verts,
            line_width=linewidth,
            draw_text="box 0th vertex is here",
            text_scale=(0.1, 0.1, 0.1),
        )
    mayavi.mlab.close(fig)


@skip_if_not_mayavi  # type: ignore
def test_plot_points_3D_mayavi() -> None:
    """Visualize 3D point cloud with Mayavi.

    Note
    -----
    To see result, simply add the following line:

    .. code-block:: python

        mayavi.mlab.show()
    """
    fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(2000, 1000))
    point_arr = np.array([[0, 1, 2], [1, -1, 4], [-1, -1, 4], [0, -1, 2]])
    fig = plot_points_3D_mayavi(point_arr, fig, fixed_color=(1, 0, 0))

    point_arr = np.random.randn(100000, 3)
    fig = plot_points_3D_mayavi(point_arr, fig, fixed_color=(1, 0, 0))
    mayavi.mlab.close(fig)


@skip_if_not_mayavi  # type: ignore
def test_plot_points_3d_argoverse() -> None:
    """Render a LiDAR sweep from Argoverse, loaded from a .txt file."""
    fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(2000, 1000))
    point_arr = np.loadtxt(_TEST_DIR / "test_data/sample_argoverse_sweep.txt")
    fig = plot_points_3D_mayavi(point_arr, fig, fixed_color=(1, 0, 0))
    mayavi.mlab.close(fig)


@skip_if_not_mayavi  # type: ignore
def test_draw_lidar_argoverse() -> None:
    """Test :ref:`draw_lidar_simple`."""
    pc = np.loadtxt(_TEST_DIR / "test_data/sample_argoverse_sweep.txt")
    fig = draw_lidar(pc)
    mayavi.mlab.close(fig)


@skip_if_not_mayavi  # type: ignore
def test_draw_coordinate_frame_at_origin() -> None:
    """Test :ref:`draw_coordinate_frame_at_origin`."""
    fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(2000, 1000))
    fig = draw_coordinate_frame_at_origin(fig)
    mayavi.mlab.close(fig)
