# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Visualization utils unit tests"""

import pathlib
from typing import Iterator

import matplotlib.pyplot as plt
import pytest

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.visualization import visualization_utils

TEST_DATA_LOC = str(pathlib.Path(__file__).parent.parent / "tests" / "test_data" / "tracking")


@pytest.fixture  # type: ignore
def axes() -> Iterator[plt.Axes]:
    fig = plt.gcf()
    yield plt.gca()
    plt.close(fig)


@pytest.fixture  # type: ignore
def axes3d() -> Iterator[plt.Axes]:
    fig = plt.gcf()
    yield fig.add_subplot(111, projection="3d")
    plt.close(fig)


@pytest.fixture  # type: ignore
def data_loader() -> ArgoverseTrackingLoader:
    return ArgoverseTrackingLoader(TEST_DATA_LOC)


def test_draw_point_cloud_no_error(data_loader: ArgoverseTrackingLoader, axes: plt.Axes) -> None:
    visualization_utils.draw_point_cloud(axes, "title!", data_loader, 0, axes=[1, 0])


def test_draw_point_cloud_3d_no_error(data_loader: ArgoverseTrackingLoader, axes3d: plt.Axes) -> None:
    visualization_utils.draw_point_cloud(axes3d, "title!", data_loader, 0)


# def test_draw_point_cloud_trajectory_no_error(data_loader: ArgoverseTrackingLoader, axes: plt.Axes) -> None:
#     visualization_utils.draw_point_cloud_trajectory(axes, "title!", data_loader, 0, [1, 0])


def test_draw_point_cloud_trajectory_no_error(data_loader: ArgoverseTrackingLoader, axes3d: plt.Axes) -> None:
    visualization_utils.draw_point_cloud_trajectory(axes3d, "title!", data_loader, 0)


def test_make_grid_ring_camera_no_error(data_loader: ArgoverseTrackingLoader) -> None:
    visualization_utils.make_grid_ring_camera(data_loader, 0)
