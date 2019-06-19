# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Tracking Loader unit tests"""

import glob
import pathlib

import numpy as np
import pytest
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import CAMERA_LIST

TEST_DATA_LOC = pathlib.Path(__file__).parent.parent / "tests" / "test_data" / "tracking"


@pytest.fixture
def data_loader() -> ArgoverseTrackingLoader:
    return ArgoverseTrackingLoader(TEST_DATA_LOC)


def test_get_city_name(data_loader: ArgoverseTrackingLoader) -> None:
    assert data_loader.city_name == "PIT"


def test_calib(data_loader: ArgoverseTrackingLoader) -> None:
    assert data_loader.calib


def test_log_list(data_loader: ArgoverseTrackingLoader) -> None:
    assert data_loader.log_list[0] == "1"


def test_image_list(data_loader: ArgoverseTrackingLoader) -> None:
    assert set(data_loader.image_list.keys()) == set(CAMERA_LIST)


def test_image_list_sync(data_loader: ArgoverseTrackingLoader) -> None:
    assert set(data_loader.image_list_sync.keys()) == set(CAMERA_LIST)


def test_image_timestamp_sync(data_loader: ArgoverseTrackingLoader) -> None:
    assert set(data_loader.image_timestamp_list_sync.keys()) == set(CAMERA_LIST)
    for camera in CAMERA_LIST:
        assert 3 not in data_loader.image_timestamp_list_sync[camera]


def test_lidar_list(data_loader: ArgoverseTrackingLoader) -> None:
    assert len(data_loader.lidar_list) == 3


def test_labale_list(data_loader: ArgoverseTrackingLoader) -> None:
    assert len(data_loader.label_list) == 3


def test_image_timestamp_list(data_loader: ArgoverseTrackingLoader) -> None:
    assert set(data_loader.image_timestamp_list.keys()) == set(CAMERA_LIST)
    for camera in CAMERA_LIST:
        assert 3 in data_loader.image_timestamp_list[camera]


def test_timestamp_image_dict(data_loader: ArgoverseTrackingLoader) -> None:
    assert set(data_loader.timestamp_image_dict.keys()) == set(CAMERA_LIST)
    for camera in CAMERA_LIST:
        assert len(data_loader.timestamp_image_dict[camera]) == 4


def test_timestamp_lidar_map(data_loader: ArgoverseTrackingLoader) -> None:
    assert len(data_loader.timestamp_lidar_dict) == 3
    assert len(data_loader.lidar_timestamp_list) == 3


def test_data_loader_get(data_loader: ArgoverseTrackingLoader) -> None:
    data_0 = data_loader[0].current_log
    data_1 = data_loader.get("1").current_log
    assert data_0 == data_1


def test_loading_image_lidar(data_loader: ArgoverseTrackingLoader) -> None:
    camera = CAMERA_LIST[0]
    log = "1"
    image_1 = data_loader.get_image_at_timestamp(0, camera, log)
    image_2 = data_loader.get_image_list_sync(camera, log, load=True)[0]
    image_3 = data_loader.get_image(0, camera, log)
    image_4 = data_loader.get_image_sync(0, camera, log)
    assert np.array_equal(image_1, image_2) and np.array_equal(image_1, image_3) and np.array_equal(image_1, image_4)

    lidar_0 = data_loader.get_lidar(0, log)

    lidar_gt = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [2.0, 0.0, 5.0],
            [3.0, 0.0, 5.0],
            [4.0, 0.0, 5.0],
            [5.0, 0.0, 5.0],
            [6.0, 0.0, 5.0],
            [7.0, 0.0, 5.0],
            [8.0, 0.0, 5.0],
            [9.0, 0.0, 5.0],
        ]
    )
    assert np.array_equal(lidar_0, lidar_gt)


def test_label_object(data_loader: ArgoverseTrackingLoader) -> None:
    label_at_frame_0 = data_loader.get_label_object(0)
    for label in label_at_frame_0:
        assert label.label_class == "VEHICLE"
        assert label.width == 2
        assert label.height == 2
        assert label.length == 2


def test_calibration(data_loader: ArgoverseTrackingLoader) -> None:
    for camera in CAMERA_LIST:
        calib = data_loader.get_calibration(camera, "1")
        assert calib.camera == camera


def test_pose(data_loader: ArgoverseTrackingLoader) -> None:
    for idx in range(len(data_loader.lidar_list)):
        pose = data_loader.get_pose(idx)
        assert np.array_equal(
            pose.inverse().transform_point_cloud(np.array([[0, 0, 0]])),
            pose.inverse_transform_point_cloud(np.array([[0, 0, 0]])),
        )


def test_idx_from_tiemstamp(data_loader: ArgoverseTrackingLoader) -> None:
    for i in range(len(data_loader.lidar_list)):
        assert data_loader.get_idx_from_timestamp(i) == i
