# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>
"""Stereo Loader unit tests"""

import glob
import pathlib

import numpy as np
import pytest

from argoverse.data_loading.argoverse_stereo_loader import ArgoverseStereoLoader
from argoverse.utils.camera_stats import STEREO_CAMERA_LIST

TEST_DATA_LOC = str(pathlib.Path(__file__).parent.parent / "tests" / "test_data" / "stereo")

STEREO_DISPARITY_LIST = ["stereo_front_left_rect_disparity", "stereo_front_left_rect_objects_disparity"]


@pytest.fixture  # type: ignore
def data_loader() -> ArgoverseStereoLoader:
    return ArgoverseStereoLoader(TEST_DATA_LOC, split_name="test")


def test_log_list(data_loader: ArgoverseStereoLoader) -> None:
    assert data_loader.log_list[0] == "1"


def test_image_list(data_loader: ArgoverseStereoLoader) -> None:
    assert set(data_loader.image_list.keys()) == set(STEREO_CAMERA_LIST)


def test_disparity_list(data_loader: ArgoverseStereoLoader) -> None:
    assert set(data_loader.disparity_list.keys()) == set(STEREO_DISPARITY_LIST)


def test_image_timestamp_list(data_loader: ArgoverseStereoLoader) -> None:
    assert set(data_loader.image_timestamp_list.keys()) == set(STEREO_CAMERA_LIST)
    for camera in STEREO_CAMERA_LIST[3:]:
        assert 1 in data_loader.image_timestamp_list[camera]


def test_timestamp_image_dict(data_loader: ArgoverseStereoLoader) -> None:
    assert set(data_loader.timestamp_image_dict.keys()) == set(STEREO_CAMERA_LIST)
    for camera in STEREO_CAMERA_LIST[3:]:
        assert len(data_loader.timestamp_image_dict[camera]) == 1


def test_data_loader_get(data_loader: ArgoverseStereoLoader) -> None:
    data_0 = data_loader[0].current_log
    data_1 = data_loader.get("1").current_log
    assert data_0 == data_1


def test_calibration(data_loader: ArgoverseStereoLoader) -> None:
    for camera in STEREO_CAMERA_LIST:
        calib = data_loader.get_calibration(camera, "1")
        assert calib.camera == camera


def test_loading_image_disparity(data_loader: ArgoverseStereoLoader) -> None:
    camera = STEREO_CAMERA_LIST[2]
    disparity = STEREO_DISPARITY_LIST[0]
    disparity_objects = STEREO_DISPARITY_LIST[1]
    log = "1"
    image_1_at_timestamp = data_loader.get_image_at_timestamp(1, camera, log)
    image_1 = data_loader.get_image(0, camera, log)
    disparity_1 = data_loader.get_disparity_map(0, disparity, log)
    disparity_objects_1 = data_loader.get_disparity_map(0, disparity_objects, log)

    assert np.array_equal(image_1_at_timestamp, image_1)
    assert disparity_1.shape == disparity_objects_1.shape
