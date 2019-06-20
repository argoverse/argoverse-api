# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import os
import pathlib

import pytest

from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader

_TEST_DATA = pathlib.Path(__file__).parent / "test_data" / "tracking"
_LOG_ID = "1"


@pytest.fixture
def data_loader() -> SimpleArgoverseTrackingDataLoader:
    return SimpleArgoverseTrackingDataLoader(os.fspath(_TEST_DATA), os.fspath(_TEST_DATA))


def test_get_city_name(data_loader: SimpleArgoverseTrackingDataLoader) -> None:
    assert data_loader.get_city_name(_LOG_ID) == "PIT"


def test_get_log_calibration_data(data_loader: SimpleArgoverseTrackingDataLoader) -> None:
    # Just check that it doesn't raise.
    assert data_loader.get_log_calibration_data(_LOG_ID)


def test_get_city_to_egovehicle_se3(data_loader: SimpleArgoverseTrackingDataLoader) -> None:
    assert data_loader.get_city_to_egovehicle_se3(_LOG_ID, 0) is not None
    assert data_loader.get_city_to_egovehicle_se3(_LOG_ID, 100) is None


def test_get_closest_im_fpath(data_loader: SimpleArgoverseTrackingDataLoader) -> None:
    # Test data doesn't have cameras so we cannot currently test this if we
    assert data_loader.get_closest_im_fpath(_LOG_ID, "camera_name", 0) is None


def test_get_ordered_log_ply_fpaths(data_loader: SimpleArgoverseTrackingDataLoader) -> None:
    # Test data doesn't have cameras so we cannot currently test this if we
    assert len(data_loader.get_ordered_log_ply_fpaths(_LOG_ID)) == 3


def test_get_labels_at_lidar_timestamp(data_loader: SimpleArgoverseTrackingDataLoader) -> None:
    assert data_loader.get_labels_at_lidar_timestamp(_LOG_ID, 0) is not None
    assert data_loader.get_labels_at_lidar_timestamp(_LOG_ID, 100) is None
