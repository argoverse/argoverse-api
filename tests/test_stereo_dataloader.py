# <Copyright 2021, Argo AI, LLC. Released under the MIT license.>

import pathlib

import pytest

from argoverse.data_loading.stereo_dataloader import ArgoverseStereoDataLoader
from argoverse.utils.camera_stats import RECTIFIED_STEREO_CAMERA_LIST

TEST_DATA_LOC = str(pathlib.Path(__file__).parent.parent / "tests" / "test_data" / "stereo")
_LOG_ID = "1"

STEREO_FRONT_LEFT_RECT = RECTIFIED_STEREO_CAMERA_LIST[0]
STEREO_FRONT_RIGHT_RECT = RECTIFIED_STEREO_CAMERA_LIST[1]


@pytest.fixture  # type: ignore
def data_loader() -> ArgoverseStereoDataLoader:
    """Load the test data using the Argoverse stereo loader class."""
    return ArgoverseStereoDataLoader(TEST_DATA_LOC, split_name="test")


def test_get_log_calibration_data(data_loader: ArgoverseStereoDataLoader) -> None:
    """Test loading the calibration data."""
    # Just check that it doesn't raise.
    assert data_loader.get_log_calibration_data(log_id=_LOG_ID)


def test_get_ordered_log_stereo_image_fpaths(data_loader: ArgoverseStereoDataLoader) -> None:
    """Test getting the list of paths to chronologically ordered rectified stereo images in a log."""
    left_stereo_img_fpaths = data_loader.get_ordered_log_stereo_image_fpaths(
        log_id=_LOG_ID,
        camera_name=STEREO_FRONT_LEFT_RECT,
    )

    right_stereo_img_fpaths = data_loader.get_ordered_log_stereo_image_fpaths(
        log_id=_LOG_ID,
        camera_name=STEREO_FRONT_RIGHT_RECT,
    )

    # Test if the length of the lists (left and right) are 1 (length of the test lists).
    assert len(left_stereo_img_fpaths) == 1
    assert len(right_stereo_img_fpaths) == 1


def test_ordered_log_disparity_map_fpaths(data_loader: ArgoverseStereoDataLoader) -> None:
    """Test getting the list of paths to chronologically ordered disparity maps in a log."""
    disparity_map_fpaths = data_loader.get_ordered_log_disparity_map_fpaths(
        log_id=_LOG_ID,
        disparity_name="stereo_front_left_rect_disparity",
    )

    disparity_obj_map_fpaths = data_loader.get_ordered_log_disparity_map_fpaths(
        log_id=_LOG_ID,
        disparity_name="stereo_front_left_rect_objects_disparity",
    )

    # Test if the length of the lists are 1 (length of the test lists).
    assert len(disparity_map_fpaths) == 1
    assert len(disparity_obj_map_fpaths) == 1


def test_get_rectified_stereo_image(data_loader: ArgoverseStereoDataLoader) -> None:
    """Test loading a rectified stereo image."""
    left_stereo_img_fpaths = data_loader.get_ordered_log_stereo_image_fpaths(
        log_id=_LOG_ID,
        camera_name=STEREO_FRONT_LEFT_RECT,
    )

    rectified_stereo_image = data_loader.get_rectified_stereo_image(left_stereo_img_fpaths[0])

    # Check if the loaded image shape is equal to the test rectified stereo image shape (2056 x 2464 x 3)
    assert rectified_stereo_image.shape == (2056, 2464, 3)


def test_get_disparity_map(data_loader: ArgoverseStereoDataLoader) -> None:
    """Test loading a disparity map."""
    disparity_img_fpaths = data_loader.get_ordered_log_disparity_map_fpaths(
        log_id=_LOG_ID,
        disparity_name="stereo_front_left_rect_disparity",
    )

    disparity_map = data_loader.get_disparity_map(disparity_img_fpaths[0])

    # Check if the loaded image shape is equal to the test disparity map shape (2056 x 2464)
    assert disparity_map.shape == (2056, 2464)
