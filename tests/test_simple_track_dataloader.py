# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import os
import pathlib

import pytest

from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader

_TEST_DATA = pathlib.Path(__file__).parent / "test_data" / "tracking"
_LOG_ID = "1"


@pytest.fixture  # type: ignore
def data_loader() -> SimpleArgoverseTrackingDataLoader:
    return SimpleArgoverseTrackingDataLoader(os.fspath(_TEST_DATA), os.fspath(_TEST_DATA))


def test_get_city_name(data_loader: SimpleArgoverseTrackingDataLoader) -> None:
    assert data_loader.get_city_name(_LOG_ID) == "PIT"


def test_get_log_calibration_data(
    data_loader: SimpleArgoverseTrackingDataLoader,
) -> None:
    # Just check that it doesn't raise.
    assert data_loader.get_log_calibration_data(_LOG_ID)


def test_get_city_SE3_egovehicle(
    data_loader: SimpleArgoverseTrackingDataLoader,
) -> None:
    assert data_loader.get_city_SE3_egovehicle(_LOG_ID, 0) is not None
    assert data_loader.get_city_SE3_egovehicle(_LOG_ID, 100) is None


def test_get_closest_im_fpath(data_loader: SimpleArgoverseTrackingDataLoader) -> None:
    # Test data doesn't have cameras so we cannot currently test this if we
    assert data_loader.get_closest_im_fpath(_LOG_ID, "nonexisting_camera_name", 0) is None


def test_get_ordered_log_ply_fpaths(
    data_loader: SimpleArgoverseTrackingDataLoader,
) -> None:
    # Test data doesn't have cameras so we cannot currently test this if we
    assert len(data_loader.get_ordered_log_ply_fpaths(_LOG_ID)) == 3


def test_get_labels_at_lidar_timestamp(
    data_loader: SimpleArgoverseTrackingDataLoader,
) -> None:
    assert data_loader.get_labels_at_lidar_timestamp(_LOG_ID, 0) is not None
    assert data_loader.get_labels_at_lidar_timestamp(_LOG_ID, 100) is None


def test_get_closest_im_fpath_exists(
    data_loader: SimpleArgoverseTrackingDataLoader,
) -> None:
    # Test data does have ring front cameras at timestamps 0,1,2,3. Compare with ground truth (gt)
    im_fpath = data_loader.get_closest_im_fpath(_LOG_ID, "ring_front_right", 2)
    assert im_fpath is not None

    gt_im_fpath = f"test_data/tracking/{_LOG_ID}/ring_front_right/ring_front_right_2.jpg"
    assert "/".join(im_fpath.split("/")[-5:]) == gt_im_fpath


def test_get_closest_lidar_fpath_found_match(
    data_loader: SimpleArgoverseTrackingDataLoader,
) -> None:
    """ Just barely within 51 ms allowed buffer"""
    cam_timestamp = int(50 * 1e6)
    ply_fpath = data_loader.get_closest_lidar_fpath(_LOG_ID, cam_timestamp)

    assert ply_fpath is not None
    gt_ply_fpath = f"test_data/tracking/{_LOG_ID}/lidar/PC_2.ply"
    assert "/".join(ply_fpath.split("/")[-5:]) == gt_ply_fpath


def test_get_closest_lidar_fpath_no_match(
    data_loader: SimpleArgoverseTrackingDataLoader,
) -> None:
    """LiDAR rotates at 10 Hz (sensor message per 100 ms). Test if camera measurement
    just barely outside 51 ms allowed buffer. Max LiDAR timestamp in log is 2.
    51 ms, not 50 ms, is allowed to give time for occasional delay.
    """
    max_allowed_interval = ((100 / 2) + 1) * 1e6
    log_max_lidar_timestamp = 2
    cam_timestamp = int(log_max_lidar_timestamp + max_allowed_interval + 1)
    ply_fpath = data_loader.get_closest_lidar_fpath(_LOG_ID, cam_timestamp)
    assert ply_fpath is None


def test_get_ordered_log_cam_fpaths(
    data_loader: SimpleArgoverseTrackingDataLoader,
) -> None:
    """ Make sure all images for one camera in one log are returned in correct order. """
    camera_name = "ring_rear_right"
    cam_img_fpaths = data_loader.get_ordered_log_cam_fpaths(_LOG_ID, camera_name)
    gt_cam_img_fpaths = [
        f"test_data/tracking/{_LOG_ID}/ring_rear_right/ring_rear_right_0.jpg",
        f"test_data/tracking/{_LOG_ID}/ring_rear_right/ring_rear_right_1.jpg",
        f"test_data/tracking/{_LOG_ID}/ring_rear_right/ring_rear_right_2.jpg",
        f"test_data/tracking/{_LOG_ID}/ring_rear_right/ring_rear_right_3.jpg",
    ]
    relative_img_fpaths = ["/".join(fpath.split("/")[-5:]) for fpath in cam_img_fpaths]
    assert relative_img_fpaths == gt_cam_img_fpaths
