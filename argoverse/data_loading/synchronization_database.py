#!/usr/bin/env python3

# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import glob
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, cast

import numpy as np
from typing_extensions import Final

from argoverse.sensor_dataset_config import ArgoverseConfig
from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST
from argoverse.utils.metric_time import TimeUnit, to_metric_time

logger = logging.getLogger(__name__)

Millisecond = TimeUnit.Millisecond
Nanosecond = TimeUnit.Nanosecond
Second = TimeUnit.Second

RING_CAMERA_FPS: Final = ArgoverseConfig.ring_cam_fps
STEREO_CAMERA_FPS: Final = ArgoverseConfig.stereo_cam_fps
LIDAR_FRAME_RATE_HZ: Final = 10

# constants defined in milliseconds
# below evaluates to 33.3 ms
RING_CAMERA_SHUTTER_INTERVAL_MS = to_metric_time(ts=1 / RING_CAMERA_FPS, src=Second, dst=Millisecond)

# below evaluates to 200 ms
STEREO_CAMERA_SHUTTER_INTERVAL_MS = to_metric_time(ts=1 / STEREO_CAMERA_FPS, src=Second, dst=Millisecond)

# below evaluates to 100 ms
LIDAR_SWEEP_INTERVAL_MS = to_metric_time(ts=1 / LIDAR_FRAME_RATE_HZ, src=Second, dst=Millisecond)

ALLOWED_TIMESTAMP_BUFFER_MS = 2  # allow 2 ms of buffer
LIDAR_SWEEP_INTERVAL_W_BUFFER_MS = LIDAR_SWEEP_INTERVAL_MS + ALLOWED_TIMESTAMP_BUFFER_MS


def get_timestamps_from_sensor_folder(sensor_folder_wildcard: str) -> np.ndarray:
    """Timestamp always lies at end of filename

    Args:
        sensor_folder_wildcard: string to glob to find all filepaths for a particular
                    sensor files within a single log run

    Returns:
        Numpy array of integers, representing timestamps
    """

    path_generator = glob.glob(sensor_folder_wildcard)
    path_generator.sort()

    return np.array([int(Path(jpg_fpath).stem.split("_")[-1]) for jpg_fpath in path_generator])


def find_closest_integer_in_ref_arr(query_int: int, ref_arr: np.ndarray) -> Tuple[int, int]:
    """
    Find the closest integer to any integer inside a reference array, and the corresponding
    difference.

    In our use case, the query integer represents a nanosecond-discretized timestamp, and the
    reference array represents a numpy array of nanosecond-discretized timestamps.

    Instead of sorting the whole array of timestamp differences, we just
    take the minimum value (to speed up this function).

    Args:
        query_int: query integer,
        ref_arr: Numpy array of integers

    Returns:
        integer, representing the closest integer found in a reference array to a query
        integer, representing the integer difference between the match and query integers
    """
    closest_ind = np.argmin(np.absolute(ref_arr - query_int))
    closest_int = cast(int, ref_arr[closest_ind])  # mypy does not understand numpy arrays
    int_diff = np.absolute(query_int - closest_int)
    return closest_int, int_diff


class SynchronizationDB:

    # Max difference between camera and LiDAR observation would be if the LiDAR timestamp is halfway between
    # two camera observations (i.e. RING_CAMERA_SHUTTER_INTERVAL_MS / 2 milliseconds on either side)
    # then convert milliseconds to nanoseconds
    MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF = to_metric_time(
        ts=RING_CAMERA_SHUTTER_INTERVAL_MS / 2, src=Millisecond, dst=Nanosecond
    )

    # Since Stereo is more sparse, we look at (STEREO_CAMERA_SHUTTER_INTERVAL_MS / 2) milliseconds on either side
    # then convert milliseconds to nanoseconds
    MAX_LIDAR_STEREO_CAM_TIMESTAMP_DIFF = to_metric_time(
        ts=STEREO_CAMERA_SHUTTER_INTERVAL_MS / 2, src=Millisecond, dst=Nanosecond
    )

    # LiDAR is 10 Hz (once per 100 milliseconds)
    # We give an extra 2 ms buffer for the message to arrive, totaling 102 ms.
    # At any point we sample, we shouldn't be more than 51 ms away.
    # then convert milliseconds to nanoseconds
    MAX_LIDAR_ANYCAM_TIMESTAMP_DIFF = to_metric_time(
        ts=LIDAR_SWEEP_INTERVAL_W_BUFFER_MS / 2, src=Millisecond, dst=Nanosecond
    )

    def __init__(self, dataset_dir: str, collect_single_log_id: Optional[str] = None) -> None:
        """Build the SynchronizationDB.
        Note that the timestamps for each camera channel are not identical, but they are clustered together.

        Args:
            dataset_dir: path to dataset.
            collect_single_log_id: log id to process. (All if not set)

        Returns:
            None
        """
        logger.info("Building SynchronizationDB")

        if collect_single_log_id is None:
            log_fpaths = glob.glob(f"{dataset_dir}/*")
        else:
            log_fpaths = [f"{dataset_dir}/{collect_single_log_id}"]

        self.per_log_camtimestamps_index: Dict[str, Dict[str, np.ndarray]] = {}
        self.per_log_lidartimestamps_index: Dict[str, np.ndarray] = {}

        for log_fpath in log_fpaths:
            log_id = Path(log_fpath).name

            self.per_log_camtimestamps_index[log_id] = {}
            for camera_name in STEREO_CAMERA_LIST + RING_CAMERA_LIST:

                sensor_folder_wildcard = f"{dataset_dir}/{log_id}/{camera_name}/{camera_name}_*.jpg"
                cam_timestamps = get_timestamps_from_sensor_folder(sensor_folder_wildcard)

                self.per_log_camtimestamps_index[log_id][camera_name] = cam_timestamps

            sensor_folder_wildcard = f"{dataset_dir}/{log_id}/lidar/PC_*.ply"
            lidar_timestamps = get_timestamps_from_sensor_folder(sensor_folder_wildcard)
            self.per_log_lidartimestamps_index[log_id] = lidar_timestamps

    def get_valid_logs(self) -> Iterable[str]:
        """Return the log_ids for which the SynchronizationDatabase contains pose information."""
        return self.per_log_camtimestamps_index.keys()

    def get_closest_lidar_timestamp(self, cam_timestamp: int, log_id: str) -> Optional[int]:
        """Given an image timestamp, find the synchronized corresponding LiDAR timestamp.
        This LiDAR timestamp should have the closest absolute timestamp to the image timestamp.

        Args:
            cam_timestamp: integer
            log_id: string

        Returns:
            closest_lidar_timestamp: closest timestamp
        """
        if log_id not in self.per_log_lidartimestamps_index:
            return None

        lidar_timestamps = self.per_log_lidartimestamps_index[log_id]
        # catch case if no files were loaded for a particular sensor
        if not lidar_timestamps.tolist():
            return None

        closest_lidar_timestamp, timestamp_diff = find_closest_integer_in_ref_arr(cam_timestamp, lidar_timestamps)
        if timestamp_diff > self.MAX_LIDAR_ANYCAM_TIMESTAMP_DIFF:
            # convert to nanoseconds->milliseconds for readability
            logger.warning(
                "No corresponding LiDAR sweep: %s > %s ms",
                to_metric_time(ts=timestamp_diff, src=Nanosecond, dst=Millisecond),
                to_metric_time(ts=self.MAX_LIDAR_ANYCAM_TIMESTAMP_DIFF, src=Nanosecond, dst=Millisecond),
            )
            return None
        return closest_lidar_timestamp

    def get_closest_cam_channel_timestamp(self, lidar_timestamp: int, camera_name: str, log_id: str) -> Optional[int]:
        """Given a LiDAR timestamp, find the synchronized corresponding image timestamp for a particular camera.
        This image timestamp should have the closest absolute timestamp.

        Args:
            lidar_timestamp: integer
            camera_name: string, representing path to log directories
            log_id: string

        Returns:
            closest_cam_ch_timestamp: closest timestamp
        """
        if (
            log_id not in self.per_log_camtimestamps_index
            or camera_name not in self.per_log_camtimestamps_index[log_id]
        ):
            return None

        cam_timestamps = self.per_log_camtimestamps_index[log_id][camera_name]
        # catch case if no files were loaded for a particular sensor
        if not cam_timestamps.tolist():
            return None

        closest_cam_ch_timestamp, timestamp_diff = find_closest_integer_in_ref_arr(lidar_timestamp, cam_timestamps)
        if timestamp_diff > self.MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF and camera_name in RING_CAMERA_LIST:
            # convert to nanoseconds->milliseconds for readability
            logger.warning(
                "No corresponding ring image at %s: %s > %s ms",
                lidar_timestamp,
                to_metric_time(ts=timestamp_diff, src=Nanosecond, dst=Millisecond),
                to_metric_time(ts=self.MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF, src=Nanosecond, dst=Millisecond),
            )
            return None
        elif timestamp_diff > self.MAX_LIDAR_STEREO_CAM_TIMESTAMP_DIFF and camera_name in STEREO_CAMERA_LIST:
            # convert to nanoseconds->milliseconds for readability
            logger.warning(
                "No corresponding stereo image at %s: %s > %s ms",
                lidar_timestamp,
                to_metric_time(ts=timestamp_diff, src=Nanosecond, dst=Millisecond),
                to_metric_time(ts=self.MAX_LIDAR_STEREO_CAM_TIMESTAMP_DIFF, src=Nanosecond, dst=Millisecond),
            )
            return None
        return closest_cam_ch_timestamp


if __name__ == "__main__":
    # example usage

    root_dir = "./"
    camera = "ring_front_center"
    id = "c6911883-1843-3727-8eaa-41dc8cda8993"

    db = SynchronizationDB(root_dir)
    get_timestamps_from_sensor_folder(root_dir)
    db.get_valid_logs()
    db.get_closest_cam_channel_timestamp(0, camera, id)
